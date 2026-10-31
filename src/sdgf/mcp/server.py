"""sdgf as an MCP server: the tools an agent uses to run existing tasks.

    python -m sdgf.mcp --store S --tasks tasks/ --releases R            stdio (local agent)
    python -m sdgf.mcp --store S --tasks tasks/ --releases R --http     streamable HTTP

A thin adapter over sdgf.service and sdgf.jobs. The server's operator, not the agent,
fixes everything that costs money or is a person's decision: the store, the task roots,
where releases go, model overrides (--backends), job limits, and which thresholds an
agent may waive (--waivable; governance never). Agents name tasks; they never pass
paths. Approving a coverage plan and resolving review items are not tools: a person
does them with the sdgf CLI, and the tools report what is waiting.

Long steps (run, evaluate, release) are jobs: start_* returns a job id at once, and
job_status / wait_job report progress and the result. Results are brief unless
detail=True, and records come a page at a time, so replies fit an agent's context.

Quick tools (list_tasks, describe_task, validate_spec, plan, get_records, list_reviews)
compile the task in this process, which imports its hooks.py: serve only tasks you trust.

Authoring (with --drafts): agents create and edit task specs in a drafts workspace,
validate them and dry-run them as jobs, then submit them. A draft's hooks.py is code an
agent wrote: it is only ever imported in a subprocess that gets none of the server's
environment beyond the basics (no API keys), and dry runs use --dry-run-backends, are
capped at --dry-run-max-target records, run L1-L4 only, and write to the drafts'
own store. A person approves a submitted draft with `sdgf drafts approve`, which copies
it into a task directory; only then can start_run use it. Agents may write hooks.py only
with --hooks allow (the default over stdio; over --http the default is deny).

HTTP has no authentication yet, so it binds only to a loopback address.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from sdgf.drafts import DraftStore
from sdgf.jobs import DEFAULT_MAX_RUNNING, DEFAULT_TIMEOUT_S, JobManager
from sdgf.service import Outcome, RunOptions, Service, ServiceError, TaskRegistry

LOOPBACK = ("127.0.0.1", "localhost", "::1")
MAX_WAIT_S = 120.0

INSTRUCTIONS = """\
sdgf builds synthetic datasets from task specs. Typical flow:
1. list_tasks, then describe_task to see what a task produces and its release targets.
2. plan to see the coverage cells. If approval_required and not approved, a person must
   approve it before generation; tell the user (the hint has the command).
3. start_run (or start_release, which runs, evaluates and releases in rounds). Both return
   a job id at once; poll job_status or block briefly with wait_job.
4. get_records pages through accepted records; start_evaluate checks the release targets.
A job that was cancelled, timed out or lost continues where it stopped with restart_job.
To write a new task (when the draft tools are present): create_draft (often copying a
task with from_task), edit it with write_draft_file and append_seeds, check it with
validate_draft and dry_run_draft (a small free run on test models), then submit_draft.
A person approves or rejects it; read_draft shows a rejection's reason.
Errors are JSON with a code, a message and a hint saying what to do instead."""

# Keys of a step's result that stay in a brief job status; the rest need detail=True.
BRIEF_KEYS = frozenset(
    {
        "run_id",
        "accepted",
        "accepted_this_invocation",
        "complete",
        "stop_reason",
        "drops_by_layer",
        "usage",
        "cost_per_accepted",
        "passed",
        "released",
        "path",
        "failing_metrics",
        "short_cells",
        "message",
    }
)

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
STARTS_JOB = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
CHANGES_JOB = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class ServerConfig:
    store: Path
    task_roots: Sequence[Path]
    releases: Path | None = None
    backends: str | None = None  # MODULE:ATTR, as --backends on the CLI
    max_running: int = DEFAULT_MAX_RUNNING
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_target_size: int | None = None
    waivable: frozenset[str] = field(default_factory=frozenset)
    env: Mapping[str, str] = field(default_factory=dict)  # extra environment for job workers
    drafts: Path | None = None  # the drafts workspace; without it, no authoring tools
    allow_hooks: bool = False
    dry_run_backends: str | None = None  # MODULE:ATTR; without it, no dry runs
    dry_run_max_target: int = 10


def _tool_error(e: Exception) -> ToolError:
    if isinstance(e, ServiceError):
        error = e.to_dict()
    else:
        error = {"code": "step_failed", "message": f"{type(e).__name__}: {e}", "hint": None}
    return ToolError(json.dumps(error))


def _errors_as_tool_errors(fn: _F) -> _F:
    """Report the failures the CLI reports (a bad spec, an unknown run, ...) to the agent
    as an is_error result carrying the error's code, message and hint."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except (ValueError, RuntimeError, KeyError, OSError, ImportError) as e:
            raise _tool_error(e) from e

    return wrapper  # type: ignore[return-value]


def _outcome(outcome: Outcome) -> dict[str, Any]:
    return {"status": outcome.status, **outcome.data}


def _brief(status: dict[str, Any], detail: bool) -> dict[str, Any]:
    data = status.get("data")
    if detail or not isinstance(data, dict):
        return status
    brief = {k: v for k, v in data.items() if k in BRIEF_KEYS}
    gate = data.get("gate")
    if isinstance(gate, dict):
        brief["gate_failures"] = gate.get("failures", [])
    omitted = sorted(set(data) - set(brief) - {"gate"})
    return {**status, "data": brief, "omitted": omitted}


def build_server(config: ServerConfig) -> MCPServer:
    service = Service(config.store, tasks=TaskRegistry(config.task_roots), allow_paths=False)
    jobs = JobManager(
        config.store,
        service=service,
        releases=config.releases,
        backends=config.backends,
        max_running=config.max_running,
        timeout_s=config.timeout_s,
        max_target_size=config.max_target_size,
        env=config.env,
    )

    drafts: DraftStore | None = None
    dry_jobs: JobManager | None = None
    if config.drafts is not None:
        drafts = DraftStore(config.drafts, allow_hooks=config.allow_hooks, env=config.env)
        if config.dry_run_backends is not None:
            # A draft's hooks.py runs in these workers: no inherited environment, so no keys.
            dry_jobs = JobManager(
                drafts.store_root,
                releases=None,
                backends=config.dry_run_backends,
                max_running=config.max_running,
                timeout_s=config.timeout_s,
                max_target_size=config.dry_run_max_target,
                env=config.env,
                inherit_env=False,
            )

    def manager_for(job_id: str) -> JobManager:
        if dry_jobs is not None and (dry_jobs.root / job_id / "job.json").is_file():
            return dry_jobs
        return jobs

    def check_waivers(waive: Iterable[str] | None) -> list[str]:
        waive = list(waive or [])
        refused = sorted(set(waive) - config.waivable)
        if refused:
            raise ServiceError(
                f"this server doesn't let agents waive {refused}",
                code="waiver_not_allowed",
                hint=f"waivable here: {sorted(config.waivable)}; ask a person to release otherwise",
            )
        return waive

    def options(
        target_size: int | None = None,
        seed: int | None = None,
        layers: list[str] | None = None,
        max_attempts_per_cell: int | None = None,
    ) -> RunOptions:
        return RunOptions(
            seed=seed,
            target_size=target_size,
            layers=layers,
            max_attempts_per_cell=max_attempts_per_cell,
        )

    server = MCPServer("sdgf", instructions=INSTRUCTIONS)

    # ── tasks ────────────────────────────────────────────────────

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def list_tasks() -> dict[str, Any]:
        """The task specs this server can run, with their type and description."""
        return _outcome(service.list_tasks())

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def describe_task(task: str) -> dict[str, Any]:
        """A task's coverage axes, target size, models, release thresholds and budget."""
        return _outcome(service.describe_task(task))

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def validate_spec(task: str) -> dict[str, Any]:
        """Compile a task spec (task.yaml, hooks.py, seeds) and report its spec_version."""
        return _outcome(service.validate_spec(task))

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def plan(task: str, target_size: int | None = None) -> dict[str, Any]:
        """Build or show the coverage plan: the cells and how many records each needs.
        If approval_required and not approved, generation waits for a person."""
        result = _outcome(service.plan(task, target_size=target_size))
        if result["approval_required"] and not result["approved"]:
            size = f" --target-size {target_size}" if target_size else ""
            result["hint"] = (
                "a person must approve this plan before any run: "
                f"sdgf plan {task} --store {config.store}{size} --approve --reviewer NAME"
            )
        return result

    # ── jobs ─────────────────────────────────────────────────────

    @server.tool(annotations=STARTS_JOB)
    @_errors_as_tool_errors
    def start_run(
        task: str,
        target_size: int | None = None,
        seed: int | None = None,
        layers: list[str] | None = None,
        max_attempts_per_cell: int | None = None,
    ) -> dict[str, Any]:
        """Start generating and checking records (stages 0 to 3) as a job; returns its
        status at once. Calls models, so it costs what the task's models cost."""
        opts = options(target_size, seed, layers, max_attempts_per_cell)
        return jobs.start("run", task, options=opts)

    @server.tool(annotations=STARTS_JOB)
    @_errors_as_tool_errors
    def start_release(
        task: str,
        run_id: str | None = None,
        target_size: int | None = None,
        seed: int | None = None,
        waive: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start a job that runs, evaluates and releases in rounds, refilling short cells,
        and writes a release only if every threshold passes. run_id continues that run.
        waive lists thresholds that may stay unmeasured, among those this server allows."""
        return jobs.start(
            "release",
            task,
            run_id=run_id,
            options=options(target_size, seed),
            waive=check_waivers(waive),
        )

    @server.tool(annotations=STARTS_JOB)
    @_errors_as_tool_errors
    def start_evaluate(
        task: str, run_id: str | None = None, waive: list[str] | None = None
    ) -> dict[str, Any]:
        """Start a job computing a run's metrics and release check (the latest run by
        default), without writing a release."""
        return jobs.start("evaluate", task, run_id=run_id, waive=check_waivers(waive))

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def job_status(job_id: str, detail: bool = False) -> dict[str, Any]:
        """A job's state (running, finished, failed, timed_out, cancelled, lost), progress
        and result. detail=True includes the full result (e.g. every metric)."""
        return _brief(manager_for(job_id).status(job_id), detail)

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def wait_job(job_id: str, timeout_s: float = 30.0, detail: bool = False) -> dict[str, Any]:
        """Wait up to timeout_s seconds (at most 120) for a job to stop running, then
        return its status as job_status does."""
        timeout = min(max(timeout_s, 0.0), MAX_WAIT_S)
        return _brief(manager_for(job_id).wait(job_id, timeout=timeout), detail)

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def list_jobs() -> dict[str, Any]:
        """Every job, oldest first, with its state and progress; dry_run marks a draft's."""
        keys = ("job_id", "kind", "task", "state", "run_id", "created_at", "progress")
        listed = [{**{k: j[k] for k in keys}, "dry_run": False} for j in jobs.list()]
        if dry_jobs is not None:
            listed += [{**{k: j[k] for k in keys}, "dry_run": True} for j in dry_jobs.list()]
        return {"jobs": sorted(listed, key=lambda j: (j["created_at"], j["job_id"]))}

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Stop a running job. Records it already accepted are kept; restart_job continues."""
        return _brief(manager_for(job_id).cancel(job_id), False)

    @server.tool(annotations=STARTS_JOB)
    @_errors_as_tool_errors
    def restart_job(job_id: str) -> dict[str, Any]:
        """Continue a stopped job's run (cancelled, timed out, lost or failed) in a new job;
        nothing already accepted is generated again."""
        return manager_for(job_id).restart(job_id)

    # ── results ──────────────────────────────────────────────────

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def get_records(
        task: str, run_id: str | None = None, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """One page of a run's accepted records (the latest run by default), at most 100
        at a time, without their audit trail; next_offset is null on the last page."""
        return _outcome(service.records(task, run_id, offset=offset, limit=limit))

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def list_reviews(task: str, run_id: str | None = None) -> dict[str, Any]:
        """Records the judge flagged for a person to review; a person resolves them."""
        return _outcome(service.review_list(task, run_id))

    if drafts is not None:
        _authoring_tools(server, config, service, drafts, dry_jobs)
    return server


def _authoring_tools(
    server: MCPServer,
    config: ServerConfig,
    service: Service,
    drafts: DraftStore,
    dry_jobs: JobManager | None,
) -> None:
    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def list_drafts() -> dict[str, Any]:
        """Every task draft and its state: editing, submitted, rejected or approved."""
        return {"drafts": drafts.list()}

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def create_draft(name: str, from_task: str | None = None) -> dict[str, Any]:
        """Start a task draft, empty or a copy of an existing task's spec files."""
        source = service.task_dir(from_task) if from_task is not None else None
        return drafts.create(name, from_dir=source)

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def read_draft(name: str) -> dict[str, Any]:
        """A draft's task.yaml and hooks.py, a preview of its seeds, and its state (with a
        person's reason if they rejected it)."""
        return drafts.read(name)

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def write_draft_file(name: str, file: str, content: str) -> dict[str, Any]:
        """Replace one spec file of a draft: task.yaml, seeds.jsonl or hooks.py (if this
        server allows agents to write hooks). Any edit needs submitting again."""
        return drafts.write_file(name, file, content)

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def append_seeds(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Add example records to a draft's seeds.jsonl (at most 1000 per call)."""
        return drafts.append_seeds(name, records)

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def delete_draft_file(name: str, file: str) -> dict[str, Any]:
        """Remove one spec file from a draft (e.g. hooks.py, which is optional)."""
        return drafts.delete_file(name, file)

    @server.tool(annotations=READ_ONLY)
    @_errors_as_tool_errors
    def validate_draft(name: str) -> dict[str, Any]:
        """Compile a draft (in a separate process) and describe it, or say what is wrong.
        warnings include where seed text would be sent."""
        return drafts.validate(name)

    @server.tool(annotations=STARTS_JOB)
    @_errors_as_tool_errors
    def dry_run_draft(name: str, target_size: int | None = None) -> dict[str, Any]:
        """Start a small run of a draft on this server's test models, checks L1-L4 only,
        as a job (see job_status). Shows whether the spec produces records that pass."""
        if dry_jobs is None:
            raise ServiceError(
                "this server has no test models for dry runs",
                code="no_dry_runs",
                hint="validate_draft still checks that the draft compiles",
            )
        limit = config.dry_run_max_target
        if target_size is not None and not 1 <= target_size <= limit:
            raise ServiceError(
                f"a dry run makes 1 to {limit} records; got {target_size}", code="limit_exceeded"
            )
        opts = RunOptions(target_size=target_size or limit, layers=("L1", "L2", "L3", "L4"))
        return dry_jobs.start("run", drafts.spec_dir(name), options=opts, label=f"draft:{name}")

    @server.tool(annotations=CHANGES_JOB)
    @_errors_as_tool_errors
    def submit_draft(name: str) -> dict[str, Any]:
        """Validate a draft and submit exactly this content for a person to approve. Edits
        afterwards withdraw it."""
        return drafts.submit(name)


# ── command line ─────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m sdgf.mcp", description="sdgf as an MCP server")
    p.add_argument("--store", required=True, type=Path, help="artefact store root")
    p.add_argument("--tasks", required=True, nargs="+", type=Path, help="directories of task specs")
    p.add_argument("--releases", type=Path, help="release root; without it, no start_release")
    p.add_argument("--backends", metavar="MODULE:ATTR", help="model overrides for every job")
    p.add_argument("--max-running", type=int, default=DEFAULT_MAX_RUNNING)
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="per job, seconds")
    p.add_argument("--max-target-size", type=int)
    p.add_argument(
        "--waivable",
        nargs="+",
        default=[],
        metavar="THRESHOLD",
        help="thresholds agents may waive (governance can never be waived)",
    )
    p.add_argument("--drafts", type=Path, help="drafts workspace; enables the authoring tools")
    p.add_argument(
        "--hooks",
        choices=["allow", "deny"],
        help="may agents write a draft's hooks.py (default: allow over stdio, deny over --http)",
    )
    p.add_argument("--dry-run-backends", metavar="MODULE:ATTR", help="test models for dry runs")
    p.add_argument("--dry-run-max-target", type=int, default=10)
    p.add_argument("--http", action="store_true", help="streamable HTTP instead of stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.http and args.host not in LOOPBACK:
        print(
            f"sdgf: error: --host {args.host}: the HTTP server has no authentication yet, "
            f"so it binds only to {', '.join(LOOPBACK)}",
            file=sys.stderr,
        )
        return 2
    config = ServerConfig(
        store=args.store,
        task_roots=args.tasks,
        releases=args.releases,
        backends=args.backends,
        max_running=args.max_running,
        timeout_s=args.timeout,
        max_target_size=args.max_target_size,
        waivable=frozenset(args.waivable),
        drafts=args.drafts,
        allow_hooks=(args.hooks or ("deny" if args.http else "allow")) == "allow",
        dry_run_backends=args.dry_run_backends,
        dry_run_max_target=args.dry_run_max_target,
    )
    server = build_server(config)
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")
    return 0
