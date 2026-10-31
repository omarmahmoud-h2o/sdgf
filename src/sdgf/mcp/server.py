"""sdgf as an MCP server: the tools an agent uses to run, and write, task specs.

    python -m sdgf.mcp --store S --tasks tasks/ --releases R            stdio (local agent)
    python -m sdgf.mcp --store S --tasks tasks/ --releases R --http     streamable HTTP
    python -m sdgf.mcp ... --http --clients clients.json --host 0.0.0.0 --public-url URL

A thin adapter over sdgf.service and sdgf.jobs. The server's operator, not the agent,
fixes everything that costs money or is a person's decision: the store, the task roots,
where releases go, model overrides (--backends), held-out files, job limits, and which
thresholds an agent may waive (--waivable; governance never). Agents name tasks; they
never pass paths. Approving a coverage plan or a draft and resolving review items are
not tools: a person does them with the sdgf CLI, and the tools report what is waiting.

Long steps (run, evaluate, release) are jobs: start_* returns a job id at once, and
job_status / wait_job report progress and the result. Results are brief unless
detail=True, and records come a page at a time, so replies fit an agent's context.

Quick tools (list_tasks, describe_task, validate_spec, plan, get_records, list_reviews)
compile the task in this process, which imports its hooks.py: serve only tasks you trust.

Clients. Over stdio, or HTTP without --clients, every call is the "local" client. With
--clients (HTTP only) each call must carry a client's bearer token (sdgf.clients), and
a client sees and controls only its own jobs and drafts, under its own limits on running
jobs, tokens and cost; approved tasks are shared. Without --clients, HTTP binds only to a
loopback address. Bearer tokens travel in the clear over plain HTTP: put TLS in front.

Every tool call is appended to <store>/audit.jsonl: when, which client, which tool, its
arguments (long text and lists by size only) and the outcome.

Authoring (with --drafts): agents create and edit task specs in a drafts workspace (one
per client with --clients), validate them and dry-run them as jobs, then submit them. A
draft's hooks.py is code an agent wrote: it is only ever imported in a subprocess that
gets none of the server's environment beyond the basics (no API keys), and dry runs use
--dry-run-backends, are capped at --dry-run-max-target records, run L1-L4 only, and
write to the drafts' own store. A person approves a submitted draft with `sdgf drafts
approve`, which copies it into a task directory; only then can start_run use it. Agents
may write hooks.py only with --hooks allow (the default over stdio; over --http, deny),
or a client's allow_hooks.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from sdgf.clients import LOCAL, ClientPolicy, ClientRegistry, StaticTokenVerifier
from sdgf.drafts import DraftStore
from sdgf.jobs import DEFAULT_MAX_RUNNING, DEFAULT_TIMEOUT_S, RUNNING, JobManager
from sdgf.service import Outcome, RunOptions, Service, ServiceError, TaskRegistry, TraceSettings

LOOPBACK = ("127.0.0.1", "localhost", "::1")
MAX_WAIT_S = 120.0
AUDIT_FILE = "audit.jsonl"
AUDIT_TEXT_CHARS = 200  # longer string arguments are audited by size only
AUDIT_LIST_ITEMS = 20

INSTRUCTIONS = """\
sdgf builds synthetic datasets from task specs. Typical flow:
1. list_tasks, then describe_task to see what a task produces and its release targets.
2. plan to see the coverage cells. If approval_required and not approved, a person must
   approve it before generation; tell the user (the hint has the command).
3. start_run (or start_release, which runs, evaluates and releases in rounds). Both return
   a job id at once; poll job_status or block briefly with wait_job.
4. get_records pages through accepted records; start_evaluate checks the release targets.
A job that was cancelled, timed out or lost continues where it stopped with restart_job.
my_usage shows what your jobs have used against your limits.
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
    held_out: Sequence[Path] = ()  # the L4 held-out check for real runs, never for drafts
    trace: str | None = None  # a tracing service for every run and release job
    drafts: Path | None = None  # the drafts workspace; without it, no authoring tools
    allow_hooks: bool = False
    dry_run_backends: str | None = None  # MODULE:ATTR; without it, no dry runs
    dry_run_max_target: int = 10
    clients: ClientRegistry | None = None  # bearer-token clients; needs public_url
    public_url: str | None = None  # this server's base URL, e.g. https://sdgf.example


def _tool_error(e: Exception) -> ToolError:
    return ToolError(json.dumps(_error(e)))


def _error(e: Exception) -> dict[str, Any]:
    if isinstance(e, ServiceError):
        return e.to_dict()
    return {"code": "step_failed", "message": f"{type(e).__name__}: {e}", "hint": None}


def _audit_value(value: Any) -> Any:
    if isinstance(value, str) and len(value) > AUDIT_TEXT_CHARS:
        return {"chars": len(value)}
    if isinstance(value, list) and (
        len(value) > AUDIT_LIST_ITEMS or any(isinstance(v, dict | list) for v in value)
    ):
        return {"items": len(value)}
    return value


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


class _Workspace:
    """One client's drafts and the jobs that dry-run them."""

    def __init__(self, drafts: DraftStore, dry_jobs: JobManager | None):
        self.drafts = drafts
        self.dry_jobs = dry_jobs


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
        held_out=config.held_out,
    )
    audit_path = Path(config.store) / AUDIT_FILE
    audit_lock = threading.Lock()
    workspaces: dict[str, _Workspace] = {}
    local = ClientPolicy(LOCAL)

    # ── who is calling ───────────────────────────────────────────

    def caller() -> ClientPolicy:
        token = get_access_token()
        if config.clients is None:
            return local
        policy = config.clients.get(token.client_id) if token is not None else None
        if policy is None:
            raise ServiceError("this server needs a client token", code="unauthenticated")
        return policy

    def workspace(client: ClientPolicy) -> _Workspace | None:
        if config.drafts is None:
            return None
        if client.id not in workspaces:
            # Each client drafts in its own directory; the local client uses the root.
            root = config.drafts if config.clients is None else config.drafts / client.id
            allow = config.allow_hooks if client.allow_hooks is None else client.allow_hooks
            drafts = DraftStore(root, allow_hooks=allow, env=config.env)
            dry_jobs = None
            if config.dry_run_backends is not None:
                # A draft's hooks.py runs in these workers: no inherited environment, no keys.
                dry_jobs = JobManager(
                    drafts.store_root,
                    backends=config.dry_run_backends,
                    max_running=config.max_running,
                    timeout_s=config.timeout_s,
                    max_target_size=config.dry_run_max_target,
                    env=config.env,
                    inherit_env=False,
                )
            workspaces[client.id] = _Workspace(drafts, dry_jobs)
        return workspaces[client.id]

    def managers(client: ClientPolicy) -> list[JobManager]:
        ws = workspace(client)
        return [jobs] + ([ws.dry_jobs] if ws is not None and ws.dry_jobs is not None else [])

    def own_job(job_id: str) -> tuple[JobManager, dict[str, Any]]:
        """The caller's job and its manager; anyone else's job is unknown to them."""
        client = caller()
        for manager in managers(client):
            if (manager.root / job_id / "job.json").is_file():
                status = manager.status(job_id)
                if status["client"] == client.id:
                    return manager, status
        raise ServiceError(f"unknown job {job_id!r}", code="unknown_job")

    def usage(client: ClientPolicy) -> dict[str, Any]:
        totals = [m.usage(client.id) for m in managers(client)]
        running = sum(
            1 for m in managers(client) for j in m.list(client.id) if j["state"] == RUNNING
        )
        return {
            "client": client.id,
            "running": running,
            "runs": sum(t["runs"] for t in totals),
            "tokens": sum(t["tokens"] for t in totals),
            "cost_usd": sum(t["cost_usd"] for t in totals),
            "unpriced_runs": sum(t["unpriced_runs"] for t in totals),
            "limits": client.limits(),
        }

    def check_limits(client: ClientPolicy) -> None:
        used = usage(client)
        if client.max_running is not None and used["running"] >= client.max_running:
            raise ServiceError(
                f"{used['running']} of your jobs are running, your limit is {client.max_running}",
                code="too_many_jobs",
                hint="wait for or cancel one (list_jobs)",
            )
        over = [
            f"{key} {used[key]:g} of {limit:g}"
            for key, limit in (("tokens", client.max_tokens), ("cost_usd", client.max_cost_usd))
            if limit is not None and used[key] >= limit
        ]
        if over:
            raise ServiceError(
                f"your jobs have used {', '.join(over)}",
                code="budget_exhausted",
                hint="ask the server's operator to raise your limit",
            )

    def trace() -> TraceSettings | None:
        return TraceSettings(config.trace) if config.trace is not None else None

    # ── every tool: errors with codes, and an audit line ─────────

    def audit(entry: Mapping[str, Any]) -> None:
        line = json.dumps(entry, default=str, sort_keys=True) + "\n"
        with audit_lock:
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(line)

    def tool(fn: _F) -> _F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            entry: dict[str, Any] = {
                "at": datetime.now(UTC).isoformat(),
                "tool": fn.__name__,
                "args": {k: _audit_value(v) for k, v in kwargs.items()},
            }
            try:
                entry["client"] = caller().id
                result = fn(*args, **kwargs)
            except (ValueError, RuntimeError, KeyError, OSError, ImportError) as e:
                entry.update(ok=False, error=_error(e)["code"])
                audit(entry)
                raise _tool_error(e) from e
            entry["ok"] = True
            audit(entry)
            return result

        return wrapper  # type: ignore[return-value]

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

    def start(kind: str, task: str, **kwargs: Any) -> dict[str, Any]:
        client = caller()
        check_limits(client)
        return jobs.start(kind, task, client=client.id, **kwargs)

    auth = None
    verifier = None
    if config.clients is not None:
        if config.public_url is None:
            raise ServiceError("--clients needs --public-url", code="bad_config")
        base = config.public_url.rstrip("/")
        verifier = StaticTokenVerifier(config.clients)
        auth = AuthSettings(
            issuer_url=base, resource_server_url=f"{base}/mcp", validate_token_resource=False
        )
    server = MCPServer("sdgf", instructions=INSTRUCTIONS, token_verifier=verifier, auth=auth)

    # ── tasks ────────────────────────────────────────────────────

    @server.tool(annotations=READ_ONLY)
    @tool
    def list_tasks() -> dict[str, Any]:
        """The task specs this server can run, with their type and description."""
        return _outcome(service.list_tasks())

    @server.tool(annotations=READ_ONLY)
    @tool
    def describe_task(task: str) -> dict[str, Any]:
        """A task's coverage axes, target size, models, release thresholds and budget."""
        return _outcome(service.describe_task(task))

    @server.tool(annotations=READ_ONLY)
    @tool
    def validate_spec(task: str) -> dict[str, Any]:
        """Compile a task spec (task.yaml, hooks.py, seeds) and report its spec_version."""
        return _outcome(service.validate_spec(task))

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def plan(task: str, target_size: int | None = None) -> dict[str, Any]:
        """Build or show the coverage plan: the cells and how many records each needs.
        If approval_required and not approved, generation waits for a person."""
        result = _outcome(service.plan(task, target_size=target_size))
        if result["approval_required"] and not result["approved"]:
            size = f" --target-size {target_size}" if target_size else ""
            store = config.store if config.clients is None else "<store>"
            result["hint"] = (
                "a person must approve this plan before any run: "
                f"sdgf plan {task} --store {store}{size} --approve --reviewer NAME"
            )
        return result

    # ── jobs ─────────────────────────────────────────────────────

    @server.tool(annotations=STARTS_JOB)
    @tool
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
        return start("run", task, options=opts, trace=trace())

    @server.tool(annotations=STARTS_JOB)
    @tool
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
        return start(
            "release",
            task,
            run_id=run_id,
            options=options(target_size, seed),
            waive=check_waivers(waive),
            trace=trace(),
        )

    @server.tool(annotations=STARTS_JOB)
    @tool
    def start_evaluate(
        task: str, run_id: str | None = None, waive: list[str] | None = None
    ) -> dict[str, Any]:
        """Start a job computing a run's metrics and release check (the latest run by
        default), without writing a release."""
        return start("evaluate", task, run_id=run_id, waive=check_waivers(waive))

    @server.tool(annotations=READ_ONLY)
    @tool
    def job_status(job_id: str, detail: bool = False) -> dict[str, Any]:
        """A job's state (running, finished, failed, timed_out, cancelled, lost), progress
        and result. detail=True includes the full result (e.g. every metric)."""
        return _brief(own_job(job_id)[1], detail)

    @server.tool(annotations=READ_ONLY)
    @tool
    def wait_job(job_id: str, timeout_s: float = 30.0, detail: bool = False) -> dict[str, Any]:
        """Wait up to timeout_s seconds (at most 120) for a job to stop running, then
        return its status as job_status does."""
        manager, _ = own_job(job_id)
        timeout = min(max(timeout_s, 0.0), MAX_WAIT_S)
        return _brief(manager.wait(job_id, timeout=timeout), detail)

    @server.tool(annotations=READ_ONLY)
    @tool
    def list_jobs() -> dict[str, Any]:
        """Your jobs, oldest first, with their state and progress; dry_run marks a draft's."""
        client = caller()
        keys = ("job_id", "kind", "task", "state", "run_id", "created_at", "progress")
        listed = []
        for manager in managers(client):
            dry = manager is not jobs
            listed += [{**{k: j[k] for k in keys}, "dry_run": dry} for j in manager.list(client.id)]
        return {"jobs": sorted(listed, key=lambda j: (j["created_at"], j["job_id"]))}

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def cancel_job(job_id: str) -> dict[str, Any]:
        """Stop a running job. Records it already accepted are kept; restart_job continues."""
        manager, _ = own_job(job_id)
        return _brief(manager.cancel(job_id), False)

    @server.tool(annotations=STARTS_JOB)
    @tool
    def restart_job(job_id: str) -> dict[str, Any]:
        """Continue a stopped job's run (cancelled, timed out, lost or failed) in a new job;
        nothing already accepted is generated again."""
        manager, _ = own_job(job_id)
        check_limits(caller())
        return manager.restart(job_id)

    @server.tool(annotations=READ_ONLY)
    @tool
    def my_usage() -> dict[str, Any]:
        """What your jobs have used (tokens, cost, runs, running jobs) and your limits."""
        return usage(caller())

    # ── results ──────────────────────────────────────────────────

    @server.tool(annotations=READ_ONLY)
    @tool
    def get_records(
        task: str, run_id: str | None = None, offset: int = 0, limit: int = 20
    ) -> dict[str, Any]:
        """One page of a run's accepted records (the latest run by default), at most 100
        at a time, without their audit trail; next_offset is null on the last page."""
        return _outcome(service.records(task, run_id, offset=offset, limit=limit))

    @server.tool(annotations=READ_ONLY)
    @tool
    def list_reviews(task: str, run_id: str | None = None) -> dict[str, Any]:
        """Records the judge flagged for a person to review; a person resolves them."""
        return _outcome(service.review_list(task, run_id))

    if config.drafts is not None:
        _authoring_tools(server, config, service, tool, caller, workspace, check_limits)
    return server


def _authoring_tools(
    server: MCPServer,
    config: ServerConfig,
    service: Service,
    tool: Callable[[_F], _F],
    caller: Callable[[], ClientPolicy],
    workspace: Callable[[ClientPolicy], _Workspace | None],
    check_limits: Callable[[ClientPolicy], None],
) -> None:
    def drafts() -> DraftStore:
        ws = workspace(caller())
        assert ws is not None  # the tools exist only when config.drafts is set
        return ws.drafts

    @server.tool(annotations=READ_ONLY)
    @tool
    def list_drafts() -> dict[str, Any]:
        """Your task drafts and their state: editing, submitted, rejected or approved."""
        return {"drafts": drafts().list()}

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def create_draft(name: str, from_task: str | None = None) -> dict[str, Any]:
        """Start a task draft, empty or a copy of an existing task's spec files."""
        source = service.task_dir(from_task) if from_task is not None else None
        return drafts().create(name, from_dir=source)

    @server.tool(annotations=READ_ONLY)
    @tool
    def read_draft(name: str) -> dict[str, Any]:
        """A draft's task.yaml and hooks.py, a preview of its seeds, and its state (with a
        person's reason if they rejected it)."""
        return drafts().read(name)

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def write_draft_file(name: str, file: str, content: str) -> dict[str, Any]:
        """Replace one spec file of a draft: task.yaml, seeds.jsonl or hooks.py (if this
        server allows agents to write hooks). Any edit needs submitting again."""
        return drafts().write_file(name, file, content)

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def append_seeds(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
        """Add example records to a draft's seeds.jsonl (at most 1000 per call)."""
        return drafts().append_seeds(name, records)

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def delete_draft_file(name: str, file: str) -> dict[str, Any]:
        """Remove one spec file from a draft (e.g. hooks.py, which is optional)."""
        return drafts().delete_file(name, file)

    @server.tool(annotations=READ_ONLY)
    @tool
    def validate_draft(name: str) -> dict[str, Any]:
        """Compile a draft (in a separate process) and describe it, or say what is wrong.
        warnings include where seed text would be sent."""
        return drafts().validate(name)

    @server.tool(annotations=STARTS_JOB)
    @tool
    def dry_run_draft(name: str, target_size: int | None = None) -> dict[str, Any]:
        """Start a small run of a draft on this server's test models, checks L1-L4 only,
        as a job (see job_status). Shows whether the spec produces records that pass."""
        client = caller()
        ws = workspace(client)
        if ws is None or ws.dry_jobs is None:
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
        check_limits(client)
        opts = RunOptions(target_size=target_size or limit, layers=("L1", "L2", "L3", "L4"))
        spec = ws.drafts.spec_dir(name)
        return ws.dry_jobs.start("run", spec, options=opts, label=f"draft:{name}", client=client.id)

    @server.tool(annotations=CHANGES_JOB)
    @tool
    def submit_draft(name: str) -> dict[str, Any]:
        """Validate a draft and submit exactly this content for a person to approve. Edits
        afterwards withdraw it."""
        return drafts().submit(name)


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
    p.add_argument(
        "--held-out", nargs="+", default=[], type=Path, metavar="PATH", help="L4 held-out files"
    )
    p.add_argument("--trace", choices=["langsmith"], help="trace every run and release job")
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
    p.add_argument("--clients", type=Path, help="clients file (sdgf.clients); HTTP only")
    p.add_argument("--public-url", help="the URL clients reach this server at (with --clients)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    def fail(message: str) -> int:
        print(f"sdgf: error: {message}", file=sys.stderr)
        return 2

    if args.clients is not None and not args.http:
        return fail("--clients needs --http: stdio has no tokens to check")
    if args.http and args.clients is None and args.host not in LOOPBACK:
        return fail(
            f"--host {args.host}: without --clients the HTTP server has no authentication, "
            f"so it binds only to {', '.join(LOOPBACK)}"
        )
    try:
        clients = ClientRegistry.load(args.clients) if args.clients is not None else None
    except ServiceError as e:
        return fail(str(e))
    config = ServerConfig(
        store=args.store,
        task_roots=args.tasks,
        releases=args.releases,
        backends=args.backends,
        max_running=args.max_running,
        timeout_s=args.timeout,
        max_target_size=args.max_target_size,
        waivable=frozenset(args.waivable),
        held_out=args.held_out,
        trace=args.trace,
        drafts=args.drafts,
        allow_hooks=(args.hooks or ("deny" if args.http else "allow")) == "allow",
        dry_run_backends=args.dry_run_backends,
        dry_run_max_target=args.dry_run_max_target,
        clients=clients,
        public_url=args.public_url or f"http://{args.host}:{args.port}",
    )
    server = build_server(config)
    if args.http:
        server.run("streamable-http", host=args.host, port=args.port)
    else:
        server.run("stdio")
    return 0
