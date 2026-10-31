"""The sdgf service: every pipeline step as a call that returns data instead of printing.

The CLI, and later the MCP server and Python harnesses, are thin adapters over Service,
so they can't drift apart. A call returns an Outcome (status + a JSON-ready dict) or
raises ServiceError for a request it can't act on; errors from the pipeline itself
(PipelineError, ArtefactError, ...) propagate unchanged.

    status            CLI exit code
    ok                0
    incomplete        1   a run stopped short, a failed gate, no release
    approval_pending  3   the coverage plan awaits a person's approval

Tasks are named through a TaskRegistry. allow_paths=True also accepts a task directory
or task.yaml path, which the CLI needs and an agent-facing adapter should not allow.

A run's options (RunOptions) are saved in its directory on first use, so resume,
evaluate and release reuse them; passing a different value for a saved option is an
error. Held-out paths, model overrides and tracing are run-time only and never saved.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from sdgf.coverage.axes import keyword_sources
from sdgf.coverage.plan import DEFAULT_SEED, load_or_build_plan, plan_stage_name
from sdgf.evaluation.gate import evaluate_gate
from sdgf.evaluation.metrics import metrics_for_run
from sdgf.hitl.queue import ApprovalPending, GoldSet, ReviewQueue, approve_plan, plan_approved
from sdgf.models.base import ModelBackend
from sdgf.models.registry import REGISTRY as MODELS
from sdgf.pipeline import DEFAULT_MAX_ROUNDS, Pipeline, RunResult
from sdgf.service.tasks import TaskRegistry
from sdgf.spec.compile import CompiledSpec, compile_spec
from sdgf.store.artefacts import ArtefactStore, new_run_id
from sdgf.tools.registry import REGISTRY as TOOLS
from sdgf.tracing import Tracer, make_tracer

log = logging.getLogger(__name__)

# The stage name predates the service; runs already on disk keep their options under it.
OPTIONS_STAGE = "cli_options"
RUN_OPTIONS: tuple[str, ...] = (
    "seed",
    "target_size",
    "plan_seed",
    "layers",
    "max_attempts_per_cell",
)
_RUN_DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "target_size": None,
    "plan_seed": DEFAULT_SEED,
    "layers": None,
    "max_attempts_per_cell": None,
}

MAX_PAGE = 100  # records per records() call
PROVENANCE_KEY = "_provenance"

BackendsFactory = Callable[[CompiledSpec], Mapping[str, ModelBackend]]
Status = Literal["ok", "incomplete", "approval_pending"]
TaskRef = str | Path


class ServiceError(ValueError):
    """A request the service can't act on: bad arguments, an unknown name, or run state.

    code is a stable identifier for callers to branch on; hint says what to do instead."""

    def __init__(self, message: str, *, code: str = "invalid_request", hint: str | None = None):
        super().__init__(message)
        self.code = code
        self.hint = hint

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "hint": self.hint}


@dataclass(frozen=True)
class Outcome:
    status: Status
    data: dict[str, Any]


@dataclass(frozen=True)
class RunOptions:
    """Options saved with a run on first use. None means the saved value, else the default."""

    seed: int | None = None
    target_size: int | None = None
    plan_seed: int | None = None
    layers: Sequence[str] | None = None
    max_attempts_per_cell: int | None = None

    def given(self) -> dict[str, Any]:
        given = {k: getattr(self, k) for k in RUN_OPTIONS}
        return {k: (list(v) if k == "layers" else v) for k, v in given.items() if v is not None}


DEFAULT_OPTIONS = RunOptions()


@dataclass(frozen=True)
class TraceSettings:
    kind: str
    project: str | None = None  # default sdgf-<task name>
    tags: Sequence[str] = ()


class Service:
    def __init__(
        self,
        store: ArtefactStore | str | Path | None = None,
        *,
        tasks: TaskRegistry | None = None,
        allow_paths: bool = True,
        notify: Callable[[str], None] | None = None,
    ):
        self.store = (
            store if store is None or isinstance(store, ArtefactStore) else ArtefactStore(store)
        )
        self.tasks = tasks
        self.allow_paths = allow_paths
        # Messages a person running the step should see (e.g. where traces go).
        self.notify = notify if notify is not None else log.info

    # ── tasks ────────────────────────────────────────────────────

    def compile(self, task: TaskRef) -> CompiledSpec:
        return compile_spec(self.task_dir(task), tool_registry=TOOLS)

    def list_tasks(self) -> Outcome:
        if self.tasks is None:
            raise ServiceError("no task registry configured", code="no_registry")
        listed: list[dict[str, Any]] = []
        for name in self.tasks.names():
            try:
                compiled = self.compile(name)
            except Exception as e:  # noqa: BLE001 - one broken spec must not hide the others
                listed.append({"name": name, "error": f"{type(e).__name__}: {e}"})
                continue
            listed.append(
                {
                    "name": name,
                    "task_type": compiled.spec.task.type,
                    "spec_version": compiled.spec_version,
                    "description": compiled.spec.task.description,
                }
            )
        return Outcome("ok", {"tasks": listed})

    def validate_spec(self, task: TaskRef) -> Outcome:
        return Outcome("ok", {"ok": True, **_spec_summary(self.compile(task))})

    def describe_task(self, task: TaskRef) -> Outcome:
        compiled = self.compile(task)
        spec = compiled.spec
        return Outcome(
            "ok",
            {
                **_spec_summary(compiled),
                "description": spec.task.description,
                "target_size": spec.coverage.target_size,
                "axes": [
                    {"name": a.name, "source": a.source, "values": a.values}
                    for a in spec.coverage.axes
                ],
                "models": {
                    stage: {"backend": c.backend, "model": c.model, "hosting": c.hosting}
                    for stage, c in spec.models
                    if c is not None
                },
                "thresholds": spec.thresholds.model_dump(),
                "budget": spec.budget.model_dump(),
                "approval_required": spec.hitl.approve_coverage_plan,
            },
        )

    # ── coverage plan ────────────────────────────────────────────

    def plan(
        self,
        task: TaskRef,
        *,
        target_size: int | None = None,
        plan_seed: int = DEFAULT_SEED,
        approve_as: str | None = None,
        note: str | None = None,
        backends: BackendsFactory | None = None,
    ) -> Outcome:
        """Build the coverage plan, or load it when cached. approve_as names the person
        approving it as it is now; only a person may, so adapters for agents never pass it."""
        compiled = self.compile(task)
        store = self._require_store()
        stage = plan_stage_name(compiled, target_size, plan_seed)
        backend = None
        if keyword_sources(compiled.spec.coverage) and not store.has_shared(
            compiled.spec_version, stage
        ):
            # Only a keyword expansion needs a model, and only when no plan is cached yet.
            backend = _overrides(backends, compiled).get("expansion")
            if backend is None:
                config = compiled.spec.models.expansion
                if config is None:
                    raise ServiceError(
                        "the coverage plan needs keyword expansion; set models.expansion",
                        code="spec_incomplete",
                    )
                backend = MODELS.create(config)
                backend.setup()
        plan, built = load_or_build_plan(
            store, compiled, target_size=target_size, seed=plan_seed, backend=backend
        )
        if approve_as is not None:
            approve_plan(store, compiled.spec_version, stage, reviewer=approve_as, note=note)
        return Outcome(
            "ok",
            {
                "spec_version": compiled.spec_version,
                "plan_stage": stage,
                "path": str(store.shared_path(compiled.spec_version, stage)),
                "built": built,
                "target_size": plan.target_size,
                "cells": [{"id": c.id, "quota": c.quota} for c in plan.cells],
                "approval_required": compiled.spec.hitl.approve_coverage_plan,
                "approved": plan_approved(store, compiled.spec_version, stage),
            },
        )

    # ── runs ─────────────────────────────────────────────────────

    def run(
        self,
        task: TaskRef,
        run_id: str | None = None,
        *,
        options: RunOptions = DEFAULT_OPTIONS,
        held_out: Sequence[str | Path] | None = None,
        backends: BackendsFactory | None = None,
        trace: TraceSettings | None = None,
    ) -> Outcome:
        """Stages 0 to 3. Reusing an existing run_id continues that run."""
        compiled = self.compile(task)
        return self._run(compiled, run_id or new_run_id(), options, held_out, backends, trace)

    def resume(
        self,
        task: TaskRef,
        run_id: str | None = None,
        *,
        options: RunOptions = DEFAULT_OPTIONS,
        held_out: Sequence[str | Path] | None = None,
        backends: BackendsFactory | None = None,
        trace: TraceSettings | None = None,
    ) -> Outcome:
        """Continue run_id, or the latest run of the spec_version, with its saved options."""
        compiled = self.compile(task)
        run_id = self.existing_run_id(compiled, run_id)
        return self._run(compiled, run_id, options, held_out, backends, trace)

    def evaluate(
        self,
        task: TaskRef,
        run_id: str | None = None,
        *,
        options: RunOptions = DEFAULT_OPTIONS,
        held_out: Sequence[str | Path] | None = None,
        backends: BackendsFactory | None = None,
        waive: Iterable[str] = (),
    ) -> Outcome:
        """Stages 4 and 5 over a run, read only: metrics and the release check."""
        compiled = self.compile(task)
        store = self._require_store()
        run_id = self.existing_run_id(compiled, run_id)
        opts = self._run_options(compiled, run_id, options)
        pipe = self._pipeline(compiled, opts, held_out, backends, None)
        run = store.open_run(compiled.spec_version, run_id)
        if not run.has_stage("cells"):
            raise ServiceError(
                f"run {run_id!r} has not generated anything yet", code="run_not_started"
            )
        metrics = metrics_for_run(
            compiled,
            run,
            calibration=pipe.calibration,
            held_out_paths=pipe.held_out_paths,
            seed=pipe.seed,
        )
        gate = evaluate_gate(metrics, compiled.spec.thresholds, waive=list(waive))
        return Outcome(
            "ok" if gate.passed else "incomplete",
            {
                "run_id": run_id,
                "passed": gate.passed,
                "gate": gate.to_dict(),
                "metrics": metrics.to_dict(),
            },
        )

    def release(
        self,
        task: TaskRef,
        releases: str | Path,
        run_id: str | None = None,
        *,
        options: RunOptions = DEFAULT_OPTIONS,
        held_out: Sequence[str | Path] | None = None,
        backends: BackendsFactory | None = None,
        trace: TraceSettings | None = None,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        waive: Iterable[str] = (),
        version: str | None = None,
    ) -> Outcome:
        """Stages 0 to 5 with refill rounds; releases under releases on a passed gate."""
        compiled = self.compile(task)
        store = self._require_store()
        run_id = run_id or new_run_id()
        opts = self._run_options(compiled, run_id, options)
        pipe = self._pipeline(compiled, opts, held_out, backends, trace)
        self._save_options(store, compiled, run_id, opts)
        try:
            result = pipe.release(
                releases, run_id, max_rounds=max_rounds, waive=tuple(waive), version=version
            )
        except ApprovalPending as e:
            return _pending(e)
        return Outcome(
            "ok" if result.released else "incomplete",
            {
                "run_id": run_id,
                "released": result.released,
                "path": str(result.path),
                "stop_reason": result.stop_reason,
                "failing_metrics": result.gate.failing_metrics,
                "short_cells": dict(result.gate.short_cells),
                "rounds": result.rounds,
            },
        )

    # ── results ──────────────────────────────────────────────────

    def records(
        self,
        task: TaskRef,
        run_id: str | None = None,
        *,
        offset: int = 0,
        limit: int = 20,
        provenance: bool = False,
    ) -> Outcome:
        """One page of a run's accepted records (the latest run by default), without their
        audit trail unless provenance. Works while the run is still going."""
        if offset < 0 or not 1 <= limit <= MAX_PAGE:
            raise ServiceError(
                f"offset must be >= 0 and limit 1 to {MAX_PAGE}; got {offset}, {limit}",
                code="bad_page",
            )
        compiled = self.compile(task)
        run_id = self.existing_run_id(compiled, run_id)
        run = self._require_store().open_run(compiled.spec_version, run_id)
        rows = run.read_jsonl("accepted")
        page = rows[offset : offset + limit]
        if not provenance:
            page = [{k: v for k, v in r.items() if k != PROVENANCE_KEY} for r in page]
        end = offset + len(page)
        return Outcome(
            "ok",
            {
                "run_id": run_id,
                "total": len(rows),
                "offset": offset,
                "records": page,
                "next_offset": end if end < len(rows) else None,
            },
        )

    # ── human review queue ───────────────────────────────────────

    def review_list(self, task: TaskRef, run_id: str | None = None) -> Outcome:
        run_id, queue, _ = self._queue(task, run_id)
        return Outcome(
            "ok",
            {
                "run_id": run_id,
                "pending": [
                    {
                        "id": i,
                        "cell_id": d.get("cell_id"),
                        "layer": d.get("layer"),
                        "code": d.get("code"),
                        "intended_label": d.get("intended_label"),
                        "reason": d.get("reason"),
                    }
                    for i, d in queue.pending().items()
                ],
                "resolved": len(queue.resolutions()),
            },
        )

    def review_resolve(
        self,
        task: TaskRef,
        item: str,
        action: str,
        *,
        reviewer: str,
        run_id: str | None = None,
        reason: str | None = None,
        new_label: Any = None,
    ) -> Outcome:
        """A person's decision on one review item; only a person may, as with approve_as."""
        run_id, queue, gold = self._queue(task, run_id)
        resolution = queue.resolve(
            item, action, reviewer=reviewer, reason=reason, new_label=new_label
        )
        return Outcome("ok", {"run_id": run_id, "gold": str(gold.path), **resolution.to_dict()})

    # ── helpers ──────────────────────────────────────────────────

    def task_dir(self, task: TaskRef) -> Path:
        if self.tasks is not None and isinstance(task, str):
            found = self.tasks.find(task)
            if found is not None:
                return found
        if self.allow_paths:
            return Path(task)
        raise ServiceError(
            f"unknown task {str(task)!r}",
            code="unknown_task",
            hint="list_tasks names the tasks available",
        )

    def _require_store(self) -> ArtefactStore:
        if self.store is None:
            raise ServiceError("this call needs an artefact store", code="no_store")
        return self.store

    def _run(
        self,
        compiled: CompiledSpec,
        run_id: str,
        options: RunOptions,
        held_out: Sequence[str | Path] | None,
        backends: BackendsFactory | None,
        trace: TraceSettings | None,
    ) -> Outcome:
        store = self._require_store()
        opts = self._run_options(compiled, run_id, options)
        pipe = self._pipeline(compiled, opts, held_out, backends, trace)
        self._save_options(store, compiled, run_id, opts)
        try:
            result = pipe.run(run_id)
        except ApprovalPending as e:
            return _pending(e)
        return Outcome("ok" if result.complete else "incomplete", _run_report(result))

    def _run_options(
        self, compiled: CompiledSpec, run_id: str, options: RunOptions
    ) -> dict[str, Any]:
        """The run's saved options, checked against any given explicitly, else the given ones."""
        store = self._require_store()
        given = options.given()
        if run_id in store.runs(compiled.spec_version):
            run = store.open_run(compiled.spec_version, run_id)
            if run.has_stage(OPTIONS_STAGE):
                saved = run.read_stage(OPTIONS_STAGE)
                clash = sorted(k for k, v in given.items() if saved.get(k) != v)
                if clash:
                    raise ServiceError(
                        f"run {run_id!r} was started with "
                        + ", ".join(f"{k}={saved.get(k)!r}" for k in clash)
                        + "; start a new run to change them",
                        code="options_clash",
                    )
                return {k: saved.get(k, _RUN_DEFAULTS[k]) for k in RUN_OPTIONS}
        return {**_RUN_DEFAULTS, **given}

    @staticmethod
    def _save_options(
        store: ArtefactStore, compiled: CompiledSpec, run_id: str, options: Mapping[str, Any]
    ) -> None:
        """Called once the pipeline is built, so bad arguments leave no run directory behind."""
        run = store.open_run(compiled.spec_version, run_id)
        if not run.has_stage(OPTIONS_STAGE):
            run.write_stage(OPTIONS_STAGE, dict(options))

    def _pipeline(
        self,
        compiled: CompiledSpec,
        options: Mapping[str, Any],
        held_out: Sequence[str | Path] | None,
        backends: BackendsFactory | None,
        trace: TraceSettings | None,
    ) -> Pipeline:
        return Pipeline(
            compiled,
            self._require_store(),
            model_overrides=_overrides(backends, compiled),
            seed=options["seed"],
            target_size=options["target_size"],
            plan_seed=options["plan_seed"],
            layers=options["layers"],
            max_attempts_per_cell=options["max_attempts_per_cell"],
            held_out_paths=held_out or None,
            ignore_unused_overrides=True,
            tracer=self._tracer(trace, compiled),
        )

    def _tracer(self, trace: TraceSettings | None, compiled: CompiledSpec) -> Tracer | None:
        if trace is None:
            return None
        project = trace.project or f"sdgf-{compiled.spec.task.name}"
        tracer = make_tracer(trace.kind, project=project, tags=trace.tags)
        sink = tracer.sink or {}
        self.notify(
            f"tracing to {trace.kind} ({sink.get('endpoint')}), project {project!r}, "
            f"{sink.get('content', 'full')} content"
        )
        return tracer

    def existing_run_id(self, compiled: CompiledSpec, run_id: str | None) -> str:
        """run_id if given and present, else the latest run of this spec_version."""
        store = self._require_store()
        if run_id is not None:
            if run_id not in store.runs(compiled.spec_version):
                raise ServiceError(
                    f"no run {run_id!r} for spec_version {compiled.spec_version}",
                    code="unknown_run",
                )
            return run_id
        latest = store.latest_run(compiled.spec_version)
        if latest is None:
            raise ServiceError(
                f"no runs for spec_version {compiled.spec_version} in {store.root}",
                code="unknown_run",
            )
        return latest

    def _queue(self, task: TaskRef, run_id: str | None) -> tuple[str, ReviewQueue, GoldSet]:
        compiled = self.compile(task)
        store = self._require_store()
        run_id = self.existing_run_id(compiled, run_id)
        run = store.open_run(compiled.spec_version, run_id)
        gold = GoldSet.for_task(store, compiled.spec.task.name)
        return run_id, ReviewQueue.for_run(run, gold), gold


def _spec_summary(compiled: CompiledSpec) -> dict[str, Any]:
    spec = compiled.spec
    return {
        "task": spec.task.name,
        "task_version": spec.task.version,
        "task_type": spec.task.type,
        "generation_mode": spec.task.generation_mode,
        "spec_version": compiled.spec_version,
        "seeds": len(compiled.seeds),
        "layers": list(spec.validation.layers),
    }


def _overrides(backends: BackendsFactory | None, compiled: CompiledSpec) -> dict[str, ModelBackend]:
    if backends is None:
        return {}
    overrides = dict(backends(compiled))
    bad = [st for st, b in overrides.items() if not isinstance(b, ModelBackend)]
    if bad:
        raise ServiceError(f"backends returned non-backends for stages {bad}")
    return overrides


def _pending(e: ApprovalPending) -> Outcome:
    return Outcome("approval_pending", {"message": str(e), "path": str(e.path)})


def _run_report(result: RunResult) -> dict[str, Any]:
    n = sum(result.counts.values())
    cost = result.usage.get("total", {}).get("cost_usd")
    return {
        "run_id": result.run.run_id,
        "run_dir": str(result.run.path),
        "spec_version": result.run.spec_version,
        "accepted_this_invocation": len(result.accepted),
        "accepted": n,
        "counts": result.counts,
        "stop_reason": result.stop_reason,
        "complete": result.complete,
        "layers": list(result.layers),
        "skipped_layers": list(result.skipped_layers),
        "drops_by_layer": result.drops.by_layer(),
        "usage": result.usage.get("total"),
        "cost_per_accepted": None if cost is None or not n else cost / n,
    }
