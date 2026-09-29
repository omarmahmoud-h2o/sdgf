"""The sdgf command line: one subcommand per pipeline step.

    sdgf validate-spec TASK                      stage 0: compile task.yaml + hooks + seeds
    sdgf plan TASK --store S [--approve ...]     stage 1: build or show the coverage plan
    sdgf run TASK --store S [--run-id R]         stages 0 to 3
    sdgf resume TASK --store S [--run-id R]      continue a run (the latest by default)
    sdgf evaluate TASK --store S [--run-id R]    stages 4 and 5 over a run, read only
    sdgf review TASK --store S list|resolve      the run's HITL review queue
    sdgf release TASK --store S --releases DIR   stages 0 to 5 with refill rounds

Every command prints one JSON object on stdout. Exit codes: 0 done, 1 finished but not
complete (a run stopped short, a failed gate, no release), 2 an error, 3 the coverage plan
awaits approval (sdgf plan --approve).

Models come from spec.models. --backends MODULE:ATTR (or path/to/file.py:ATTR) names a
callable taking the CompiledSpec and returning {stage: ModelBackend}, which replaces
those stages; that is how the mock backend is used from the command line. Stages the run
doesn't use are ignored, so one plugin can serve every command.

A run's options (seed, target size, plan seed, layers, per-cell attempt cap) are saved in
its directory as cli_options.json on first use, so resume, evaluate and release reuse
them; passing a different value for a saved option is an error. Held-out paths are
never saved (they stay run-time only), so pass --held-out again on each command.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from sdgf.coverage.axes import keyword_sources
from sdgf.coverage.plan import DEFAULT_SEED, load_or_build_plan, plan_stage_name
from sdgf.evaluation.gate import evaluate_gate
from sdgf.evaluation.metrics import metrics_for_run
from sdgf.hitl.queue import ApprovalPending, GoldSet, ReviewQueue, approve_plan, plan_approved
from sdgf.models.base import ModelBackend
from sdgf.models.registry import REGISTRY as MODELS
from sdgf.pipeline import DEFAULT_MAX_ROUNDS, Pipeline, RunResult
from sdgf.spec.compile import CompiledSpec, compile_spec
from sdgf.store.artefacts import ArtefactStore, new_run_id
from sdgf.tools.registry import REGISTRY as TOOLS

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_ERROR = 2
EXIT_APPROVAL = 3

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

BackendsFactory = Callable[[CompiledSpec], Mapping[str, ModelBackend]]


class CLIError(ValueError):
    """Bad arguments or state the command can't act on."""


# ── helpers ──────────────────────────────────────────────────────


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def load_backends(ref: str) -> BackendsFactory:
    """MODULE:ATTR or path/to/file.py:ATTR -> the callable it names."""
    target, sep, attr = ref.rpartition(":")
    if not sep or not target or not attr:
        raise CLIError(f"--backends {ref!r}: expected MODULE:ATTR or FILE.py:ATTR")
    if target.endswith(".py"):
        path = Path(target)
        if not path.is_file():
            raise CLIError(f"--backends {ref!r}: no file {path}")
        spec = importlib.util.spec_from_file_location(f"_sdgf_backends_{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    factory = getattr(module, attr, None)
    if not callable(factory):
        raise CLIError(f"--backends {ref!r}: {attr!r} is not a callable in {target}")
    return factory


def _overrides(args: argparse.Namespace, compiled: CompiledSpec) -> dict[str, ModelBackend]:
    if not getattr(args, "backends", None):
        return {}
    overrides = dict(load_backends(args.backends)(compiled))
    bad = [st for st, b in overrides.items() if not isinstance(b, ModelBackend)]
    if bad:
        raise CLIError(f"--backends returned non-backends for stages {bad}")
    return overrides


def _compile(args: argparse.Namespace) -> CompiledSpec:
    return compile_spec(args.task, tool_registry=TOOLS)


def _store(args: argparse.Namespace) -> ArtefactStore:
    return ArtefactStore(args.store)


def _run_exists(store: ArtefactStore, spec_version: str, run_id: str) -> bool:
    return run_id in store.runs(spec_version)


def _run_options(
    args: argparse.Namespace,
    store: ArtefactStore,
    compiled: CompiledSpec,
    run_id: str,
) -> dict[str, Any]:
    """The run's saved options, checked against any passed explicitly, else the given ones."""
    given = {k: getattr(args, k, None) for k in RUN_OPTIONS}
    given = {k: (list(v) if k == "layers" else v) for k, v in given.items() if v is not None}
    if _run_exists(store, compiled.spec_version, run_id):
        run = store.open_run(compiled.spec_version, run_id)
        if run.has_stage(OPTIONS_STAGE):
            saved = run.read_stage(OPTIONS_STAGE)
            clash = sorted(k for k, v in given.items() if saved.get(k) != v)
            if clash:
                raise CLIError(
                    f"run {run_id!r} was started with "
                    + ", ".join(f"{k}={saved.get(k)!r}" for k in clash)
                    + "; start a new run to change them"
                )
            return {k: saved.get(k, _RUN_DEFAULTS[k]) for k in RUN_OPTIONS}
    return {**_RUN_DEFAULTS, **given}


def _save_options(
    store: ArtefactStore, compiled: CompiledSpec, run_id: str, options: Mapping[str, Any]
) -> None:
    """Called once the pipeline is built, so bad arguments leave no run directory behind."""
    run = store.open_run(compiled.spec_version, run_id)
    if not run.has_stage(OPTIONS_STAGE):
        run.write_stage(OPTIONS_STAGE, dict(options))


def _pipeline(
    args: argparse.Namespace,
    store: ArtefactStore,
    compiled: CompiledSpec,
    options: Mapping[str, Any],
) -> Pipeline:
    return Pipeline(
        compiled,
        store,
        model_overrides=_overrides(args, compiled),
        seed=options["seed"],
        target_size=options["target_size"],
        plan_seed=options["plan_seed"],
        layers=options["layers"],
        max_attempts_per_cell=options["max_attempts_per_cell"],
        held_out_paths=args.held_out or None,
        ignore_unused_overrides=True,
    )


def _existing_run_id(args: argparse.Namespace, store: ArtefactStore, compiled: CompiledSpec) -> str:
    """--run-id if given and present, else the latest run of this spec_version."""
    if args.run_id is not None:
        if not _run_exists(store, compiled.spec_version, args.run_id):
            raise CLIError(f"no run {args.run_id!r} for spec_version {compiled.spec_version}")
        return args.run_id
    latest = store.latest_run(compiled.spec_version)
    if latest is None:
        raise CLIError(f"no runs for spec_version {compiled.spec_version} in {store.root}")
    return latest


def _run_report(result: RunResult) -> dict[str, Any]:
    return {
        "run_id": result.run.run_id,
        "run_dir": str(result.run.path),
        "spec_version": result.run.spec_version,
        "accepted_this_invocation": len(result.accepted),
        "accepted": sum(result.counts.values()),
        "counts": result.counts,
        "stop_reason": result.stop_reason,
        "complete": result.complete,
        "layers": list(result.layers),
        "skipped_layers": list(result.skipped_layers),
        "drops_by_layer": result.drops.by_layer(),
    }


def _parse_label(text: str) -> Any:
    """A relabel value as JSON (true, 3, "x"), or the bare string when it isn't JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ── commands ─────────────────────────────────────────────────────


def cmd_validate_spec(args: argparse.Namespace) -> int:
    compiled = _compile(args)
    spec = compiled.spec
    _print(
        {
            "ok": True,
            "task": spec.task.name,
            "task_version": spec.task.version,
            "task_type": spec.task.type,
            "generation_mode": spec.task.generation_mode,
            "spec_version": compiled.spec_version,
            "seeds": len(compiled.seeds),
            "layers": list(spec.validation.layers),
        }
    )
    return EXIT_OK


def cmd_plan(args: argparse.Namespace) -> int:
    compiled = _compile(args)
    store = _store(args)
    stage = plan_stage_name(compiled, args.target_size, args.plan_seed)
    backend = None
    if keyword_sources(compiled.spec.coverage) and not store.has_shared(
        compiled.spec_version, stage
    ):
        # Only a keyword expansion needs a model, and only when no plan is cached yet.
        backend = _overrides(args, compiled).get("expansion")
        if backend is None:
            config = compiled.spec.models.expansion
            if config is None:
                raise CLIError("the coverage plan needs keyword expansion; set models.expansion")
            backend = MODELS.create(config)
            backend.setup()
    plan, built = load_or_build_plan(
        store, compiled, target_size=args.target_size, seed=args.plan_seed, backend=backend
    )
    if args.approve:
        if not args.reviewer:
            raise CLIError("--approve needs --reviewer")
        approve_plan(store, compiled.spec_version, stage, reviewer=args.reviewer, note=args.note)
    _print(
        {
            "spec_version": compiled.spec_version,
            "plan_stage": stage,
            "path": str(store.shared_path(compiled.spec_version, stage)),
            "built": built,
            "target_size": plan.target_size,
            "cells": [{"id": c.id, "quota": c.quota} for c in plan.cells],
            "approval_required": compiled.spec.hitl.approve_coverage_plan,
            "approved": plan_approved(store, compiled.spec_version, stage),
        }
    )
    return EXIT_OK


def _do_run(args: argparse.Namespace, run_id: str) -> int:
    compiled = _compile(args)
    store = _store(args)
    options = _run_options(args, store, compiled, run_id)
    pipe = _pipeline(args, store, compiled, options)
    _save_options(store, compiled, run_id, options)
    result = pipe.run(run_id)
    _print(_run_report(result))
    return EXIT_OK if result.complete else EXIT_INCOMPLETE


def cmd_run(args: argparse.Namespace) -> int:
    return _do_run(args, args.run_id or new_run_id())


def cmd_resume(args: argparse.Namespace) -> int:
    compiled = _compile(args)
    return _do_run(args, _existing_run_id(args, _store(args), compiled))


def cmd_evaluate(args: argparse.Namespace) -> int:
    compiled = _compile(args)
    store = _store(args)
    run_id = _existing_run_id(args, store, compiled)
    options = _run_options(args, store, compiled, run_id)
    pipe = _pipeline(args, store, compiled, options)
    run = store.open_run(compiled.spec_version, run_id)
    if not run.has_stage("cells"):
        raise CLIError(f"run {run_id!r} has not generated anything yet")
    metrics = metrics_for_run(
        compiled,
        run,
        calibration=pipe.calibration,
        held_out_paths=pipe.held_out_paths,
        seed=pipe.seed,
    )
    gate = evaluate_gate(metrics, compiled.spec.thresholds, waive=args.waive)
    _print(
        {
            "run_id": run_id,
            "passed": gate.passed,
            "gate": gate.to_dict(),
            "metrics": metrics.to_dict(),
        }
    )
    return EXIT_OK if gate.passed else EXIT_INCOMPLETE


def cmd_release(args: argparse.Namespace) -> int:
    compiled = _compile(args)
    store = _store(args)
    run_id = args.run_id or new_run_id()
    options = _run_options(args, store, compiled, run_id)
    pipe = _pipeline(args, store, compiled, options)
    _save_options(store, compiled, run_id, options)
    result = pipe.release(
        args.releases,
        run_id,
        max_rounds=args.max_rounds,
        waive=args.waive,
        version=args.version,
    )
    _print(
        {
            "run_id": run_id,
            "released": result.released,
            "path": str(result.path),
            "stop_reason": result.stop_reason,
            "failing_metrics": result.gate.failing_metrics,
            "short_cells": dict(result.gate.short_cells),
            "rounds": result.rounds,
        }
    )
    return EXIT_OK if result.released else EXIT_INCOMPLETE


def cmd_review(args: argparse.Namespace) -> int:
    compiled = _compile(args)
    store = _store(args)
    run_id = _existing_run_id(args, store, compiled)
    run = store.open_run(compiled.spec_version, run_id)
    gold = GoldSet.for_task(store, compiled.spec.task.name)
    queue = ReviewQueue.for_run(run, gold)
    if args.review_command == "list":
        pending = queue.pending()
        _print(
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
                    for i, d in pending.items()
                ],
                "resolved": len(queue.resolutions()),
            }
        )
        return EXIT_OK
    resolution = queue.resolve(
        args.item,
        args.action,
        reviewer=args.reviewer,
        reason=args.reason,
        new_label=None if args.new_label is None else _parse_label(args.new_label),
    )
    _print({"run_id": run_id, "gold": str(gold.path), **resolution.to_dict()})
    return EXIT_OK


# ── parser ───────────────────────────────────────────────────────


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdgf", description="Governed, spec-driven synthetic data generation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def task_command(name: str, help_: str, *, store: bool = True) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_, description=help_)
        p.add_argument("task", help="task directory or task.yaml path")
        if store:
            p.add_argument("--store", required=True, help="artefact store root")
        return p

    def models(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--backends",
            metavar="MODULE:ATTR",
            help="callable(CompiledSpec) -> {stage: ModelBackend} replacing spec.models stages",
        )

    def run_options(p: argparse.ArgumentParser) -> None:
        # Defaults are None so a resumed run's saved options can be told from explicit ones.
        p.add_argument("--run-id", help="run id; a new one is made if omitted")
        p.add_argument("--seed", type=int, help="run seed (default 0)")
        p.add_argument("--target-size", type=_positive, help="override coverage.target_size")
        p.add_argument("--plan-seed", type=int, help=f"coverage plan seed (default {DEFAULT_SEED})")
        p.add_argument("--layers", nargs="+", metavar="LAYER", help="validation layers to run")
        p.add_argument("--max-attempts-per-cell", type=_positive, help="per-cell attempt cap")
        p.add_argument(
            "--held-out",
            nargs="+",
            metavar="PATH",
            help="held-out files for the L4 check; never saved or sent to a model",
        )
        models(p)

    def waive(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--waive",
            nargs="+",
            default=[],
            metavar="THRESHOLD",
            help="thresholds whose metric may be unmeasured (never governance)",
        )

    p = task_command("validate-spec", "compile the spec (stage 0) and report it", store=False)
    p.set_defaults(func=cmd_validate_spec)

    p = task_command("plan", "build or show the coverage plan (stage 1)")
    p.add_argument("--target-size", type=_positive)
    p.add_argument("--plan-seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--approve", action="store_true", help="approve the plan as it is now")
    p.add_argument("--reviewer", help="who approves the plan")
    p.add_argument("--note", help="approval note")
    models(p)
    p.set_defaults(func=cmd_plan)

    p = task_command("run", "generate and validate a run (stages 0 to 3)")
    run_options(p)
    p.set_defaults(func=cmd_run)

    p = task_command("resume", "continue a run with its saved options (default: the latest)")
    run_options(p)
    p.set_defaults(func=cmd_resume)

    p = task_command("evaluate", "compute metrics and gate a run without writing (stages 4, 5)")
    run_options(p)
    waive(p)
    p.set_defaults(func=cmd_evaluate)

    p = task_command("release", "run, evaluate, gate and release, refilling short cells")
    p.add_argument("--releases", required=True, help="release root directory")
    p.add_argument("--max-rounds", type=_positive, default=DEFAULT_MAX_ROUNDS)
    p.add_argument("--version", help="release version (default derived from spec and run)")
    run_options(p)
    waive(p)
    p.set_defaults(func=cmd_release)

    p = task_command("review", "list or resolve a run's HITL review queue")
    p.add_argument("--run-id", help="run id (default: the latest)")
    review = p.add_subparsers(dest="review_command", required=True)
    review.add_parser("list", help="pending review items")
    r = review.add_parser("resolve", help="accept, reject or relabel one item")
    r.add_argument("item", help="review item id")
    r.add_argument("--action", required=True, choices=["accept", "reject", "relabel"])
    r.add_argument("--reviewer", required=True)
    r.add_argument("--reason")
    r.add_argument("--new-label", help="the label for relabel, as JSON (e.g. true)")
    p.set_defaults(func=cmd_review)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ApprovalPending as e:
        print(f"sdgf: {e}", file=sys.stderr)
        return EXIT_APPROVAL
    except (ValueError, RuntimeError, KeyError, OSError, ImportError) as e:
        print(f"sdgf: error: {e}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
