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
awaits approval (sdgf plan --approve). Each command is a thin adapter over sdgf.service:
it maps arguments to a Service call and prints the Outcome.

Models come from spec.models. --backends MODULE:ATTR (or path/to/file.py:ATTR) names a
callable taking the CompiledSpec and returning {stage: ModelBackend}, which replaces
those stages; that is how the mock backend is used from the command line. Stages the run
doesn't use are ignored, so one plugin can serve every command.

A run's options (seed, target size, plan seed, layers, per-cell attempt cap) are saved in
its directory as cli_options.json on first use, so resume, evaluate and release reuse
them; passing a different value for a saved option is an error. Held-out paths are
never saved (they stay run-time only), so pass --held-out again on each command.

--trace langsmith sends a trace per candidate to LangSmith (LANGSMITH_API_KEY, optional
LANGSMITH_ENDPOINT). Like --held-out it is run-time only: a run can be resumed with
tracing on or off. A traced run records LangSmith as a data destination.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sdgf.coverage.plan import DEFAULT_SEED
from sdgf.hitl.queue import ApprovalPending
from sdgf.pipeline import DEFAULT_MAX_ROUNDS
from sdgf.service import (
    OPTIONS_STAGE,
    BackendsFactory,
    Outcome,
    RunOptions,
    Service,
    TraceSettings,
)
from sdgf.tracing import LANGSMITH

__all__ = ["OPTIONS_STAGE", "build_parser", "load_backends", "main"]

EXIT_OK = 0
EXIT_INCOMPLETE = 1
EXIT_ERROR = 2
EXIT_APPROVAL = 3

_EXIT = {"ok": EXIT_OK, "incomplete": EXIT_INCOMPLETE, "approval_pending": EXIT_APPROVAL}


class CLIError(ValueError):
    """Bad arguments or state the command can't act on."""


# ── helpers ──────────────────────────────────────────────────────


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True, default=str))


def _emit(outcome: Outcome) -> int:
    if outcome.status == "approval_pending":
        print(f"sdgf: {outcome.data['message']}", file=sys.stderr)
    else:
        _print(outcome.data)
    return _EXIT[outcome.status]


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


def _backends(args: argparse.Namespace) -> BackendsFactory | None:
    """--backends, loaded only when the service first needs a model, as before."""
    ref = getattr(args, "backends", None)
    if not ref:
        return None
    return lambda compiled: load_backends(ref)(compiled)


def _service(args: argparse.Namespace) -> Service:
    return Service(
        getattr(args, "store", None),
        notify=lambda message: print(f"sdgf: {message}", file=sys.stderr),
    )


def _options(args: argparse.Namespace) -> RunOptions:
    return RunOptions(
        seed=args.seed,
        target_size=args.target_size,
        plan_seed=args.plan_seed,
        layers=args.layers,
        max_attempts_per_cell=args.max_attempts_per_cell,
    )


def _trace(args: argparse.Namespace) -> TraceSettings | None:
    kind = getattr(args, "trace", None)
    if kind is None:
        return None
    return TraceSettings(kind, project=args.trace_project, tags=tuple(args.trace_tags or ()))


def _parse_label(text: str) -> Any:
    """A relabel value as JSON (true, 3, "x"), or the bare string when it isn't JSON."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# ── commands ─────────────────────────────────────────────────────


def cmd_validate_spec(args: argparse.Namespace) -> int:
    return _emit(_service(args).validate_spec(args.task))


def cmd_plan(args: argparse.Namespace) -> int:
    if args.approve and not args.reviewer:
        raise CLIError("--approve needs --reviewer")
    return _emit(
        _service(args).plan(
            args.task,
            target_size=args.target_size,
            plan_seed=args.plan_seed,
            approve_as=args.reviewer if args.approve else None,
            note=args.note,
            backends=_backends(args),
        )
    )


def cmd_run(args: argparse.Namespace) -> int:
    return _emit(
        _service(args).run(
            args.task,
            args.run_id,
            options=_options(args),
            held_out=args.held_out,
            backends=_backends(args),
            trace=_trace(args),
        )
    )


def cmd_resume(args: argparse.Namespace) -> int:
    return _emit(
        _service(args).resume(
            args.task,
            args.run_id,
            options=_options(args),
            held_out=args.held_out,
            backends=_backends(args),
            trace=_trace(args),
        )
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    return _emit(
        _service(args).evaluate(
            args.task,
            args.run_id,
            options=_options(args),
            held_out=args.held_out,
            backends=_backends(args),
            waive=args.waive,
        )
    )


def cmd_release(args: argparse.Namespace) -> int:
    return _emit(
        _service(args).release(
            args.task,
            args.releases,
            args.run_id,
            options=_options(args),
            held_out=args.held_out,
            backends=_backends(args),
            trace=_trace(args),
            max_rounds=args.max_rounds,
            waive=args.waive,
            version=args.version,
        )
    )


def cmd_review(args: argparse.Namespace) -> int:
    service = _service(args)
    if args.review_command == "list":
        return _emit(service.review_list(args.task, args.run_id))
    return _emit(
        service.review_resolve(
            args.task,
            args.item,
            args.action,
            reviewer=args.reviewer,
            run_id=args.run_id,
            reason=args.reason,
            new_label=None if args.new_label is None else _parse_label(args.new_label),
        )
    )


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

    def trace(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--trace",
            choices=[LANGSMITH],
            help="trace every candidate (model calls, tool calls, checks L1-L6) to this service",
        )
        p.add_argument("--trace-project", help="tracing project (default sdgf-<task name>)")
        p.add_argument("--trace-tags", nargs="+", metavar="TAG", help="extra tags on every trace")

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
    trace(p)
    p.set_defaults(func=cmd_run)

    p = task_command("resume", "continue a run with its saved options (default: the latest)")
    run_options(p)
    trace(p)
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
    trace(p)
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
