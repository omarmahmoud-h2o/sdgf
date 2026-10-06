"""Stress test a task: where does a whole run break? (README "Stress testing a task").

    python -m sdgf.stress TASK --backends FILE.py:ATTR [--timeout 60] [--out stress_report.md]
    python -m sdgf.stress TASK --real

Each scenario runs one small run (target 8) in its own process, with a fresh temp store
and a wall-clock timeout, after injecting one failure into a healthy run: a model reply
that is garbage or cut off, a backend error, a crashing hook, a cell that never fills,
a killed run, and so on. The --backends plugin supplies the healthy models (the same
format as `sdgf run --backends`; it must return at least a generator and a judge).
Faults wrap those backends, so sdgf itself is not changed.

Each scenario ends as one of

    COMPLETED              every quota met
    STOPPED <reason>       the run stopped on its own (stalled, budget:...)
    CRASHED <Error> at f:l the run raised; f:l is the innermost sdgf frame
    HUNG > Ns              still running after the timeout, so it was killed

and a verdict: OK when that is what the scenario expects, BUG otherwise, with one line
saying what should have happened. The report is the table; it never fails a build.

--real runs one small run (target 4) on the spec's own models instead. It asks before
starting, because it spends money and sends data to those models.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import stat
import sys
import tempfile
import time
import traceback
import urllib.error
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import yaml

from sdgf.models.base import ModelBackend, ModelBackendError, ModelResponse, ToolSpec

TARGET = 8
REAL_TARGET = 4
DEFAULT_TIMEOUT = 60.0
SENSITIVE = "Jane Citizen of 12 Example Street, Fakeville; internal note: overdraft review flagged"


class Kill(BaseException):
    """Stands in for a killed process (Ctrl-C / SIGKILL) part-way through a run."""


# ── faults ───────────────────────────────────────────────────────


class FaultyBackend(ModelBackend):
    """A backend that misbehaves on chosen calls. `when(n, temperature)` picks the calls
    (n counts from 1); `mode` says how. Every call's prompt is kept in `prompts`."""

    def __init__(
        self,
        inner: ModelBackend,
        mode: str | None = None,
        when: Callable[[int, float], bool] = lambda n, t: True,
        hosting: str | None = None,
    ):
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self.model = inner.model
        self.hosting = hosting or inner.hosting
        self.mode = mode
        self.when = when
        self.n = 0
        self.prompts: list[str] = []

    def setup(self) -> None:
        self.inner.setup()

    def call(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        tools: list[ToolSpec] | None = None,
    ) -> ModelResponse:
        self.n += 1
        self.prompts.append(prompt)
        fire = self.mode is not None and self.when(self.n, temperature)
        if fire and self.mode == "garbage":
            return ModelResponse(text="Sorry, I can't produce that as JSON right now.")
        if fire and self.mode == "empty":
            return ModelResponse(text="")
        if fire and self.mode == "raise":
            raise ModelBackendError(f"injected backend failure on call {self.n}")
        if fire and self.mode == "http429":
            raise urllib.error.HTTPError(
                "https://api.example.invalid/v1/chat/completions",
                429,
                "Too Many Requests",
                None,  # type: ignore[arg-type]
                None,
            )
        if fire and self.mode == "kill":
            raise Kill(f"killed on call {self.n}")
        if fire and self.mode == "drop_temperature":
            return self.inner.call(prompt, max_tokens, 0.0, tools)
        response = self.inner.call(prompt, max_tokens, temperature, tools)
        if fire and self.mode == "truncate" and response.text:
            return replace(response, text=response.text[: len(response.text) // 2])
        if fire and self.mode == "low_confidence" and response.text:
            reply = json.loads(response.text)
            reply["confidence"] = {k: 0.3 for k in reply.get("confidence", {"verdict": 0})}
            return replace(response, text=json.dumps(reply))
        return response


class FlipAtTemperature(ModelBackend):
    """A judge that answers differently at temperature 0.8, as a sampling model would."""

    def __init__(self, inner: ModelBackend):
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self.model, self.hosting = inner.model, inner.hosting

    def call(self, prompt, max_tokens, temperature, tools=None):  # type: ignore[no-untyped-def]
        response = self.inner.call(prompt, max_tokens, temperature, tools)
        if temperature == 0.8 and response.text:
            reply = json.loads(response.text)
            reply["verdict"] = "no_breach" if reply["verdict"] == "breach" else "breach"
            return replace(response, text=json.dumps(reply))
        return response


def every(k: int) -> Callable[[int, float], bool]:
    return lambda n, t: n % k == 0


def on_call(k: int) -> Callable[[int, float], bool]:
    return lambda n, t: n == k


def voting(n: int, t: float) -> bool:
    return t > 0  # L6 votes run at 0.7/0.8/0.9; L5 judges at the spec's 0.0


# ── scenarios ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    expect: str  # what should happen, shown on a BUG row
    faults: dict[str, tuple[str, Callable[[int, float], bool]]]  # stage -> (mode, when)
    ok: Callable[[dict[str, Any]], bool]
    edit_spec: Callable[[dict[str, Any]], None] | None = None
    hook_fault: tuple[str, type[BaseException], int] | None = None  # hook, error, on call
    special: str | None = None


def _completed(r: dict[str, Any]) -> bool:
    return r["result"] == "COMPLETED"


def _named_backend_error(r: dict[str, Any]) -> bool:
    # documented: a model API that fails stops the run with a named error; resume it
    return r["result"] == "COMPLETED" or r.get("error_type") in {
        "ModelBackendError",
        "JevRequestError",
    }


def _with_fallback_reasons(spec: dict[str, Any]) -> None:
    spec["rubric"]["reason_required"] = "flagged"
    spec["models"]["fallback_judge"] = dict(spec["models"]["judge"])


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "1",
        "generator returns non-JSON every time",
        "stop soon as stalled, naming the cells",
        {"generator": ("garbage", lambda n, t: True)},
        lambda r: r["result"] == "STOPPED" and r["stop_reason"] == "stalled",
    ),
    Scenario(
        "2",
        "generator returns empty text (every 3rd call)",
        "send back, then carry on",
        {"generator": ("empty", every(3))},
        _completed,
    ),
    Scenario(
        "3",
        "generator reply cut off mid-JSON (every 3rd call)",
        "send back, then carry on",
        {"generator": ("truncate", every(3))},
        _completed,
    ),
    Scenario(
        "4",
        "generator backend error on call 3",
        "stop with a named backend error (resumable), or skip the candidate",
        {"generator": ("raise", on_call(3))},
        _named_backend_error,
    ),
    Scenario(
        "5",
        "generator HTTP 429 once (as openai_compat raises it)",
        "retry a transient rate limit and finish",
        {"generator": ("http429", on_call(3))},
        _completed,
    ),
    Scenario(
        "6",
        "judge returns garbage (every 3rd call)",
        "drop those as judge_error and finish",
        {"judge": ("garbage", every(3))},
        _completed,
    ),
    Scenario(
        "7",
        "judge backend error on call 4",
        "stop with a named backend error (resumable), or drop the candidate",
        {"judge": ("raise", on_call(4))},
        _named_backend_error,
    ),
    Scenario(
        "8",
        "judge confidence always low (0.3)",
        "escalate to L6 votes and finish",
        {"judge": ("low_confidence", lambda n, t: True)},
        _completed,
    ),
    Scenario(
        "9",
        "reason writer (fallback_judge) raises",
        "keep the verdict without a reason and finish",
        {"fallback_judge": ("raise", lambda n, t: True)},
        _completed,
        edit_spec=_with_fallback_reasons,
    ),
    Scenario(
        "10",
        "an L6 voter raises (2nd vote)",
        "count the failed vote as an abstention and finish",
        {
            "judge": (
                "raise",
                lambda n, t, _c=[0]: t > 0 and (_c.__setitem__(0, _c[0] + 1) or _c[0] == 2),
            )
        },
        _completed,
    ),
    Scenario(
        "11a",
        "label_rule hook raises AttributeError once",
        "send the candidate back (a record error), not crash",
        {},
        _completed,
        hook_fault=("label_rule", AttributeError, 3),
    ),
    Scenario(
        "11b",
        "label_rule hook raises KeyError once",
        "send the candidate back (a record error), not crash",
        {},
        _completed,
        hook_fault=("label_rule", KeyError, 3),
    ),
    Scenario(
        "12",
        "one cell can never fill",
        "stop as stalled after a bounded number of tries, naming the cell",
        {},
        lambda r: r["result"] == "STOPPED" and r["stop_reason"] == "stalled",
        special="cell_never_fills",
    ),
    Scenario(
        "13",
        "run killed mid-way, then resumed",
        "resume finishes with every record once (none lost, none duplicated)",
        {},
        lambda r: r["result"] == "COMPLETED" and r["extra"].get("duplicates") == 0,
        special="kill_resume",
    ),
    Scenario(
        "14",
        "store folder not writable",
        "a clear error before any model call",
        {},
        lambda r: r.get("error_type") == "PermissionError" and r["calls"] == 0,
        special="read_only_store",
    ),
    Scenario(
        "15",
        "judge ignores temperature (L6 votes have no spread)",
        "detect that L6 votes are not independent (warn or refuse)",
        {"judge": ("drop_temperature", lambda n, t: True)},
        lambda r: r["extra"].get("identical_vote_share", 0) < 1.0 or r["extra"].get("warned"),
        special="temperature_judge",
    ),
    Scenario(
        "16",
        "sensitive seed text, external generator",
        "mask or refuse seed text the stage 0 scan can't recognise before it leaves",
        {},
        lambda r: not r["extra"].get("leaked"),
        special="sensitive_seed",
    ),
)


# ── one scenario, in a child process ─────────────────────────────


# Frames that only pass a call through (metering, concurrency caps, tracing, this tool),
# skipped so the location names the sdgf code that made the failing call.
_PASS_THROUGH = ("stress.py", "usage.py", f"models{os.sep}base.py", "tracing.py", "mock.py")


def _innermost_sdgf_frame(tb: Any) -> str | None:
    frames = [
        f
        for f in traceback.extract_tb(tb)
        if f"{os.sep}sdgf{os.sep}" in f.filename and not f.filename.endswith(_PASS_THROUGH)
    ]
    if not frames:
        return None
    f = frames[-1]
    return f"{Path(f.filename).name}:{f.lineno}"


def _sdgf_traceback(tb: Any) -> str:
    keep = [f for f in traceback.extract_tb(tb) if f"{os.sep}sdgf{os.sep}" in f.filename]
    return "".join(traceback.format_list(keep))


def _load_factory(ref: str):  # type: ignore[no-untyped-def]
    from sdgf.cli import load_backends

    target = ref.rpartition(":")[0]
    if target.endswith(".py"):
        # plugins such as tests/cli_backends.py import their neighbours
        sys.path.insert(0, str(Path(target).resolve().parent))
    return load_backends(ref)


def _copy_task(task: Path, root: Path, edit: Callable[[dict[str, Any]], None] | None) -> Path:
    src = task if task.is_dir() else task.parent
    dst = root / "task"
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "data"))
    if edit is not None:
        path = dst / "task.yaml"
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        edit(spec)
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return dst


def _add_sensitive_seed(task_dir: Path) -> None:
    path = task_dir / "seeds.jsonl"
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    seed = json.loads(lines[0])
    seed["id"] = "SEED-STRESS-SENSITIVE"
    seed["messages"][-1]["content"] += f" {SENSITIVE}."
    path.write_text("\n".join([json.dumps(seed), *lines]) + "\n", encoding="utf-8")


def _hooked(compiled: Any, name: str, error: type[BaseException], on: int) -> Any:
    original = getattr(compiled.hooks, name)
    count = [0]

    def faulty(record: Any) -> Any:
        count[0] += 1
        if count[0] == on:
            raise error(f"injected {error.__name__} in {name} on call {on}")
        return original(record)

    return replace(compiled, hooks=replace(compiled.hooks, **{name: faulty}))


def _summary(result: Any) -> dict[str, Any]:
    by_code = result.snapshot.get("drops", {}).get("by_code", {})
    return {
        "stop_reason": result.stop_reason,
        "kept": sum(result.counts.values()),
        "dropped": sum(result.snapshot.get("drops", {}).get("by_layer", {}).values()),
        "top_codes": dict(Counter(by_code).most_common(3)),
    }


def _vote_spread(records: list[dict[str, Any]]) -> dict[str, Any]:
    voted = identical = 0
    for r in records:
        for lr in r.get("_provenance", {}).get("layer_results", ()):
            votes = [b.get("vote") for b in lr.get("ballots", ())]
            if lr.get("layer") == "L6" and len(votes) > 1:
                voted += 1
                identical += len(set(votes)) == 1
    return {
        "voted_records": voted,
        "identical_vote_share": round(identical / voted, 2) if voted else 0.0,
        "warned": False,  # sdgf has no check for votes without spread
    }


def run_scenario(
    key: str, task: str, backends: str, workdir: str, target: int = TARGET
) -> dict[str, Any]:
    """Run one scenario in this process and describe how it ended."""
    from sdgf.pipeline import Pipeline
    from sdgf.spec.compile import compile_spec

    sc = next(s for s in SCENARIOS if s.key == key)
    root = Path(workdir)
    task_dir = _copy_task(Path(task), root, sc.edit_spec)
    if sc.special == "sensitive_seed":
        _add_sensitive_seed(task_dir)
    out: dict[str, Any] = {"result": None, "stop_reason": None, "calls": 0, "extra": {}}
    wrapped: dict[str, FaultyBackend] = {}
    try:
        compiled = compile_spec(task_dir)
        if sc.hook_fault is not None:
            compiled = _hooked(compiled, *sc.hook_fault)
        healthy = dict(_load_factory(backends)(compiled))
        if sc.special == "temperature_judge":
            healthy["judge"] = FlipAtTemperature(healthy["judge"])
        armed = [False]
        if sc.special == "cell_never_fills":
            compiled = _never_fill(compiled, armed)
        for stage in ("generator", "judge"):
            mode, when = sc.faults.get(stage, (None, lambda n, t: False))
            hosting = (
                "provider_api" if sc.special == "sensitive_seed" and stage == "generator" else None
            )
            wrapped[stage] = FaultyBackend(healthy[stage], mode, when, hosting)
        if "fallback_judge" in sc.faults:
            mode, when = sc.faults["fallback_judge"]
            wrapped["fallback_judge"] = FaultyBackend(healthy["judge"], mode, when)

        store = root / "store"
        if sc.special == "read_only_store":
            store.mkdir()
            store.chmod(stat.S_IREAD | stat.S_IEXEC)
            store = store / "inner"

        def pipeline(overrides: dict[str, ModelBackend]) -> Any:
            return Pipeline(compiled, store, model_overrides=overrides, target_size=target)

        if sc.special == "kill_resume":
            killer = dict(
                wrapped, generator=FaultyBackend(healthy["generator"], "kill", on_call(10))
            )
            try:
                pipeline(killer).run("stress")
            except Kill:
                out["extra"]["killed_after_calls"] = killer["generator"].n
            result = pipeline(wrapped).run("stress")
            lines = (result.run.path / "accepted.jsonl").read_text(encoding="utf-8").splitlines()
            bodies = [
                json.dumps(
                    {k: v for k, v in json.loads(ln).items() if k != "_provenance"}, sort_keys=True
                )
                for ln in lines
            ]
            out["extra"]["duplicates"] = len(bodies) - len(set(bodies))
        else:
            pipe = pipeline(wrapped)
            if sc.special == "cell_never_fills":
                pipe.plan()  # stage 1 also calls sampler_constraints; fail only afterwards
                armed[0] = True
            result = pipe.run("stress")
        out.update(_summary(result))
        out["result"] = "COMPLETED" if result.stop_reason == "complete" else "STOPPED"
        if sc.special == "temperature_judge":
            records = [
                json.loads(ln)
                for ln in (result.run.path / "accepted.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            out["extra"].update(_vote_spread(records))
        if sc.special == "cell_never_fills":
            out["extra"]["stalled_cells"] = result.snapshot.get("stalled_cells")
    except BaseException as e:  # a crash is a finding, not a tool failure
        if isinstance(e, KeyboardInterrupt):
            raise
        out["result"] = "CRASHED"
        out["error_type"] = type(e).__name__
        out["message"] = str(e)[:300]
        out["where"] = _innermost_sdgf_frame(e.__traceback__)
        out["traceback"] = _sdgf_traceback(e.__traceback__)
    finally:
        out["calls"] = sum(b.n for b in wrapped.values())
        if sc.special == "sensitive_seed" and "generator" in wrapped:
            out["extra"]["leaked"] = any(SENSITIVE in p for p in wrapped["generator"].prompts)
        if sc.special == "read_only_store":
            (root / "store").chmod(stat.S_IRWXU)
    return out


def _never_fill(compiled: Any, armed: list[bool]) -> Any:
    """Once armed (after planning), the first cell the run asks about is rejected by
    sampler_constraints every time, so it can never fill (each try is an invalid_cell
    drop); other cells are unchanged."""
    original = compiled.hooks.sampler_constraints
    first: list[str] = []

    def sampler(cell: dict[str, Any], rng: Any) -> Any:
        key = json.dumps(cell, sort_keys=True, default=str)
        if not armed[0]:
            return original(cell, rng) if original is not None else dict(cell)
        if not first:
            first.append(key)
        if key == first[0]:
            return None
        return original(cell, rng) if original is not None else dict(cell)

    return replace(compiled, hooks=replace(compiled.hooks, sampler_constraints=sampler))


def _child(key: str, task: str, backends: str, workdir: str, queue: Any) -> None:
    queue.put(run_scenario(key, task, backends, workdir))


def run_isolated(
    key: str, task: str, backends: str, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """run_scenario in a separate process; a run still going after `timeout` is HUNG."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"sdgf-stress-{key}-") as workdir:
        proc = ctx.Process(target=_child, args=(key, task, backends, workdir, queue))
        proc.start()
        proc.join(timeout)
        seconds = round(time.monotonic() - started, 1)
        if proc.is_alive():
            proc.kill()
            proc.join()
            return {"result": "HUNG", "seconds": seconds, "calls": None, "extra": {}}
        if queue.empty():
            return {
                "result": "CRASHED",
                "error_type": f"exit code {proc.exitcode}",
                "seconds": seconds,
                "calls": None,
                "extra": {},
            }
        out = queue.get()
        out["seconds"] = seconds
        return out


# ── report ───────────────────────────────────────────────────────


def describe(r: dict[str, Any], timeout: float) -> str:
    if r["result"] == "STOPPED":
        return f"STOPPED {r['stop_reason']}"
    if r["result"] == "CRASHED":
        where = f" at {r['where']}" if r.get("where") else ""
        return f"CRASHED {r.get('error_type')}{where}"
    if r["result"] == "HUNG":
        return f"HUNG > {timeout:g}s"
    return "COMPLETED"


def verdict(sc: Scenario, r: dict[str, Any]) -> bool:
    try:
        return bool(sc.ok(r))
    except Exception:
        return False


def report(
    task: str, rows: list[tuple[Scenario, dict[str, Any]]], timeout: float
) -> tuple[str, str]:
    """(terminal table, markdown report)."""
    bugs = sum(not verdict(sc, r) for sc, r in rows)
    head = f"{Path(task).name} stress: {len(rows)} scenarios, {bugs} BUG"
    lines = [head, f"{'#':>3}  {'scenario':<52} {'result':<58} {'kept/drop':>9}  verdict"]
    md = [
        f"# {head}",
        "",
        "| # | scenario | result | kept/drop | calls | verdict |",
        "|---|---|---|---|---|---|",
    ]
    details = []
    for sc, r in rows:
        ok = verdict(sc, r)
        res = describe(r, timeout)
        kd = f"{r.get('kept', '-')}/{r.get('dropped', '-')}"
        v = "OK" if ok else f"BUG  should {sc.expect}"
        lines.append(f"{sc.key:>3}  {sc.title[:52]:<52} {res[:58]:<58} {kd:>9}  {v}")
        md.append(
            f"| {sc.key} | {sc.title} | {res} | {kd} | {r.get('calls')} | {'OK' if ok else 'BUG: should ' + sc.expect} |"
        )
        if not ok or r.get("extra"):
            block = [f"### {sc.key}. {sc.title}", "", f"- result: {res} ({r.get('seconds')}s)"]
            if r.get("message"):
                block.append(f"- error: {r['message']}")
            if r.get("top_codes"):
                block.append(f"- top drop codes: {r['top_codes']}")
            if r.get("extra"):
                block.append(f"- measured: {json.dumps(r['extra'], sort_keys=True)}")
            if r.get("traceback"):
                block += ["", "```", r["traceback"].rstrip(), "```"]
            details.append("\n".join(block))
    if details:
        md += ["", "## Details", "", "\n\n".join(details)]
    return "\n".join(lines), "\n".join(md) + "\n"


# ── real-model pass ──────────────────────────────────────────────


def real_pass(task: str, timeout: float) -> dict[str, Any]:
    from sdgf.pipeline import Pipeline

    with tempfile.TemporaryDirectory(prefix="sdgf-stress-real-") as workdir:
        out: dict[str, Any] = {"extra": {}}
        started = time.monotonic()
        try:
            pipe = Pipeline(task, Path(workdir) / "store", target_size=REAL_TARGET)
            result = pipe.run("stress-real")
            out.update(_summary(result))
            out["result"] = "COMPLETED" if result.stop_reason == "complete" else "STOPPED"
            out["usage"] = result.usage.get("total", {})
            records = [
                json.loads(ln)
                for ln in (result.run.path / "accepted.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            out["extra"].update(_vote_spread(records))
        except Exception as e:
            out["result"] = "CRASHED"
            out["error_type"] = type(e).__name__
            out["message"] = str(e)[:300]
            out["where"] = _innermost_sdgf_frame(e.__traceback__)
        out["seconds"] = round(time.monotonic() - started, 1)
        return out


# ── entry point ──────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sdgf.stress", description=__doc__.split("\n\n")[0])
    p.add_argument("task", help="task directory")
    p.add_argument("--backends", help="healthy mock models, FILE.py:ATTR or MODULE:ATTR")
    p.add_argument("--real", action="store_true", help="one small run on the spec's own models")
    p.add_argument("--only", nargs="+", metavar="N", help="run only these scenarios")
    p.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds per scenario")
    p.add_argument("--out", default="stress_report.md", help="markdown report path")
    p.add_argument("--yes", action="store_true", help="don't ask before the real-model pass")
    args = p.parse_args(argv)

    if args.real:
        if not args.yes:
            answer = input(
                f"--real runs {REAL_TARGET} records on {args.task}'s configured models: it costs "
                "money and sends data to them. Continue? [y/N] "
            )
            if answer.strip().lower() != "y":
                return 1
        r = real_pass(args.task, args.timeout)
        print(json.dumps(r, indent=2, sort_keys=True, default=str))
        return 0

    if not args.backends:
        p.error("--backends is required for the mock scenarios (or use --real)")
    chosen = [s for s in SCENARIOS if not args.only or s.key in args.only]
    rows = []
    for sc in chosen:
        print(f"  running {sc.key}: {sc.title} ...", file=sys.stderr, flush=True)
        rows.append((sc, run_isolated(sc.key, args.task, args.backends, args.timeout)))
    table, md = report(args.task, rows, args.timeout)
    print(table)
    Path(args.out).write_text(md, encoding="utf-8")
    print(f"\nreport: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
