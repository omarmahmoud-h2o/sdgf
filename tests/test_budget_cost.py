"""Budget and cost tracking: tokens and estimated cost metered per model stage, a clean
and resumable stop when spec.budget runs out, and cost per accepted record reported."""

import json
import shutil
import threading
from pathlib import Path

import pytest
import yaml

from sdgf.evaluation.gate import evaluate_gate
from sdgf.evaluation.metrics import metrics_for_run
from sdgf.evaluation.reports import dataset_card, governance_report
from sdgf.generate.scheduler import Cell, Scheduler
from sdgf.models.base import ModelResponse, ToolCall
from sdgf.models.mock import MockBackend
from sdgf.models.usage import (
    USAGE_STAGE,
    MeteredBackend,
    Pricing,
    StageUsage,
    UsageLedger,
    UsageMeter,
    estimate_tokens,
)
from sdgf.pipeline import ACCEPTED_STREAM, Pipeline, PipelineError
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import BudgetSection, SpecValidationError, parse_spec
from sdgf.store.provenance import split
from test_pipeline import NO_JUDGE, fag_reply, recipe_from_prompt, valid_backend
from test_pipeline_concurrency import OrderFreeWorld

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 20
IN, OUT = 300, 700  # tokens the metered mock reports per call
# unmeasured without a judge or embedder; governance can't be waived
WAIVE = ("fidelity_min", "kappa_min", "residual_error_max", "semantic_diversity_min")


def fag_spec(root: Path, *, budget=None, generator=None, judge=None, concurrency=1):
    """A copy of the FAG task with the given budget and generator/judge config changes."""
    task = root / "fag"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))
    spec = yaml.safe_load((task / "task.yaml").read_text())
    if budget is not None:
        spec["budget"] = budget
    spec["models"]["generator"].update(generator or {})
    spec["models"]["judge"].update(judge or {})
    spec["models"]["generator"]["concurrency"] = concurrency
    spec["models"]["judge"]["concurrency"] = concurrency
    (task / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return compile_spec(task)


def counted_backend():
    """A valid FAG generator that reports its token counts, as provider APIs do."""

    def reply(call):
        text = json.dumps(fag_reply(recipe_from_prompt(call.prompt)))
        return ModelResponse(text=text, input_tokens=IN, output_tokens=OUT)

    return MockBackend(reply)


def pipeline(compiled, root, backend, **kw):
    return Pipeline(
        compiled,
        root / "store",
        model_overrides={"generator": backend},
        target_size=TARGET,
        layers=NO_JUDGE,
        **kw,
    )


# ── metering ────────────────────────────────────────────────────


def test_estimate_and_pricing():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert Pricing(2.0, 10.0).cost(1_000_000, 500_000) == pytest.approx(7.0)


def test_metered_backend_uses_reported_tokens_and_prices_them():
    meter = UsageMeter()
    inner = MockBackend([ModelResponse(text="hi", input_tokens=100, output_tokens=50)])
    backend = MeteredBackend(inner, "generator", meter, Pricing(1.0, 3.0))
    assert (backend.name, backend.model, backend.hosting) == ("mock", "mock", "local")
    assert backend.call("prompt", 10, 0.0).text == "hi"
    usage = meter.take_loose().stages["generator"]
    assert usage.to_dict() == {
        "calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "tokens": 150,
        "estimated_calls": 0,
        "cost_usd": pytest.approx(250 / 1e6),
    }
    assert meter.take_loose().stages == {}  # drained


def test_unreported_tokens_are_estimated_and_unpriced_cost_is_none():
    meter = UsageMeter()
    backend = MeteredBackend(MockBackend(["x" * 40, ToolCall("calc", {"a": 1})]), "judge", meter)
    backend.call("p" * 400, 10, 0.0)
    backend.call("q" * 8, 10, 0.0, tools=[{"name": "calc"}])
    usage = meter.take_loose().stages["judge"]
    assert usage.calls == usage.estimated_calls == 2
    assert usage.input_tokens > 100 + 2  # the offered tools count as input too
    assert usage.output_tokens > 10  # the tool call's JSON counts as output
    assert usage.cost_usd is None


def test_capture_is_per_thread_and_outside_calls_are_loose():
    meter = UsageMeter()
    backend = MeteredBackend(
        MockBackend([ModelResponse("r", input_tokens=1, output_tokens=1)], cycle=True),
        "generator",
        meter,
    )
    held = {}

    def slot(name, calls):
        with meter.capture() as ledger:
            for _ in range(calls):
                backend.call("p", 1, 0.0)
        held[name] = ledger

    threads = [threading.Thread(target=slot, args=(n, c)) for n, c in (("a", 3), ("b", 5))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    backend.call("p", 1, 0.0)
    assert held["a"].stages["generator"].calls == 3
    assert held["b"].stages["generator"].calls == 5
    assert meter.take_loose().stages["generator"].calls == 1


def test_ledger_sums_and_keeps_unpriced_as_none():
    ledger = UsageLedger()
    ledger.add(UsageLedger({"generator": StageUsage(1, 10, 5, 0, None)}))
    assert ledger.cost_usd is None and ledger.tokens == 15
    ledger.add(UsageLedger({"judge": StageUsage(2, 4, 4, 2, 0.5)}))
    assert ledger.cost_usd == 0.5 and ledger.total().calls == 3
    assert UsageLedger.from_dict(ledger.to_dict()).to_dict() == ledger.to_dict()


def test_prices_are_both_or_neither(fag_raw):
    fag_raw["models"]["generator"]["input_cost_per_mtok"] = 1.0
    del fag_raw["models"]["generator"]["output_cost_per_mtok"]
    with pytest.raises(SpecValidationError, match="output_cost_per_mtok"):
        parse_spec(fag_raw)


@pytest.fixture
def fag_raw():
    return yaml.safe_load((FAG_DIR / "task.yaml").read_text())


# ── the pipeline meters every stage ─────────────────────────────


def test_run_records_usage_per_stage(tmp_path):
    compiled = fag_spec(
        tmp_path, generator={"input_cost_per_mtok": 1.0, "output_cost_per_mtok": 2.0}
    )
    backend = counted_backend()
    result = pipeline(compiled, tmp_path, backend).run("u1")
    assert result.complete
    calls = len(backend.calls)
    ledger = result.run.read_stage(USAGE_STAGE)
    assert ledger == result.usage
    assert ledger["invocations"] == 1
    gen = ledger["stages"]["generator"]
    assert gen["calls"] == calls and gen["estimated_calls"] == 0
    assert gen["tokens"] == calls * (IN + OUT)
    assert gen["cost_usd"] == pytest.approx(calls * (IN * 1.0 + OUT * 2.0) / 1e6)
    assert ledger["total"]["tokens"] == gen["tokens"]
    summary = result.run.read_stage("summary")
    assert summary["usage"]["tokens"] == gen["tokens"]  # what the budget was charged
    assert summary["usage_by_stage"] == ledger["stages"]


def test_cost_per_accepted_record_in_metrics_and_card(tmp_path):
    compiled = fag_spec(
        tmp_path, generator={"input_cost_per_mtok": 1.0, "output_cost_per_mtok": 2.0}
    )
    backend = counted_backend()
    result = pipeline(compiled, tmp_path, backend).run("u1")
    overall = metrics_for_run(compiled, result.run).overall
    cost = len(backend.calls) * (IN + 2 * OUT) / 1e6
    assert overall.cost_per_record == pytest.approx(cost / TARGET)
    assert overall.tokens_per_record == len(backend.calls) * (IN + OUT) / TARGET
    stage = overall.usage_by_stage["generator"]
    assert stage["cost_per_record"] == pytest.approx(cost / TARGET)

    metrics = metrics_for_run(compiled, result.run)
    gate = evaluate_gate(metrics, compiled.spec.thresholds, waive=WAIVE)
    card = dataset_card(
        compiled, "v1", "u1", metrics, gate, governance_report(compiled, metrics, gate, [], []), "t"
    )
    assert "## Cost" in card
    assert f"| generator | {len(backend.calls)} | {stage['tokens']} | 0 |" in card


def test_unpriced_stages_leave_cost_unmeasured(tmp_path):
    compiled = fag_spec(
        tmp_path, generator={"input_cost_per_mtok": None, "output_cost_per_mtok": None}
    )
    result = pipeline(compiled, tmp_path, valid_backend()).run("u1")
    overall = metrics_for_run(compiled, result.run).overall
    assert overall.cost_per_record is None  # not priced is not $0
    assert overall.tokens_per_record > 0  # estimated from text
    assert overall.usage_by_stage["generator"]["estimated_calls"] > 0


def test_a_cost_budget_needs_every_used_stage_priced(tmp_path):
    compiled = fag_spec(
        tmp_path,
        budget={"max_cost_usd": 1.0},
        generator={"input_cost_per_mtok": None, "output_cost_per_mtok": None},
    )
    with pytest.raises(PipelineError, match=r"max_cost_usd.*\['generator'\]"):
        pipeline(compiled, tmp_path, valid_backend())


def test_judge_stage_is_metered_separately(tmp_path):
    compiled = fag_spec(tmp_path)
    world = OrderFreeWorld()
    backends = world.backends()
    pipe = Pipeline(compiled, tmp_path / "store", model_overrides=backends, target_size=TARGET)
    result = pipe.run("j1")
    assert result.complete
    stages = result.usage["stages"]
    assert stages["generator"]["calls"] == len(backends["generator"].calls)
    assert stages["judge"]["calls"] == len(backends["judge"].calls)
    assert stages["judge"]["cost_usd"] == 0.0  # FAG prices its local models at 0


def test_concurrent_and_sequential_runs_meter_the_same(tmp_path):
    ledgers = []
    for name, n in (("seq", 1), ("con", 6)):
        root = tmp_path / name
        pipe = Pipeline(
            fag_spec(root, concurrency=n),
            root / "store",
            model_overrides=OrderFreeWorld().backends(),
            target_size=TARGET,
        )
        ledgers.append(pipe.run("c1").usage["stages"])
    assert ledgers[0] == ledgers[1]


# ── budget stops are clean and resumable ────────────────────────


def test_token_budget_stops_cleanly_and_resumes(tmp_path):
    per_call = IN + OUT
    compiled = fag_spec(tmp_path, budget={"max_tokens": 6 * per_call})
    first = pipeline(compiled, tmp_path, counted_backend()).run("b1")
    assert first.stop_reason == "budget:max_tokens"
    assert 0 < len(first.accepted) < TARGET
    summary = first.run.read_stage("summary")
    assert summary["stop_reason"] == "budget:max_tokens"
    assert summary["usage"]["tokens"] >= 6 * per_call
    assert len(first.run.read_jsonl(ACCEPTED_STREAM)) == len(first.accepted)

    # Each resume gets the budget afresh and continues from what was accepted.
    results = [first]
    while not results[-1].complete:
        assert len(results) < TARGET, "resume made no progress"
        results.append(pipeline(compiled, tmp_path, counted_backend()).run("b1"))
        assert results[-1].run.resumed and results[-1].accepted
    stored = results[-1].run.read_jsonl(ACCEPTED_STREAM)
    assert len(stored) == TARGET
    assert len({split(r)[1].seed for r in stored}) == TARGET  # nothing regenerated
    assert results[-1].run.read_stage(USAGE_STAGE)["invocations"] == len(results)


def test_usage_ledger_accumulates_across_invocations(tmp_path):
    compiled = fag_spec(tmp_path, budget={"max_tokens": 6 * (IN + OUT)})
    per_invocation = []
    result = None
    while result is None or not result.complete:
        result = pipeline(compiled, tmp_path, counted_backend()).run("b1")
        per_invocation.append(result.run.read_stage("summary")["usage_by_stage"])
    ledger = result.run.read_stage(USAGE_STAGE)
    assert ledger["invocations"] == len(per_invocation)
    assert ledger["stages"]["generator"]["calls"] == sum(
        u["generator"]["calls"] for u in per_invocation
    )
    assert ledger["total"]["tokens"] == sum(u["generator"]["tokens"] for u in per_invocation)
    # stage 4 reads the whole run's cost, not the last invocation's
    overall = metrics_for_run(compiled, result.run).overall
    assert overall.tokens_per_record == ledger["total"]["tokens"] / TARGET


def test_cost_budget_stops_the_run(tmp_path):
    compiled = fag_spec(
        tmp_path,
        budget={"max_cost_usd": 0.01},
        generator={"input_cost_per_mtok": 5.0, "output_cost_per_mtok": 5.0},
    )
    result = pipeline(compiled, tmp_path, counted_backend()).run("b1")
    assert result.stop_reason == "budget:max_cost_usd"
    assert result.usage["total"]["cost_usd"] >= 0.01
    assert len(result.accepted) < TARGET


def test_budget_stop_ends_the_release_loop(tmp_path):
    compiled = fag_spec(tmp_path, budget={"max_tokens": 6 * (IN + OUT)})
    pipe = pipeline(compiled, tmp_path, counted_backend())
    released = pipe.release(tmp_path / "releases", "b1", waive=WAIVE)
    assert not released.released
    assert released.stop_reason == "budget:max_tokens"
    assert (released.run.path / "shortfall.json").is_file()


# ── wave sizing ─────────────────────────────────────────────────


def cells(n=10):
    return [Cell(f"c{i}", {}, 5) for i in range(n)]


def settle(scheduler, tokens=0, cost=0.0):
    cell = scheduler.next_cell()
    scheduler.charge(tokens=tokens, cost_usd=cost)
    scheduler.accept(cell.id)


def test_wave_allowance_is_unbounded_without_a_usage_budget():
    assert Scheduler(cells(), BudgetSection(max_candidates=5)).wave_allowance() is None


def test_wave_allowance_probes_then_divides_the_remaining_budget():
    scheduler = Scheduler(cells(), BudgetSection(max_tokens=1000))
    assert scheduler.wave_allowance() == 1  # nothing settled yet
    settle(scheduler, tokens=100)
    assert scheduler.wave_allowance() == 9  # 900 left at 100 per candidate
    for _ in range(8):
        settle(scheduler, tokens=100)
    assert scheduler.wave_allowance() == 1  # at least one while budget is left
    settle(scheduler, tokens=100)
    assert scheduler.next_cell() is None and scheduler.stop_reason == "budget:max_tokens"


def test_wave_allowance_takes_the_tightest_budget_and_ignores_free_stages():
    scheduler = Scheduler(cells(), BudgetSection(max_tokens=10_000, max_cost_usd=1.0))
    settle(scheduler, tokens=100, cost=0.25)
    assert scheduler.wave_allowance() == 3  # cost: 0.75 left at 0.25 each
    free = Scheduler(cells(), BudgetSection(max_cost_usd=1.0))
    settle(free, tokens=100, cost=0.0)
    assert free.wave_allowance() is None  # nothing costs anything


def test_restored_usage_counts_against_the_budget_but_not_the_average():
    scheduler = Scheduler(cells(), BudgetSection(max_tokens=1000))
    scheduler.restore_usage(tokens=500)
    settle(scheduler, tokens=100)
    assert scheduler.wave_allowance() == 4  # 400 left at 100 per candidate
