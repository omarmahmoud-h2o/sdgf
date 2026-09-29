"""M8 checkpoint: HITL, CLI, concurrency and budget together on FAG. A priced, concurrent,
review-enabled run waits for plan approval, stops on its cost budget and resumes to
completion with the same records as an unbudgeted sequential run; its review queue is
resolved into the gold set (also read through the CLI), the gold set calibrates the judge,
and the run releases with its cost reported. MockBackends only."""

import dataclasses
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from sdgf.evaluation.reports import CARD, METRICS, verify_release
from sdgf.hitl.queue import ApprovalPending, GoldSet, ReviewQueue, approve_plan
from sdgf.judge.calibration import gold_from_records, run_calibration
from sdgf.judge.llm_judge import LLMJudge
from sdgf.models.usage import USAGE_STAGE
from sdgf.pipeline import ACCEPTED_STREAM, REVIEW_STREAM, Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import split
from test_cli import sdgf
from test_evaluation_gate import calibration
from test_pipeline_concurrency import OrderFreeWorld

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 40
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
REVIEWER = "Test Reviewer"
PRICE = 2.0  # USD per million tokens, both directions, both stages (mock tokens are estimated)
MAX_COST = 0.25  # about a third of what the whole run costs (~$0.72), so it stops and resumes
WAIVE = ("semantic_diversity_min",)  # no embedder in tests


def fag_spec(root: Path, *, concurrency: int, budget: dict | None) -> object:
    task = root / "fag"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))
    spec = yaml.safe_load((task / "task.yaml").read_text())
    for stage in ("generator", "judge"):
        spec["models"][stage].update(
            concurrency=concurrency, input_cost_per_mtok=PRICE, output_cost_per_mtok=PRICE
        )
    spec["hitl"].update(approve_coverage_plan=True, review_flagged=True)
    spec["budget"] = budget or {}
    (task / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return compile_spec(task)


def trusted(compiled):
    # The judge the run builds from OrderFreeWorld's MockBackend is mock:mock.
    return dataclasses.replace(calibration(compiled), judge_id="mock:mock")


def pipeline(compiled, root, world):
    return Pipeline(
        compiled,
        root / "store",
        model_overrides=world.backends(),
        target_size=TARGET,
        calibration=trusted(compiled),
    )


def records(result):
    """The accepted records with their cell, seed and repair count, in a stable order."""
    out = []
    for line in result.run.read_jsonl(ACCEPTED_STREAM):
        record, prov = split(line)
        out.append((prov.cell_id, prov.seed, prov.repair_count, json.dumps(record, sort_keys=True)))
    return sorted(out)


def run_to_completion(compiled, root, run_id):
    """Approve the plan when asked, then rerun the run id until it completes."""
    world = OrderFreeWorld(unsure=True)
    pipe = pipeline(compiled, root, world)
    with pytest.raises(ApprovalPending):
        pipe.run(run_id)
    assert world.generating.peak == 0  # nothing generated before approval
    approve_plan(pipe.store, compiled.spec_version, pipe.plan_stage, reviewer=REVIEWER, now=NOW)
    results = [pipe.run(run_id)]
    while not results[-1].complete:
        assert len(results) < TARGET, "resume made no progress"
        results.append(pipeline(compiled, root, OrderFreeWorld(unsure=True)).run(run_id))
        assert results[-1].run.resumed
    return world, results


@pytest.fixture(scope="module")
def budgeted(tmp_path_factory):
    root = tmp_path_factory.mktemp("budgeted")
    compiled = fag_spec(root, concurrency=4, budget={"max_cost_usd": MAX_COST})
    world, results = run_to_completion(compiled, root, "m8")
    return compiled, root, world, results


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    root = tmp_path_factory.mktemp("reference")
    compiled = fag_spec(root, concurrency=1, budget=None)
    _, results = run_to_completion(compiled, root, "m8")
    assert len(results) == 1
    return results[0]


def test_budget_stops_then_resumes_to_the_sequential_result(budgeted, reference):
    _, _, world, results = budgeted
    assert len(results) > 1
    assert all(r.stop_reason == "budget:max_cost_usd" for r in results[:-1])
    assert results[-1].complete
    assert 1 < world.generating.peak <= 4  # the first invocation ran concurrently
    # budget stops, resumes and 4 workers don't change what is accepted
    assert records(results[-1]) == records(reference)
    assert len(records(reference)) == TARGET
    assert sum(json.loads(r[3])["label"] for r in records(reference)) == TARGET // 2


def test_usage_ledger_covers_every_invocation(budgeted):
    compiled, _, _, results = budgeted
    ledger = results[-1].run.read_stage(USAGE_STAGE)
    assert ledger["invocations"] == len(results)
    assert set(ledger["stages"]) == {"generator", "judge"}
    assert all(s["estimated_calls"] == s["calls"] for s in ledger["stages"].values())
    total = ledger["total"]
    assert total["cost_usd"] == pytest.approx(total["tokens"] * PRICE / 1e6)
    assert total["cost_usd"] > MAX_COST  # the whole run cost more than any one invocation may


def test_review_queue_resolves_into_gold_and_calibrates(budgeted):
    compiled, root, _, results = budgeted
    run = results[-1].run
    store = ArtefactStore(root / "store")
    gold = GoldSet.for_task(store, compiled.spec.task.name)
    queue = ReviewQueue.for_run(run, gold)
    pending = queue.pending()
    assert pending and len(pending) == len(run.read_jsonl(REVIEW_STREAM))
    assert {p["code"] for p in pending.values()} == {"low_confidence"}

    # the CLI reads the same queue the in-process run wrote
    code, listed, err = sdgf("review", root / "fag", "--store", root / "store", "list")
    assert code == 0, err
    assert listed["run_id"] == "m8" and {p["id"] for p in listed["pending"]} == set(pending)

    (first, a), (second, b) = list(pending.items())[:2]
    queue.resolve(first, "accept", reviewer=REVIEWER, now=NOW)
    queue.resolve(second, "relabel", reviewer=REVIEWER, new_label=not b["intended_label"], now=NOW)
    labels = [g.label for g in gold_from_records(gold.records())]
    assert labels == [a["intended_label"], not b["intended_label"]]
    assert len(queue.pending()) == len(pending) - 2

    judge = LLMJudge.from_spec(compiled, OrderFreeWorld().backends()["judge"])
    result = run_calibration(
        compiled, judge, gold_from_records(gold.records()), judge_name="mock:mock"
    )
    assert result.n == 2 and not result.passed
    assert any("min_gold" in p for p in result.problems)  # two items can't make a judge trusted


def test_completed_run_releases_with_its_cost(budgeted):
    compiled, root, _, results = budgeted
    pipe = pipeline(compiled, root, OrderFreeWorld(unsure=True))
    released = pipe.release(root / "releases", "m8", waive=WAIVE, now=NOW)
    assert released.released, released.gate.failures
    assert verify_release(released.path)["records"] == TARGET
    overall = json.loads((released.path / METRICS).read_text())["metrics"]["overall"]
    ledger = results[-1].run.read_stage(USAGE_STAGE)
    assert overall["cost_per_record"] == pytest.approx(ledger["total"]["cost_usd"] / TARGET)
    assert "## Cost" in (released.path / CARD).read_text()
