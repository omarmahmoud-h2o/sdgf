"""Stages 4 and 5 in the pipeline: gate, release, and refill rounds for short cells.
FAG runs on MockBackends only; no model or API calls."""

import dataclasses
import json

import pytest

from sdgf.evaluation.reports import verify_release
from sdgf.generate.scheduler import Cell, Scheduler, SchedulerError
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ROUNDS_STAGE, Pipeline, PipelineError
from sdgf.spec.schema import BudgetSection
from test_evaluation_gate import calibration
from test_m4_checkpoint import World
from test_pipeline import NO_JUDGE, fag, fag_reply, recipe_from_prompt  # noqa: F401

TARGET = 20
WAIVE = ["semantic_diversity_min"]  # no embedder in tests
FLAKY = {"product_scope": "corps_act", "label": True, "conversation_length": "single_turn"}
FLAKY_ID = "corps_act|true|single_turn"  # quota 2 at TARGET 20


def trusted_calibration(fag):  # noqa: F811
    # The judge the run builds from World's MockBackend is mock:mock.
    return dataclasses.replace(calibration(fag), judge_id="mock:mock")


class FlakyWorld(World):
    """World whose generator writes unparseable output for FLAKY's cell for the first
    `failures` calls, so that cell's first slot is dropped after its repairs."""

    def __init__(self, failures: int, **kw):
        super().__init__(**kw)
        self.failures = failures
        self.flaky_calls = 0

    def generate(self, call) -> str:
        recipe = recipe_from_prompt(call.prompt)
        if all(recipe.get(k) == v for k, v in FLAKY.items()) and self.failures > 0:
            self.failures -= 1
            self.flaky_calls += 1
            return "not json at all"
        return super().generate(call)


def pipeline(fag, root, world, **kw):  # noqa: F811
    return Pipeline(
        fag,
        root,
        model_overrides=world.backends(),
        target_size=TARGET,
        calibration=trusted_calibration(fag),
        **kw,
    )


@pytest.fixture(scope="module")
def released(fag, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("rel")
    pipe = pipeline(fag, root / "store", World())
    return pipe, pipe.release(root / "releases", "ok", waive=WAIVE)


@pytest.fixture(scope="module")
def recovered(fag, tmp_path_factory):  # noqa: F811
    root = tmp_path_factory.mktemp("rec")
    repair_tries = fag.spec.validation.repair_tries
    world = FlakyWorld(failures=repair_tries + 1)  # exactly one dropped slot
    pipe = pipeline(fag, root / "store", world, max_attempts_per_cell=2)
    return world, pipe, pipe.release(root / "releases", "flaky", waive=WAIVE)


# ── pass first time ──────────────────────────────────────────────


def test_fag_releases_in_one_round(fag, released):  # noqa: F811
    pipe, result = released
    assert result.released and result.stop_reason == "released"
    assert result.gate.passed and result.path.is_dir()
    manifest = verify_release(result.path)
    assert manifest["records"] == TARGET and manifest["spec_version"] == fag.spec_version
    assert [r["round"] for r in result.rounds] == [1]
    assert result.rounds[0]["cells"] is None and result.rounds[0]["outcome"] == "released"
    assert result.run.read_stage(ROUNDS_STAGE) == result.rounds
    assert not result.run.has_stage("shortfall")


def test_release_metrics_use_the_run_calibration(fag, released):  # noqa: F811
    pipe, result = released
    assert pipe.calibration is not None
    assert result.metrics.overall.kappa == trusted_calibration(fag).kappa
    assert result.metrics.overall.residual_error is not None


def test_calibration_for_another_judge_is_not_used_for_metrics(fag, tmp_path):  # noqa: F811
    pipe = Pipeline(
        fag,
        tmp_path,
        model_overrides=World().backends(),
        target_size=TARGET,
        calibration=calibration(fag),  # judge_id mock:judge, not the mock:mock built
    )
    assert pipe.calibration is None


# ── fail, then recover ───────────────────────────────────────────


def test_failed_gate_refills_only_short_cells_and_recovers(fag, recovered):  # noqa: F811
    world, _, result = recovered
    assert world.flaky_calls == fag.spec.validation.repair_tries + 1 and world.failures == 0
    first, second = result.rounds
    assert first["scheduler_stop"] == "stalled" and not first["passed"]
    assert "coverage_min_cell_fill" in first["failing_metrics"]
    assert first["short_cells"] == {FLAKY_ID: 1} and first["outcome"] == "refill"
    assert second["cells"] == [FLAKY_ID] and second["accepted"] == 1
    assert second["passed"] and second["outcome"] == "released"
    assert result.released and verify_release(result.path)["records"] == TARGET


def test_refill_round_generates_only_in_short_cells(recovered):
    _, _, result = recovered
    accepted = result.run.read_jsonl("accepted")
    assert len(accepted) == TARGET
    last = accepted[-1]["_provenance"]["cell_id"]
    assert last == FLAKY_ID
    summary = result.run.read_stage("summary")
    assert summary["only"] == [FLAKY_ID] and list(summary["cells"]) == [FLAKY_ID]
    # usage carries over rounds, so it counts every candidate of the run
    assert summary["usage"]["candidates"] == TARGET + 1
    assert result.run.read_stage(ROUNDS_STAGE) == result.rounds


def test_refill_rounds_stop_at_max_rounds_with_a_shortfall(fag, tmp_path):  # noqa: F811
    world = FlakyWorld(failures=10**6)  # the cell never recovers
    pipe = pipeline(fag, tmp_path / "store", world, max_attempts_per_cell=2)
    result = pipe.release(tmp_path / "releases", "stuck", waive=WAIVE, max_rounds=2)
    assert not result.released and result.stop_reason == "max_rounds"
    assert [r["outcome"] for r in result.rounds] == ["refill", "max_rounds"]
    assert result.rounds[1]["cells"] == [FLAKY_ID]
    assert result.path.name == "shortfall.json"
    shortfall = result.run.read_stage("shortfall")
    assert shortfall["short_cells"] == {FLAKY_ID: 2}
    assert not (tmp_path / "releases").exists()


def test_failures_other_than_coverage_are_not_refilled(fag, tmp_path):  # noqa: F811
    def reply(call):
        return json.dumps(fag_reply(recipe_from_prompt(call.prompt)))

    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(reply)},
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    result = pipe.release(tmp_path / "releases", "nojudge", waive=WAIVE)
    assert result.stop_reason == "no_short_cells" and len(result.rounds) == 1
    assert "fidelity_min" in result.gate.failing_metrics
    assert result.run.has_stage("shortfall")


def test_budget_exhaustion_stops_refilling(fag, tmp_path):  # noqa: F811
    budget = BudgetSection(max_candidates=10)
    spec = dataclasses.replace(fag, spec=fag.spec.model_copy(update={"budget": budget}))
    pipe = pipeline(spec, tmp_path / "store", World())
    result = pipe.release(tmp_path / "releases", "broke", waive=WAIVE)
    assert result.stop_reason == "budget:max_candidates" and len(result.rounds) == 1
    assert not result.released


def test_release_rejects_bad_options(fag, tmp_path):  # noqa: F811
    pipe = pipeline(fag, tmp_path / "store", World())
    with pytest.raises(PipelineError, match="max_rounds"):
        pipe.release(tmp_path / "releases", max_rounds=0)
    with pytest.raises(PipelineError, match="not in run"):
        pipe.run("r", only=["no|such|cell"])


# ── scheduler usage carry-over ───────────────────────────────────


def test_restored_usage_counts_against_the_budget():
    now = [100.0]
    s = Scheduler(
        [Cell("a", {}, 5)], BudgetSection(max_candidates=3, max_seconds=50), clock=lambda: now[0]
    )
    s.restore_usage(candidates=2, tokens=7, cost_usd=0.5, seconds=10)
    assert s.elapsed() == 10 and s.usage.tokens == 7 and s.usage.cost_usd == 0.5
    assert s.next_cell() is not None
    s.accept("a")
    assert s.next_cell() is None and s.stop_reason == "budget:max_candidates"
    with pytest.raises(SchedulerError):
        s.restore_usage(candidates=-1)
