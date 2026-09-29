"""Release gate against known metric values: pass, fail, hard fail, waivers, short cells,
and a FAG mock run gated with its own spec thresholds. No model or API calls."""

import json

import pytest

from sdgf.evaluation.diversity import DiversityScores
from sdgf.evaluation.gate import GateError, evaluate_gate
from sdgf.evaluation.metrics import AxisBalance, Metrics, MetricsReport, metrics_for_run
from sdgf.judge.calibration import CalibrationResult
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from test_m4_checkpoint import World
from test_pipeline import NO_JUDGE, fag, fag_reply, recipe_from_prompt  # noqa: F401

THRESHOLDS = {
    "fidelity_min": 0.95,
    "kappa_min": 0.70,
    "coverage_min_cell_fill": 0.90,
    "balance_tolerance": 0.05,
    "distinct_n_min": 0.30,
    "self_bleu_max": 0.60,
    "semantic_diversity_min": 1.0,
    "residual_error_max": 0.05,
    "governance_violations_max": 0,
    "overlap_max": 0.80,
    "cost_per_record_max": 0.05,
}


def diversity(d2=0.5, bleu=0.3, entropy=1.5):
    return DiversityScores(
        size=10, distinct={1: 0.4, 2: d2}, self_bleu=bleu, cluster_entropy=entropy
    )


def cell(quota=10, accepted=10, **kw):
    base = dict(
        accepted=accepted,
        quota=quota,
        fidelity=1.0,
        judged=accepted,
        fill=accepted / quota,
        min_cell_fill=accepted / quota,
        balance={"label": AxisBalance({"True": 1.0}, {"True": 1.0}, 0.0)},
        balance_max_deviation=0.0,
        diversity=diversity(),
        residual_error=0.02,
        governance_violations=0,
        overlap={"seed": 0.4},
        overlap_max=0.4,
    )
    return Metrics(**(base | kw))


def report(overall=None, **cells):
    cells = cells or {"a": cell(), "b": cell()}
    base = dict(
        accepted=sum(c.accepted for c in cells.values()),
        quota=sum(c.quota for c in cells.values()),
        fidelity=1.0,
        judged=20,
        kappa=0.8,
        fill=1.0,
        cells_at_quota=1.0,
        min_cell_fill=min(c.fill for c in cells.values()),
        balance_max_deviation=0.0,
        diversity=diversity(),
        residual_error=0.02,
        governance_violations=0,
        overlap={"seed": 0.4},
        overlap_max=0.4,
        cost_per_record=0.01,
    )
    return MetricsReport("sv", Metrics(**(base | (overall or {}))), cells)


def failures(result):
    return {(f.metric, f.reason, f.cell) for f in result.failures}


# ── pass ─────────────────────────────────────────────────────────


def test_every_threshold_met_passes():
    result = evaluate_gate(report(), THRESHOLDS)
    assert result.passed and not result.hard_fail
    assert result.failures == () and result.short_cells == {} and result.failing_metrics == []
    assert result.spec_version == "sv"


def test_values_at_the_threshold_pass():
    r = report(
        {
            "fidelity": 0.95,
            "kappa": 0.70,
            "balance_max_deviation": 0.05,
            "diversity": diversity(d2=0.30, bleu=0.60, entropy=1.0),
            "residual_error": 0.05,
            "overlap": {"seed": 0.80},
            "cost_per_record": 0.05,
        },
        a=cell(quota=10, accepted=9),
        b=cell(),
    )
    assert evaluate_gate(r, THRESHOLDS).passed


def test_unset_thresholds_are_not_checked():
    r = report({"kappa": None, "cost_per_record": 9.0})
    assert evaluate_gate(r, THRESHOLDS | {"kappa_min": None, "cost_per_record_max": None}).passed


def test_accepts_the_spec_thresholds_section(fag):  # noqa: F811
    assert evaluate_gate(report(), fag.spec.thresholds).passed


# ── fail ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "overall, metric, reason",
    [
        ({"fidelity": 0.9}, "fidelity_min", "below_min"),
        ({"kappa": 0.5}, "kappa_min", "below_min"),
        ({"balance_max_deviation": 0.08}, "balance_tolerance", "above_max"),
        ({"diversity": diversity(d2=0.2)}, "distinct_n_min", "below_min"),
        ({"diversity": diversity(bleu=0.7)}, "self_bleu_max", "above_max"),
        ({"diversity": diversity(entropy=0.5)}, "semantic_diversity_min", "below_min"),
        ({"residual_error": 0.1}, "residual_error_max", "above_max"),
        ({"overlap": {"seed": 0.85}}, "overlap_max", "above_max"),
        ({"overlap": {"held_out": 0.9}}, "overlap_max", "above_max"),
        ({"cost_per_record": 0.2}, "cost_per_record_max", "above_max"),
        ({"kappa": None}, "kappa_min", "not_measured"),
        ({"diversity": None}, "self_bleu_max", "not_measured"),
    ],
)
def test_each_missed_threshold_fails_by_name(overall, metric, reason):
    result = evaluate_gate(report(overall), THRESHOLDS)
    assert not result.passed and not result.hard_fail
    assert (metric, reason, None) in failures(result)
    assert metric in result.failing_metrics


def test_distinct_n_order_is_selectable():
    r = report({"diversity": diversity(d2=0.5)})
    assert evaluate_gate(r, THRESHOLDS, distinct_n=2).passed
    result = evaluate_gate(r, THRESHOLDS | {"distinct_n_min": 0.45}, distinct_n=1)
    assert ("distinct_n_min", "below_min", None) in failures(result)
    # an order the report didn't compute is unmeasured
    assert ("distinct_n_min", "not_measured", None) in failures(
        evaluate_gate(r, THRESHOLDS, distinct_n=3)
    )


def test_embedding_overlap_scores_are_not_gated_on_overlap_max():
    r = report({"overlap": {"seed": 0.4, "embedding:seed": 0.95}})
    assert evaluate_gate(r, THRESHOLDS).passed


def test_short_cells_are_listed_with_records_missing():
    r = report({"fill": 0.75, "min_cell_fill": 0.5}, a=cell(quota=10, accepted=5), b=cell())
    result = evaluate_gate(r, THRESHOLDS)
    assert not result.passed
    assert result.short_cells == {"a": 5}
    assert ("coverage_min_cell_fill", "below_min", None) in failures(result)
    assert ("coverage_min_cell_fill", "below_min", "a") in failures(result)


def test_cells_above_the_fill_threshold_but_below_quota_are_not_short():
    r = report(a=cell(quota=10, accepted=9), b=cell())
    result = evaluate_gate(r, THRESHOLDS)
    assert result.passed and result.short_cells == {}


def test_per_cell_failures_name_the_cell():
    r = report(
        a=cell(fidelity=0.8, residual_error=0.2, overlap={"seed": 0.9}),
        b=cell(balance_max_deviation=0.5),
    )
    result = evaluate_gate(r, THRESHOLDS)
    assert failures(result) >= {
        ("fidelity_min", "below_min", "a"),
        ("residual_error_max", "above_max", "a"),
        ("overlap_max", "above_max", "a"),
        ("balance_tolerance", "above_max", "b"),
    }
    assert all(f.cell in {None, "a", "b"} for f in result.failures)


def test_unmeasured_per_cell_values_are_left_to_the_overall_check():
    r = report(a=cell(fidelity=None, residual_error=None), b=cell())
    assert evaluate_gate(r, THRESHOLDS).passed


def test_to_dict_is_json_safe():
    result = evaluate_gate(report({"fidelity": 0.5}), THRESHOLDS)
    d = json.loads(json.dumps(result.to_dict()))
    assert d["passed"] is False and d["failing_metrics"] == ["fidelity_min"]
    assert d["failures"][0] == {
        "metric": "fidelity_min",
        "reason": "below_min",
        "value": 0.5,
        "threshold": 0.95,
        "cell": None,
    }


# ── waivers ──────────────────────────────────────────────────────


def test_waived_metrics_may_be_unmeasured_but_are_checked_when_measured():
    unmeasured = report({"kappa": None, "residual_error": None})
    waive = ["kappa_min", "residual_error_max"]
    result = evaluate_gate(unmeasured, THRESHOLDS, waive=waive)
    assert result.passed and result.waived == ("kappa_min", "residual_error_max")
    measured = evaluate_gate(report({"kappa": 0.4}), THRESHOLDS, waive=waive)
    assert ("kappa_min", "below_min", None) in failures(measured)


def test_governance_and_unknown_names_cannot_be_waived():
    with pytest.raises(GateError, match="can't be waived"):
        evaluate_gate(report(), THRESHOLDS, waive=["governance_violations_max"])
    with pytest.raises(GateError, match="unknown thresholds"):
        evaluate_gate(report(), THRESHOLDS, waive=["vibes_min"])


# ── hard fail ────────────────────────────────────────────────────


def test_any_governance_violation_is_a_hard_fail():
    r = report({"governance_violations": 1}, a=cell(governance_violations=1), b=cell())
    result = evaluate_gate(r, THRESHOLDS)
    assert not result.passed and result.hard_fail
    assert failures(result) == {
        ("governance_violations_max", "governance", None),
        ("governance_violations_max", "governance", "a"),
    }


def test_governance_fails_hard_even_if_every_quality_metric_passes_and_threshold_unset():
    r = report({"governance_violations": 2})
    result = evaluate_gate(r, THRESHOLDS | {"governance_violations_max": None})
    # an unset governance threshold still means zero (§8: 0, hard)
    assert result.hard_fail and "governance_violations_max" in result.failing_metrics


def test_unmeasured_governance_is_a_hard_fail():
    result = evaluate_gate(report({"governance_violations": None}), THRESHOLDS)
    assert result.hard_fail
    assert ("governance_violations_max", "governance", None) in failures(result)
    per_cell = evaluate_gate(report(a=cell(governance_violations=None), b=cell()), THRESHOLDS)
    assert per_cell.hard_fail


# ── FAG runs ─────────────────────────────────────────────────────


def calibration(fag, kappa=0.9):  # noqa: F811
    return CalibrationResult(
        spec_version=fag.spec_version,
        judge_id="mock:judge",
        n=40,
        unparseable=0,
        accuracy=1.0,
        kappa=kappa,
        ece=0.02,
        bins=(),
        kappa_min=0.7,
        ece_max=0.1,
        min_gold=30,
        confusion={"True": {"True": 20}, "False": {"False": 20}},
    )


def test_fag_judged_run_passes_its_own_thresholds_with_calibration(fag, tmp_path):  # noqa: F811
    pipe = Pipeline(fag, tmp_path / "store", model_overrides=World().backends(), target_size=40)
    run = pipe.run("gate").run
    m = metrics_for_run(fag, run, calibration=calibration(fag), usage={"cost_usd": 0.4})
    # No embedder in tests, so cluster entropy is unmeasured and waived explicitly.
    result = evaluate_gate(m, fag.spec.thresholds, waive=["semantic_diversity_min"])
    assert result.passed, result.failures
    assert result.waived == ("semantic_diversity_min",) and result.short_cells == {}
    assert result.spec_version == fag.spec_version
    # without the waiver, the missing embedder is reported rather than silently passed
    strict = evaluate_gate(m, fag.spec.thresholds)
    assert failures(strict) == {("semantic_diversity_min", "not_measured", None)}


def test_fag_unjudged_run_fails_on_unmeasured_quality(fag, tmp_path):  # noqa: F811
    def reply(call):
        return json.dumps(fag_reply(recipe_from_prompt(call.prompt)))

    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(reply)},
        target_size=20,
        layers=NO_JUDGE,
    )
    m = metrics_for_run(fag, pipe.run("nojudge").run)
    result = evaluate_gate(m, fag.spec.thresholds)
    assert not result.passed and not result.hard_fail
    assert {"fidelity_min", "kappa_min", "residual_error_max"} <= set(result.failing_metrics)
    assert all(
        f.reason == "not_measured"
        for f in result.failures
        if f.metric in {"fidelity_min", "kappa_min", "residual_error_max"}
    )
