"""§8 metrics on hand-built records and drops, then on FAG mock pipeline runs. All text is
synthetic; no model or API calls."""

import json

import pytest

from sdgf.evaluation.metrics import (
    MetricsError,
    compute_metrics,
    drop_tries,
    judge_agreed,
    metrics_for_run,
    record_tries,
    residual_by_label,
)
from sdgf.generate.scheduler import Cell
from sdgf.judge.calibration import CalibrationResult
from sdgf.pipeline import Pipeline
from sdgf.store.provenance import PROVENANCE_KEY
from sdgf.validate.base import Layer, ValidationIssue
from sdgf.validate.l4_overlap import OverlapEngine, OverlapLayer, ShingleIndex
from sdgf.validate.repair import Drop
from test_m4_checkpoint import World
from test_pipeline import NO_JUDGE, fag, fag_reply, recipe_from_prompt  # noqa: F401
from sdgf.models.mock import MockBackend

CELLS = [
    Cell("a", {"label": True, "scope": "x"}, 2),
    Cell("b", {"label": False, "scope": "x"}, 2),
]


def lr(layer, outcome="pass", attempt=0):
    return {"layer": layer, "outcome": outcome, "attempt": attempt, "errors": []}


def passes(attempt=0, layers=("L1", "L2", "L3", "L4", "L5")):
    return [lr(layer, attempt=attempt) for layer in layers]


def rec(cell, label, text, *, results=None, repairs=0, trace=()):
    return {
        "messages": [
            {"turn": 1, "role": "customer", "content": f"Question about {text}?"},
            {"turn": 2, "role": "assistant", "content": f"Here is how {text} works."},
        ],
        "label": label,
        "spans": [],
        PROVENANCE_KEY: {
            "cell_id": cell,
            "repair_count": repairs,
            "layer_results": passes() if results is None else results,
            "tool_trace": list(trace),
        },
    }


def calibration(confusion, accuracy=0.9, kappa=0.8):
    return CalibrationResult(
        spec_version="sv",
        judge_id="mock:judge",
        n=sum(sum(h.values()) for h in confusion.values()),
        unparseable=0,
        accuracy=accuracy,
        kappa=kappa,
        ece=0.05,
        bins=(),
        kappa_min=0.7,
        ece_max=0.1,
        min_gold=1,
        confusion=confusion,
    )


# ── per-record facts ─────────────────────────────────────────────


def test_record_tries_names_each_failed_try_and_generation_failures():
    results = [lr("L1"), lr("L2", "fail_repairable")] + passes(attempt=2)
    # attempt 1 left no layer results: the model returned no parseable record
    assert record_tries(rec("a", True, "fees", results=results, repairs=2)) == [
        "L2",
        "generate",
        None,
    ]
    assert record_tries(rec("a", True, "fees")) == [None]


def test_judge_agreed_reads_the_final_l5_outcome():
    assert judge_agreed(rec("a", True, "fees")) is True
    assert judge_agreed(rec("a", True, "fees", results=passes(layers=("L1", "L2")))) is None
    results = [lr("L5", "fail_repairable")] + passes(attempt=1)
    assert judge_agreed(rec("a", True, "fees", results=results, repairs=1)) is True
    assert judge_agreed(rec("a", True, "fees", results=[lr("L5", "fail_repairable")])) is False


def test_drop_tries_uses_history_and_falls_back_for_old_drops():
    drop = Drop(
        "a",
        "L2",
        ("x",),
        "r",
        attempts=3,
        history=(("generate", ("g",)), ("L1", ()), ("L2", ("x",))),
    )
    assert drop_tries(drop) == ["generate", "L1", "L2"]
    assert drop_tries(drop.to_dict()) == ["generate", "L1", "L2"]
    old = {"cell_id": "a", "layer": "L2", "attempts": 2}
    assert drop_tries(old) == ["L2", "L2"]
    assert drop_tries({"cell_id": "a", "layer": "generate", "attempts": 0}) == []


def test_residual_by_label_is_one_minus_precision():
    cal = calibration({"True": {"True": 9, "False": 1}, "False": {"False": 10}})
    assert residual_by_label(cal) == pytest.approx({"True": 0.1, "False": 0.0})


# ── compute_metrics ──────────────────────────────────────────────


def test_error_rates_count_every_try_and_yield():
    repaired = [lr("L1"), lr("L2", "fail_repairable")] + passes(attempt=1)
    accepted = [
        rec("a", True, "fees", results=repaired, repairs=1),
        rec("b", False, "rates"),
    ]
    drops = [
        Drop("a", "L3", ("pii_tfn",), "r", attempts=1, hard=True, history=(("L3", ("pii_tfn",)),)),
        Drop("b", "L2", ("x",), "r", attempts=3, history=(("L2", ("x",)),) * 3),
    ]
    m = compute_metrics(accepted, drops, cells=CELLS)
    o = m.overall
    assert o.candidates == 2 + 1 + 1 + 3
    assert o.rejections == {"L2": 4, "L3": 1}
    assert o.error_rates == pytest.approx({"L2": 4 / 7, "L3": 1 / 7})
    assert o.yield_ == pytest.approx(2 / 7)
    assert m.per_cell["a"].candidates == 3 and m.per_cell["a"].rejections == {"L2": 1, "L3": 1}
    assert m.per_cell["b"].yield_ == pytest.approx(1 / 4)


def test_coverage_fill_short_cells_and_zero_quota():
    cells = CELLS + [Cell("c", {"label": True, "scope": "y"}, 0)]
    accepted = [rec("a", True, "fees"), rec("a", True, "rates"), rec("b", False, "cards")]
    m = compute_metrics(accepted, cells=cells)
    assert m.per_cell["a"].fill == 1.0 and m.per_cell["b"].fill == 0.5
    assert m.per_cell["c"].fill == 1.0  # nothing asked of it, so it is full
    assert m.overall.quota == 4 and m.overall.fill == pytest.approx(3 / 4)
    assert m.overall.cells_at_quota == pytest.approx(2 / 3)
    assert m.overall.min_cell_fill == 0.5
    assert m.overall.short_cells == {"b": 1}
    assert m.per_cell["b"].short_cells == {"b": 1}


def test_unknown_cells_are_refused():
    with pytest.raises(MetricsError, match="zzz"):
        compute_metrics([rec("zzz", True, "fees")], cells=CELLS)


def test_balance_against_planned_labels_and_relabels():
    accepted = [rec("a", True, "fees"), rec("a", True, "rates"), rec("b", False, "cards")]
    accepted.append(rec("b", True, "loans"))  # relabelled by a person: its own label counts
    m = compute_metrics(accepted, cells=CELLS)
    bal = m.overall.balance["label"]
    assert bal.target == {"true": 0.5, "false": 0.5}
    assert bal.actual == {"false": 0.25, "true": 0.75}
    assert m.overall.balance_max_deviation == pytest.approx(0.25)
    assert m.per_cell["a"].balance_max_deviation == 0.0
    assert m.per_cell["b"].balance_max_deviation == pytest.approx(0.5)


def test_balance_on_an_explicit_non_label_axis():
    cells = [
        Cell("a", {"label": True, "scope": "x"}, 1),
        Cell("b", {"label": True, "scope": "y"}, 1),
    ]
    m = compute_metrics(
        [rec("a", True, "fees"), rec("a", True, "rates")],
        cells=cells,
        balance={"scope": {"x": 0.5, "y": 0.5}},
    )
    assert m.overall.balance["scope"].actual == {"x": 1.0, "y": 0.0}
    assert m.overall.balance_max_deviation == pytest.approx(0.5)


def test_fidelity_counts_only_judged_records():
    accepted = [
        rec("a", True, "fees"),
        rec("a", True, "rates"),
        rec("b", False, "cards", results=[lr("L5", "fail_repairable")]),
        rec("b", False, "loans", results=passes(layers=("L1", "L2"))),
    ]
    m = compute_metrics(accepted, cells=CELLS)
    assert m.overall.judged == 3 and m.overall.fidelity == pytest.approx(2 / 3)
    assert m.per_cell["a"].fidelity == 1.0
    assert m.per_cell["b"].judged == 1 and m.per_cell["b"].fidelity == 0.0


def test_kappa_and_residual_error_from_calibration():
    cal = calibration({"True": {"True": 9, "False": 1}, "False": {"False": 10}}, accuracy=0.95)
    accepted = [rec("a", True, "fees"), rec("a", True, "rates"), rec("b", False, "cards")]
    accepted.append(rec("b", "maybe", "loans"))  # no gold precision: falls back to 1 - accuracy
    m = compute_metrics(accepted, cells=CELLS, calibration=cal)
    assert m.overall.kappa == 0.8
    assert m.overall.residual_error == pytest.approx((0.1 + 0.1 + 0.0 + 0.05) / 4)
    assert m.per_cell["a"].kappa is None  # the gold set isn't per cell
    assert m.per_cell["a"].residual_error == pytest.approx(0.1)


def test_unmeasured_metrics_are_none_not_zero():
    m = compute_metrics([rec("a", True, "fees")], cells=CELLS)
    o = m.overall
    assert o.kappa is None and o.residual_error is None
    assert o.governance_violations is None and o.overlap == {} and o.overlap_max is None
    assert o.cost_per_record is None and o.tokens_per_record is None
    empty = compute_metrics([], cells=CELLS)
    assert empty.overall.fidelity is None and empty.overall.yield_ is None
    assert empty.overall.balance["label"].max_deviation is None


class SecretLayer(Layer):
    name = "L3"

    def __init__(self):
        self.traces = []

    def check(self, record, context):
        assert PROVENANCE_KEY not in record  # scanners see the released record only
        self.traces.append(context.extra["tool_trace"])
        text = json.dumps(record)
        issues = [ValidationIssue("secret_token", "found")] if "TOKEN" in text else []
        return self.verdict(issues, repairable=False)


def test_governance_violations_rescan_the_released_set():
    layer = SecretLayer()
    trace = [{"tool": "calculator", "sensitivity": "public"}]
    accepted = [
        rec("a", True, "fees", trace=trace),
        rec("b", False, "TOKEN abc"),
        rec("b", False, "x"),
    ]
    m = compute_metrics(accepted, cells=CELLS, governance=layer)
    assert m.overall.governance_violations == 1
    assert m.overall.governance_codes == {"secret_token": 1}
    assert m.per_cell["a"].governance_violations == 0
    assert m.per_cell["b"].governance_violations == 1
    assert layer.traces[0] == trace


def test_overlap_is_the_max_similarity_to_seeds_and_held_out():
    seed_text = "Question about term deposit rates? Here is how term deposit rates works."
    layer = OverlapLayer(
        [OverlapEngine(build=ShingleIndex, thresholds={"seed": 0.8, "held_out": 0.8})],
        seeds=[("seed:0", seed_text)],
        held_out=[("held:0", "Completely unrelated synthetic text about zebras and kites.")],
    )
    accepted = [rec("a", True, "term deposit rates"), rec("b", False, "merchant terminals")]
    m = compute_metrics(accepted, cells=CELLS, overlap=layer)
    assert m.overall.overlap["seed"] == pytest.approx(1.0)
    assert m.overall.overlap["held_out"] < 0.2
    assert m.overall.overlap_max == pytest.approx(1.0)
    assert m.per_cell["b"].overlap["seed"] < 1.0


def test_cost_per_record_from_usage():
    accepted = [rec("a", True, "fees"), rec("b", False, "cards")]
    usage = {"tokens": 1000, "cost_usd": 0.5, "seconds": 8.0}
    o = compute_metrics(accepted, cells=CELLS, usage=usage).overall
    assert (o.tokens_per_record, o.cost_per_record, o.seconds_per_record) == (500, 0.25, 4.0)


def test_diversity_overall_and_per_cell():
    accepted = [rec("a", True, "fees"), rec("a", True, "fees"), rec("b", False, "cards")]
    m = compute_metrics(accepted, cells=CELLS)
    assert m.per_cell["a"].diversity.self_bleu == pytest.approx(1.0)
    assert m.overall.diversity.size == 3
    assert m.overall.diversity.distinct[1] < 1.0


def test_without_cells_groups_by_provenance_cell():
    m = compute_metrics([rec("a", True, "fees")], [Drop("q", "L1", (), "r", attempts=1)])
    assert set(m.per_cell) == {"a", "q"}
    assert m.overall.fill is None and m.per_cell["a"].quota is None


def test_report_is_json_serialisable():
    d = compute_metrics([rec("a", True, "fees")], cells=CELLS, spec_version="sv").to_dict()
    again = json.loads(json.dumps(d))
    assert again["spec_version"] == "sv"
    assert "yield" in again["overall"] and "yield_" not in again["overall"]
    assert again["overall"]["balance"]["label"]["target"] == {"true": 0.5, "false": 0.5}


# ── FAG runs ─────────────────────────────────────────────────────


def test_fag_run_metrics_from_the_run_directory(fag, tmp_path):  # noqa: F811
    def reply(call):
        repairing = "previous attempt was rejected" in call.prompt
        return json.dumps(fag_reply(recipe_from_prompt(call.prompt), reword=not repairing))

    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(reply)},
        target_size=20,
        layers=NO_JUDGE,
    )
    result = pipe.run("r1")
    breaches = sum(r["label"] for r in result.accepted)
    m = metrics_for_run(fag, result.run)
    o = m.overall
    assert m.spec_version == fag.spec_version
    assert o.accepted == 20 and o.fill == 1.0 and o.cells_at_quota == 1.0 and o.short_cells == {}
    assert o.balance_max_deviation == 0.0  # BREACH_RATE 0.5 held exactly
    assert o.candidates == 20 + breaches  # every breach's first try reworded a span
    assert o.rejections == {"L2": breaches}
    assert o.yield_ == pytest.approx(20 / (20 + breaches))
    assert o.fidelity is None and o.judged == 0  # no judge in this run
    assert o.governance_violations == 0
    assert 0 < o.overlap["seed"] < fag.spec.thresholds.overlap_max
    assert o.seconds_per_record is not None
    assert set(m.per_cell) == {c.id for c in pipe.plan().cells}


def test_fag_judged_run_has_full_fidelity_and_counts_l5_repairs(fag, tmp_path):  # noqa: F811
    world = World()
    pipe = Pipeline(fag, tmp_path / "store", model_overrides=world.backends(), target_size=40)
    result = pipe.run("m7")
    o = metrics_for_run(fag, result.run).overall
    assert o.judged == 40 and o.fidelity == 1.0
    assert o.rejections.get("L5") == world.advised > 0
    assert o.kappa is None and o.residual_error is None  # no calibration passed
