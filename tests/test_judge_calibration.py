import json
from pathlib import Path

import pytest

from sdgf.judge.calibration import (
    UNPARSEABLE,
    CalibrationError,
    CalibrationResult,
    CalibrationStore,
    GoldItem,
    calibrate,
    cohen_kappa,
    expected_calibration_error,
    gold_from_records,
    judge_id,
    reliability_bins,
    run_calibration,
    spec_judge_id,
    trust_for,
)
from sdgf.judge.interface import Judge, JudgeParseError, JudgeResult, compile_rubric
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import ModelConfig, RubricSection
from sdgf.store.artefacts import ArtefactStore
from sdgf.validate.base import ValidationContext
from sdgf.validate.l5_judge import JudgeLayer
from sdgf.validate.l6_consistency import ConsistencyLayer

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
LABELS = {"breach": True, "no_breach": False}


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def seeds():
    return [json.loads(line) for line in (FAG_DIR / "seeds.jsonl").read_text().splitlines()]


def result(verdict, conf=1.0):
    return JudgeResult(verdict, {}, {"verdict": conf})


def run(results, human, **kw):
    kw.setdefault("spec_version", "v1")
    kw.setdefault("judge", "mock:judge")
    kw.setdefault("kappa_min", 0.7)
    kw.setdefault("labels", LABELS)
    kw.setdefault("min_gold", 4)
    return calibrate(results, human, **kw)


def balanced(n=40, conf=1.0):
    human = [i % 2 == 0 for i in range(n)]
    return [result("breach" if h else "no_breach", conf) for h in human], human


# ── Cohen's kappa ────────────────────────────────────────────────


def test_kappa_textbook_example():
    # 50 items: yes/yes 20, yes/no 5, no/yes 10, no/no 15 -> po 0.7, pe 0.5, kappa 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_perfect_and_inverse():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == -1.0


def test_kappa_chance_level_is_zero():
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_kappa_undefined_or_bad_input_raises():
    with pytest.raises(CalibrationError, match="undefined"):
        cohen_kappa(["a", "a"], ["a", "a"])
    with pytest.raises(CalibrationError, match="at least one"):
        cohen_kappa([], [])
    with pytest.raises(CalibrationError, match="paired"):
        cohen_kappa([1], [1, 0])


def test_kappa_keeps_bools_apart_from_ints():
    assert cohen_kappa([True, False], [1, 0]) < 1.0


# ── reliability bins and ECE ─────────────────────────────────────


def test_bins_and_ece_known_example():
    bins = reliability_bins([0.9, 0.9, 0.6, 0.6], [True, False, True, True], n_bins=10)
    assert len(bins) == 10
    assert (bins[9].count, bins[9].accuracy, bins[9].mean_confidence) == (2, 0.5, 0.9)
    assert (bins[6].count, bins[6].accuracy, bins[6].mean_confidence) == (2, 1.0, 0.6)
    assert bins[0].count == 0 and bins[0].accuracy is None
    # 0.5 * |0.5 - 0.9| + 0.5 * |1.0 - 0.6|
    assert expected_calibration_error(bins) == pytest.approx(0.4)


def test_bin_edges():
    bins = reliability_bins([0.0, 0.5, 1.0], [True, True, True], n_bins=2)
    assert [b.count for b in bins] == [1, 2]
    assert (bins[0].lower, bins[0].upper, bins[1].upper) == (0.0, 0.5, 1.0)


def test_perfectly_calibrated_ece_is_zero():
    bins = reliability_bins([1.0] * 5, [True] * 5)
    assert expected_calibration_error(bins) == 0.0


def test_bins_reject_bad_input():
    with pytest.raises(CalibrationError, match="outside"):
        reliability_bins([1.2], [True])
    with pytest.raises(CalibrationError, match="one correctness flag"):
        reliability_bins([0.5], [True, False])
    with pytest.raises(CalibrationError, match="at least one"):
        expected_calibration_error(reliability_bins([], []))


# ── calibrate ────────────────────────────────────────────────────


def test_accurate_calibrated_judge_passes():
    results, human = balanced()
    r = run(results, human)
    assert r.passed and r.problems == ()
    assert (r.kappa, r.ece, r.accuracy, r.n, r.unparseable) == (1.0, 0.0, 1.0, 40, 0)
    assert r.confusion == {"True": {"True": 20}, "False": {"False": 20}}
    assert r.trusts("v1", "mock:judge")


def test_small_gold_set_fails():
    results, human = balanced(10)
    r = run(results, human, min_gold=30)
    assert not r.passed
    assert r.problems == ("gold set has 10 items, fewer than min_gold 30",)


def test_low_kappa_fails():
    results, human = balanced()
    results = [result("breach")] * 30 + results[30:]  # 15 no_breach items misjudged
    r = run(results, human)
    assert r.kappa < 0.7
    assert any("below kappa_min" in p for p in r.problems)


def test_accurate_but_underconfident_judge_fails_on_ece():
    results, human = balanced(conf=0.55)
    r = run(results, human)
    assert r.kappa == 1.0
    assert r.ece == pytest.approx(0.45)
    assert r.problems == ("expected calibration error 0.450 is above ece_max 0.1",)


def test_unparseable_answers_count_wrong_and_skip_bins():
    results, human = balanced()
    results[0] = None
    r = run(results, human)
    assert r.unparseable == 1
    assert r.accuracy == pytest.approx(39 / 40)
    assert sum(b.count for b in r.bins) == 39
    assert r.ece == 0.0
    assert r.confusion[repr(UNPARSEABLE)] == {"True": 1}


def test_all_unparseable_fails_without_ece():
    _, human = balanced()
    r = run([None] * 40, human)
    assert r.ece is None and r.kappa == 0.0
    assert any("ECE can't be measured" in p for p in r.problems)


def test_undefined_kappa_fails():
    r = run([result("breach")] * 5, [True] * 5)
    assert r.kappa is None
    assert any("undefined" in p for p in r.problems)


def test_label_map_is_strict_about_bools():
    results, _ = balanced()
    r = run(results, [1 if i % 2 == 0 else 0 for i in range(40)])
    assert r.accuracy == 0.0


def test_identity_map_and_unmapped_verdicts():
    r = run([result("yes"), result("no")] * 10, ["yes", "no"] * 10, labels=None)
    assert r.passed
    r = run([result("unclear")] + [result("breach"), result("no_breach")] * 10, [True] * 21)
    assert r.accuracy == pytest.approx(10 / 21)
    assert "'<unmapped>'" in r.confusion


def test_calibrate_rejects_bad_input():
    with pytest.raises(CalibrationError, match="pair up"):
        run([result("breach")], [True, False])
    with pytest.raises(CalibrationError, match="empty"):
        run([], [])


def test_trusts_needs_same_spec_version_and_judge():
    results, human = balanced()
    r = run(results, human)
    assert not r.trusts("v2", "mock:judge")
    assert not r.trusts("v1", "mock:other")


def test_result_round_trips():
    results, human = balanced(conf=0.8)
    results[3] = None
    r = run(results, human)
    back = CalibrationResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert back == r
    with pytest.raises(CalibrationError, match="version"):
        CalibrationResult.from_dict({**r.to_dict(), "version": 99})


# ── persistence ──────────────────────────────────────────────────


def test_store_saves_and_loads_per_spec_version_and_judge(tmp_path):
    store = CalibrationStore(ArtefactStore(tmp_path))
    results, human = balanced()
    r = run(results, human, judge="openai_compat:Test/Model-7B")
    path = store.save(r)
    assert path.parent == tmp_path / "v1" / "shared"
    assert store.load("v1", "openai_compat:Test/Model-7B") == r
    assert store.trusted("v1", "openai_compat:Test/Model-7B")
    assert store.load("v1", "openai_compat:Other") is None
    assert store.load("v2", "openai_compat:Test/Model-7B") is None
    assert not store.trusted("v2", "openai_compat:Test/Model-7B")


def test_store_failed_result_is_not_trusted(tmp_path):
    store = CalibrationStore(ArtefactStore(tmp_path))
    results, human = balanced(conf=0.55)
    store.save(run(results, human))
    assert store.load("v1", "mock:judge") is not None
    assert not store.trusted("v1", "mock:judge")


def test_saved_result_is_a_shared_run_artefact(tmp_path):
    artefacts = ArtefactStore(tmp_path)
    results, human = balanced()
    r = run(results, human)
    path = CalibrationStore(artefacts).save(r)
    run_dir = artefacts.open_run("v1", "r1")
    assert run_dir.read_stage(path.stem, shared=True) == r.to_dict()


# ── against the FAG spec ─────────────────────────────────────────


class LookupJudge(Judge):
    """Answers from a table keyed by the conversation, recording what it was shown."""

    name = "lookup"

    def __init__(self, answers, conf=1.0):
        super().__init__(compile_rubric(RubricSection(verdict={"values": list(LABELS)})))
        self.answers, self.conf = answers, conf
        self.seen = []

    def judge(self, record):
        self.seen.append(record)
        verdict = self.answers.get(json.dumps(record["messages"]))
        if verdict is None:
            raise JudgeParseError(["<root>: no answer"])
        return result(verdict, self.conf)


def answers_for(records):
    return {json.dumps(r["messages"]): "breach" if r["label"] else "no_breach" for r in records}


def test_judge_ids(fag):
    assert judge_id(ModelConfig(backend="mock", model="m")) == "mock:m"
    spec = fag.spec.models.judge
    assert spec_judge_id(fag) == f"{spec.backend}:{spec.model}"


def test_gold_from_records(seeds):
    gold = gold_from_records(seeds)
    assert [g.label for g in gold] == [s["label"] for s in seeds]
    with pytest.raises(CalibrationError, match="no 'label'"):
        gold_from_records([{"messages": []}])


def test_run_calibration_on_fag_seeds_is_blind(fag, seeds):
    judge = LookupJudge(answers_for(seeds))
    r = run_calibration(fag, judge, gold_from_records(seeds))
    assert all(set(view) == {"messages"} for view in judge.seen)
    assert (r.spec_version, r.judge_id) == (fag.spec_version, spec_judge_id(fag))
    assert (r.kappa, r.ece, r.kappa_min) == (1.0, 0.0, fag.spec.thresholds.kappa_min)
    # FAG's six seeds are too few to trust a judge on.
    assert r.problems == ("gold set has 6 items, fewer than min_gold 30",)


def test_run_calibration_passes_with_a_big_enough_gold_set(fag, seeds):
    gold = [GoldItem(s, s["label"]) for s in seeds] * 5
    judge = LookupJudge(answers_for(seeds))
    r = run_calibration(fag, judge, gold)
    assert r.passed and r.n == 30
    assert trust_for(fag, r)


def test_run_calibration_counts_judge_failures(fag, seeds):
    judge = LookupJudge(answers_for(seeds[1:]))
    r = run_calibration(fag, judge, gold_from_records(seeds), judge_name="mock:x")
    assert r.unparseable == 1 and r.judge_id == "mock:x"


# ── L5 and L6 read the result ────────────────────────────────────


def passing(fag):
    results, human = balanced()
    return run(results, human, spec_version=fag.spec_version, judge=spec_judge_id(fag))


def test_trust_for(fag):
    assert trust_for(fag, passing(fag))
    assert not trust_for(fag, None)
    assert not trust_for(fag, passing(fag), judge="mock:other")
    results, human = balanced()
    assert not trust_for(fag, run(results, human, judge=spec_judge_id(fag)))  # spec v1


def test_l5_reports_trust_from_calibration(fag, seeds):
    judge = LookupJudge(answers_for(seeds))
    ctx = ValidationContext(recipe={"label": seeds[0]["label"]})
    trusted = JudgeLayer.from_spec(fag, judge, calibration=passing(fag))
    assert trusted.check(seeds[0], ctx).details["trusted"] is True
    untrusted = JudgeLayer.from_spec(fag, judge)
    assert untrusted.check(seeds[0], ctx).details["trusted"] is False


def test_l6_trusts_a_calibrated_judge_and_skips_votes(fag, seeds):
    judge = LookupJudge(answers_for(seeds))
    l5 = JudgeLayer.from_spec(fag, judge, calibration=passing(fag))
    record = seeds[0]
    ctx = ValidationContext(recipe={"label": record["label"], "contestable": True})
    v5 = l5.check(record, ctx)
    assert v5.passed and v5.details["escalate"]
    ctx6 = ValidationContext(recipe=ctx.recipe, previous=(v5,))

    judge.seen.clear()
    l6 = ConsistencyLayer.from_spec(fag, judge=judge, calibration=passing(fag))
    v6 = l6.check(record, ctx6)
    assert v6.passed and v6.details["method"] == "confidence"
    assert judge.seen == []

    stale = ConsistencyLayer.from_spec(
        fag, judge=judge, calibration=passing(fag), judge_name="mock:other"
    )
    v6 = stale.check(record, ctx6)
    assert v6.details["method"] == "votes"
    assert len(judge.seen) == fag.spec.validation.consistency_k


def test_l6_untrusted_without_calibration(fag, seeds):
    judge = LookupJudge(answers_for(seeds))
    assert not ConsistencyLayer.from_spec(fag, judge=judge).trusted
