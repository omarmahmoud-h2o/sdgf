"""M4 checkpoint: the FAG pipeline end to end with the judge layers, L1-L6, on MockBackends.

The generator mock sometimes writes a "no breach" conversation whose assistant advises,
the §12.2 blind spot; the blind judge mock reads the conversation and says breach, so L5
repairs it in its cell. Hard and contestable records escalate to L6 and get K judge votes,
unless a stored calibration makes the judge trusted. Low-confidence verdicts go to the
review stream when hitl.review_flagged is on, and a spec selecting Jev fails clearly.
"""

import dataclasses
import json
from collections import Counter
from pathlib import Path

import pytest

from sdgf.judge.calibration import CalibrationResult, CalibrationStore
from sdgf.judge.jev import JevNotImplementedError
from sdgf.judge.llm_judge import RECORD_HEADER
from sdgf.models.mock import MockBackend
from sdgf.pipeline import DROPS_STREAM, REVIEW_STREAM, Pipeline, PipelineError, fixed_axis_cells
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import split
from test_pipeline import fag_reply, recipe_from_prompt

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 40
LAYERS = ("L1", "L2", "L3", "L4", "L5", "L6")
REPAIR = "previous attempt was rejected"
ADVICE = "Honestly, this account is ideal for your business, so move everything onto it."
K = 5


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


class World:
    """Generator and judge mocks sharing what the generator wrote, so the judge can be right.

    The judge is blind (it sees only the conversation), so it recovers the truth from the
    conversation text: advice wording reads as breach, otherwise the label the conversation
    was written for.
    """

    def __init__(self, *, advise_every=4, confidence=0.9):
        self.truth: dict[str, bool] = {}
        self.first = 0
        self.advised = 0
        self.advise_every = advise_every
        self.confidence = confidence
        self.judge_prompts: list[str] = []

    def generate(self, call) -> str:
        recipe = recipe_from_prompt(call.prompt)
        r = fag_reply(recipe)
        if REPAIR not in call.prompt:
            self.first += 1
            if self.advise_every and not recipe["label"] and self.first % self.advise_every == 0:
                r["messages"][-1]["content"] += " " + ADVICE
                self.advised += 1
        self.truth[r["messages"][0]["content"]] = recipe["label"]
        return json.dumps(r)

    def judge(self, call) -> str:
        prompt = call.prompt
        self.judge_prompts.append(prompt)
        record = prompt.split(RECORD_HEADER, 1)[1]
        if ADVICE in record:
            breach = True
        else:
            (breach,) = {v for k, v in self.truth.items() if json.dumps(k)[1:-1] in record}
        tier = "PERSONAL_ADVICE" if breach else "FACTUAL_INFORMATION"
        conf = self.confidence
        return json.dumps(
            {
                "verdict": "breach" if breach else "no_breach",
                "scores": {"advice_tier": tier, "realism": 4},
                "confidence": {"verdict": conf, "advice_tier": conf, "realism": 0.9},
            }
        )

    def backends(self):
        return {"generator": MockBackend(self.generate), "judge": MockBackend(self.judge)}


def run(fag, root, world, **kw):
    pipe = Pipeline(fag, root, model_overrides=world.backends(), target_size=TARGET, **kw)
    return pipe, pipe.run("m4")


@pytest.fixture(scope="module")
def judged_run(fag, tmp_path_factory):
    world = World()
    pipe, result = run(fag, tmp_path_factory.mktemp("m4") / "store", world)
    return world, pipe, result


def escalated(record) -> bool:
    return record["difficulty"] == "HARD" or record["contestable"] is True


def test_full_run_uses_l1_to_l6_and_fills_every_cell(fag, judged_run):
    _, pipe, result = judged_run
    assert result.layers == LAYERS and result.skipped_layers == ()
    assert result.complete and len(result.accepted) == TARGET
    assert result.counts == {c.id: c.quota for c in fixed_axis_cells(fag.spec.coverage, TARGET)}
    bare = [split(r)[0] for r in result.accepted]
    assert Counter(r["label"] for r in bare) == {True: 20, False: 20}  # BREACH_RATE 0.5
    spec = result.run.read_stage("spec")
    assert [m["stage"] for m in spec["models"]] == ["generator", "judge"]  # D12: no fallback
    assert spec["judge_trusted"] is False
    for row in result.accepted:
        _, prov = split(row)
        prov.check_accepted(LAYERS)
        assert {m.stage for m in prov.models} == {"generator", "judge"}


def test_advice_wording_on_a_non_breach_is_caught_and_repaired_at_l5(judged_run):
    world, _, result = judged_run
    assert world.advised > 0
    bare = [split(r)[0] for r in result.accepted]
    assert not any(ADVICE in m["content"] for r in bare for m in r["messages"])
    repaired = [split(r)[1] for r in result.accepted if split(r)[1].repair_count]
    assert len(repaired) == world.advised  # every advising first attempt was repaired
    for prov in repaired:
        first = [lr for lr in prov.layer_results if lr.attempt == 0]
        assert first[-1].layer == "L5" and first[-1].outcome == "fail_repairable"
    assert result.run.read_jsonl(DROPS_STREAM) == []


def test_the_judge_is_blind_to_the_label(judged_run):
    world, _, _ = judged_run
    for prompt in world.judge_prompts:
        record = prompt.split(RECORD_HEADER, 1)[1]
        for field in ("label", "spans", "signal_categories", "advice_tier", "_provenance"):
            assert f'"{field}"' not in record


def test_escalated_records_get_k_votes_at_l6(judged_run):
    world, _, result = judged_run
    bare = [split(r)[0] for r in result.accepted]
    n_escalated = sum(escalated(r) for r in bare)
    assert 0 < n_escalated < TARGET
    # One L5 call per judged attempt, plus K votes for each escalated record that reached L6.
    attempts = TARGET + world.advised
    assert len(world.judge_prompts) >= attempts + K * n_escalated
    assert len(world.judge_prompts) <= attempts + K * (n_escalated + world.advised)


def test_a_stored_calibration_makes_the_judge_trusted_and_skips_votes(fag, tmp_path):
    root = tmp_path / "store"
    CalibrationStore(ArtefactStore(root)).save(
        CalibrationResult(
            spec_version=fag.spec_version,
            judge_id="mock:mock",  # the judge the run actually builds
            n=30,
            unparseable=0,
            accuracy=1.0,
            kappa=1.0,
            ece=0.02,
            bins=(),
            kappa_min=0.7,
            ece_max=0.1,
            min_gold=30,
        )
    )
    world = World(advise_every=0)
    pipe, result = run(fag, root, world)
    assert pipe.trusted and result.run.read_stage("spec")["judge_trusted"] is True
    assert result.complete and len(result.accepted) == TARGET
    assert len(world.judge_prompts) == TARGET  # L6 used the confidence, no votes


def test_a_calibration_for_another_judge_is_not_trusted(fag, tmp_path):
    root = tmp_path / "store"
    other = CalibrationResult(
        spec_version=fag.spec_version,
        judge_id="openai_compat:Qwen3.5-4B",
        n=30,
        unparseable=0,
        accuracy=1.0,
        kappa=1.0,
        ece=0.02,
        bins=(),
        kappa_min=0.7,
        ece_max=0.1,
        min_gold=30,
    )
    CalibrationStore(ArtefactStore(root)).save(other)
    pipe = Pipeline(fag, root, model_overrides=World().backends(), target_size=TARGET)
    assert pipe.trusted is False


def test_low_confidence_goes_to_the_review_stream_when_review_is_on(fag, tmp_path):
    spec = fag.spec.model_copy(
        update={"hitl": fag.spec.hitl.model_copy(update={"review_flagged": True})}
    )
    reviewed = dataclasses.replace(fag, spec=spec)
    world = World(advise_every=0, confidence=0.5)
    pipe = Pipeline(
        reviewed,
        tmp_path / "store",
        model_overrides=world.backends(),
        target_size=TARGET,
        max_attempts_per_cell=2,
    )
    result = pipe.run("m4")
    assert result.accepted == []  # every verdict is low confidence: nothing auto-accepted
    drops = result.run.read_jsonl(DROPS_STREAM)
    assert drops and {(d["layer"], tuple(d["codes"])) for d in drops} == {
        ("L5", ("sent_to_review",))
    }
    review = result.run.read_jsonl(REVIEW_STREAM)
    assert len(review) == len(drops)
    assert all(
        item["code"] == "low_confidence" and "_provenance" not in item["record"] for item in review
    )


def test_review_is_off_by_default_so_no_review_stream(judged_run):
    _, pipe, result = judged_run
    assert pipe.review_sink is None
    assert not result.run.jsonl_path(REVIEW_STREAM).exists()


def test_selecting_jev_fails_clearly(fag, tmp_path):
    models = fag.spec.models
    judge = models.judge.model_copy(update={"backend": "jev", "params": {}})
    spec = fag.spec.model_copy(update={"models": models.model_copy(update={"judge": judge})})
    with pytest.raises(JevNotImplementedError, match=r"models\.judge.*§7\.3"):
        Pipeline(
            dataclasses.replace(fag, spec=spec),
            tmp_path,
            model_overrides={"generator": MockBackend(["{}"])},
        )


def test_overrides_for_stages_the_run_does_not_use_are_refused(fag, tmp_path):
    overrides = {**World().backends(), "fallback_judge": MockBackend(["x"])}
    with pytest.raises(PipelineError, match="fallback_judge"):
        Pipeline(fag, tmp_path, model_overrides=overrides)
