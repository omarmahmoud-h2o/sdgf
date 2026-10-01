"""The CFA spec (M9): the answer-emergent use case runs end to end on the same core as FAG.

MockBackends only; no model or API calls. The world keeps the correct option of every
question it writes, so the blind judge and the K voters can answer the question itself.
"""

import json
import random
from pathlib import Path

import jsonschema
import pytest

from sdgf.coverage.axes import BLOOM_LEVELS
from sdgf.evaluation.reports import DATASET, verify_release
from sdgf.generate.prompts import CELL_HEADER
from sdgf.judge.llm_judge import RECORD_HEADER
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import PROVENANCE_KEY
from sdgf.tasktypes.registry import REGISTRY
from sdgf.tasktypes.sft_qa import MULTIPLE_CHOICE
from sdgf.validate.repair import FEEDBACK_HEADER
from test_m4_checkpoint import World as FagWorld
from test_pipeline import fag  # noqa: F401

CFA_DIR = Path(__file__).resolve().parents[1] / "tasks" / "cfa"
TARGET = 12
# No gold set (kappa, residual error) and no embedder (semantic diversity) in tests.
WAIVE = ["kappa_min", "residual_error_max", "semantic_diversity_min"]

VOCAB = (
    "coupon duration convexity yield spread premium discount maturity accrual callable "
    "putable sinking hedge swap forward future option strike expiry volatility gamma "
    "delta theta vega beta alpha sharpe treynor jensen tracking benchmark index passive "
    "active factor momentum value growth dividend payout retention buyback leverage "
    "solvency liquidity turnover margin accrual depreciation amortisation goodwill "
    "impairment inventory receivable payable covenant collateral tranche senior junior "
    "mezzanine arbitrage basis carry rollover liability surplus annuity pension endowment "
    "trust fiduciary custody disclosure compliance conduct priority independence"
).split()


def cell_params(prompt: str) -> dict:
    section = prompt.split(CELL_HEADER, 1)[1].split("\n\n", 1)[0]
    return dict(
        (key, json.loads(value))
        for key, value in (line[2:].split(": ", 1) for line in section.strip().splitlines())
    )


class CfaWorld:
    """Expansion, generator and judge mocks. The generator's first try is wrong for every
    `wrong_every`-th question, so the blind judge disagrees and repair fixes it."""

    def __init__(self, *, wrong_every=4, unreadable_vote=True):
        self.truth: dict[str, str] = {}
        self.first = 0
        self.wrong_every = wrong_every
        self.unreadable_vote = unreadable_vote
        self.expansion_prompts: list[str] = []
        self.judge_prompts: list[str] = []
        self.answer_prompts: list[str] = []

    def expand(self, call) -> str:
        self.expansion_prompts.append(call.prompt)
        if call.prompt.rstrip().endswith("Core Keywords:"):
            return "duration, credit_spread"
        if call.prompt.rstrip().endswith("Prerequisite Concepts:"):
            return "time_value_of_money"
        return "option_greeks"

    def generate(self, call) -> str:
        params = cell_params(call.prompt)
        repair = FEEDBACK_HEADER in call.prompt
        if not repair:
            self.first += 1
        rng = random.Random(f"{params['keyword']}|{params['bloom_level']}|{self.first}")
        options = ["A", "B", "C", "D"]
        stem = " ".join(rng.sample(VOCAB, 12))
        lines = [f"For {params['keyword']} at the {params['bloom_level']} level: {stem}?"]
        lines += [f"{o}) {' '.join(rng.sample(VOCAB, 4))}" for o in options]
        question = "\n".join(lines)
        correct = rng.choice(options)
        self.truth[question] = correct
        given = correct
        if not repair and self.wrong_every and self.first % self.wrong_every == 0:
            given = options[(options.index(correct) + 1) % 4]
        response = f"Reason: {' '.join(rng.sample(VOCAB, 10))}.\nAnswer: {given}"
        return json.dumps({"question": question, "response": response, "answer": given})

    def _question_in(self, text: str) -> str:
        (question,) = [q for q in self.truth if json.dumps(q)[1:-1] in text or q in text]
        return question

    def judge(self, call) -> str:
        if RECORD_HEADER in call.prompt:
            self.judge_prompts.append(call.prompt)
            record = call.prompt.split(RECORD_HEADER, 1)[1]
            verdict = self.truth[self._question_in(record)]
            return json.dumps(
                {
                    "verdict": verdict,
                    "scores": {"question_quality": 4},
                    "confidence": {"verdict": 0.9, "question_quality": 0.8},
                }
            )
        # An L6 vote: the question alone plus the multiple-choice suffix.
        self.answer_prompts.append(call.prompt)
        vote = len(self.answer_prompts)
        if self.unreadable_vote and vote % 5 == 0:
            return "I am not sure."
        return f"Reason: worked through the options.\nAnswer: {self.truth[self._question_in(call.prompt)]}"

    def backends(self):
        return {
            "expansion": MockBackend(self.expand),
            "generator": MockBackend(self.generate),
            "judge": MockBackend(self.judge),
        }


@pytest.fixture(scope="module")
def cfa():
    return compile_spec(CFA_DIR)


@pytest.fixture(scope="module")
def released(cfa, tmp_path_factory):
    root = tmp_path_factory.mktemp("cfa")
    world = CfaWorld()
    pipe = Pipeline(cfa, root / "store", model_overrides=world.backends(), target_size=TARGET)
    return world, pipe, pipe.release(root / "releases", "cfa", waive=WAIVE)


def released_records(result) -> list[dict]:
    lines = (result.path / DATASET).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


# ── spec and seeds ───────────────────────────────────────────────


def test_cfa_spec_compiles_as_answer_emergent_sft_qa(cfa):
    spec = cfa.spec
    assert (spec.task.type, spec.task.generation_mode) == ("sft_qa", "answer_emergent")
    assert REGISTRY.resolve(spec.task).format is MULTIPLE_CHOICE
    # The DS²-Instruct cfa task description, ported verbatim.
    assert spec.task.description.startswith(
        "Your task is to answer CFA exam questions in a multi-choice form."
    )
    axes = {a.name: a.source for a in spec.coverage.axes}
    assert axes == {"keyword": "keyword_expansion", "bloom_level": "bloom"}
    assert spec.models.expansion is not None
    assert spec.validation.consistency_k == 5 and spec.validation.escalation.always
    assert "L6" in spec.validation.layers
    assert spec.rubric.verdict.values == ["A", "B", "C", "D"]
    assert not cfa.hooks.source  # no hooks: nothing is fixed by code but the cell


def test_cfa_seeds_are_valid_sft_qa_records(cfa):
    task_type = REGISTRY.resolve(cfa.spec.task)
    schema = task_type.output_schema(cfa.spec.output_schema)
    assert len(cfa.seeds) == 5
    for seed in cfa.seeds:
        jsonschema.validate(seed, schema)
        assert all(v(seed) == [] for v in task_type.default_validators())
        assert seed["answer"] == MULTIPLE_CHOICE.extractor(seed["response"])
        assert seed["bloom_level"] in BLOOM_LEVELS
    assert len({s["question"] for s in cfa.seeds}) == 5
    assert len({s["bloom_level"] for s in cfa.seeds}) == 5


# ── end to end ───────────────────────────────────────────────────


def test_cfa_releases_sft_pairs(released):
    world, pipe, result = released
    assert result.released and result.stop_reason == "released"
    assert verify_release(result.path)["records"] == TARGET
    records = released_records(result)
    assert len(records) == TARGET
    for r in records:
        assert set(r) >= {"question", "response", "answer", "keyword", "bloom_level"}
        assert r["answer"] == MULTIPLE_CHOICE.extractor(r["response"])
        # the released answer is the one the world knows is right
        assert r["answer"] == world.truth[r["question"]]


def test_cfa_plan_crosses_expanded_keywords_with_bloom_levels(released):
    world, pipe, result = released
    plan = pipe.plan()
    keywords = plan.keywords["keyword_expansion"]
    assert {"bonds", "capm", "duration", "option_greeks"} <= set(keywords)
    assert len(plan.cells) == len(keywords) * len(BLOOM_LEVELS)
    assert sum(c.quota for c in plan.cells) == TARGET
    # every expansion prompt carries the keywords found so far (§12.1)
    assert all("Existing Keywords: portfolio_theory" in p for p in world.expansion_prompts)


def test_cfa_generation_prompt_carries_bloom_guidance_and_answer_form(released):
    world, pipe, result = released
    prompt = pipe.generator.prompts.build({"keyword": "bonds", "bloom_level": "Apply"}).text
    assert BLOOM_LEVELS["Apply"] in prompt
    assert MULTIPLE_CHOICE.suffix in prompt
    assert "work out the answer yourself" in prompt


def test_cfa_judge_and_voters_never_see_the_answer(released):
    world, pipe, result = released
    assert world.judge_prompts and world.answer_prompts
    for prompt in world.judge_prompts + world.answer_prompts:
        assert "Answer: " not in prompt.replace(MULTIPLE_CHOICE.suffix, "")
        assert '"response"' not in prompt and '"answer"' not in prompt
    assert all(p.endswith(MULTIPLE_CHOICE.suffix) for p in world.answer_prompts)


def test_cfa_every_record_is_voted_and_passes_l6(released):
    world, pipe, result = released
    accepted = result.run.read_jsonl("accepted")
    assert len(accepted) == TARGET
    for r in accepted:
        final = r[PROVENANCE_KEY]["repair_count"]
        (l6,) = [
            x
            for x in r[PROVENANCE_KEY]["layer_results"]
            if x["layer"] == "L6" and x["attempt"] == final
        ]
        # With the record's answer checked against the majority, a pass means they agree.
        assert l6["outcome"] == "pass"
    # escalation.always: K = 5 votes for every record that reached L6, not only hard ones
    reached = sum(
        1 for r in accepted for x in r[PROVENANCE_KEY]["layer_results"] if x["layer"] == "L6"
    )
    assert len(world.answer_prompts) == 5 * reached >= 5 * TARGET
    # every fifth vote is unreadable and abstains instead of counting against the
    # majority (§12.1), and still no record fails L6
    assert all(
        x["outcome"] == "pass"
        for r in accepted
        for x in r[PROVENANCE_KEY]["layer_results"]
        if x["layer"] == "L6"
    )
    # each vote is in provenance with the stage and temperature it was sampled at
    for r in accepted:
        for x in r[PROVENANCE_KEY]["layer_results"]:
            if x["layer"] == "L6":
                assert [b["temperature"] for b in x["ballots"]] == [0.7, 0.8, 0.9, 0.7, 0.8]
                assert {b["stage"] for b in x["ballots"]} == {"judge"}
                assert [b["vote"] for b in x["ballots"]].count(None) >= 1


def test_cfa_wrong_first_answers_are_repaired_at_l5(released):
    world, pipe, result = released
    accepted = result.run.read_jsonl("accepted")
    repaired = [r for r in accepted if r[PROVENANCE_KEY]["repair_count"] > 0]
    assert repaired
    for r in repaired:
        failed = [x for x in r[PROVENANCE_KEY]["layer_results"] if x["outcome"] != "pass"]
        assert {x["layer"] for x in failed} == {"L5"}
    assert result.metrics.overall.fidelity == 1.0


def test_cfa_release_lists_only_local_endpoints(released):
    world, pipe, result = released
    endpoints = pipe.models.endpoints()
    # L6's answers come from the judge stage, so no extra endpoint receives data
    assert {e["stage"] for e in endpoints} == {"expansion", "generator", "judge"}
    assert all(e["hosting"] == "local" for e in endpoints)
    assert pipe.models.external_endpoints() == []


def test_both_generation_modes_run_on_the_same_pipeline(fag, cfa, tmp_path):  # noqa: F811
    fag_pipe = Pipeline(
        fag, tmp_path / "fag", model_overrides=FagWorld().backends(), target_size=10
    )
    fag_result = fag_pipe.run("fag")
    cfa_pipe = Pipeline(cfa, tmp_path / "cfa", model_overrides=CfaWorld().backends(), target_size=6)
    cfa_result = cfa_pipe.run("cfa")
    assert fag_result.complete and cfa_result.complete
    assert type(fag_pipe.cascade) is type(cfa_pipe.cascade)
    assert fag_pipe.cascade.names == cfa_pipe.cascade.names == ("L1", "L2", "L3", "L4", "L5", "L6")
    assert all("label" in r for r in fag_result.accepted)
    assert all("answer" in r and "label" not in r for r in cfa_result.accepted)
