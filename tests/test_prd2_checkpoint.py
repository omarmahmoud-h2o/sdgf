"""PRD 2 checkpoint: a mock groundness run whose L6 votes differ across temperatures.

The groundness fixture spec (tests/fixtures/groundness, a snapshot of tasks/groundness with
mock models and consistency_k 5) sets models.judge.temperature 0.0 and no
models.consistency_judge, so L5 judges at 0.0 and L6 votes on the judge backend at the
default vote temperatures (0.7, 0.8, 0.9). The blind judge mock recovers the truth from the conversation it wrote
and flips its verdict at 0.8 only, so every escalated record gets votes that disagree
with each other while the majority still matches the label. The run's L6 details and
provenance record each vote's model and temperature. MockBackends only; never the live
tasks/groundness spec, which the user edits. The fixture is built from tasks/groundness,
which holds production-derived data, so like it the fixture stays local (git-ignored) and
this module skips when it is absent.
"""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml
from test_pipeline import filler, recipe_from_prompt

from sdgf.judge.llm_judge import RECORD_HEADER
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import split

TASK = Path(__file__).resolve().parent / "fixtures" / "groundness"
pytestmark = pytest.mark.skipif(not TASK.exists(), reason="tests/fixtures/groundness not present")

TARGET = 24
FLIP = 0.8  # the one vote temperature at which the judge mock gets it wrong
# consistency_k 5 cycling the default vote temperatures 0.7, 0.8, 0.9
VOTE_TEMPERATURES = [0.7, 0.8, 0.9, 0.7, 0.8]


@pytest.fixture(scope="module")
def groundness():
    return compile_spec(TASK)


def tools_for(agent: str) -> list[str]:
    params = yaml.safe_load((TASK / "task.yaml").read_text())["coverage"]["params"]
    return params["agents"][agent]["tools"]


class World:
    """A recipe-driven generator and a blind, temperature-sensitive judge."""

    def __init__(self) -> None:
        self.truth: dict[str, bool] = {}
        self.judge_temperatures: list[float] = []

    def generate(self, call) -> str:
        recipe = recipe_from_prompt(call.prompt)
        turns, agent = recipe["turn_count"], recipe["agent"]
        tool = tools_for(agent)[0]
        n_calls = 2 if recipe["defect_type"] == "sources_conflict" else 1
        claim = f"The {recipe['primary_topic'].replace('_', ' ')} detail is as shown"
        messages = []
        for turn in range(1, turns + 1):
            if turn % 2:
                content = f"Customer asks about {filler(recipe, turn)}."
                messages.append({"turn": turn, "role": "user", "content": content})
                continue
            msg = {"turn": turn, "role": "assistant"}
            if turn == turns:
                msg["content"] = f"{claim}. Context: {filler(recipe, turn)}."
                msg["tool_calls"] = [
                    {
                        "name": tool,
                        "caller": agent,
                        "inputs": {},
                        "outputs": f"result {i}: {filler(recipe, turn + i, 6)}",
                    }
                    for i in range(n_calls)
                ]
            else:
                msg["content"] = f"Earlier answer: {filler(recipe, turn)}."
            messages.append(msg)
        self.truth[messages[0]["content"]] = recipe["label"]
        span = {
            "turn": turns,
            "text": claim,
            "category": recipe["target_category"],
            "support_level": recipe["target_support_level"],
            "is_target": True,
        }
        return json.dumps(
            {"messages": messages, "spans": [span], "reasoning_summary": "Fictional mock."}
        )

    def judge(self, call) -> str:
        record = call.prompt.split(RECORD_HEADER, 1)[1]
        (block,) = {v for k, v in self.truth.items() if json.dumps(k)[1:-1] in record}
        self.judge_temperatures.append(call.temperature)
        if call.temperature == FLIP:
            block = not block
        return json.dumps(
            {
                "verdict": "block" if block else "pass",
                "scores": {"weakest_support_level": "fully supported", "realism": 4},
                "confidence": {"verdict": 0.9, "weakest_support_level": 0.9, "realism": 0.9},
            }
        )

    def backends(self):
        return {"generator": MockBackend(self.generate), "judge": MockBackend(self.judge)}


@pytest.fixture(scope="module")
def mock_run(groundness, tmp_path_factory):
    world = World()
    pipe = Pipeline(
        groundness,
        tmp_path_factory.mktemp("prd2") / "store",
        model_overrides=world.backends(),
        target_size=TARGET,
    )
    return world, pipe, pipe.run("prd2")


def l6_results(row):
    _, prov = split(row)
    return [r for r in prov.layer_results if r.layer == "L6"]


def voted_results(result):
    return [r for row in result.accepted for r in l6_results(row) if r.ballots]


def test_groundness_votes_on_the_judge_at_the_default_temperatures(mock_run):
    _, _, result = mock_run
    voted = voted_results(result)
    assert voted, "no record escalated to L6"
    for res in voted:
        assert [b.temperature for b in res.ballots] == VOTE_TEMPERATURES
        assert {(b.stage, b.model) for b in res.ballots} == {("judge", "mock")}


def test_mock_groundness_run_fills_every_cell(mock_run):
    _, _, result = mock_run
    assert result.complete and len(result.accepted) == TARGET
    labels = Counter(split(r)[0]["label"] for r in result.accepted)
    assert set(labels) == {True, False}


def test_escalated_records_get_votes_that_differ_across_temperatures(mock_run):
    world, _, result = mock_run
    for res in voted_results(result):
        assert res.outcome == "pass"
        flipped = [b.vote for b in res.ballots if b.temperature == FLIP]
        kept = [b.vote for b in res.ballots if b.temperature != FLIP]
        # the 0.8 votes disagree with the others, which still hold the majority
        assert len(set(flipped)) == len(set(kept)) == 1 and flipped[0] != kept[0]
        assert (len(kept), len(flipped)) == (3, 2)
    # L5 judged at 0.0, L6 at the vote temperatures: never one verdict repeated K times.
    assert set(world.judge_temperatures) == {0.0, 0.7, 0.8, 0.9}


def test_unescalated_records_take_no_votes(mock_run):
    _, _, result = mock_run
    for row in result.accepted:
        bare, _ = split(row)
        if bare["difficulty"] == "easy" and not bare["contestable"]:
            assert all(not r.ballots for r in l6_results(row))
