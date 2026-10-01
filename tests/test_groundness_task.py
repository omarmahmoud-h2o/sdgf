"""The groundness task's judge_context: the judge reads its own view of the task.

task.description is written for the generator (spans, is_target, the fixed parameters,
the synthesis style); rubric.judge_context gives the judge the support levels, category
minimums and worked examples without those instructions. MockBackends only.
"""

import json
from pathlib import Path

import pytest

from sdgf.generate.prompts import build_static_prefix
from sdgf.judge.llm_judge import RECORD_HEADER, LLMJudge
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.tasktypes.registry import REGISTRY

TASK = Path(__file__).resolve().parents[1] / "tasks" / "groundness" / "task.yaml"

pytestmark = pytest.mark.skipif(not TASK.exists(), reason="tasks/groundness not present")

SCORES = {"weakest_support_level": "fully supported", "realism": 4}
REPLY = json.dumps(
    {
        "verdict": "pass",
        "scores": SCORES,
        "confidence": {"verdict": 0.9, **{name: 0.9 for name in SCORES}},
    }
)
GENERATOR_ONLY = ["is_target", "fixed parameters", "annotate", "exclamation", "Write the assistant"]
CATEGORIES = [
    "essential_fact",
    "action_taken",
    "agent_capability",
    "interpretive",
    "other",
    "clarifying_question",
    "caveats_limitations",
    "general_info_navigation",
    "conversational_social",
]


@pytest.fixture(scope="module")
def groundness():
    return compile_spec(TASK)


def judge_for(compiled):
    backend = MockBackend([REPLY], cycle=True)
    return LLMJudge.from_spec(compiled, backend), backend


def test_judge_context_is_set_and_differs_from_the_description(groundness):
    rubric = groundness.spec.rubric
    assert rubric.judge_context
    assert rubric.judge_context.strip() != groundness.spec.task.description.strip()
    assert judge_for(groundness)[0].context == rubric.judge_context.strip()


def test_judge_context_has_no_generator_instructions(groundness):
    context = groundness.spec.rubric.judge_context
    for phrase in GENERATOR_ONLY:
        assert phrase not in context, phrase
    # The generator's description does carry them, so the check means something.
    assert "is_target" in groundness.spec.task.description


def test_judge_context_covers_support_levels_minimums_and_examples(groundness):
    context = groundness.spec.rubric.judge_context
    for level in groundness.spec.rubric.criteria[0].values:
        assert level in context, level
    for category in CATEGORIES:
        assert category in context, category
    for category in ("essential_fact", "action_taken", "agent_capability"):
        assert f"{category} (fully supported)" in context
    for category in ("interpretive", "other"):
        assert f"{category} (partially supported)" in context
    assert "Worked examples" in context
    # The rubric's hedge cases: a valid inference passes, a dropped hedge blocks.
    assert '"within 3 days"' in context
    assert '"up to 3 days"' in context


def test_judge_prompt_gets_judge_context_and_generation_gets_description(groundness):
    judge, backend = judge_for(groundness)
    seed = groundness.seeds[0]
    judge.judge(seed)
    prompt = backend.calls[0].prompt
    description = groundness.spec.task.description.strip()
    context = groundness.spec.rubric.judge_context.strip()
    assert "## Context\n" + context in prompt
    assert description not in prompt

    generation = build_static_prefix(groundness, REGISTRY.resolve(groundness.spec.task), [])
    assert description in generation
    assert context not in generation


def test_judge_prompt_is_blind_to_label_and_spans(groundness):
    judge, backend = judge_for(groundness)
    for seed in groundness.seeds:
        judge.judge(seed)
        judge.judge({**seed, "label": not seed["label"], "spans": []})
        assert backend.calls[-2].prompt == backend.calls[-1].prompt
        record = backend.calls[-1].prompt.split(RECORD_HEADER, 1)[1]
        assert '"label"' not in record
        assert '"spans"' not in record
        assert '"is_target"' not in record
