import json
import random
import shutil
from pathlib import Path

import pytest

from sdgf.generate.generator import Generator, GeneratorError, extract_json, merge
from sdgf.generate.scheduler import Cell
from sdgf.models.base import ModelResponse, ToolCall
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CELL = {"product_scope": "corps_act", "label": True, "conversation_length": "single_turn"}


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def model_reply(**overrides):
    """A model reply that also tries to set code-owned fields, which must be ignored."""
    body = {
        "customer_intent": "Asks which account suits the business.",
        "messages": [
            {
                "turn": 1,
                "role": "customer",
                "content": "Which account should Acme Test Pty Ltd use?",
            },
            {
                "turn": 2,
                "role": "assistant",
                "content": "For your business, pick the Everyday account.",
            },
        ],
        "spans": [
            {"turn": 2, "text": "pick the Everyday account", "category": "PRODUCT_RECOMMENDATION"}
        ],
        "problematic_turns": [2],
        "reasoning_summary": "Recommends a product for the customer's business.",
        "expected_ai_behavior": "Give facts and decline to recommend.",
        # code-owned fields the model must not control
        "label": False,
        "advice_tier": "FACTUAL_INFORMATION",
        "product_scope": "non_corps_act",
        "_provenance": {"forged": True},
    }
    body.update(overrides)
    return json.dumps(body)


# ------------------------------------------------------------- extract_json


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        '```json\n{"a": 1}\n```',
        '```\n{"a": 1}\n```',
        'Here you go: {"a": 1} hope that helps',
        '  \n{"a": 1}\n  ',
    ],
)
def test_extract_json_tolerates_fences_and_chatter(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize(
    "text", [None, "", "no json here", "{not json}", "} backwards {", "[1, 2]"]
)
def test_extract_json_returns_none_when_unparseable(text):
    assert extract_json(text) is None


def test_extract_json_matches_the_original(monkeypatch):
    import importlib.util
    import sys

    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    scripts = FAG_DIR.parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("_orig_utils", scripts / "utils.py")
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    for text in ['```json\n{"a": [1, {"b": 2}]}\n```', "x {bad} y", "", '{"k": "v"} tail']:
        assert extract_json(text) == orig.extract_json(text)


# -------------------------------------------------------------------- merge


def test_merge_recipe_wins_and_private_keys_dropped():
    merged = merge({"label": True, "tier": "X"}, {"label": False, "prose": "p", "_x": 1})
    assert merged == {"label": True, "tier": "X", "prose": "p"}


# ---------------------------------------------------------------- generator


def test_labels_come_from_the_cell_never_the_model(fag):
    backend = MockBackend([model_reply()])
    result = Generator(fag, backend).generate(Cell("c1", CELL, 3), random.Random(0))
    assert result.ok and result.error is None
    rec = result.record
    assert rec["label"] is True
    assert rec["product_scope"] == "corps_act"
    assert rec["advice_tier"] == result.recipe["advice_tier"] != "FACTUAL_INFORMATION"
    assert fag.hooks.label_rule(rec) is True
    assert "_provenance" not in rec
    # prose comes from the model
    assert rec["reasoning_summary"].startswith("Recommends")
    assert rec["messages"][1]["role"] == "assistant"
    assert result.cell_id == "c1"


def test_every_recipe_field_is_in_the_record(fag):
    result = Generator(fag, MockBackend([model_reply()])).generate(CELL, random.Random(1))
    for key, value in result.recipe.items():
        assert result.record[key] == value
    assert result.cell_id is None


def test_sampler_constraints_applied_and_seeded(fag):
    def recipe(seed):
        backend = MockBackend([model_reply()])
        return Generator(fag, backend).generate(CELL, random.Random(seed)).recipe

    assert recipe(7) == recipe(7)
    r = recipe(7)
    for key in (
        "advice_tier",
        "signal_categories",
        "severity",
        "is_corps_question",
        "denial_present",
    ):
        assert key in r
    assert r["severity"] is not None  # breach => severity


def test_prompt_contains_recipe_and_uses_spec_settings(fag):
    backend = MockBackend([model_reply()])
    gen = Generator(fag, backend)
    result = gen.generate(CELL, random.Random(2))
    call = backend.calls[0]
    assert call.prompt == result.prompt.text
    assert call.prompt.startswith(gen.prompts.static_prefix)
    assert json.dumps(result.recipe["advice_tier"]) in result.prompt.cell
    assert call.max_tokens == fag.spec.models.generator.max_tokens
    assert call.temperature == fag.spec.models.generator.temperature
    assert call.tools is None


def test_overrides_for_tokens_and_temperature(fag):
    backend = MockBackend([model_reply()])
    Generator(fag, backend, max_tokens=123, temperature=0.1).generate(CELL, random.Random(0))
    assert (backend.calls[0].max_tokens, backend.calls[0].temperature) == (123, 0.1)


def test_fenced_reply_is_accepted(fag):
    backend = MockBackend([f"```json\n{model_reply()}\n```"])
    assert Generator(fag, backend).generate(CELL, random.Random(0)).ok


@pytest.mark.parametrize(
    "reply, error",
    [
        ("sorry, I can't do that", "no_json"),
        ('{"messages": [', "no_json"),
        (ModelResponse(text=None), "no_text"),
        (ModelResponse(text="", tool_calls=(ToolCall("lookup", {"q": "x"}),)), "tool_calls"),
    ],
)
def test_failures_are_returned_with_machine_readable_errors(fag, reply, error):
    result = Generator(fag, MockBackend([reply])).generate(Cell("c", CELL, 1), random.Random(0))
    assert not result.ok
    assert result.error == error and result.detail
    assert result.cell_id == "c"
    assert result.recipe["label"] is True  # the recipe is kept so the retry stays in-cell
    assert result.response is not None


def test_invalid_cell_skips_the_model(fag):
    backend = MockBackend([model_reply()])
    # general advice on a non-Corps product is permitted, so it can't be a breach
    bad = {**CELL, "product_scope": "non_corps_act", "advice_tier": "GENERAL_ADVICE"}
    result = Generator(fag, backend).generate(bad, random.Random(0))
    assert result.error == "invalid_cell" and result.recipe is None
    assert backend.calls == []


def test_complete_appends_extra_after_the_prompt(fag):
    backend = MockBackend([model_reply()])
    gen = Generator(fag, backend)
    recipe = gen.recipe(CELL, random.Random(0))
    prompt = gen.prompts.build(recipe)
    result = gen.complete("c", recipe, prompt, extra="Fix: span not found verbatim in turn 2")
    assert result.ok
    assert backend.calls[0].prompt == prompt.text + "\n\nFix: span not found verbatim in turn 2"
    assert result.prompt.prefix_hash == gen.prompts.prefix_hash


def _copy_task(tmp_path, hooks_source):
    task = tmp_path / "task"
    shutil.copytree(FAG_DIR, task)
    (task / "hooks.py").write_text(hooks_source, encoding="utf-8")
    return compile_spec(task)


def test_label_first_recipe_without_label_raises(tmp_path):
    compiled = _copy_task(
        tmp_path, "def sampler_constraints(cell, rng):\n    return {'topic': 'x'}\n"
    )
    with pytest.raises(GeneratorError, match="label"):
        Generator(compiled, MockBackend([model_reply()])).generate(CELL, random.Random(0))


def test_no_sampler_hook_uses_cell_params_as_recipe(tmp_path):
    compiled = _copy_task(tmp_path, "")
    result = Generator(compiled, MockBackend([model_reply()])).generate(CELL, random.Random(0))
    assert result.recipe == CELL
    assert result.record["label"] is True
