"""Repair loop: re-prompt with validator errors, same cell, drop with a logged reason."""

import json
from pathlib import Path

import pytest

from sdgf.generate.generator import Generator
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import JsonlWriter, iter_jsonl
from sdgf.store.provenance import ProvenanceBuilder
from sdgf.validate.base import Layer, ValidationIssue
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.repair import (
    FEEDBACK_HEADER,
    GENERATE_STAGE,
    DropLog,
    RepairLoop,
    repair_feedback,
)

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
MODEL_FIELDS = ("messages", "spans", "problematic_turns", "customer_intent", "reasoning_summary")
CELL_ID = "corps_act|true|short"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture
def seed(fag):
    return json.loads(json.dumps(next(s for s in fag.seeds if s["label"] is True)))


def split_seed(seed):
    """The code-owned recipe and the model-written part of a seed record."""
    recipe = {k: v for k, v in seed.items() if k not in MODEL_FIELDS}
    model = {k: seed[k] for k in MODEL_FIELDS}
    return recipe, model


def reworded(model):
    bad = json.loads(json.dumps(model))
    bad["spans"][0]["text"] = "A paraphrase that is not in the assistant turn."
    return bad


def fag_loop(fag, replies, **kw):
    backend = MockBackend([json.dumps(r) if isinstance(r, dict) else r for r in replies])
    gen = Generator(fag, backend)
    cascade = Cascade([SchemaLayer.from_spec(fag), RulesLayer.from_spec(fag)])
    return RepairLoop(gen, cascade, **kw), backend


def builder():
    models = [{"stage": "generator", "backend": "mock", "model": "mock", "hosting": "local"}]
    return ProvenanceBuilder("v1", CELL_ID, 7, models)


# ── FAG: fixed on the second try ─────────────────────────────


def test_fag_fixed_on_second_try_is_accepted(fag, seed):
    recipe, model = split_seed(seed)
    loop, backend = fag_loop(fag, [reworded(model), model])
    prompt = loop.generator.prompts.build(recipe)
    prov = builder()
    out = loop.run(CELL_ID, recipe, prompt, prov)

    assert out.accepted and out.attempts == 2 and out.repairs == 1
    assert out.history == [("L2", ("span_not_verbatim",))]
    assert out.record["label"] is True and out.record["spans"] == seed["spans"]
    assert len(loop.drop_log) == 0

    # the repair prompt is the original prompt plus the specific errors
    first, second = (c.prompt for c in backend.calls)
    assert first == prompt.text
    assert second.startswith(prompt.text + "\n\n" + FEEDBACK_HEADER)
    assert "span_not_verbatim at spans[0].text" in second

    p = prov.build()
    assert p.repair_count == 1
    assert [(r.layer, r.outcome, r.attempt) for r in p.layer_results] == [
        ("L1", "pass", 0),
        ("L2", "fail_repairable", 0),
        ("L1", "pass", 1),
        ("L2", "pass", 1),
    ]
    p.check_accepted(["L1", "L2"])


def test_repair_keeps_the_cell_recipe(fag, seed):
    """A repaired reply that tries to change the label still gets the cell's label."""
    recipe, model = split_seed(seed)
    loop, _ = fag_loop(fag, [reworded(model), {**model, "label": False, "severity": None}])
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    assert out.accepted and out.record["label"] is True and out.record["severity"] == "HIGH"


def test_fag_exhausted_drops_with_reason_per_cell_and_layer(fag, seed):
    recipe, model = split_seed(seed)
    loop, backend = fag_loop(fag, [reworded(model)] * 3, repair_tries=2)
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))

    assert not out.accepted and out.attempts == 3 and len(backend.calls) == 3
    assert out.drop.layer == "L2" and out.drop.cell_id == CELL_ID and not out.drop.hard
    assert "span_not_verbatim" in out.drop.codes
    assert "2 repair(s) exhausted" in out.drop.reason
    assert loop.drop_log.by_cell_layer() == {(CELL_ID, "L2"): 1}
    assert loop.drop_log.by_code()["span_not_verbatim"] == 1


def test_repair_tries_default_from_spec(fag, seed):
    recipe, model = split_seed(seed)
    loop, backend = fag_loop(fag, [reworded(model)])
    backend.cycle = True
    assert loop.repair_tries == fag.spec.validation.repair_tries
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    assert out.attempts == fag.spec.validation.repair_tries + 1


def test_zero_repair_tries_drops_on_first_failure(fag, seed):
    recipe, model = split_seed(seed)
    loop, backend = fag_loop(fag, [reworded(model), model], repair_tries=0)
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    assert not out.accepted and len(backend.calls) == 1


def test_first_try_pass_needs_no_repair(fag, seed):
    recipe, model = split_seed(seed)
    loop, backend = fag_loop(fag, [model])
    prov = builder()
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe), prov)
    assert out.accepted and out.attempts == 1 and out.history == []
    assert prov.build().repair_count == 0 and len(backend.calls) == 1


def test_unparseable_reply_is_fed_back_and_repaired(fag, seed):
    recipe, model = split_seed(seed)
    loop, backend = fag_loop(fag, ["sorry, here you go: not json", model])
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    assert out.accepted and out.history == [(GENERATE_STAGE, ("no_json",))]
    assert "no_json: " in backend.calls[1].prompt


def test_unparseable_every_time_drops_at_generate(fag, seed):
    recipe, _ = split_seed(seed)
    loop, _ = fag_loop(fag, ["nope"] * 2, repair_tries=1)
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    assert out.drop.layer == GENERATE_STAGE and out.drop.codes == ("no_json",)
    assert out.result is None


# ── hard failures and the drop log ───────────────────────────


class HardLayer(Layer):
    name = "L3"

    def check(self, record, context):
        return self.verdict([ValidationIssue("pii_tfn", "fictional TFN found")], repairable=False)


def test_hard_failure_is_dropped_without_repair(fag, seed):
    recipe, model = split_seed(seed)
    backend = MockBackend([json.dumps(model)] * 3)
    cascade = Cascade([SchemaLayer.from_spec(fag), RulesLayer.from_spec(fag), HardLayer()])
    loop = RepairLoop(Generator(fag, backend), cascade, repair_tries=2)
    out = loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    assert len(backend.calls) == 1 and out.attempts == 1
    assert out.drop.hard and out.drop.layer == "L3" and "not repaired" in out.drop.reason
    assert loop.drop_log.by_layer() == {"L3": 1}


def test_drop_log_streams_to_jsonl(fag, seed, tmp_path):
    recipe, model = split_seed(seed)
    path = tmp_path / "drops.jsonl"
    with JsonlWriter(path) as writer:
        loop, _ = fag_loop(fag, [reworded(model)], repair_tries=0, drop_log=DropLog(writer))
        loop.run(CELL_ID, recipe, loop.generator.prompts.build(recipe))
    (row,) = list(iter_jsonl(path))
    assert row["cell_id"] == CELL_ID and row["layer"] == "L2" and row["attempts"] == 1
    assert row["errors"][0]["code"] == "span_not_verbatim"


def test_repair_feedback_lists_each_error():
    text = repair_feedback([ValidationIssue("a", "first", path="x"), "b: second"])
    assert text.splitlines()[0] == FEEDBACK_HEADER
    assert "- a at x: first" in text and "- b: second" in text


def test_negative_repair_tries_rejected(fag):
    loop_args = (Generator(fag, MockBackend(["{}"])), Cascade([SchemaLayer.from_spec(fag)]))
    with pytest.raises(ValueError):
        RepairLoop(*loop_args, repair_tries=-1)
