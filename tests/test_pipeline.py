"""Pipeline stages 0, 2 and 3 (L1, L2) end to end on the FAG spec with a MockBackend."""

import json
from collections import Counter
from pathlib import Path

import pytest

from sdgf.generate.prompts import CELL_HEADER
from sdgf.models.mock import MockBackend
from sdgf.pipeline import (
    ACCEPTED_STREAM,
    DROPS_STREAM,
    Pipeline,
    PipelineError,
    candidate_seed,
    fixed_axis_cells,
)
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import CoverageSection
from sdgf.store.provenance import split

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 20


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def recipe_from_prompt(prompt: str) -> dict:
    """The fixed parameters the pipeline put at the end of the prompt."""
    section = prompt.split(CELL_HEADER, 1)[1].split("\n\n", 1)[0]
    recipe = {}
    for line in section.strip().splitlines():
        key, value = line[2:].split(": ", 1)
        recipe[key] = json.loads(value)
    return recipe


def fag_reply(recipe: dict, *, reword: bool = False) -> dict:
    """A valid model reply for a FAG recipe: one advisory sentence per declared signal."""
    turns = recipe["turn_count"]
    sentences = [f"On point {i + 1}, the Business Flex account suits you." for i in range(3)]
    signals = recipe["signal_categories"]
    advisory = [f"Here is what I'd flag ({sig.lower()}) for Acme Test Pty Ltd." for sig in signals]
    messages = []
    for turn in range(1, turns + 1):
        if turn % 2:
            content = f"Customer question {turn} about the {recipe['primary_topic']}."
        elif turn == turns:
            content = " ".join(["Thanks for asking.", *advisory, sentences[0]])
        else:
            content = f"Factual answer {turn}: the monthly fee is $10."
        messages.append(
            {"turn": turn, "role": "customer" if turn % 2 else "assistant", "content": content}
        )
    spans = []
    if recipe["label"]:
        spans = [
            {"turn": turns, "text": text, "category": sig} for text, sig in zip(advisory, signals)
        ]
        if reword and spans:
            spans[0]["text"] = "A paraphrase that is not in the assistant turn."
    return {
        "messages": messages,
        "spans": spans,
        "problematic_turns": [turns] if recipe["label"] else [],
        "customer_intent": "Wants to know about an account.",
        "reasoning_summary": "Fictional mock reasoning.",
        "expected_ai_behavior": "Give facts, decline personal advice.",
        # The model tries to set the label; the cell must win.
        "label": not recipe["label"],
    }


def valid_backend(**kw):
    return MockBackend(lambda call: json.dumps(fag_reply(recipe_from_prompt(call.prompt))), **kw)


def run(fag, tmp_path, backend, **kw):
    run_id = kw.pop("run_id", "r1")
    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": backend},
        target_size=kw.pop("target_size", TARGET),
        **kw,
    )
    return pipe, pipe.run(run_id)


# ── cells ───────────────────────────────────────────────────────


def test_fixed_axis_cells_cross_axes_and_sum_to_target(fag):
    cells = fixed_axis_cells(fag.spec.coverage, TARGET)
    assert len(cells) == 2 * 2 * 4
    assert sum(c.quota for c in cells) == TARGET
    assert cells[0].id == "corps_act|true|single_turn"
    assert cells[0].params == {
        "product_scope": "corps_act",
        "label": True,
        "conversation_length": "single_turn",
    }


def test_weighted_quotas_follow_axis_weights(fag):
    cells = fixed_axis_cells(fag.spec.coverage, 1000)
    by_scope = Counter()
    by_label = Counter()
    for c in cells:
        by_scope[c.params["product_scope"]] += c.quota
        by_label[c.params["label"]] += c.quota
    assert by_scope == {"corps_act": 650, "non_corps_act": 350}
    assert by_label == {True: 500, False: 500}  # BREACH_RATE 0.5


def test_even_quota_policy():
    cov = CoverageSection(
        target_size=7,
        quota_policy="even",
        axes=[{"name": "a", "values": [1, 2], "weights": [0.9, 0.1]}],
    )
    assert [c.quota for c in fixed_axis_cells(cov)] == [4, 3]


def test_non_fixed_axis_needs_coverage_plan():
    cov = CoverageSection(target_size=5, axes=[{"name": "topic", "source": "keyword_expansion"}])
    with pytest.raises(PipelineError, match="coverage plan"):
        fixed_axis_cells(cov)


def test_candidate_seed_is_stable_and_distinct():
    assert candidate_seed(0, "c", 1) == candidate_seed(0, "c", 1)
    assert len({candidate_seed(0, "c", i) for i in range(50)}) == 50
    assert candidate_seed(0, "c", 1) != candidate_seed(1, "c", 1)


# ── end to end ──────────────────────────────────────────────────


def test_fag_end_to_end_20_records(fag, tmp_path):
    backend = valid_backend()
    pipe, result = run(fag, tmp_path, backend)

    assert result.complete
    assert len(result.accepted) == TARGET
    assert len(backend.calls) == TARGET  # every candidate accepted first time
    assert len(result.drops) == 0
    cells = {c.id: c.quota for c in fixed_axis_cells(fag.spec.coverage, TARGET)}
    assert result.counts == cells
    assert result.layers == ("L1", "L2")
    assert result.skipped_layers == ("L3", "L4", "L5", "L6")

    stored = result.run.read_jsonl(ACCEPTED_STREAM)
    assert stored == result.accepted
    assert result.run.read_jsonl(DROPS_STREAM) == []
    for rec in stored:
        bare, prov = split(rec)
        prov.check_accepted(("L1", "L2"))
        assert prov.spec_version == fag.spec_version
        assert prov.run_id == "r1"
        assert prov.repair_count == 0
        assert prov.model("generator").backend == "mock"
        cell = dict(zip(("product_scope", "label", "conversation_length"), prov.cell_id.split("|")))
        assert bare["product_scope"] == cell["product_scope"]
        assert json.dumps(bare["label"]) == cell["label"]  # label came from the cell
        assert bare["label"] == fag.hooks.label_rule(bare)
        assert "policy_categories" in bare  # post_process ran
        assert bare["policy_categories"] == fag.hooks.post_process(bare)["policy_categories"]
        assert fag.hooks.extra_validators(bare) == []

    spec = result.run.read_stage("spec")
    assert spec["task_type"] == "classification_spans"
    assert spec["skipped_layers"] == ["L3", "L4", "L5", "L6"]
    summary = result.run.read_stage("summary")
    assert summary["stop_reason"] == "complete"
    assert summary["drops"]["by_layer"] == {}


def test_repairable_failures_are_repaired_in_the_same_cell(fag, tmp_path):
    def reply(call):
        # Every first attempt rewords a span; the repair prompt carries the error.
        repairing = "previous attempt was rejected" in call.prompt
        recipe = recipe_from_prompt(call.prompt)
        return json.dumps(fag_reply(recipe, reword=not repairing))

    _, result = run(fag, tmp_path, MockBackend(reply))
    assert result.complete and len(result.accepted) == TARGET
    assert len(result.drops) == 0
    repairs = [split(r)[1].repair_count for r in result.accepted]
    breaches = [split(r)[0]["label"] for r in result.accepted]
    assert repairs == [1 if b else 0 for b in breaches]


def test_drops_are_logged_and_requeued_without_changing_quotas(fag, tmp_path):
    good = valid_backend()
    calls = []

    def reply(call):
        calls.append(call)
        return "not json at all" if len(calls) <= 3 else good.call(call.prompt, 1, 0.0).text

    _, result = run(fag, tmp_path, MockBackend(reply))
    assert result.complete and len(result.accepted) == TARGET
    assert result.counts == {c.id: c.quota for c in fixed_axis_cells(fag.spec.coverage, TARGET)}
    drops = result.run.read_jsonl(DROPS_STREAM)
    assert len(drops) == 1  # repair_tries 2: three bad replies drop one slot
    assert drops[0]["layer"] == "generate" and drops[0]["codes"] == ["no_json"]
    assert drops[0]["attempts"] == 3
    # the retry of the dropped slot is in the same cell
    first_cell = split(result.accepted[0])[1].cell_id
    assert drops[0]["cell_id"] == first_cell
    assert result.run.read_stage("summary")["drops"]["by_layer"] == {"generate": 1}


def test_same_seed_gives_the_same_records(fag, tmp_path):
    _, a = run(fag, tmp_path / "a", valid_backend())
    _, b = run(fag, tmp_path / "b", valid_backend())
    _, c = run(fag, tmp_path / "c", valid_backend(), seed=7)
    strip = lambda res: [split(r)[0] for r in res.accepted]  # noqa: E731
    assert strip(a) == strip(b)
    assert strip(a) != strip(c)


def test_resume_finishes_a_killed_run(fag, tmp_path):
    good = valid_backend()
    count = []

    def dies_after_eight(call):
        count.append(1)
        if len(count) > 8:
            raise RuntimeError("killed")
        return good.call(call.prompt, 1, 0.0).text

    with pytest.raises(RuntimeError, match="killed"):
        run(fag, tmp_path, MockBackend(dies_after_eight))

    _, resumed = run(fag, tmp_path, valid_backend())
    assert resumed.run.resumed
    assert resumed.complete
    assert len(resumed.accepted) == TARGET - 8
    stored = resumed.run.read_jsonl(ACCEPTED_STREAM)
    assert len(stored) == TARGET
    cells = {c.id: c.quota for c in fixed_axis_cells(fag.spec.coverage, TARGET)}
    assert Counter(split(r)[1].cell_id for r in stored) == Counter(cells)
    seeds = [split(r)[1].seed for r in stored]
    assert len(set(seeds)) == TARGET  # no candidate regenerated with a used seed


def test_resume_with_a_different_target_is_refused(fag, tmp_path):
    run(fag, tmp_path, valid_backend())
    with pytest.raises(PipelineError, match="start a new run"):
        run(fag, tmp_path, valid_backend(), target_size=TARGET + 1)


def test_unimplemented_layers_are_refused(fag, tmp_path):
    with pytest.raises(PipelineError, match="not implemented"):
        Pipeline(fag, tmp_path, model_overrides={"generator": valid_backend()}, layers=["L1", "L3"])


def test_explicit_layer_subset(fag, tmp_path):
    _, result = run(fag, tmp_path, valid_backend(), layers=["L1"])
    assert result.layers == ("L1",)
    assert result.skipped_layers == ("L2", "L3", "L4", "L5", "L6")
    split(result.accepted[0])[1].check_accepted(("L1",))
