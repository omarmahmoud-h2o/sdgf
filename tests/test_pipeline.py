"""Pipeline stages 0, 2 and 3 end to end on the FAG spec with a MockBackend.

Most tests run L1-L4 with only a generator mock; the judge layers (L5, L6) are covered in
tests/test_m4_checkpoint.py, which pass a mock judge too.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sdgf.coverage.axes import cross, resolve_axes
from sdgf.coverage.plan import assign_quotas, build_plan
from sdgf.generate.prompts import CELL_HEADER
from sdgf.models.mock import MockBackend
from sdgf.pipeline import (
    ACCEPTED_STREAM,
    DROPS_STREAM,
    Pipeline,
    PipelineError,
    candidate_seed,
)
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import CoverageSection
from sdgf.store.provenance import split
from sdgf.validate.base import ValidationContext

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 20
NO_JUDGE = ("L1", "L2", "L3", "L4")  # runs that pass no judge mock leave L5/L6 out


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


WORDS = (
    "invoice ledger payroll merchant terminal settlement overdraft facility statement "
    "transfer deposit withdrawal branch online portal card limit review schedule quarter "
    "supplier customer inventory warehouse delivery contract renewal notice balance "
    "interest fee waiver rebate cashflow forecast budget expense receipt audit "
    "morning evening weekly monthly annual regional metro rural coastal inland "
    "bakery joinery nursery studio workshop clinic cafe garage florist printer"
).split()


def filler(recipe: dict, turn: int, n: int = 12) -> str:
    """Words drawn from a hash of the recipe, so distinct recipes don't read alike at L4."""
    digest = hashlib.sha256(f"{json.dumps(recipe, sort_keys=True)}:{turn}".encode()).digest()
    return " ".join(WORDS[b % len(WORDS)] for b in digest[:n])


def fag_reply(recipe: dict, *, reword: bool = False) -> dict:
    """A valid model reply for a FAG recipe: one advisory sentence per declared signal."""
    turns = recipe["turn_count"]
    sentences = [f"On point {i + 1}, the Business Flex account suits you." for i in range(3)]
    signals = recipe["signal_categories"]
    advisory = [f"Here is what I'd flag ({sig.lower()}) for Acme Test Pty Ltd." for sig in signals]
    messages = []
    for turn in range(1, turns + 1):
        if turn % 2:
            content = f"Question about the {recipe['primary_topic']}: {filler(recipe, turn)}."
        elif turn == turns:
            content = " ".join([f"Thanks: {filler(recipe, turn)}.", *advisory, sentences[0]])
        else:
            content = f"Factual answer, the monthly fee is $10: {filler(recipe, turn)}."
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


def run(fag, tmp_path, backend, *, judge=None, **kw):
    run_id = kw.pop("run_id", "r1")
    overrides = {"generator": backend}
    if judge is not None:
        overrides["judge"] = judge
    else:
        kw.setdefault("layers", NO_JUDGE)
    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides=overrides,
        target_size=kw.pop("target_size", TARGET),
        **kw,
    )
    return pipe, pipe.run(run_id)


# ── cells ───────────────────────────────────────────────────────


def test_fag_plan_cells_cross_axes_and_sum_to_target(fag):
    cells = build_plan(fag, target_size=TARGET).cells
    assert len(cells) == 2 * 2 * 4
    assert sum(c.quota for c in cells) == TARGET
    assert cells[0].id == "corps_act|true|single_turn"
    assert cells[0].params == {
        "product_scope": "corps_act",
        "label": True,
        "conversation_length": "single_turn",
    }


def test_weighted_quotas_follow_axis_weights(fag):
    cells = build_plan(fag, target_size=1000).cells
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
    axes = resolve_axes(cov)
    assert assign_quotas(cross(axes), axes, {}, cov.target_size) == [4, 3]


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
    cells = {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
    assert result.counts == cells
    assert result.layers == ("L1", "L2", "L3", "L4")
    assert result.skipped_layers == ("L5", "L6")

    stored = result.run.read_jsonl(ACCEPTED_STREAM)
    assert stored == result.accepted
    assert result.run.read_jsonl(DROPS_STREAM) == []
    for rec in stored:
        bare, prov = split(rec)
        prov.check_accepted(("L1", "L2", "L3", "L4"))
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
    assert spec["skipped_layers"] == ["L5", "L6"]
    assert spec["held_out_check"] is False
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
    assert result.counts == {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
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
    cells = {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
    assert Counter(split(r)[1].cell_id for r in stored) == Counter(cells)
    seeds = [split(r)[1].seed for r in stored]
    assert len(set(seeds)) == TARGET  # no candidate regenerated with a used seed


def test_resume_with_a_different_target_is_refused(fag, tmp_path):
    run(fag, tmp_path, valid_backend())
    with pytest.raises(PipelineError, match="start a new run"):
        run(fag, tmp_path, valid_backend(), target_size=TARGET + 1)


def test_unknown_layers_are_refused(fag, tmp_path):
    with pytest.raises(PipelineError, match="not implemented"):
        Pipeline(fag, tmp_path, model_overrides={"generator": valid_backend()}, layers=["L1", "L7"])


def test_explicit_layer_subset(fag, tmp_path):
    _, result = run(fag, tmp_path, valid_backend(), layers=["L1"])
    assert result.layers == ("L1",)
    assert result.skipped_layers == ("L2", "L3", "L4", "L5", "L6")
    split(result.accepted[0])[1].check_accepted(("L1",))


# ── L3 and L4 in the pipeline ──────────────────────────────────


def test_governance_failure_is_dropped_without_repair_and_refilled(fag, tmp_path):
    calls = []

    def reply(call):
        calls.append(call)
        r = fag_reply(recipe_from_prompt(call.prompt))
        if len(calls) == 1:
            r["messages"][0]["content"] += " My TFN is 000 000 000."  # fictional
        return json.dumps(r)

    _, result = run(fag, tmp_path, MockBackend(reply))
    assert result.complete and len(result.accepted) == TARGET
    drops = result.run.read_jsonl(DROPS_STREAM)
    assert len(drops) == 1
    assert drops[0]["layer"] == "L3" and drops[0]["codes"] == ["pii_tfn"]
    assert drops[0]["hard"] is True and drops[0]["attempts"] == 1  # never repaired
    assert "000 000 000" not in json.dumps(drops)  # the drop log doesn't copy the TFN
    assert not any("previous attempt was rejected" in c.prompt for c in calls)


def test_near_duplicate_of_an_accepted_record_is_dropped(fag, tmp_path):
    template = {}

    def reply(call):
        # Non-breach candidates copy the text of the first non-breach reply of their length.
        recipe = recipe_from_prompt(call.prompt)
        r = fag_reply(recipe)
        if not recipe["label"]:
            r["messages"] = template.setdefault(len(r["messages"]), r["messages"])
        return json.dumps(r)

    _, result = run(fag, tmp_path, MockBackend(reply), max_attempts_per_cell=3)
    drops = result.run.read_jsonl(DROPS_STREAM)
    assert drops and {d["layer"] for d in drops} == {"L4"}
    assert {c for d in drops for c in d["codes"]} == {"near_duplicate"}
    assert all(d["hard"] and d["attempts"] == 1 for d in drops)
    assert all("|false|" in d["cell_id"] for d in drops)
    lengths = Counter(
        len(split(r)[0]["messages"]) for r in result.accepted if not split(r)[0]["label"]
    )
    assert all(n == 1 for n in lengths.values())  # one accepted copy per template


def test_resume_rebuilds_the_near_duplicate_corpus(fag, tmp_path):
    _, done = run(fag, tmp_path, valid_backend())
    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": valid_backend()},
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    assert pipe.overlap is not None and pipe.overlap.corpus_size == 0
    resumed = pipe.run("r1")  # already complete: nothing new, but the corpus is rebuilt
    assert resumed.accepted == []
    assert pipe.overlap.corpus_size == TARGET
    verdict = pipe.overlap.check(split(done.accepted[0])[0], ValidationContext(cell_id="t"))
    assert verdict.outcome == "fail_hard"
    assert verdict.errors[0].code == "near_duplicate"


def test_held_out_check_is_opt_in(fag, tmp_path):
    _, plain = run(fag, tmp_path / "a", valid_backend())
    assert plain.run.read_stage("spec")["held_out_check"] is False
    held = tmp_path / "held_out.jsonl"
    first = split(plain.accepted[0])[0]
    held.write_text(json.dumps({"messages": first["messages"]}) + "\n")

    pipe = Pipeline(
        fag,
        tmp_path / "b",
        model_overrides={"generator": valid_backend()},
        held_out_paths=[held],
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    assert pipe.overlap is not None and pipe.overlap.held_out_enabled
    result = pipe.run("r1")
    assert result.complete and len(result.accepted) == TARGET
    spec = result.run.read_stage("spec")
    assert spec["held_out_check"] is True
    assert str(held) not in json.dumps(spec)  # the path isn't recorded
    drops = result.run.read_jsonl(DROPS_STREAM)
    # The same seed regenerates the held-out copy first; it drops and the cell refills.
    assert [(d["layer"], d["codes"]) for d in drops] == [("L4", ["held_out_overlap"])]
    assert first["messages"] not in [split(r)[0]["messages"] for r in result.accepted]


def test_held_out_paths_need_l4(fag, tmp_path):
    with pytest.raises(PipelineError, match="needs L4"):
        Pipeline(
            fag,
            tmp_path,
            model_overrides={"generator": valid_backend()},
            layers=["L1", "L2"],
            held_out_paths=[tmp_path / "x.jsonl"],
        )
