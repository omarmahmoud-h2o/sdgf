"""M3 checkpoint: the FAG pipeline end to end on a MockBackend, with every L1-L4 failure mode.

One run mixes clean replies with a wrong turn order (L1), a reworded span (L2), a fictional
TFN and a secret-shaped token (L3) and a copied seed (L4). L1/L2 failures must be repaired
in their cell; L3/L4 failures must drop at once, without repair and without their matched
text in the drop log. Every cell still fills to exactly its quota, and accepted records pass
a fresh L1 -> L4 cascade.
"""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sdgf.coverage.plan import build_plan
from sdgf.governance.profile import GLOBAL_PROFILE, profile_for
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ACCEPTED_STREAM, DROPS_STREAM
from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import split
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.l3_governance import GovernanceLayer
from sdgf.validate.l4_overlap import OverlapLayer
from test_pipeline import fag_reply, recipe_from_prompt, run

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 40
LAYERS = ("L1", "L2", "L3", "L4")
REPAIR = "previous attempt was rejected"
TFN = "000 000 000"  # fictional
TOKEN = "sk-" + "0" * 32  # fictional, secret-shaped
KINDS = ("ok", "swap", "reword", "tfn", "secret", "seed_copy")


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def swap_roles(reply: dict) -> dict:
    for m in reply["messages"]:
        m["role"] = "assistant" if m["role"] == "customer" else "customer"
    return reply


def make_mixed_reply(fag):
    """Deterministic per first-attempt index, cycling through KINDS.

    Repair prompts always get a valid reply. A seed copy is only possible for a non-breach
    recipe (a breach recipe's spans must cite its own text), so breach recipes get a valid
    reply instead.
    """
    seed_messages = next(s for s in fag.seeds if not s["label"])["messages"]
    state = {"first": 0}

    def reply(call) -> str:
        recipe = recipe_from_prompt(call.prompt)
        r = fag_reply(recipe)
        if REPAIR in call.prompt:
            return json.dumps(r)
        kind = KINDS[state["first"] % len(KINDS)]
        state["first"] += 1
        state.setdefault("kinds", Counter())[kind] += 1
        if kind == "swap":
            swap_roles(r)
        elif kind == "reword":
            r = fag_reply(recipe, reword=True)
        elif kind == "tfn":
            r["messages"][0]["content"] += f" My TFN is {TFN}."
        elif kind == "secret":
            r["messages"][0]["content"] += f" Our integration key is {TOKEN}."
        elif kind == "seed_copy" and not recipe["label"]:
            r["messages"] = copy.deepcopy(seed_messages)
        return json.dumps(r)

    reply.state = state
    return reply


@pytest.fixture(scope="module")
def mixed_run(fag, tmp_path_factory):
    reply = make_mixed_reply(fag)
    backend = MockBackend(reply)
    _, result = run(fag, tmp_path_factory.mktemp("m3"), backend, target_size=TARGET)
    return backend, result


def test_fag_compiles_through_stage_0_with_the_global_profile(fag):
    assert profile_for(fag) == GLOBAL_PROFILE
    assert len(fag.seeds) == 6
    assert fag.spec.thresholds.unset() == []


def test_mixed_run_runs_l1_to_l4_and_fills_every_cell(fag, mixed_run):
    _, result = mixed_run
    assert result.layers == LAYERS
    assert result.skipped_layers == ("L5", "L6")
    assert result.complete and len(result.accepted) == TARGET
    quotas = {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
    assert result.counts == quotas
    bare = [split(r)[0] for r in result.accepted]
    assert Counter(r["label"] for r in bare) == {True: 20, False: 20}  # BREACH_RATE 0.5


def test_l3_and_l4_failures_drop_at_once_without_matched_text(mixed_run):
    backend, result = mixed_run
    drops = result.run.read_jsonl(DROPS_STREAM)
    codes = Counter((d["layer"], c) for d in drops for c in d["codes"])
    assert codes[("L3", "pii_tfn")] > 0
    assert codes[("L3", "secrets_api_key")] > 0
    assert codes[("L4", "seed_overlap")] > 0
    assert {d["layer"] for d in drops} == {"L3", "L4"}  # L1/L2 were all repaired
    assert all(d["hard"] and d["attempts"] == 1 for d in drops)
    dumped = json.dumps(drops)
    assert TFN not in dumped and TOKEN not in dumped
    # Hard drops are never repaired: no repair prompt carries an L3/L4 code.
    repair_prompts = [c.prompt for c in backend.calls if REPAIR in c.prompt]
    assert not any(code in p for p in repair_prompts for _, code in codes)
    summary = result.run.read_stage("summary")
    assert summary["stop_reason"] == "complete"
    assert set(summary["drops"]["by_layer"]) == {"L3", "L4"}


def test_l1_and_l2_failures_are_repaired_in_their_cell(fag, mixed_run):
    backend, result = mixed_run
    repair_prompts = [c.prompt for c in backend.calls if REPAIR in c.prompt]
    assert any("first_role" in p for p in repair_prompts)
    assert any("span_not_verbatim" in p for p in repair_prompts)
    repairs = Counter(split(r)[1].repair_count for r in result.accepted)
    assert set(repairs) == {0, 1} and repairs[1] > 0
    quotas = {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
    for d in result.run.read_jsonl(DROPS_STREAM):
        assert result.counts[d["cell_id"]] == quotas[d["cell_id"]]


def test_accepted_records_pass_a_fresh_l1_to_l4_cascade(fag, mixed_run):
    _, result = mixed_run
    overlap = OverlapLayer.from_spec(fag)
    cascade = Cascade.from_config(
        list(LAYERS),
        {
            "L1": SchemaLayer.from_spec(fag),
            "L2": RulesLayer.from_spec(fag),
            "L3": GovernanceLayer.from_spec(fag),
            "L4": overlap,
        },
    )
    assert result.run.read_jsonl(ACCEPTED_STREAM) == result.accepted
    for i, row in enumerate(result.accepted):
        bare, prov = split(row)
        prov.check_accepted(LAYERS)
        verdict = cascade.run(bare, ValidationContext(cell_id=prov.cell_id))
        assert verdict.outcome == "pass", verdict.errors
        overlap.remember(bare, key=f"accepted:{i}")  # accepted set has no near-duplicates
        assert TFN not in json.dumps(bare) and TOKEN not in json.dumps(bare)
