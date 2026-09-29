"""M2 checkpoint: the FAG pipeline end to end on a MockBackend, with every M2 failure mode.

One run mixes clean replies with unparseable ones, a wrong turn order (L1) and a reworded
span (L2). The run must still fill every cell to exactly its quota, repair or drop each bad
reply in its own cell, and give accepted records that pass a fresh L1 -> L2 cascade and the
original scripts/utils validator.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from sdgf.coverage.plan import build_plan
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ACCEPTED_STREAM, DROPS_STREAM
from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import split
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from test_pipeline import fag_reply, recipe_from_prompt, run

SDGF_DIR = Path(__file__).resolve().parents[1]
FAG_DIR = SDGF_DIR / "tasks" / "fag"
SCRIPTS_DIR = SDGF_DIR.parent / "scripts"
TARGET = 40
REPAIR = "previous attempt was rejected"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def orig_utils():
    # scripts/ is read-only here: import without writing bytecode into it.
    sys.path.insert(0, str(SCRIPTS_DIR))
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import utils
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.dont_write_bytecode = dont_write
    return utils


def swap_roles(reply: dict) -> dict:
    for m in reply["messages"]:
        m["role"] = "assistant" if m["role"] == "customer" else "customer"
    return reply


def mixed_reply(call) -> str:
    """Deterministic per first-attempt index: 0 junk, 1 swapped roles, 2 reworded span, 3 ok.

    Repair prompts get a valid reply, except junk, which stays junk so its slot drops.
    """
    recipe = recipe_from_prompt(call.prompt)
    kind = mixed_reply.first % 4
    if REPAIR not in call.prompt:
        mixed_reply.first += 1
        mixed_reply.kind = kind
    else:
        kind = mixed_reply.kind
        if kind != 0:
            return json.dumps(fag_reply(recipe))
    if kind == 0:
        return "Sorry, I can't produce JSON today."
    if kind == 1:
        return json.dumps(swap_roles(fag_reply(recipe)))
    return json.dumps(fag_reply(recipe, reword=kind == 2))


@pytest.fixture
def mixed_run(fag, tmp_path):
    mixed_reply.first = 0
    mixed_reply.kind = 0
    backend = MockBackend(mixed_reply)
    _, result = run(fag, tmp_path, backend, target_size=TARGET)
    return backend, result


def test_mixed_run_fills_every_cell_exactly(fag, mixed_run):
    _, result = mixed_run
    assert result.complete
    assert len(result.accepted) == TARGET
    quotas = {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
    assert result.counts == quotas
    bare = [split(r)[0] for r in result.accepted]
    assert Counter(r["label"] for r in bare) == {True: 20, False: 20}  # BREACH_RATE 0.5
    assert Counter(r["product_scope"] for r in bare) == {"corps_act": 26, "non_corps_act": 14}


def test_mixed_run_drops_junk_and_repairs_the_rest(fag, mixed_run):
    backend, result = mixed_run
    drops = result.run.read_jsonl(DROPS_STREAM)
    # Every fourth first attempt is junk and stays junk through repair_tries, so drops.
    assert drops and all(d["layer"] == "generate" and d["codes"] == ["no_json"] for d in drops)
    assert all(d["attempts"] == fag.spec.validation.repair_tries + 1 for d in drops)
    summary = result.run.read_stage("summary")
    assert summary["stop_reason"] == "complete"
    assert summary["drops"]["by_layer"] == {"generate": len(drops)}

    repairs = Counter(split(r)[1].repair_count for r in result.accepted)
    assert set(repairs) == {0, 1}
    assert repairs[1] > 0  # L1 swaps and L2 rewordings were repaired, not dropped
    # Each repair prompt carries the validator's error codes back to the model.
    repair_prompts = [c.prompt for c in backend.calls if REPAIR in c.prompt]
    assert any("first_role" in p for p in repair_prompts)
    assert any("span_not_verbatim" in p for p in repair_prompts)

    # Drops are refilled in their own cell: each dropped cell still met its quota.
    quotas = {c.id: c.quota for c in build_plan(fag, target_size=TARGET).cells}
    for d in drops:
        assert result.counts[d["cell_id"]] == quotas[d["cell_id"]]


def test_accepted_records_pass_a_fresh_cascade(fag, mixed_run):
    _, result = mixed_run
    cascade = Cascade.from_config(
        ["L1", "L2"], {"L1": SchemaLayer.from_spec(fag), "L2": RulesLayer.from_spec(fag)}
    )
    assert result.run.read_jsonl(ACCEPTED_STREAM) == result.accepted
    for row in result.accepted:
        bare, prov = split(row)
        prov.check_accepted(("L1", "L2"))
        verdict = cascade.run(bare, ValidationContext(cell_id=prov.cell_id))
        assert verdict.outcome == "pass", verdict.errors
        assert bare["label"] == fag.hooks.label_rule(bare)
        assert bare["policy_categories"] == fag.hooks.post_process(bare)["policy_categories"]


def test_accepted_records_pass_the_original_validator(mixed_run, orig_utils):
    """Mapped back to the scripts/ field names, the new pipeline's output is valid there too."""
    _, result = mixed_run
    for row in result.accepted:
        bare, _ = split(row)
        rec = {k: v for k, v in bare.items() if k not in ("label", "spans")}
        rec["financial_advice_breach"] = bare["label"]
        rec["problematic_spans"] = bare["spans"]
        ok, errors = orig_utils.validate_conversation(rec)
        assert ok, errors
