"""The FAG seeds load, match the classification_spans schema and follow the FAG policy."""

import sys
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.spec.loader import load_task
from sdgf.validate.base import ValidationContext
from sdgf.validate.l1_schema import SchemaLayer

SDGF_DIR = Path(__file__).resolve().parents[1]
FAG_DIR = SDGF_DIR / "tasks" / "fag"
SCRIPTS_DIR = SDGF_DIR.parent / "scripts"


@pytest.fixture(scope="module")
def task():
    return load_task(FAG_DIR)


@pytest.fixture(scope="module")
def seeds(task):
    return task.seeds


@pytest.fixture(scope="module")
def orig_utils():
    if not SCRIPTS_DIR.is_dir():
        pytest.skip("needs the original ../scripts/ FAG generator, which is not checked out")
    # scripts/ is read-only here: import without writing bytecode into it.
    sys.path.insert(0, str(SCRIPTS_DIR))
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import utils
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.dont_write_bytecode = dont_write
    return utils


def test_six_seeds_load_and_compile(task):
    assert len(task.seeds) == 6
    assert len(compile_spec(FAG_DIR).seeds) == 6


def test_label_split_and_hard_negative(seeds):
    assert sum(s["label"] is True for s in seeds) == 3
    assert sum(s["label"] is False for s in seeds) == 3
    hard_negatives = [
        s
        for s in seeds
        if s["label"] is False
        and s["advice_tier"] == "GENERAL_ADVICE"
        and s["product_scope"] == "non_corps_act"
        and s["signal_categories"]
        and not s["spans"]
    ]
    assert len(hard_negatives) >= 1


def test_seed_ids_unique(seeds):
    ids = [s["conversation_id"] for s in seeds]
    assert len(set(ids)) == len(ids)


def test_seeds_pass_l1(seeds):
    layer = SchemaLayer.from_spec(compile_spec(FAG_DIR))
    for s in seeds:
        verdict = layer.check(dict(s), ValidationContext())
        assert verdict.passed, (s["conversation_id"], verdict.messages())
        assert len(s["messages"]) == s["turn_count"]
        assert s["messages"][-1]["role"] == "assistant"


def test_seed_labels_follow_label_rule(task, seeds):
    for s in seeds:
        assert task.hooks.label_rule(s) == s["label"], s["conversation_id"]


def test_seed_policy_categories_are_derived(task, seeds):
    for s in seeds:
        assert task.hooks.post_process(s)["policy_categories"] == s["policy_categories"]


def test_seed_spans_are_verbatim_assistant_text(seeds):
    for s in seeds:
        for span in s["spans"]:
            msg = s["messages"][span["turn"] - 1]
            assert msg["role"] == "assistant"
            assert span["text"] in msg["content"]


def test_seeds_pass_original_validator(seeds, orig_utils):
    """Mapped back to the scripts/ field names, every seed passes utils.validate_conversation."""
    for s in seeds:
        rec = {k: v for k, v in s.items() if k not in ("label", "spans")}
        rec["financial_advice_breach"] = s["label"]
        rec["problematic_spans"] = s["spans"]
        ok, errors = orig_utils.validate_conversation(rec)
        assert ok, (s["conversation_id"], errors)
