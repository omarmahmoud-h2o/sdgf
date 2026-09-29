"""L3 governance layer: every scanner runs, every finding is a hard drop, and sensitive tool
traces get stricter handling. All identifiers and secrets are fictional placeholders."""

import copy
import json
from pathlib import Path

import pytest

from sdgf.governance.profile import GovernanceProfile, merge_profile
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import GovernanceSection
from sdgf.store.provenance import ToolTraceEntry
from sdgf.validate.base import Layer, ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.l3_governance import GovernanceLayer, is_sensitive, sensitive_entries

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CTX = ValidationContext()
L3 = GovernanceLayer.from_profile()


def record(assistant="The monthly fee is $10 and there is no setup cost.", **extra):
    base = {
        "messages": [
            {"turn": 1, "role": "customer", "content": "What does the Acme Test account cost?"},
            {"turn": 2, "role": "assistant", "content": assistant},
        ],
        "label": False,
        "spans": [],
    }
    return {**base, **extra}


def traced(*entries):
    return ValidationContext(extra={"tool_trace": list(entries)})


def seeds():
    lines = (FAG_DIR / "seeds.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR / "task.yaml")


def test_clean_record_passes():
    verdict = L3.check(record(), CTX)
    assert verdict.passed and verdict.layer == "L3"


@pytest.mark.parametrize(
    "text, code",
    [
        ("Your TFN on file is 000 000 000.", "pii_tfn"),
        ("Email us at nobody@example.invalid for details.", "pii_email"),
        ("Pay into account 000-000 00000000 today.", "pii_account_number"),
        ("This fee is complete bullshit.", "toxicity_profanity"),
        ("Use key sk-ant-EXAMPLE0000000000000000000000 to connect.", "secrets_api_key"),
    ],
)
def test_each_scanner_is_a_hard_fail(text, code):
    verdict = L3.check(record(text), CTX)
    assert verdict.hard and not verdict.repairable
    assert code in verdict.codes
    issue = next(e for e in verdict.errors if e.code == code)
    assert issue.path == "messages[1].content"


def test_denied_entity_is_a_hard_fail():
    profile = merge_profile(GovernanceSection(entity_deny=["Fictional Megabank"]))
    layer = GovernanceLayer.from_profile(profile)
    verdict = layer.check(record("Fictional Megabank offers a cheaper account."), CTX)
    assert verdict.hard and verdict.codes == ("entity_denied",)
    assert verdict.errors[0].details["rule"] == "Fictional Megabank"


def test_issues_never_repeat_the_matched_text():
    verdict = L3.check(record("Your TFN on file is 000 000 000."), CTX)
    for issue in verdict.errors:
        dumped = json.dumps(issue.to_dict())
        assert "000 000 000" not in dumped and "000000000" not in dumped


def test_every_finding_is_reported_across_scanners_and_fields():
    rec = record("Your TFN is 000 000 000, you idiot.", summary="Mail nobody@example.invalid")
    codes = set(L3.check(rec, CTX).codes)
    assert {"pii_tfn", "toxicity_insult", "pii_email"} <= codes
    paths = {e.path for e in L3.check(rec, CTX).errors}
    assert {"messages[1].content", "summary"} <= paths


def test_private_keys_are_not_scanned():
    rec = record(_provenance={"note": "Your TFN is 000 000 000"})
    assert L3.check(rec, CTX).passed


def test_from_spec_uses_the_task_profile(fag):
    layer = GovernanceLayer.from_spec(fag)
    for seed in seeds():
        assert layer.check(seed, CTX).passed


def test_needs_a_scanner():
    with pytest.raises(ValueError):
        GovernanceLayer([])


# ── sensitive tool traces ────────────────────────────────────────


@pytest.mark.parametrize(
    "label, sensitive",
    [
        (None, False),
        ("", False),
        ("public", False),
        ("internal", True),
        ("restricted", True),
        ("made-up-level", True),
    ],
)
def test_sensitivity_labels(label, sensitive):
    assert is_sensitive(label) is sensitive


def test_sensitive_entries_accept_dataclasses_and_dicts():
    trace = [
        ToolTraceEntry("catalogue", {"q": "fees"}, {"fee": "$10"}, "public"),
        {"tool": "ledger", "result": {"balance": "1234"}, "sensitivity": "confidential"},
    ]
    assert [e["tool"] for e in sensitive_entries(trace)] == ["ledger"]
    with pytest.raises(TypeError):
        sensitive_entries(["not an entry"])


LEDGER = ToolTraceEntry(
    "ledger",
    {"customer": "Acme Test Pty Ltd"},
    {"note": "Overdraft review flagged in Q3", "ref": 90210555, "active": True},
    "confidential",
)


def test_sensitive_value_repeated_in_record_is_a_leak():
    rec = record("Your file says: overdraft   REVIEW flagged in Q3.")
    assert L3.check(rec, CTX).passed  # no trace, nothing to leak
    verdict = L3.check(rec, traced(LEDGER))
    assert verdict.hard and verdict.codes == ("tool_data_leak",)
    issue = verdict.errors[0]
    assert issue.path == "messages[1].content"
    assert issue.details == {"tool": "ledger", "sensitivity": "confidential", "length": 30}
    assert "Overdraft" not in json.dumps(issue.to_dict())


def test_same_value_from_a_public_tool_is_not_a_leak():
    public = ToolTraceEntry("catalogue", {}, LEDGER.result, "public")
    rec = record("Your file says: overdraft review flagged in Q3.")
    assert L3.check(rec, traced(public)).passed


def test_short_values_are_not_leaks():
    entry = {"tool": "ledger", "result": {"currency": "AUD"}, "sensitivity": "internal"}
    assert L3.check(record("All fees are in AUD."), traced(entry)).passed


def test_sensitive_trace_catches_any_identifier_shaped_number():
    rec = record("Your reference is 1234-5678 and the fee is $10.")
    assert L3.check(rec, CTX).passed  # no PII rule knows this format
    verdict = L3.check(rec, traced(LEDGER))
    assert verdict.codes == ("sensitive_identifier",)
    assert verdict.errors[0].details == {"start": 18, "end": 27}


def test_numeric_tool_value_leak_and_identifier_both_reported():
    verdict = L3.check(record("Ref 90210555 is on your file."), traced(LEDGER))
    assert set(verdict.codes) == {"sensitive_identifier", "tool_data_leak"}


def test_sensitive_identifier_does_not_double_report_pii():
    verdict = L3.check(record("Your TFN is 000 000 000."), traced(LEDGER))
    assert verdict.codes == ("pii_tfn",)


def test_clean_record_with_sensitive_trace_passes():
    assert L3.check(record(), traced(LEDGER)).passed


# ── in the cascade ───────────────────────────────────────────────


class Boom(Layer):
    name = "L4"

    def check(self, record, context):
        raise AssertionError("cascade must stop at L3")


def test_fag_seed_with_tfn_passes_l1_l2_and_drops_at_l3(fag):
    seed = copy.deepcopy(seeds()[3])
    seed["messages"][0]["content"] += " My TFN is 000 000 000."
    cascade = Cascade(
        [
            SchemaLayer.from_spec(fag),
            RulesLayer.from_spec(fag),
            GovernanceLayer.from_spec(fag),
            Boom(),
        ]
    )
    result = cascade.run(seed, CTX)
    assert result.failed_layer == "L3" and result.hard and not result.repairable
    assert result.layers_run == ("L1", "L2", "L3")
    assert [e.code for e in result.errors] == ["pii_tfn"]


def test_code_built_profile_is_honoured():
    profile = GovernanceProfile(entity_deny=frozenset({"Fictional Megabank"}))
    layer = GovernanceLayer.from_profile(profile)
    assert layer.check(record("Try fictional  megabank instead."), CTX).codes == ("entity_denied",)
