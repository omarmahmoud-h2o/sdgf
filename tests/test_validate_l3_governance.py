"""L3 governance layer: every scanner runs, every finding is a hard drop, and sensitive tool
traces get stricter handling. All identifiers and secrets are fictional placeholders."""

import copy
import json
import textwrap
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import ToolTraceEntry
from sdgf.validate.base import Layer, ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.l3_governance import GovernanceLayer, sensitive_entries

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CTX = ValidationContext()
L3 = GovernanceLayer.from_spec(compile_spec(FAG_DIR / "task.yaml"))


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


TASK_YAML = """\
task:
  name: governed-toy
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Toy task whose governance section tightens or documents exceptions.
output_schema:
  turns:
    roles: [customer, assistant]
    first_role: customer
rubric:
  verdict:
    values: [pass, fail]
seeds:
  path: seeds.jsonl
coverage:
  target_size: 10
  axes:
    - name: label
      values: [true, false]
models:
  generator:
    backend: mock
    model: mock-1
validation:
  layers: [L1, L2, L3]
thresholds:
  fidelity_min: 0.95
  kappa_min: 0.8
  coverage_min_cell_fill: 0.9
  balance_tolerance: 0.05
  distinct_n_min: 0.3
  self_bleu_max: 0.6
  semantic_diversity_min: 1.0
  residual_error_max: 0.05
  overlap_max: 0.8
  cost_per_record_max: 0.05
"""


def governed(tmp_path, governance=""):
    """The safety scan of a task whose task.yaml carries `governance` (YAML text)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "task.yaml").write_text(TASK_YAML + textwrap.dedent(governance), encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text(json.dumps(record()) + "\n", encoding="utf-8")
    return GovernanceLayer.from_spec(compile_spec(tmp_path))


def test_task_pii_pattern_from_task_yaml_is_dropped(tmp_path):
    layer = governed(
        tmp_path,
        """\
        governance:
          extra_pii_patterns:
            member_no: 'MBR-\\d{6}'
        """,
    )
    verdict = layer.check(record("Thanks, member MBR-000000, your fee is waived."), CTX)
    assert verdict.hard and verdict.codes == ("pii_member_no",)


def test_clean_fag_seed_passes_the_safety_scan(fag):
    verdict = GovernanceLayer.from_spec(fag).check(seeds()[0], CTX)
    assert verdict.passed and verdict.layer == "L3"


def test_tfn_in_a_turn_is_dropped_with_its_offsets(fag):
    verdict = GovernanceLayer.from_spec(fag).check(record("Your TFN on file is 000 000 000."), CTX)
    assert verdict.hard and not verdict.repairable
    assert verdict.codes == ("pii_tfn",)
    issue = verdict.errors[0]
    assert issue.path == "messages[1].content"
    assert (issue.details["start"], issue.details["end"]) == (20, 31)


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


def test_issues_never_repeat_the_matched_text():
    verdict = L3.check(record("Your TFN on file is 000 000 000."), CTX)
    dumped = json.dumps([issue.to_dict() for issue in verdict.errors])
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


@pytest.mark.parametrize("label", ["made-up-level", "internal", "restricted"])
def test_unknown_or_sensitive_label_fails_closed_on_identifier_numbers(label):
    entry = {"tool": "ledger", "result": {"status": "ok"}, "sensitivity": label}
    verdict = L3.check(record("Ref 1234-5678"), traced(entry))
    assert verdict.hard and verdict.codes == ("sensitive_identifier",)
    assert verdict.errors[0].details == {"start": 4, "end": 13}


@pytest.mark.parametrize("label", [None, "", "public"])
def test_public_or_unlabelled_trace_leaves_identifier_numbers_alone(label):
    entry = {"tool": "catalogue", "result": {"status": "ok"}, "sensitivity": label}
    assert L3.check(record("Ref 1234-5678"), traced(entry)).passed


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


TOXICITY_EXCEPTION = """\
governance:
  toxicity_exceptions: [profanity]
  exceptions:
    - rule: toxicity:profanity
      reason: a toxicity-detection task needs profane customer turns
"""


def test_documented_toxicity_exception_lets_profanity_through(tmp_path):
    rec = record("This is bullshit")
    assert governed(tmp_path / "plain").check(rec, CTX).codes == ("toxicity_profanity",)
    assert governed(tmp_path / "excepted", TOXICITY_EXCEPTION).check(rec, CTX).passed


def test_denied_entity_from_task_yaml_is_dropped_whatever_its_spelling(tmp_path):
    layer = governed(tmp_path, "governance:\n  entity_deny: [Fictional Megabank]\n")
    verdict = layer.check(record("If the fee bothers you, try fictional   MEGABANK."), CTX)
    assert verdict.hard and verdict.codes == ("entity_denied",)
    assert verdict.errors[0].details["rule"] == "Fictional Megabank"
