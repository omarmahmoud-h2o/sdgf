"""L1 schema/layout layer: JSON schema, turn structure and task-type validators."""

import copy
from pathlib import Path

import pytest
from jsonschema.exceptions import SchemaError

from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import TurnStructure
from sdgf.tasktypes.classification_spans import CLASSIFICATION_SPANS
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer, format_path, validator_code

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CTX = ValidationContext()
TURNS = TurnStructure(roles=["customer", "assistant"], first_role="customer")


def good_record():
    return {
        "messages": [
            {"turn": 1, "role": "customer", "content": "What does the Acme Test card cost?"},
            {"turn": 2, "role": "assistant", "content": "The annual fee is $0 in this example."},
        ],
        "label": False,
        "spans": [],
    }


@pytest.fixture
def layer():
    return SchemaLayer(
        CLASSIFICATION_SPANS.output_schema(),
        TURNS,
        CLASSIFICATION_SPANS.default_validators(),
    )


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def codes(verdict):
    return list(verdict.codes)


def test_valid_record_passes(layer):
    verdict = layer.check(good_record(), CTX)
    assert verdict.passed and verdict.layer == "L1"


def test_format_path():
    assert format_path(["messages", 3, "content"]) == "messages[3].content"
    assert format_path([]) is None
    assert format_path([0, "text"]) == "[0].text"


# ── stage 1: schema ──────────────────────────────────────────────


def test_missing_required_field(layer):
    record = good_record()
    del record["label"]
    verdict = layer.check(record, CTX)
    assert verdict.repairable
    assert codes(verdict) == ["schema_required"]
    assert "'label'" in verdict.errors[0].message


def test_wrong_field_type(layer):
    record = good_record()
    record["messages"][1]["turn"] = "2"
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["schema_type"]
    assert verdict.errors[0].path == "messages[1].turn"


def test_empty_messages(layer):
    record = good_record()
    record["messages"] = []
    assert codes(layer.check(record, CTX)) == ["schema_minItems"]


def test_empty_content(layer):
    record = good_record()
    record["messages"][0]["content"] = ""
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["schema_minLength"]
    assert verdict.errors[0].path == "messages[0].content"


def test_non_object_record(layer):
    verdict = layer.check(["not", "a", "record"], CTX)
    assert codes(verdict) == ["schema_type"]
    assert verdict.errors[0].path is None


def test_spec_enum_field(fag):
    layer = SchemaLayer.from_spec(fag)
    record = copy.deepcopy(fag.seeds[0])
    record["product_scope"] = "offshore"
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["schema_enum"]
    assert verdict.errors[0].path == "product_scope"


def test_all_schema_errors_reported_together(layer):
    record = good_record()
    del record["spans"]
    record["messages"][0]["turn"] = 0
    assert sorted(codes(layer.check(record, CTX))) == ["schema_minimum", "schema_required"]


def test_schema_errors_suppress_later_stages(layer):
    record = good_record()
    del record["label"]
    record["messages"][0]["role"] = "assistant"  # would be a turn error
    assert codes(layer.check(record, CTX)) == ["schema_required"]


def test_invalid_schema_rejected():
    with pytest.raises(SchemaError):
        SchemaLayer({"type": "no-such-type"})


# ── stage 2: turn structure ──────────────────────────────────────


def test_turns_not_numbered_from_one(layer):
    record = good_record()
    record["messages"][0]["turn"] = 2
    record["messages"][1]["turn"] = 3
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["turn_numbering", "turn_numbering"]
    assert verdict.errors[0].details == {"expected": 1, "got": 2}


def test_turn_gap(layer):
    record = good_record()
    record["messages"][1]["turn"] = 3
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["turn_numbering"]
    assert verdict.errors[0].path == "messages[1].turn"


def test_first_role_wrong(layer):
    record = good_record()
    record["messages"][0]["role"] = "assistant"
    record["messages"][1]["role"] = "customer"
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["first_role", "role_alternation"]


def test_roles_do_not_alternate(layer):
    record = good_record()
    record["messages"].append({"turn": 3, "role": "assistant", "content": "Anything else?"})
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["role_alternation"]
    assert verdict.errors[0].path == "messages[2].role"
    assert "expected 'customer'" in verdict.errors[0].message


def test_unknown_role(layer):
    record = good_record()
    record["messages"][1]["role"] = "banker"
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["unknown_role"]
    assert verdict.errors[0].details["allowed"] == ["customer", "assistant"]


def test_non_alternating_only_checks_first_role():
    structure = TurnStructure(
        roles=["customer", "assistant"], first_role="customer", alternating=False
    )
    layer = SchemaLayer(CLASSIFICATION_SPANS.output_schema(), structure)
    record = good_record()
    record["messages"].append({"turn": 3, "role": "assistant", "content": "Anything else?"})
    assert layer.check(record, CTX).passed
    record["messages"][0]["role"] = "assistant"
    assert codes(layer.check(record, CTX)) == ["first_role"]


def test_numbered_from_zero():
    structure = TurnStructure(
        roles=["customer", "assistant"], first_role="customer", numbered_from=0
    )
    layer = SchemaLayer({"type": "object"}, structure)
    record = good_record()
    assert codes(layer.check(record, CTX)) == ["turn_numbering", "turn_numbering"]
    for i, msg in enumerate(record["messages"]):
        msg["turn"] = i
    assert layer.check(record, CTX).passed


def test_no_turn_structure_skips_role_checks():
    layer = SchemaLayer(CLASSIFICATION_SPANS.output_schema())
    record = good_record()
    record["messages"][0]["role"] = "someone"
    assert layer.check(record, CTX).passed


# ── stage 3: task-type validators ────────────────────────────────


def test_span_citing_missing_turn(layer):
    record = good_record()
    record["spans"] = [{"turn": 9, "text": "annual fee", "category": "X"}]
    verdict = layer.check(record, CTX)
    assert codes(verdict) == ["span_turn"]
    assert verdict.repairable
    assert "turn=9" in verdict.errors[0].message


def test_task_type_validators_run_after_turns(layer):
    record = good_record()
    record["messages"][0]["role"] = "assistant"
    record["messages"][1]["role"] = "customer"
    record["spans"] = [{"turn": 9, "text": "annual fee", "category": "X"}]
    assert "span_turn" not in codes(layer.check(record, CTX))


def test_validator_code():
    def custom_errors(record):
        return []

    assert validator_code(custom_errors) == "custom"


def test_custom_validator_errors_are_issues():
    layer = SchemaLayer({"type": "object"}, validators=[lambda r: ["bad thing"]])
    verdict = layer.check({}, CTX)
    assert verdict.repairable and verdict.errors[0].message == "bad thing"


# ── with the FAG spec ────────────────────────────────────────────


def test_fag_seeds_pass(fag):
    layer = SchemaLayer.from_spec(fag)
    for seed in fag.seeds:
        verdict = layer.check(dict(seed), CTX)
        assert verdict.passed, verdict.messages()


def test_fag_wrong_turn_order_fails_in_cascade(fag):
    cascade = Cascade.from_config(["L1"], [SchemaLayer.from_spec(fag)])
    record = copy.deepcopy(fag.seeds[0])
    record["messages"][0], record["messages"][1] = record["messages"][1], record["messages"][0]
    result = cascade.run(record)
    assert result.failed_layer == "L1" and result.repairable
    assert "turn_numbering" in {e.code for e in result.errors}
