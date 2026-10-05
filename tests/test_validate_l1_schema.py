"""L1 schema/layout layer: JSON schema, turn structure and task-type validators."""

import copy
import shutil
from pathlib import Path

import pytest
from jsonschema.exceptions import SchemaError

from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import TurnStructure
from sdgf.tasktypes.classification_spans import CLASSIFICATION_SPANS
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer

SDGF_DIR = Path(__file__).resolve().parents[1]
FAG_DIR = SDGF_DIR / "tasks" / "fag"
CFA_DIR = SDGF_DIR / "tasks" / "cfa"
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


# ── stage 1: schema ──────────────────────────────────────────────


def fag_seed(fag):
    return copy.deepcopy(fag.seeds[0])


def test_fag_candidate_missing_label_is_sent_back(fag):
    record = fag_seed(fag)
    del record["label"]
    verdict = SchemaLayer.from_spec(fag).check(record, CTX)
    assert verdict.outcome == "fail_repairable"
    assert codes(verdict) == ["schema_required"]
    assert "'label'" in verdict.errors[0].message


def test_fag_turn_given_as_string_is_a_schema_type_error(fag):
    record = fag_seed(fag)
    record["messages"][1]["turn"] = "2"
    verdict = SchemaLayer.from_spec(fag).check(record, CTX)
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
    record["messages"][0]["turn"] = "1"
    assert sorted(codes(layer.check(record, CTX))) == ["schema_required", "schema_type"]


def test_fag_schema_error_hides_turn_errors(fag):
    record = fag_seed(fag)
    del record["label"]
    record["messages"][0]["role"] = "assistant"  # would be turn errors
    record["messages"][1]["role"] = "customer"
    assert codes(SchemaLayer.from_spec(fag).check(record, CTX)) == ["schema_required"]


def test_invalid_schema_rejected():
    with pytest.raises(SchemaError):
        SchemaLayer({"type": "no-such-type"})


# ── stage 2: turn structure ──────────────────────────────────────


def test_fag_turns_not_numbered_from_one(fag):
    record = fag_seed(fag)
    record["messages"] = record["messages"][:2]
    record["spans"] = []
    record["messages"][0]["turn"] = 2
    record["messages"][1]["turn"] = 3
    verdict = SchemaLayer.from_spec(fag).check(record, CTX)
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


def fag_spec_numbered_from_zero(tmp_path):
    task_dir = tmp_path / "fag"
    shutil.copytree(FAG_DIR, task_dir, ignore=shutil.ignore_patterns("__pycache__"))
    task_yaml = task_dir / "task.yaml"
    text = task_yaml.read_text()
    assert "numbered_from: 1" in text
    task_yaml.write_text(text.replace("numbered_from: 1", "numbered_from: 0"))
    return compile_spec(task_dir)


def test_classification_spans_record_numbered_from_zero_passes(tmp_path):
    compiled = fag_spec_numbered_from_zero(tmp_path)
    record = copy.deepcopy(compiled.seeds[0])
    record["messages"] = [
        {"turn": 0, "role": "customer", "content": "Which business account should I be on?"},
        {"turn": 1, "role": "assistant", "content": "Given your cash flow, take the Flex account."},
    ]
    record["spans"] = [
        {"turn": 0, "text": "Which business account", "category": "NEED_BASED_FRAMING"}
    ]
    verdict = SchemaLayer.from_spec(compiled).check(record, CTX)
    assert verdict.passed, verdict.messages()


def test_no_turn_structure_skips_role_checks():
    layer = SchemaLayer(CLASSIFICATION_SPANS.output_schema())
    record = good_record()
    record["messages"][0]["role"] = "someone"
    assert layer.check(record, CTX).passed


# ── stage 3: task-type validators ────────────────────────────────


def test_fag_span_citing_missing_turn_is_sent_back(fag):
    record = fag_seed(fag)
    record["messages"] = record["messages"][:2]
    record["spans"] = [{"turn": 9, "text": "monthly fee", "category": "NEED_BASED_FRAMING"}]
    verdict = SchemaLayer.from_spec(fag).check(record, CTX)
    assert codes(verdict) == ["span_turn"]
    assert verdict.repairable
    assert "turn=9" in verdict.errors[0].message


def test_task_type_validators_run_after_turns(layer):
    record = good_record()
    record["messages"][0]["role"] = "assistant"
    record["messages"][1]["role"] = "customer"
    record["spans"] = [{"turn": 9, "text": "annual fee", "category": "X"}]
    assert "span_turn" not in codes(layer.check(record, CTX))


def test_custom_validator_errors_are_issues():
    layer = SchemaLayer({"type": "object"}, validators=[lambda r: ["bad thing"]])
    verdict = layer.check({}, CTX)
    assert verdict.repairable and verdict.errors[0].message == "bad thing"


# ── with the FAG spec ────────────────────────────────────────────


@pytest.mark.parametrize("task", ["fag", "cfa", "groundness"])
def test_every_seed_passes_l1(task):
    task_dir = SDGF_DIR / "tasks" / task
    if not (task_dir / "task.yaml").exists():
        pytest.skip(f"tasks/{task} not present")
    compiled = compile_spec(task_dir)
    layer = SchemaLayer.from_spec(compiled)
    assert compiled.seeds
    for seed in compiled.seeds:
        verdict = layer.check(copy.deepcopy(seed), CTX)
        assert verdict.passed, verdict.messages()


def test_fag_wrong_turn_order_fails_in_cascade(fag):
    cascade = Cascade.from_config(["L1"], [SchemaLayer.from_spec(fag)])
    record = copy.deepcopy(fag.seeds[0])
    record["messages"][0], record["messages"][1] = record["messages"][1], record["messages"][0]
    result = cascade.run(record)
    assert result.failed_layer == "L1" and result.repairable
    assert "turn_numbering" in {e.code for e in result.errors}


# ── with the CFA spec (sft_qa) ───────────────────────────────────


def test_cfa_answer_that_disagrees_with_the_response_is_sent_back():
    cfa = compile_spec(CFA_DIR)
    record = copy.deepcopy(cfa.seeds[0])
    assert record["response"].endswith("Answer: B")
    record["answer"] = "C"
    verdict = SchemaLayer.from_spec(cfa).check(record, CTX)
    assert verdict.outcome == "fail_repairable"
    assert codes(verdict) == ["response_answer"]
