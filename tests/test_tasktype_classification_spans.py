import copy

import jsonschema
import pytest

from sdgf.spec.schema import FieldSpec, OutputSchemaSection, TaskSection
from sdgf.tasktypes.base import TaskTypeError
from sdgf.tasktypes.classification_spans import (
    ClassificationSpans,
    span_turn_errors,
)
from sdgf.tasktypes.registry import REGISTRY, get_task_type

RECORD = {
    "messages": [
        {"turn": 1, "role": "customer", "content": "Acme Test Pty Ltd needs a loan."},
        {"turn": 2, "role": "assistant", "content": "This loan is ideal for your business."},
    ],
    "label": True,
    "spans": [{"turn": 2, "text": "ideal for your business", "category": "SUITABILITY"}],
}


def _schema():
    return ClassificationSpans().output_schema()


def _valid(record) -> bool:
    return jsonschema.Draft202012Validator(_schema()).is_valid(record)


def _rec():
    return copy.deepcopy(RECORD)


def test_registered_on_import():
    import sdgf.tasktypes  # noqa: F401

    assert "classification_spans" in REGISTRY
    tt = get_task_type("classification_spans")
    assert isinstance(tt, ClassificationSpans)
    assert tt.generation_modes == ("label_first",)


def test_resolve_rejects_answer_emergent():
    task = TaskSection(
        name="t",
        version="1",
        type="classification_spans",
        generation_mode="answer_emergent",
        description="d",
    )
    with pytest.raises(TaskTypeError, match="answer_emergent"):
        REGISTRY.resolve(task)


def test_schema_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(_schema())


def test_valid_record_passes_schema():
    assert _valid(_rec())


@pytest.mark.parametrize("label", [True, False, "breach", 3])
def test_label_types(label):
    r = _rec()
    r["label"] = label
    assert _valid(r)


def test_non_breach_with_empty_spans_passes():
    r = _rec()
    r["label"] = False
    r["spans"] = []
    assert _valid(r)


@pytest.mark.parametrize("field", ["messages", "label", "spans"])
def test_missing_top_level_field_fails(field):
    r = _rec()
    del r[field]
    assert not _valid(r)


def test_empty_messages_fails():
    r = _rec()
    r["messages"] = []
    assert not _valid(r)


@pytest.mark.parametrize("field", ["turn", "role", "content"])
def test_message_missing_field_fails(field):
    r = _rec()
    del r["messages"][0][field]
    assert not _valid(r)


def test_message_empty_content_fails():
    r = _rec()
    r["messages"][0]["content"] = ""
    assert not _valid(r)


@pytest.mark.parametrize("field", ["turn", "text", "category"])
def test_span_missing_field_fails(field):
    r = _rec()
    del r["spans"][0][field]
    assert not _valid(r)


def test_label_of_wrong_type_fails():
    r = _rec()
    r["label"] = None
    assert not _valid(r)


def test_spec_fields_layer_on_top():
    section = OutputSchemaSection(
        fields={"severity": FieldSpec(type="string", enum=["low", "high"], required=False)}
    )
    schema = ClassificationSpans().output_schema(section)
    assert schema["properties"]["severity"]["enum"] == ["low", "high"]
    assert "severity" not in schema["required"]


def test_spec_cannot_redefine_label():
    section = OutputSchemaSection(fields={"label": FieldSpec(type="string")})
    with pytest.raises(TaskTypeError, match="label"):
        ClassificationSpans().output_schema(section)


def test_default_validators_pass_valid_record():
    for v in ClassificationSpans().default_validators():
        assert v(_rec()) == []


def test_span_cites_missing_turn():
    r = _rec()
    r["spans"][0]["turn"] = 5
    errors = span_turn_errors(r)
    assert errors and "turn=5" in errors[0]
