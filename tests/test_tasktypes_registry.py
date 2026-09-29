import pytest

from sdgf.spec.schema import Axis, FieldSpec, OutputSchemaSection, TaskSection
from sdgf.tasktypes.base import TaskType, TaskTypeError
from sdgf.tasktypes.registry import (
    REGISTRY,
    TaskTypeRegistry,
    UnknownTaskTypeError,
    get_task_type,
)


class LabelType(TaskType):
    name = "toy_label"
    generation_modes = ("label_first",)

    def base_schema(self):
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}, "label": {"type": "boolean"}},
            "required": ["text", "label"],
        }

    def default_axes(self):
        return [Axis(name="label", values=[True, False])]

    def default_validators(self):
        return [lambda r: [] if r.get("text") else ["text is empty"]]


class AnswerType(TaskType):
    name = "toy_answer"
    generation_modes = ("answer_emergent",)

    def base_schema(self):
        return {"type": "object", "properties": {"answer": {"type": "string"}}}

    def answer_extractor(self):
        return lambda text: text.strip() or None


def _task(type_name: str, mode: str = "label_first") -> TaskSection:
    return TaskSection(
        name="t", version="1", type=type_name, generation_mode=mode, description="toy task"
    )


def test_register_and_get():
    reg = TaskTypeRegistry()
    tt = reg.register(LabelType())
    assert reg.get("toy_label") is tt
    assert "toy_label" in reg
    assert reg.names() == ["toy_label"]


def test_unknown_type_raises_clear_error():
    reg = TaskTypeRegistry()
    reg.register(LabelType())
    with pytest.raises(UnknownTaskTypeError) as e:
        reg.get("nope")
    msg = str(e.value)
    assert "unknown task type 'nope'" in msg
    assert "toy_label" in msg
    assert isinstance(e.value, TaskTypeError)


def test_unknown_type_on_empty_registry():
    with pytest.raises(UnknownTaskTypeError, match="registered task types: none"):
        TaskTypeRegistry().get("x")


def test_duplicate_registration_rejected_unless_replace():
    reg = TaskTypeRegistry()
    reg.register(LabelType())
    with pytest.raises(TaskTypeError, match="already registered"):
        reg.register(LabelType())
    new = LabelType()
    assert reg.register(new, replace=True) is new
    assert reg.get("toy_label") is new


def test_resolve_checks_generation_mode():
    reg = TaskTypeRegistry()
    reg.register(LabelType())
    reg.register(AnswerType())
    assert reg.resolve(_task("toy_label")).name == "toy_label"
    assert reg.resolve(_task("toy_answer", "answer_emergent")).name == "toy_answer"
    with pytest.raises(TaskTypeError, match="does not support generation_mode 'answer_emergent'"):
        reg.resolve(_task("toy_label", "answer_emergent"))
    with pytest.raises(UnknownTaskTypeError):
        reg.resolve(_task("missing"))


def test_answer_emergent_requires_extractor():
    class NoExtractor(AnswerType):
        name = "no_extractor"

        def answer_extractor(self):
            return None

    with pytest.raises(TaskTypeError, match="no answer extractor"):
        TaskTypeRegistry().register(NoExtractor())


def test_missing_name_or_bad_modes_rejected():
    class NoName(LabelType):
        name = ""

    class BadMode(LabelType):
        name = "bad_mode"
        generation_modes = ("guesswork",)

    class NoModes(LabelType):
        name = "no_modes"
        generation_modes = ()

    for cls, match in (
        (NoName, "non-empty name"),
        (BadMode, "generation_modes"),
        (NoModes, "generation_modes"),
    ):
        with pytest.raises(TaskTypeError, match=match):
            TaskTypeRegistry().register(cls())


def test_base_schema_must_be_object():
    class NotObject(LabelType):
        name = "not_object"

        def base_schema(self):
            return {"type": "array"}

    with pytest.raises(TaskTypeError, match="object schema"):
        TaskTypeRegistry().register(NotObject())


def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        TaskType()


def test_output_schema_merges_spec_fields():
    tt = LabelType()
    section = OutputSchemaSection(
        fields={
            "severity": FieldSpec(type="string", enum=["low", "high"], description="how bad"),
            "note": FieldSpec(type="string", required=False),
        }
    )
    schema = tt.output_schema(section)
    assert schema["properties"]["severity"] == {
        "type": "string",
        "description": "how bad",
        "enum": ["low", "high"],
    }
    assert schema["properties"]["note"] == {"type": "string"}
    assert schema["required"] == ["text", "label", "severity"]
    # the type's own schema is not mutated
    assert "severity" not in tt.base_schema()["properties"]


def test_output_schema_nullable_field_allows_null():
    section = OutputSchemaSection(
        fields={"severity": FieldSpec(type="string", nullable=True, enum=["low", "high"])}
    )
    schema = LabelType().output_schema(section)
    assert schema["properties"]["severity"] == {
        "type": ["string", "null"],
        "enum": ["low", "high", None],
    }
    assert "severity" in schema["required"]


def test_output_schema_without_fields_is_base():
    tt = LabelType()
    assert tt.output_schema() == tt.base_schema()
    assert tt.output_schema(OutputSchemaSection()) == tt.base_schema()


def test_output_schema_rejects_redefining_owned_field():
    section = OutputSchemaSection(fields={"label": FieldSpec(type="string")})
    with pytest.raises(TaskTypeError, match=r"\['label'\].*toy_label"):
        LabelType().output_schema(section)


def test_axes_spec_overrides_defaults_by_name():
    tt = LabelType()
    assert [a.name for a in tt.axes([Axis(name="topic", values=["a"])])] == ["topic", "label"]
    override = Axis(name="label", values=[True])
    merged = tt.axes([override])
    assert merged == [override]


def test_defaults_and_extractor():
    lt, at = LabelType(), AnswerType()
    (check,) = lt.default_validators()
    assert check({"text": "hi"}) == []
    assert check({"text": ""}) == ["text is empty"]
    assert lt.answer_extractor() is None
    assert at.default_axes() == [] and at.default_validators() == []
    assert at.answer_extractor()("  B ") == "B"
    assert at.answer_extractor()("   ") is None


def test_module_level_registry():
    assert get_task_type.__module__ == "sdgf.tasktypes.registry"
    with pytest.raises(UnknownTaskTypeError):
        get_task_type("definitely_not_registered")
    assert isinstance(REGISTRY, TaskTypeRegistry)
