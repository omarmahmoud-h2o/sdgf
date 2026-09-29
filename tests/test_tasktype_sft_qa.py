import jsonschema
import pytest

from sdgf.spec.schema import TaskSection
from sdgf.tasktypes.base import TaskTypeError
from sdgf.tasktypes.registry import REGISTRY, get_task_type
from sdgf.tasktypes.sft_qa import (
    ANSWER_FORMATS,
    AnswerFormat,
    SftQA,
    extract_boxed_math_answer,
    extract_multiple_choice_answer,
    extract_numeric_answer,
    extract_yes_no_maybe_answer,
    get_answer_format,
    option_labels,
    register_answer_format,
)

QUESTION = (
    "A fictional bond pays a 5% annual coupon on a face value of 100. What is the coupon?\n"
    "A) 0.5\nB) 5\nC) 50\nD) 500"
)
RECORD = {
    "question": QUESTION,
    "response": "Reason: 5% of 100 is 5.\nAnswer: B",
    "answer": "B",
}


def _task(**kw) -> TaskSection:
    base = dict(
        name="t", version="1", type="sft_qa", generation_mode="answer_emergent", description="d"
    )
    return TaskSection(**{**base, **kw})


def _valid(task_type, record) -> bool:
    return jsonschema.Draft202012Validator(task_type.output_schema()).is_valid(record)


def _errors(task_type, record) -> list[str]:
    return [e for v in task_type.default_validators() for e in v(record)]


# ── extractors ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Reason: ...\nAnswer: C", "C"),
        ("answer: a", None),  # the letter must be upper case
        ("I think B is right", "B"),
        ("no letter here", None),
    ],
)
def test_multiple_choice(text, expected):
    assert extract_multiple_choice_answer(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Reason: ...\nAnswer: Maybe", "maybe"),
        ("Yes, because the abstract says so.", "yes"),
        ("It is unclear.", None),
    ],
)
def test_yes_no_maybe(text, expected):
    assert extract_yes_no_maybe_answer(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("3 apples plus 4\nfinal answer: 1,200 dollars", "1200"),
        ("Final Answer: seven", "seven"),
        ("first 3 then 12", "12"),
        ("no numbers", None),
    ],
)
def test_numeric(text, expected):
    assert extract_numeric_answer(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (r"so $\boxed{1}$ then $\boxed{2,000}$", "2000"),
        ("The final answer is: x = 3", "x = 3"),
        ("roughly 2.5 or 7", "7"),
        ("nothing", None),
    ],
)
def test_boxed_math(text, expected):
    assert extract_boxed_math_answer(text) == expected


def test_builtin_formats_registered():
    assert {"multiple_choice", "yes_no_maybe", "numeric", "boxed_math"} <= set(ANSWER_FORMATS)


def test_unknown_format_is_clear():
    with pytest.raises(TaskTypeError, match="unknown answer format 'nope'"):
        get_answer_format("nope")


def test_custom_format_is_pluggable():
    fmt = AnswerFormat("upper_word", lambda t: t.strip().upper() or None, "One word.")
    register_answer_format(fmt)
    try:
        with pytest.raises(TaskTypeError, match="already registered"):
            register_answer_format(fmt)
        tt = REGISTRY.resolve(_task(answer_format="upper_word"))
        assert tt.answer_extractor()(" yes ") == "YES"
        assert tt.answer_suffix() == "One word."
    finally:
        ANSWER_FORMATS.pop("upper_word")


# ── task type ────────────────────────────────────────────────────


def test_registered_on_import():
    tt = get_task_type("sft_qa")
    assert isinstance(tt, SftQA)
    assert tt.generation_modes == ("answer_emergent",)
    assert tt.format.name == "multiple_choice"


def test_resolve_picks_the_spec_format():
    tt = REGISTRY.resolve(_task(answer_format="numeric"))
    assert tt.format.name == "numeric"
    assert tt.answer_extractor() is extract_numeric_answer
    assert "final answer:" in tt.answer_suffix()
    assert get_task_type("sft_qa").format.name == "multiple_choice"  # registry untouched


def test_resolve_default_format_and_label_first_rejected():
    assert REGISTRY.resolve(_task()) is get_task_type("sft_qa")
    with pytest.raises(TaskTypeError, match="does not support generation_mode 'label_first'"):
        REGISTRY.resolve(_task(generation_mode="label_first"))
    with pytest.raises(TaskTypeError, match="unknown answer format"):
        REGISTRY.resolve(_task(answer_format="nope"))


def test_other_types_reject_answer_format():
    task = _task(
        type="classification_spans", generation_mode="label_first", answer_format="numeric"
    )
    with pytest.raises(TaskTypeError, match="takes no answer_format"):
        REGISTRY.resolve(task)


def test_schema_accepts_valid_record_and_answer_is_optional():
    tt = SftQA()
    assert _valid(tt, RECORD)
    assert _valid(tt, {k: v for k, v in RECORD.items() if k != "answer"})


@pytest.mark.parametrize("missing", ["question", "response"])
def test_schema_requires(missing):
    assert not _valid(SftQA(), {k: v for k, v in RECORD.items() if k != missing})


def test_schema_closed_answer_set():
    assert not _valid(SftQA(), {**RECORD, "answer": "E"})
    assert _valid(SftQA("numeric"), {**RECORD, "answer": "42"})
    assert not _valid(SftQA("yes_no_maybe"), {**RECORD, "answer": "B"})


def test_default_axes_add_bloom():
    (axis,) = SftQA().default_axes()
    assert axis.name == "bloom_level" and axis.source == "bloom"


def test_judge_sees_only_the_question():
    assert SftQA().judge_fields() == ("question",)


def test_valid_record_has_no_validator_errors():
    assert _errors(SftQA(), RECORD) == []


def test_unreadable_response():
    errors = _errors(SftQA("numeric"), {"question": "q", "response": "no digits at all"})
    assert errors == ["response has no numeric answer the extractor can read"]


def test_answer_must_match_response():
    errors = _errors(SftQA(), {**RECORD, "answer": "C"})
    assert errors == ["answer='C' but the response's final answer reads 'B'"]


def test_multiple_choice_needs_options():
    errors = _errors(SftQA(), {"question": "What is 5% of 100?", "response": "Answer: B"})
    assert any("at least two options" in e for e in errors)


def test_multiple_choice_answer_must_be_an_option():
    record = {"question": "Pick one.\nA) x\nB) y", "response": "Answer: D"}
    assert _errors(SftQA(), record) == [
        "answer 'D' is not one of the question's options ['A', 'B']"
    ]


def test_option_labels_forms():
    assert option_labels("Q\nA) x\n(B) y\nC. z\nD: w") == {"A", "B", "C", "D"}
    assert option_labels("A company ... B2B.") == set()


def test_option_check_only_for_multiple_choice():
    record = {"question": "What is 2 + 3?", "response": "final answer: 5"}
    assert _errors(SftQA("numeric"), record) == []


# ── label field and derived answer ───────────────────────────────


def test_label_field_is_the_answer_and_the_judge_never_sees_it():
    sft = SftQA()
    assert sft.label_field() == "answer"
    assert sft.answer_suffix() == sft.format.suffix
    assert "answer" not in sft.judge_fields() and "response" not in sft.judge_fields()


def test_derive_fields_reads_a_missing_answer_from_the_response():
    sft = SftQA()
    record = {k: v for k, v in RECORD.items() if k != "answer"}
    assert sft.derive_fields(record) == {**record, "answer": "B"}
    assert "answer" not in record  # the input isn't modified


def test_derive_fields_keeps_a_given_answer_and_unreadable_responses():
    sft = SftQA()
    wrong = {**RECORD, "answer": "C"}
    assert sft.derive_fields(wrong) == wrong  # left for the validator to reject
    unreadable = {"question": QUESTION, "response": "no idea"}
    assert sft.derive_fields(unreadable) == unreadable


def test_other_task_types_keep_the_label_field_and_derive_nothing():
    spans = get_task_type("classification_spans")
    assert spans.label_field() == "label" and spans.answer_suffix() == ""
    record = {"label": True}
    assert spans.derive_fields(record) is record
