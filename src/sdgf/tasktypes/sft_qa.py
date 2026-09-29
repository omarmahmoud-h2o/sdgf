"""Answer-emergent SFT question-answer pairs (FRAMEWORK_DESIGN.md §7.1, §11).

A record is a question the model wrote, the model's own response to it, and the answer
read out of that response:

    question  the instruction or exam question, options included for multiple choice
    response  reasoning ending in the final answer in the format's required form
    answer    optional; when present it must equal what the extractor reads from response

Nothing is fixed by code before generation: the answer emerges, and L6 votes K fresh
answers to the question, so the majority becomes the answer (the DS²-Instruct method).
How an answer is read out of text is the answer format, chosen per spec with
task.answer_format and pluggable via register_answer_format(). The built-in formats port
the extractors and response suffixes in src_original/DS2-Instruct/scripts/utils.py and
prompts.py.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sdgf.spec.schema import Axis, TaskSection
from sdgf.tasktypes.base import AnswerExtractor, Record, TaskType, TaskTypeError, Validator
from sdgf.tasktypes.registry import REGISTRY

# ── extractors (ported from DS2-Instruct scripts/utils.py) ────────


def extract_boxed_math_answer(resp: str) -> str | None:
    resp = resp.replace(",", "")
    m = re.findall(r"\\boxed\{(.*?)\}", resp)
    if m:
        return m[-1].strip()
    m = re.findall(r"final answer is:?\s*(.*)", resp, re.IGNORECASE)
    if m:
        return m[-1].strip()
    nums = re.findall(r"[-+]?\d*\.?\d+", resp)
    return nums[-1] if nums else None


def extract_numeric_answer(resp: str) -> str | None:
    m = re.findall(r"[Ff]inal [Aa]nswer:?\s*(.+)", resp)
    if m:
        nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", m[-1])
        return nums[0].replace(",", "") if nums else m[-1].strip()
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", resp)
    return nums[-1].replace(",", "") if nums else None


def extract_multiple_choice_answer(resp: str) -> str | None:
    m = re.search(r"[Aa]nswer:\s*([A-D])", resp)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-D])\b", resp)
    return m.group(1) if m else None


def extract_yes_no_maybe_answer(resp: str) -> str | None:
    m = re.search(r"[Aa]nswer:\s*(yes|no|maybe)", resp, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    low = resp.strip().lower()
    for ans in ("yes", "no", "maybe"):
        if low.startswith(ans):
            return ans
    return None


# ── answer formats ───────────────────────────────────────────────

_MC_SUFFIX = (
    "Return exactly two lines and nothing else:\n"
    "Reason: <1-3 sentence explanation>\n"
    "Answer: <A|B|C|D>"
)


@dataclass(frozen=True)
class AnswerFormat:
    name: str
    extractor: AnswerExtractor
    suffix: str  # appended to the question when answering, so answers are extractable
    choices: tuple[str, ...] | None = None  # closed answer set, if any


ANSWER_FORMATS: dict[str, AnswerFormat] = {}


def register_answer_format(fmt: AnswerFormat, *, replace: bool = False) -> AnswerFormat:
    if fmt.name in ANSWER_FORMATS and not replace:
        raise TaskTypeError(f"answer format {fmt.name!r} is already registered")
    ANSWER_FORMATS[fmt.name] = fmt
    return fmt


def get_answer_format(name: str) -> AnswerFormat:
    try:
        return ANSWER_FORMATS[name]
    except KeyError:
        known = ", ".join(sorted(ANSWER_FORMATS))
        raise TaskTypeError(f"unknown answer format {name!r}; registered: {known}") from None


MULTIPLE_CHOICE = register_answer_format(
    AnswerFormat(
        "multiple_choice", extract_multiple_choice_answer, _MC_SUFFIX, ("A", "B", "C", "D")
    )
)
YES_NO_MAYBE = register_answer_format(
    AnswerFormat(
        "yes_no_maybe",
        extract_yes_no_maybe_answer,
        "Return exactly two lines and nothing else:\n"
        "Reason: <1-3 sentence explanation>\n"
        "Answer: <yes|no|maybe>",
        ("yes", "no", "maybe"),
    )
)
NUMERIC = register_answer_format(
    AnswerFormat(
        "numeric",
        extract_numeric_answer,
        "Provide a step-by-step reasoning process and then write the final "
        "numerical answer on a new line in the format: final answer: <answer>.",
    )
)
BOXED_MATH = register_answer_format(
    AnswerFormat(
        "boxed_math",
        extract_boxed_math_answer,
        "Provide a step-by-step reasoning process and then write the final "
        r"answer in the LaTeX boxed tag: $\boxed{answer}$.",
    )
)

# ── task type ────────────────────────────────────────────────────


def _text(record: Record, key: str) -> str | None:
    value = record.get(key)
    return value if isinstance(value, str) else None


def option_labels(question: str) -> set[str]:
    """Option letters the question lists, as `A)`, `A.`, `(A)` or `A:` at a line start."""
    return set(re.findall(r"(?m)^\s*\(?([A-D])[).:]\s", question))


class SftQA(TaskType):
    name = "sft_qa"
    generation_modes = ("answer_emergent",)

    def __init__(self, answer_format: str = "multiple_choice"):
        self.format = get_answer_format(answer_format)

    def base_schema(self) -> dict[str, Any]:
        answer: dict[str, Any] = {"type": "string", "minLength": 1}
        if self.format.choices is not None:
            answer["enum"] = list(self.format.choices)
        return {
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 1},
                "response": {"type": "string", "minLength": 1},
                "answer": answer,
            },
            "required": ["question", "response"],
        }

    def configure(self, task: TaskSection) -> TaskType:
        if task.answer_format is None or task.answer_format == self.format.name:
            return self
        return SftQA(task.answer_format)

    def default_axes(self) -> list[Axis]:
        return [Axis(name="bloom_level", source="bloom")]

    def answer_extractor(self) -> AnswerExtractor:
        return self.format.extractor

    def answer_suffix(self) -> str:
        return self.format.suffix

    def label_field(self) -> str:
        # L5 compares a blind judge's answer with the model's own answer.
        return "answer"

    def derive_fields(self, record: Record) -> Record:
        """Fill a missing answer from the response, so L5 and L6 have one to compare."""
        response = _text(record, "response")
        if record.get("answer") is None and response is not None:
            extracted = self.format.extractor(response)
            if extracted is not None:
                return {**record, "answer": extracted}
        return record

    def default_validators(self) -> list[Validator]:
        validators = [self.response_answer_errors]
        if self.format is MULTIPLE_CHOICE:
            validators.append(self.option_errors)
        return validators

    def judge_fields(self) -> tuple[str, ...]:
        # The judge and the K voters answer the question afresh, so they never see the
        # generator's response or answer.
        return ("question",)

    def response_answer_errors(self, record: Record) -> list[str]:
        response = _text(record, "response")
        if response is None:
            return []
        extracted = self.format.extractor(response)
        if extracted is None:
            return [f"response has no {self.format.name} answer the extractor can read"]
        answer = record.get("answer")
        if answer is not None and answer != extracted:
            return [f"answer={answer!r} but the response's final answer reads {extracted!r}"]
        return []

    def option_errors(self, record: Record) -> list[str]:
        question, response = _text(record, "question"), _text(record, "response")
        if question is None:
            return []
        labels = option_labels(question)
        if len(labels) < 2:
            return ["a multiple-choice question must list at least two options, A) B) ..."]
        chosen = record.get("answer") or (response and self.format.extractor(response))
        if chosen is not None and chosen not in labels:
            return [f"answer {chosen!r} is not one of the question's options {sorted(labels)}"]
        return []


SFT_QA = REGISTRY.register(SftQA())
