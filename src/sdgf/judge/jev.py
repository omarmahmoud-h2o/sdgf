"""Jev (TypeSafe AI, System One) as the decision judge (FRAMEWORK_DESIGN.md §7.3, D11).

Jev doesn't generate text. It takes a *state* and a set of typed *questions* and returns
typed answers with probabilities. Two parts:

    JevBackend   the models.<stage> backend `jev`: POSTs {model, state, questions} to the
                 System One endpoint and returns the answers. Defaults to OpenRouter's
                 System One API (https://openrouter.ai/api/v1/systemone) with the key in
                 OPENROUTER_API_KEY; point params.api_base at https://api.typesafe.ai for
                 TypeSafe's own endpoint. It answers decide(), never call().
    JevJudge     a Judge that compiles any task's rubric into Jev questions, so it works
                 for every use case without task-specific code:

                     verdict                  Choice over rubric.verdict.values
                     enum criterion           Choice over its values
                     int criterion, <= 10     Score with one level per integer
                     int criterion, > 10      Choice over the integers as strings

                 One request per record carries every question, so they're answered in
                 parallel over the same state. The answers map back into the same
                 JudgeResult (verdict, scores, per-field confidence) LLMJudge returns, so
                 L5, L6, calibration and metrics don't change.

The state Jev sees is {"task": judge context, "worked_examples": [...], "record": the
judge_fields view}. Like LLMJudge it never carries the label, spans or "_" keys.

A Score answer's `score` is an expected value over levels; the rubric needs one integer,
so the judge takes the most probable level. Confidence comes from Jev's own answer
distribution (Choice and Score carry one); TypeSafe doesn't claim it is calibrated, so a
Jev judge is trusted only after judge/calibration.py passes, as for any judge.

Jev can't write reasons (writes_reasons False): a rubric with reason_required needs
models.fallback_judge on a text model. Jev is deterministic, so it can't cast K
independent L6 votes; stage 0 rejects a spec that would ask it to.

params: api_base (default https://openrouter.ai/api), api_key_env (default
OPENROUTER_API_KEY), timeout (default 60), max_retries (default 3, on 429/529/5xx and
connection errors), backoff (seconds, default 1.0, doubled per retry).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Mapping

from sdgf.judge.interface import (
    Judge,
    JudgeError,
    JudgeField,
    JudgeParseError,
    JudgeResult,
    JudgeSchema,
    Record,
    compile_rubric,
)
from sdgf.judge.llm_judge import judge_view
from sdgf.models._util import api_key_from_env
from sdgf.models.base import DecisionResponse, ModelBackend, ModelBackendError, ModelResponse
from sdgf.spec.schema import Hosting, ModelConfig, RubricExample

JEV_BACKEND = "jev"
DEFAULT_API_BASE = "https://openrouter.ai/api"
DEFAULT_KEY_ENV = "OPENROUTER_API_KEY"
SYSTEM_ONE_PATH = "/v1/systemone"
MAX_SCORE_LEVELS = 10  # the System One API accepts 2..10 Score levels
RETRYABLE = frozenset({429, 500, 502, 503, 504, 529})
# A request the service refuses as invalid for this record (too large, malformed state):
# the judge can't answer it, which is a judge error for the record, not a crash.
UNANSWERABLE = frozenset({400, 413, 422})

# (method, url, headers, body, timeout) -> (status, parsed JSON body)
Transport = Callable[[str, str, Mapping[str, str], Any, float], tuple[int, Any]]


class JevRequestError(ModelBackendError):
    """The System One endpoint refused a request or kept failing."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def urllib_transport(
    method: str, url: str, headers: Mapping[str, str], body: Any, timeout: float
) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=dict(headers), method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, {"error": raw}


def _error_text(payload: Any) -> str:
    if isinstance(payload, dict) and "error" in payload:
        err = payload["error"]
        return err.get("message", str(err)) if isinstance(err, dict) else str(err)
    return str(payload)[:500]


# ── backend ──────────────────────────────────────────────────────


class JevBackend(ModelBackend):
    name = JEV_BACKEND
    default_hosting = "provider_api"

    def __init__(
        self,
        model: str,
        hosting: Hosting | None = None,
        params: dict | None = None,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        super().__init__(model, hosting)
        params = dict(params or {})
        params.setdefault("api_key_env", DEFAULT_KEY_ENV)
        self.api_base = str(params.get("api_base", DEFAULT_API_BASE)).rstrip("/")
        self.url = self.api_base + SYSTEM_ONE_PATH
        self.api_key = api_key_from_env(params, self.name)
        self.timeout = float(params.get("timeout", 60))
        self.max_retries = int(params.get("max_retries", 3))
        self.backoff = float(params.get("backoff", 1.0))
        if self.max_retries < 0:
            raise ModelBackendError("jev: max_retries must be >= 0")
        self.transport = transport or urllib_transport
        self.sleep = sleep

    def call(self, prompt, max_tokens, temperature, tools=None) -> ModelResponse:
        raise ModelBackendError(
            "jev is a decision model: it answers typed questions, not prompts, so it can "
            "only serve a judge stage (models.judge or models.consistency_judge)"
        )

    def decide(self, state: Any, questions: dict[str, dict[str, Any]]) -> DecisionResponse:
        body = {"model": self.model, "state": state, "questions": questions}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        status, payload, error = None, None, ""
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.sleep(self.backoff * 2 ** (attempt - 1))
            try:
                status, payload = self.transport("POST", self.url, headers, body, self.timeout)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                status, payload, error = None, None, f"{type(e).__name__}: {e}"
                continue
            if status == 200:
                return self._response(payload)
            error = _error_text(payload)
            if status not in RETRYABLE:
                break
        where = f"HTTP {status}" if status is not None else "no response"
        raise JevRequestError(f"jev: System One request failed ({where}): {error}", status)

    @staticmethod
    def _response(payload: Any) -> DecisionResponse:
        if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
            raise JevRequestError("jev: response has no answers object", 200)
        usage = payload.get("usage") or {}
        return DecisionResponse(
            answers=payload["answers"],
            model=payload.get("model"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
        )


def factory(config: ModelConfig) -> ModelBackend:
    """Model-registry factory for backend: jev."""
    return JevBackend(config.model, config.hosting, config.params)


# ── rubric -> questions ──────────────────────────────────────────


def _instructions(question: str, description: str) -> Any:
    """A plain question, or {question, definition} when the rubric defines the field."""
    description = " ".join(description.split())
    return {"question": question, "definition": description} if description else question


def _uses_score(field: JudgeField) -> bool:
    return field.kind == "int" and field.size <= MAX_SCORE_LEVELS


def field_question(field: JudgeField, *, verdict: bool = False) -> dict[str, Any]:
    """The System One question for one rubric field."""
    if verdict:
        ask = "Which verdict does `record` deserve under the rubric in `task`?"
    else:
        ask = f"Under the rubric in `task`, what is {field.name} for `record`?"
    question: dict[str, Any] = {"instructions": _instructions(ask, field.description)}
    if field.kind == "enum":
        question.update(type="choice", criteria={c: None for c in field.choices})
    elif _uses_score(field):
        assert field.min is not None and field.max is not None
        levels = [
            f"{field.name} = {n} (scale {field.min} lowest to {field.max} highest)"
            for n in range(field.min, field.max + 1)
        ]
        question.update(type="score", criteria=levels)
    else:
        question.update(type="choice", criteria={str(v): None for v in field.values()})
    return question


def compile_questions(schema: JudgeSchema) -> dict[str, dict[str, Any]]:
    """Every rubric field as a System One question, keyed by field name."""
    questions = {schema.verdict.name: field_question(schema.verdict, verdict=True)}
    for c in schema.criteria:
        questions[c.name] = field_question(c)
    return questions


def _most_probable(probabilities: Mapping[str, Any], order: Iterable[str]) -> str | None:
    """The most probable key; ties go to the first in `order`, so decoding is deterministic."""
    best, best_p = None, -1.0
    for key in order:
        p = probabilities.get(key)
        if isinstance(p, (int, float)) and not isinstance(p, bool) and p > best_p:
            best, best_p = key, float(p)
    return best


def decode_answer(field: JudgeField, answer: Any) -> tuple[Any, float | None, list[str]]:
    """(rubric value, confidence, errors) for one field's answer."""
    name = field.name
    if not isinstance(answer, dict):
        return None, None, [f"{name}: no answer"]
    confidence = answer.get("confidence")
    if confidence is not None and not isinstance(confidence, (int, float)):
        return None, None, [f"{name}: confidence is not a number"]
    if _uses_score(field):
        if answer.get("type") != "score":
            return None, None, [f"{name}: expected a score answer, got {answer.get('type')!r}"]
        assert field.min is not None
        levels = [str(i) for i in range(field.size)]
        level = _most_probable(answer.get("probabilities") or {}, levels)
        if level is None:
            score = answer.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                return None, None, [f"{name}: score answer has no probabilities or score"]
            level = str(min(max(round(score), 0), field.size - 1))
        return field.min + int(level), confidence, []
    if answer.get("type") != "choice":
        return None, None, [f"{name}: expected a choice answer, got {answer.get('type')!r}"]
    choice = answer.get("choice")
    allowed = [str(v) for v in field.values()]
    if choice not in allowed:
        return None, None, [f"{name}: {choice!r} is not one of {allowed}"]
    return (int(choice) if field.kind == "int" else choice), confidence, []


# ── judge ────────────────────────────────────────────────────────


class JevJudge(Judge):
    """A blind decision judge on Jev: every rubric field as one typed question."""

    name = JEV_BACKEND
    writes_reasons = False
    temperature = None  # Jev has no sampling temperature; L6 ballots record None

    def __init__(
        self,
        schema: JudgeSchema,
        backend: ModelBackend,
        *,
        fields: Iterable[str],
        context: str = "",
        examples: Iterable[RubricExample] = (),
        stage: str | None = None,
    ):
        super().__init__(schema)
        self.backend = backend
        self.fields = tuple(fields)
        if not self.fields:
            raise JudgeError("a judge needs at least one record field to look at")
        self.context = context.strip()
        self.stage = stage  # the models stage this judge calls, for provenance
        self.examples = tuple(examples)
        for i, ex in enumerate(self.examples):
            extra = sorted(set(ex.record) - set(self.fields))
            if extra:
                raise JudgeError(f"example {i} shows fields the judge may not see: {extra}")
        self.questions = compile_questions(schema)

    @classmethod
    def from_spec(
        cls,
        compiled: Any,
        backend: ModelBackend,
        *,
        stage: str = "judge",
        task_type: Any = None,
        **kwargs: Any,
    ) -> JevJudge:
        """A Jev judge for a compiled spec, with the same context, examples and judge view
        as LLMJudge.from_spec. Sampling options (temperature, max_tokens) don't apply."""
        from sdgf.tasktypes.registry import REGISTRY

        spec = compiled.spec
        task_type = task_type or REGISTRY.resolve(spec.task)
        kwargs.pop("temperature", None)
        kwargs.pop("max_tokens", None)
        kwargs.pop("parse_retries", None)
        kwargs.setdefault("fields", task_type.judge_fields())
        kwargs.setdefault("context", spec.rubric.judge_context or spec.task.description)
        kwargs.setdefault("examples", spec.rubric.examples)
        kwargs.setdefault("stage", stage)
        return cls(compile_rubric(spec.rubric), backend, **kwargs)

    def state(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """What Jev sees: the task context, any worked examples, and the blind record view."""
        state: dict[str, Any] = {}
        if self.context:
            state["task"] = self.context
        if self.examples:
            state["worked_examples"] = [
                {
                    "record": ex.record,
                    "expected": {
                        "verdict": ex.verdict,
                        **({"scores": ex.scores} if ex.scores else {}),
                    },
                    **({"why": ex.note.strip()} if ex.note.strip() else {}),
                }
                for ex in self.examples
            ]
        state["record"] = judge_view(record, self.fields)
        return state

    def judge(self, record: Record) -> JudgeResult:
        try:
            response = self.backend.decide(self.state(record), self.questions)
        except JevRequestError as e:
            if e.status in UNANSWERABLE:
                raise JudgeParseError([f"<request>: {e}"]) from e
            raise
        return self.result(response.answers)

    def result(self, answers: Mapping[str, Any]) -> JudgeResult:
        """Map System One answers back into the rubric's typed JudgeResult."""
        errors: list[str] = []
        values: dict[str, Any] = {}
        confidence: dict[str, float] = {}
        for field in self.schema.fields:
            value, conf, errs = decode_answer(field, answers.get(field.name))
            errors += errs
            values[field.name] = value
            if conf is not None:
                confidence[field.name] = float(conf)
            elif not errs:
                errors.append(f"{field.name}: answer has no confidence")
        if errors:
            raise JudgeParseError(errors)
        verdict = values.pop(self.schema.verdict.name)
        data = {"verdict": verdict, "scores": values, "confidence": confidence}
        return self.schema.parse(data, with_reason=False)
