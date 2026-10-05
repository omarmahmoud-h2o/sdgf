"""The Jev decision judge: rubric -> System One questions, the OpenRouter request, answers
-> JudgeResult, error handling, selection and the stage 0 rules. A fake transport stands in
for the HTTP endpoint; nothing calls the network."""

import json

import pytest

from sdgf.judge.interface import Judge, JudgeError, JudgeParseError, compile_rubric
from sdgf.judge.jev import (
    DEFAULT_API_BASE,
    JEV_BACKEND,
    JevBackend,
    JevJudge,
    JevRequestError,
    compile_questions,
    factory,
)
from sdgf.judge.llm_judge import LLMJudge
from sdgf.judge.select import build_judge, judge_from_spec
from sdgf.models.base import BoundedBackend, ModelBackendError
from sdgf.models.mock import MockBackend
from sdgf.models.registry import REGISTRY
from sdgf.models.usage import MeteredBackend, Pricing, UsageMeter
from sdgf.spec.compile import Stage0Error, compile_spec
from sdgf.spec.schema import ModelConfig, RubricSection

KEY_ENV = "SDGF_TEST_OPENROUTER_KEY"

TASK_YAML = """\
task:
  name: toy
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Toy task for Jev tests.
output_schema:
  turns:
    roles: [customer, assistant]
    first_role: customer
rubric:
  verdict:
    values: [pass, fail]
    description: fail if the assistant breaks the rule.
  criteria:
    - name: tier
      values: [low, high]
    - name: realism
      min: 1
      max: 5
  reason_required: {reason}
seeds:
  path: seeds.jsonl
coverage:
  target_size: 10
  axes:
    - name: label
      values: [true, false]
models:
  generator:
{generator}
  judge:
{judge}
validation:
  layers: {layers}
  consistency_k: {k}
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

JEV = f"    backend: jev\n    model: jev-1.13\n    params: {{api_key_env: {KEY_ENV}}}"
MOCK = "    backend: mock\n    model: mock-1"


def toy(tmp_path, judge=JEV, *, generator=MOCK, layers="[L1, L2, L5]", k=5, reason="never"):
    (tmp_path / "task.yaml").write_text(
        TASK_YAML.format(judge=judge, generator=generator, layers=layers, k=k, reason=reason),
        encoding="utf-8",
    )
    (tmp_path / "seeds.jsonl").write_text(
        '{"id": "seed-1", "customer": "Acme Test Pty Ltd", "label": true}\n', encoding="utf-8"
    )
    return compile_spec(tmp_path)


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "or-test-key-000")


def answers(verdict="fail", tier="high", realism_probs=None, conf=0.9):
    probs = realism_probs or {"0": 0.0, "1": 0.1, "2": 0.2, "3": 0.6, "4": 0.1}
    return {
        "verdict": {
            "type": "choice",
            "choice": verdict,
            "probabilities": {verdict: conf},
            "confidence": conf,
        },
        "tier": {"type": "choice", "choice": tier, "probabilities": {tier: 0.8}, "confidence": 0.7},
        "realism": {
            "type": "score",
            "score": sum(int(k) * v for k, v in probs.items()),
            "probabilities": probs,
            "confidence": 0.5,
        },
    }


class FakeTransport:
    """Records each request and replies from a list of (status, payload)."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, method, url, headers, body, timeout):
        self.requests.append({"method": method, "url": url, "headers": dict(headers), "body": body})
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


def ok(answer_map, model="jev-1.13.0"):
    return 200, {
        "model": model,
        "answers": answer_map,
        "usage": {"input_tokens": 300, "output_tokens": 20},
    }


def backend(*replies, **params):
    return JevBackend(
        "jev-1.13",
        params={"api_key_env": KEY_ENV, **params},
        transport=FakeTransport(*replies),
        sleep=lambda s: None,
    )


RECORD = {
    "messages": [{"turn": 1, "role": "customer", "content": "Hi"}],
    "label": True,
    "spans": [{"turn": 1, "text": "Hi", "category": "x"}],
    "_provenance": {"x": 1},
}


# ── backend ──────────────────────────────────────────────────────


def test_jev_is_registered_and_builds_an_openrouter_backend():
    assert JEV_BACKEND in REGISTRY.names()
    b = factory(ModelConfig(backend="jev", model="jev-1.13", params={"api_key_env": KEY_ENV}))
    assert isinstance(b, JevBackend)
    assert b.url == DEFAULT_API_BASE + "/v1/systemone" == "https://openrouter.ai/api/v1/systemone"
    assert b.hosting == "provider_api"


def test_default_key_env_is_openrouter_and_must_be_set(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ModelBackendError, match="OPENROUTER_API_KEY"):
        JevBackend("jev-1.13")


def test_api_base_can_point_at_typesafe_directly():
    b = backend(ok({}), api_base="https://api.typesafe.ai/")
    assert b.url == "https://api.typesafe.ai/v1/systemone"


def test_jev_answers_questions_not_prompts():
    with pytest.raises(ModelBackendError, match="decision model"):
        backend(ok({})).call("hello", 10, 0.0)
    with pytest.raises(ModelBackendError, match="text model"):
        MockBackend(["x"]).decide({}, {})


def test_request_shape_and_auth():
    b = backend(ok({"q": {"type": "noul", "noul": 0.9}}))
    r = b.decide({"record": {"a": 1}}, {"q": {"type": "noul", "instructions": "Is it?"}})
    req = b.transport.requests[0]
    assert req["method"] == "POST" and req["url"] == "https://openrouter.ai/api/v1/systemone"
    assert req["headers"]["Authorization"] == "Bearer or-test-key-000"
    assert req["body"] == {
        "model": "jev-1.13",
        "state": {"record": {"a": 1}},
        "questions": {"q": {"type": "noul", "instructions": "Is it?"}},
    }
    assert r.answers["q"]["noul"] == 0.9 and r.model == "jev-1.13.0"
    assert (r.input_tokens, r.output_tokens) == (300, 20)


def test_rate_limits_are_retried_with_backoff():
    slept = []
    b = JevBackend(
        "jev-1.13",
        params={"api_key_env": KEY_ENV, "backoff": 0.5},
        transport=FakeTransport((429, {"error": {"message": "slow down"}}), (529, {}), ok({})),
        sleep=slept.append,
    )
    b.decide({}, {})
    assert len(b.transport.requests) == 3 and slept == [0.5, 1.0]


def test_client_errors_are_not_retried():
    b = backend((401, {"error": {"message": "bad key"}}), ok({}))
    with pytest.raises(JevRequestError, match="HTTP 401.*bad key") as e:
        b.decide({}, {})
    assert e.value.status == 401 and len(b.transport.requests) == 1


def test_retries_give_up_with_the_last_error():
    b = backend((503, {"error": "down"}), max_retries=2)
    with pytest.raises(JevRequestError, match="HTTP 503"):
        b.decide({}, {})
    assert len(b.transport.requests) == 3


def test_wrappers_pass_decide_through_and_meter_it():
    meter = UsageMeter()
    inner = backend(ok({}))
    wrapped = BoundedBackend(MeteredBackend(inner, "judge", meter, Pricing(0.05, 0.0)), 2)
    assert wrapped.name == "jev"
    wrapped.decide({"record": {}}, {})
    usage = meter.take_loose().stage("judge")
    assert (usage.calls, usage.input_tokens, usage.output_tokens) == (1, 300, 20)
    assert usage.cost_usd == pytest.approx(300 * 0.05 / 1e6)


# ── rubric -> questions ──────────────────────────────────────────


def schema(**criteria):
    rubric = RubricSection.model_validate(
        {
            "verdict": {"values": ["pass", "fail"], "description": "fail if the rule is broken"},
            "criteria": [{"name": n, **c} for n, c in criteria.items()],
        }
    )
    return compile_rubric(rubric)


def test_verdict_and_enum_criteria_become_choices():
    q = compile_questions(schema(tier={"values": ["low", "high"], "description": "Advice tier."}))
    assert q["verdict"]["type"] == "choice"
    assert q["verdict"]["criteria"] == {"pass": None, "fail": None}
    assert q["verdict"]["instructions"]["definition"] == "fail if the rule is broken"
    assert q["tier"] == {
        "type": "choice",
        "criteria": {"low": None, "high": None},
        "instructions": {
            "question": "Under the rubric in `task`, what is tier for `record`?",
            "definition": "Advice tier.",
        },
    }


def test_small_int_ranges_become_scores_and_wide_ones_choices():
    q = compile_questions(schema(realism={"min": 1, "max": 5}, count={"min": 0, "max": 20}))
    assert q["realism"]["type"] == "score"
    assert len(q["realism"]["criteria"]) == 5
    assert q["realism"]["criteria"][0].startswith("realism = 1")
    assert q["count"]["type"] == "choice"
    assert list(q["count"]["criteria"]) == [str(n) for n in range(21)]


# ── judge ────────────────────────────────────────────────────────


def jev_judge(c, *replies):
    return judge_from_spec(c, backend(*replies))


def test_selection_picks_jev_for_a_jev_backend_and_llm_otherwise(tmp_path):
    c = toy(tmp_path)
    assert isinstance(jev_judge(c, ok(answers())), JevJudge)
    assert isinstance(judge_from_spec(c, MockBackend(["{}"])), LLMJudge)
    built = build_judge(c)
    assert isinstance(built, JevJudge) and isinstance(built.backend, JevBackend)


def test_judge_maps_answers_into_the_rubric_result(tmp_path):
    j = jev_judge(toy(tmp_path), ok(answers()))
    r = j.judge(RECORD)
    assert r.verdict == "fail"
    assert r.scores == {"tier": "high", "realism": 4}  # level 3 of 1..5 is the most probable
    assert r.confidence == {"verdict": 0.9, "tier": 0.7, "realism": 0.5}
    assert r.reason is None


def test_the_state_is_blind_to_label_spans_and_private_keys(tmp_path):
    j = jev_judge(toy(tmp_path), ok(answers()))
    j.judge(RECORD)
    state = j.backend.transport.requests[0]["body"]["state"]
    assert state["record"] == {"messages": RECORD["messages"]}
    assert state["task"] == "Toy task for Jev tests."
    sent = json.dumps(j.backend.transport.requests[0]["body"])
    assert '"spans"' not in sent and '"label"' not in sent and "_provenance" not in sent


def test_judge_context_and_worked_examples_reach_the_state(tmp_path):
    rubric = RubricSection.model_validate(
        {
            **toy(tmp_path).spec.rubric.model_dump(exclude={"examples", "judge_context"}),
            "judge_context": "Judge only the assistant.",
            "examples": [
                {
                    "record": {"messages": [{"turn": 1, "role": "assistant", "content": "x"}]},
                    "verdict": "fail",
                    "note": "breaks the rule",
                }
            ],
        }
    )
    j = JevJudge(
        compile_rubric(rubric),
        backend(ok(answers())),
        fields=("messages",),
        context=rubric.judge_context,
        examples=rubric.examples,
    )
    state = j.state(RECORD)
    assert state["task"] == "Judge only the assistant."
    assert state["worked_examples"][0]["expected"] == {"verdict": "fail"}
    assert state["worked_examples"][0]["why"] == "breaks the rule"


def test_one_request_carries_every_question(tmp_path):
    j = jev_judge(toy(tmp_path), ok(answers()))
    j.judge(RECORD)
    j.judge(RECORD)
    reqs = j.backend.transport.requests
    assert len(reqs) == 2
    assert set(reqs[0]["body"]["questions"]) == {"verdict", "tier", "realism"}


@pytest.mark.parametrize(
    "broken, expected",
    [
        (lambda a: a.pop("tier"), "tier: no answer"),
        (lambda a: a["verdict"].update(choice="maybe"), "not one of"),
        (lambda a: a["realism"].update(type="choice"), "expected a score answer"),
        (lambda a: a["tier"].pop("confidence"), "tier: answer has no confidence"),
    ],
)
def test_unusable_answers_are_judge_parse_errors(tmp_path, broken, expected):
    a = answers()
    broken(a)
    with pytest.raises(JudgeParseError, match=expected):
        jev_judge(toy(tmp_path), ok(a)).judge(RECORD)


def test_a_request_refused_as_invalid_is_a_judge_parse_error(tmp_path):
    j = jev_judge(toy(tmp_path), (422, {"error": {"message": "state too large"}}))
    with pytest.raises(JudgeParseError, match="state too large"):
        j.judge(RECORD)


def test_auth_failures_still_raise(tmp_path):
    j = jev_judge(toy(tmp_path), (401, {"error": "bad key"}))
    with pytest.raises(JevRequestError):
        j.judge(RECORD)


def test_jev_writes_no_reasons(tmp_path):
    j = jev_judge(toy(tmp_path), ok(answers()))
    assert isinstance(j, Judge) and j.writes_reasons is False
    with pytest.raises(JudgeError, match="fallback judge"):
        j.explain(RECORD, j.judge(RECORD))


def test_explicit_backend_override_wins_over_the_spec(tmp_path):
    c = toy(tmp_path)
    reply = json.dumps(
        {
            "verdict": "pass",
            "scores": {"tier": "low", "realism": 3},
            "confidence": {"verdict": 0.9, "tier": 0.9, "realism": 0.9},
        }
    )
    judge = build_judge(c, MockBackend([reply]))
    assert isinstance(judge, LLMJudge)
    assert judge.judge({"messages": []}).verdict == "pass"


# ── stage 0 ──────────────────────────────────────────────────────


def test_jev_can_judge_l5(tmp_path):
    assert toy(tmp_path).spec.models.judge.backend == "jev"


def test_jev_cannot_generate(tmp_path):
    with pytest.raises(Stage0Error, match="models.generator: jev is a decision model"):
        toy(tmp_path, generator=JEV)


def test_jev_cannot_cast_k_l6_votes(tmp_path):
    with pytest.raises(Stage0Error, match="models.judge: jev is deterministic.*5 votes"):
        toy(tmp_path, layers="[L1, L2, L5, L6]", k=5)
    (tmp_path / "k1").mkdir()
    assert toy(tmp_path / "k1", layers="[L1, L2, L5, L6]", k=1)


def test_reasons_need_a_text_fallback_judge(tmp_path):
    with pytest.raises(Stage0Error, match="models.fallback_judge: .*can't write reasons"):
        toy(tmp_path, JEV + "\n  fallback_judge:\n" + JEV, reason="flagged")
    (tmp_path / "ok").mkdir()
    assert toy(tmp_path / "ok", JEV + "\n  fallback_judge:\n" + MOCK, reason="flagged")
