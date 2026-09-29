import json

import pytest

from sdgf.judge.interface import Judge, JudgeError, compile_rubric
from sdgf.judge.jev import (
    JEV_BACKEND,
    NOT_IMPLEMENTED,
    JevJudge,
    JevNotImplementedError,
    factory,
)
from sdgf.judge.llm_judge import LLMJudge
from sdgf.judge.select import build_judge
from sdgf.models.base import ModelBackendError
from sdgf.models.mock import MockBackend
from sdgf.models.registry import REGISTRY, build_models
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import ModelConfig, RubricSection

TASK_YAML = """\
task:
  name: toy
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Toy task for Jev selection tests.
output_schema: {{}}
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
  judge:
{judge}
validation:
  layers: [L1, L2, L5]
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

JEV = "    backend: jev\n    model: jev-1\n    hosting: provider_api"
MOCK = "    backend: mock\n    model: mock-judge"


def toy(tmp_path, judge):
    (tmp_path / "task.yaml").write_text(TASK_YAML.format(judge=judge), encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text(
        '{"id": "seed-1", "customer": "Acme Test Pty Ltd", "label": true}\n', encoding="utf-8"
    )
    return compile_spec(tmp_path)


def assert_points_at_design(err):
    msg = str(err)
    assert "Jev" in msg
    assert "FRAMEWORK_DESIGN.md" in msg and "§7.3" in msg and "§16 Q1" in msg


def test_jev_is_registered_as_a_model_backend():
    assert JEV_BACKEND == "jev"
    assert JEV_BACKEND in REGISTRY.names()


def test_error_is_not_implemented_and_a_backend_error():
    assert issubclass(JevNotImplementedError, NotImplementedError)
    assert issubclass(JevNotImplementedError, ModelBackendError)
    assert_points_at_design(NOT_IMPLEMENTED)


def test_factory_fails_clearly():
    with pytest.raises(JevNotImplementedError) as e:
        factory(ModelConfig(backend="jev", model="jev-1"))
    assert_points_at_design(e.value)


def test_jev_judge_is_a_judge_that_cannot_be_built():
    assert issubclass(JevJudge, Judge)
    assert JevJudge.writes_reasons is False
    schema = compile_rubric(RubricSection(verdict={"values": ["pass", "fail"]}))
    with pytest.raises(NotImplementedError) as e:
        JevJudge(schema)
    assert_points_at_design(e.value)


def test_spec_selecting_jev_compiles(tmp_path):
    c = toy(tmp_path, JEV)
    assert c.spec.models.judge.backend == "jev"


def test_building_models_for_a_jev_spec_fails_naming_the_stage(tmp_path):
    c = toy(tmp_path, JEV)
    with pytest.raises(JevNotImplementedError) as e:
        build_models(c.spec.models)
    assert str(e.value).startswith("models.judge: ")
    assert_points_at_design(e.value)


def test_build_judge_for_a_jev_spec_fails_clearly(tmp_path):
    c = toy(tmp_path, JEV)
    with pytest.raises(NotImplementedError) as e:
        build_judge(c)
    assert str(e.value).startswith("models.judge: ")
    assert_points_at_design(e.value)


def test_no_silent_fallback_to_another_judge(tmp_path):
    # Selecting Jev must never quietly give an LLM judge on some other backend.
    c = toy(tmp_path, JEV)
    with pytest.raises(JevNotImplementedError):
        build_judge(c, stage="judge")


def test_explicit_backend_override_wins_over_jev(tmp_path):
    # As with ModelRegistry overrides, a hand-built backend (e.g. a test mock) is used.
    c = toy(tmp_path, JEV)
    reply = json.dumps({"verdict": "pass", "scores": {}, "confidence": {"verdict": 0.9}})
    judge = build_judge(c, MockBackend([reply]))
    assert isinstance(judge, LLMJudge)
    assert judge.judge({"messages": []}).verdict == "pass"


def test_build_judge_for_a_text_backend_gives_llm_judge(tmp_path):
    c = toy(tmp_path, MOCK)
    judge = build_judge(c)
    assert isinstance(judge, LLMJudge)
    assert judge.backend.model == "mock-judge"


def test_build_judge_without_config_raises(tmp_path):
    c = toy(tmp_path, MOCK)
    with pytest.raises(JudgeError, match="models.fallback_judge"):
        build_judge(c, stage="fallback_judge")
