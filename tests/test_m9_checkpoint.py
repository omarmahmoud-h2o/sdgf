"""M9 checkpoint: the finished framework ships two use cases on one core. Both task specs
validate through the CLI, they cover both generation modes, every built-in task type and
backend is registered, Jev builds as a decision backend (OpenRouter System One), and importing the
whole core pulls in no optional engine or model SDK. MockBackends and subprocesses only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sdgf.judge.jev import JevBackend, factory
from sdgf.models.registry import REGISTRY as MODELS
from sdgf.spec.compile import compile_spec
from sdgf.tasktypes.registry import REGISTRY as TASK_TYPES
from test_cli import sdgf

ROOT = Path(__file__).resolve().parents[1]
TASKS = {"fag": ("classification_spans", "label_first", 6), "cfa": ("sft_qa", "answer_emergent", 5)}
OPTIONAL = (
    "presidio_analyzer",
    "detoxify",
    "sentence_transformers",
    "rank_bm25",
    "anthropic",
    "openai",
    "vllm",
    "mlx_lm",
    "torch",
)


@pytest.mark.parametrize("task", sorted(TASKS))
def test_shipped_task_validates_through_cli(task):
    task_type, mode, seeds = TASKS[task]
    code, out, err = sdgf("validate-spec", ROOT / "tasks" / task)
    assert code == 0, err
    assert out["ok"] is True
    assert (out["task"], out["task_type"], out["generation_mode"]) == (task, task_type, mode)
    assert out["seeds"] == seeds
    assert out["layers"] == ["L1", "L2", "L3", "L4", "L5", "L6"]
    assert out["spec_version"] == compile_spec(ROOT / "tasks" / task).spec_version


def test_both_generation_modes_share_one_core():
    modes = {compile_spec(ROOT / "tasks" / t).spec.task.generation_mode for t in TASKS}
    assert modes == {"label_first", "answer_emergent"}
    assert {"classification_spans", "sft_qa"} <= set(TASK_TYPES.names())


def test_every_backend_is_registered_and_jev_is_a_decision_backend(monkeypatch):
    from sdgf.spec.schema import ModelConfig

    names = set(MODELS.names())
    assert {"mock", "openai_compat", "anthropic", "vllm", "mlx", "jev"} <= names
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key-000")
    assert isinstance(factory(ModelConfig(backend="jev", model="jev-1.13")), JevBackend)


def test_core_imports_no_optional_engine():
    code = (
        "import json, sys\n"
        "import sdgf.cli, sdgf.pipeline\n"
        f"print(json.dumps([m for m in {list(OPTIONAL)!r} if m in sys.modules]))\n"
    )
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == []
