import textwrap

import pytest

from sdgf.spec.compile import CompiledSpec, compile_spec
from sdgf.spec.loader import SpecLoadError, load_task
from sdgf.spec.schema import SpecValidationError

TASK_YAML = """\
# toy task for compile tests
task:
  name: toy
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Toy task for compile tests.
output_schema:
  turns:
    roles: [customer, assistant]
    first_role: customer
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
validation:
  layers: [L1, L2]
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

HOOKS = """\
def label_rule(record):
    return record["tier"] >= 2
"""

SEEDS = (
    '{"id": "seed-1", "customer": "Acme Test Pty Ltd", "label": true}\n'
    "\n"
    '{"id": "seed-2", "customer": "Acme Test Pty Ltd", "label": false}\n'
)


@pytest.fixture
def task_dir(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
    (tmp_path / "hooks.py").write_text(HOOKS, encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text(SEEDS, encoding="utf-8")
    return tmp_path


def test_compile_loads_all_three_parts(task_dir):
    c = compile_spec(task_dir)
    assert isinstance(c, CompiledSpec)
    assert c.name == "toy"
    assert c.hooks.present() == ["label_rule"]
    assert [s["id"] for s in c.seeds] == ["seed-1", "seed-2"]
    assert len(c.spec_version) == 64
    assert c.task_dir == task_dir


def test_compile_accepts_task_yaml_path(task_dir):
    assert compile_spec(task_dir / "task.yaml").spec_version == compile_spec(task_dir).spec_version


def test_compiled_spec_is_frozen(task_dir):
    c = compile_spec(task_dir)
    with pytest.raises(Exception):
        c.spec_version = "x"
    with pytest.raises(Exception):
        c.spec.task.name = "other"


def test_spec_version_is_stable(task_dir):
    assert compile_spec(task_dir).spec_version == compile_spec(task_dir).spec_version


def test_spec_version_ignores_yaml_comments_and_formatting(task_dir):
    before = compile_spec(task_dir).spec_version
    (task_dir / "task.yaml").write_text("# another comment\n" + TASK_YAML, encoding="utf-8")
    assert compile_spec(task_dir).spec_version == before


def test_changing_spec_changes_version(task_dir):
    before = compile_spec(task_dir).spec_version
    (task_dir / "task.yaml").write_text(
        TASK_YAML.replace("target_size: 10", "target_size: 11"), encoding="utf-8"
    )
    assert compile_spec(task_dir).spec_version != before


def test_changing_hooks_changes_version(task_dir):
    before = compile_spec(task_dir).spec_version
    (task_dir / "hooks.py").write_text(HOOKS.replace(">= 2", ">= 3"), encoding="utf-8")
    assert compile_spec(task_dir).spec_version != before


def test_changing_seeds_changes_version(task_dir):
    before = compile_spec(task_dir).spec_version
    (task_dir / "seeds.jsonl").write_text(SEEDS.replace("seed-2", "seed-3"), encoding="utf-8")
    assert compile_spec(task_dir).spec_version != before


def test_removing_hooks_changes_version(task_dir):
    before = compile_spec(task_dir).spec_version
    (task_dir / "hooks.py").unlink()
    c = compile_spec(task_dir)
    assert c.hooks.present() == []
    assert c.spec_version != before


def test_seeds_path_is_relative_to_task_dir(task_dir):
    (task_dir / "data").mkdir()
    (task_dir / "seeds.jsonl").rename(task_dir / "data" / "seeds.jsonl")
    (task_dir / "task.yaml").write_text(
        TASK_YAML.replace("path: seeds.jsonl", "path: data/seeds.jsonl"), encoding="utf-8"
    )
    assert len(load_task(task_dir).seeds) == 2


def test_missing_task_yaml(tmp_path):
    with pytest.raises(SpecLoadError, match="task spec not found"):
        compile_spec(tmp_path)


def test_invalid_yaml(task_dir):
    (task_dir / "task.yaml").write_text("task: [unclosed\n", encoding="utf-8")
    with pytest.raises(SpecLoadError, match="invalid YAML"):
        compile_spec(task_dir)


def test_schema_errors_propagate(task_dir):
    (task_dir / "task.yaml").write_text(TASK_YAML.split("thresholds:")[0], encoding="utf-8")
    with pytest.raises(SpecValidationError) as exc:
        compile_spec(task_dir)
    assert "thresholds" in [path for path, _ in exc.value.errors]


def test_missing_seeds_file(task_dir):
    (task_dir / "seeds.jsonl").unlink()
    with pytest.raises(SpecLoadError, match="seeds.path"):
        compile_spec(task_dir)


def test_bad_seed_line_names_line_number(task_dir):
    (task_dir / "seeds.jsonl").write_text(SEEDS + "{not json}\n", encoding="utf-8")
    with pytest.raises(SpecLoadError, match=r"seeds\.jsonl:4: invalid JSON"):
        compile_spec(task_dir)


def test_non_object_seed_rejected(task_dir):
    (task_dir / "seeds.jsonl").write_text('["a", "b"]\n', encoding="utf-8")
    with pytest.raises(SpecLoadError, match="JSON object"):
        compile_spec(task_dir)


def test_hook_errors_propagate(task_dir):
    from sdgf.spec.hooks import HookError

    (task_dir / "hooks.py").write_text(
        textwrap.dedent("""
        def label_rule(record, extra):
            return True
        """),
        encoding="utf-8",
    )
    with pytest.raises(HookError, match="label_rule"):
        compile_spec(task_dir)
