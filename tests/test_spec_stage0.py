import json
from pathlib import Path

import pytest

from sdgf.spec.compile import Stage0Error, compile_spec, stage0_problems
from test_spec_compile import HOOKS, SEEDS, TASK_YAML

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


def write_task(tmp_path, *, yaml_text=TASK_YAML, seeds=SEEDS):
    (tmp_path / "task.yaml").write_text(yaml_text, encoding="utf-8")
    (tmp_path / "hooks.py").write_text(HOOKS, encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text(seeds, encoding="utf-8")
    return tmp_path


def seed_line(**fields):
    return json.dumps({"id": "seed-x", "label": True, **fields}) + "\n"


def problems_of(tmp_path, **kwargs):
    with pytest.raises(Stage0Error) as exc:
        compile_spec(tmp_path, **kwargs)
    return exc.value.problems


def test_clean_spec_compiles(tmp_path):
    assert compile_spec(write_task(tmp_path)).name == "toy"


def test_fag_spec_passes_stage0():
    assert len(compile_spec(FAG_DIR).seeds) == 6


@pytest.mark.parametrize(
    "text, rule",
    [
        ("My TFN is 000 000 000.", "tfn"),
        ("Email me at jane@example.test please.", "email"),
        ("Our ABN is 00 000 000 000.", "abn"),
    ],
)
def test_seed_with_pii_is_rejected(tmp_path, text, rule):
    write_task(tmp_path, seeds=SEEDS + seed_line(messages=[{"content": text}]))
    problems = problems_of(tmp_path)
    assert len(problems) == 1
    assert "seed 2 (seed-x)" in problems[0]
    assert f"pii scan: rule {rule} at messages[0].content" in problems[0]


def test_seed_with_toxicity_is_rejected(tmp_path):
    write_task(tmp_path, seeds=SEEDS + seed_line(note="You are an idiot."))
    (problem,) = problems_of(tmp_path)
    assert "toxicity scan" in problem and "at note" in problem


def test_problems_never_repeat_the_matched_text(tmp_path):
    write_task(tmp_path, seeds=SEEDS + seed_line(note="TFN 000 000 000"))
    (problem,) = problems_of(tmp_path)
    assert "000 000 000" not in problem


def test_every_bad_seed_is_reported(tmp_path):
    bad = seed_line(note="TFN 000 000 000") + seed_line(note="You are an idiot.")
    write_task(tmp_path, seeds=SEEDS + bad)
    problems = problems_of(tmp_path)
    assert [p.split(" fails")[0] for p in problems] == [
        "seeds: seed 2 (seed-x)",
        "seeds: seed 3 (seed-x)",
    ]


def test_private_seed_keys_are_not_scanned(tmp_path):
    write_task(tmp_path, seeds=SEEDS + seed_line(_note="TFN 000 000 000"))
    compile_spec(tmp_path)


def test_listed_tool_must_be_registered(tmp_path):
    yaml_text = TASK_YAML + "tools:\n  - name: catalogue_lookup\n  - name: calculator\n"
    write_task(tmp_path, yaml_text=yaml_text)
    assert problems_of(tmp_path) == [
        "tools: 'catalogue_lookup' is not in the tool registry",
        "tools: 'calculator' is not in the tool registry",
    ]
    assert problems_of(tmp_path, tool_registry={"calculator"}) == [
        "tools: 'catalogue_lookup' is not in the tool registry"
    ]
    c = compile_spec(tmp_path, tool_registry={"calculator", "catalogue_lookup"})
    assert [t.name for t in c.spec.tools] == ["catalogue_lookup", "calculator"]


def test_unset_threshold_is_rejected(tmp_path):
    write_task(tmp_path, yaml_text=TASK_YAML.replace("  overlap_max: 0.8\n", ""))
    assert problems_of(tmp_path) == ["thresholds.overlap_max: release threshold is not set"]


def test_all_unset_thresholds_are_listed(tmp_path):
    write_task(tmp_path, yaml_text=TASK_YAML.split("thresholds:")[0] + "thresholds: {}\n")
    problems = problems_of(tmp_path)
    assert len(problems) == 10
    assert all(p.startswith("thresholds.") for p in problems)


def test_all_gates_report_together(tmp_path):
    yaml_text = TASK_YAML.replace("  kappa_min: 0.8\n", "") + "tools:\n  - name: calculator\n"
    write_task(tmp_path, yaml_text=yaml_text, seeds=SEEDS + seed_line(note="TFN 000 000 000"))
    problems = problems_of(tmp_path)
    assert [p.split(":")[0] for p in problems] == ["seeds", "tools", "thresholds.kappa_min"]
    with pytest.raises(Stage0Error, match="stage 0 rejected the spec"):
        compile_spec(tmp_path)


def test_seed_gate_uses_task_governance_patterns(tmp_path):
    yaml_text = TASK_YAML + ("governance:\n  extra_pii_patterns:\n    member_id: 'MEM-\\d{4}'\n")
    write_task(tmp_path, yaml_text=yaml_text, seeds=SEEDS + seed_line(note="ref MEM-0000"))
    (problem,) = problems_of(tmp_path)
    assert "rule member_id" in problem


def test_stage0_problems_on_parsed_spec():
    c = compile_spec(FAG_DIR)
    assert stage0_problems(c.spec, c.seeds) == []
    assert stage0_problems(c.spec, [{"note": "TFN 000 000 000"}])[0].startswith("seeds: seed 0 ")
