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


def with_l6(*, k=5, temperatures=None):
    lines = ["  layers: [L1, L2, L6]\n", f"  consistency_k: {k}\n"]
    if temperatures is not None:
        lines.append(f"  consistency:\n    temperatures: {temperatures}\n")
    return TASK_YAML.replace("  layers: [L1, L2]\n", "".join(lines))


@pytest.mark.parametrize("temperatures", ["[0.0]", "[0, 0.0, 0]"])
def test_l6_with_only_temperature_zero_votes_is_rejected(tmp_path, temperatures):
    write_task(tmp_path, yaml_text=with_l6(temperatures=temperatures))
    (problem,) = problems_of(tmp_path)
    assert problem.startswith("validation.consistency.temperatures: ")
    assert "5 votes" in problem and "consistency_k: 1" in problem


@pytest.mark.parametrize(
    "yaml_kwargs",
    [
        {},  # default temperatures 0.7, 0.8, 0.9
        {"temperatures": "[0.0, 0.7]"},  # one varied temperature is enough
        {"k": 1, "temperatures": "[0.0]"},  # a single vote repeats nothing
    ],
)
def test_l6_with_vote_diversity_or_one_vote_compiles(tmp_path, yaml_kwargs):
    write_task(tmp_path, yaml_text=with_l6(**yaml_kwargs))
    assert "L6" in compile_spec(tmp_path).spec.validation.layers


def test_temperature_zero_is_fine_when_l6_is_off(tmp_path):
    yaml_text = TASK_YAML.replace(
        "  layers: [L1, L2]\n",
        "  layers: [L1, L2]\n  consistency:\n    temperatures: [0.0]\n",
    )
    write_task(tmp_path, yaml_text=yaml_text)
    compile_spec(tmp_path)


# ── rubric.examples ──────────────────────────────────────────────


def with_examples(*examples):
    rubric = "rubric:\n  verdict:\n    values: [pass, fail]\n"
    block = "  examples:\n" + "".join(
        f"    - {json.dumps({'verdict': 'pass', **ex})}\n" for ex in examples
    )
    return TASK_YAML.replace(rubric, rubric + block)


def test_clean_examples_compile(tmp_path):
    ex = {"record": {"messages": [{"role": "customer", "content": "Hi"}]}, "note": "Fine."}
    write_task(tmp_path, yaml_text=with_examples(ex))
    (example,) = compile_spec(tmp_path).spec.rubric.examples
    assert example.verdict == "pass" and example.note == "Fine."


@pytest.mark.parametrize(
    "example, where",
    [
        ({"record": {"messages": [{"content": "My TFN is 000 000 000."}]}}, "record.messages"),
        ({"record": {"messages": []}, "note": "Email jane@example.test"}, "note"),
    ],
)
def test_example_with_pii_is_rejected(tmp_path, example, where):
    write_task(tmp_path, yaml_text=with_examples(example))
    (problem,) = problems_of(tmp_path)
    assert problem.startswith("rubric.examples[0] fails pii scan: rule ")
    assert f"at {where}" in problem
    assert "000 000 000" not in problem and "jane@" not in problem


def test_example_with_toxicity_is_rejected(tmp_path):
    write_task(tmp_path, yaml_text=with_examples({"record": {"messages": []}, "note": "idiot"}))
    (problem,) = problems_of(tmp_path)
    assert problem.startswith("rubric.examples[0] fails toxicity scan")


def test_example_may_not_show_the_label_or_spans(tmp_path):
    ex = {"record": {"messages": [], "label": True, "spans": []}}
    write_task(tmp_path, yaml_text=with_examples({"record": {"messages": []}}, ex))
    (problem,) = problems_of(tmp_path)
    assert problem.startswith("rubric.examples[1]: record has fields the judge may not see")
    assert "['label', 'spans']" in problem and "allowed: ['messages']" in problem


TURNS_YAML = """\
output_schema:
  turns:
    roles: [customer, assistant]
    first_role: customer
"""


def test_classification_spans_spec_without_turns_is_rejected(tmp_path):
    yaml_text = TASK_YAML.replace(TURNS_YAML, "output_schema: {}\n")
    (problem,) = problems_of(write_task(tmp_path, yaml_text=yaml_text))
    assert problem.startswith("output_schema.turns:")
    assert "classification_spans" in problem
