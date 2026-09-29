import json
import random
import shutil
from pathlib import Path

import pytest

from sdgf.generate.prompts import CELL_HEADER, PromptBuilder, build_cell_section, select_few_shot
from sdgf.generate.scheduler import Cell
from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import prompt_hash

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def builder(fag):
    return PromptBuilder(fag)


def recipe(fag, product_scope, label, seed):
    cell = {"product_scope": product_scope, "label": label, "conversation_length": "short"}
    return fag.hooks.sampler_constraints(cell, random.Random(seed))


def test_two_cells_share_the_prefix_hash(fag, builder):
    a = builder.build(Cell("corps-breach", recipe(fag, "corps_act", True, 1), 5))
    b = builder.build(recipe(fag, "non_corps_act", False, 2))
    assert a.prefix_hash == b.prefix_hash == builder.prefix_hash
    assert a.static == b.static
    assert a.cell != b.cell
    assert a.hash != b.hash


def test_prefix_hash_is_stable_across_builders(fag, builder):
    assert PromptBuilder(fag).prefix_hash == builder.prefix_hash
    assert builder.prefix_hash.startswith("sha256:")


def test_static_content_comes_first_and_cell_content_last(fag, builder):
    params = recipe(fag, "corps_act", True, 3)
    p = builder.build(params)
    assert p.text.startswith(p.static)
    assert p.text.endswith(p.cell)
    assert p.text.index(CELL_HEADER) > p.text.index("## Output format")
    assert p.hash == prompt_hash(p.text)


def test_static_prefix_holds_description_rubric_schema_and_seeds(fag, builder):
    s = builder.static_prefix
    assert fag.spec.task.description.strip() in s
    assert "Verdict values: breach, no_breach." in s
    assert "realism (integer 1..5)" in s
    assert '"problematic_turns"' in s and '"messages"' in s  # full output schema
    assert "starting with customer" in s
    assert "copied exactly" in s
    assert s.count("SEED-FAG-") == fag.spec.seeds.few_shot_count


def test_cell_values_never_leak_into_the_prefix(fag, builder):
    params = recipe(fag, "non_corps_act", False, 4)
    p = builder.build(params)
    assert CELL_HEADER not in p.static
    for key, value in params.items():
        assert f"- {key}: {json.dumps(value, sort_keys=True)}" in p.cell


def test_cell_section_keeps_parameter_order():
    text = build_cell_section({"b": 1, "a": [True, None], "c": "x"})
    assert text.splitlines() == [CELL_HEADER, "- b: 1", "- a: [true, null]", '- c: "x"']


def test_few_shot_interleaves_labels(fag):
    picked = select_few_shot(fag.seeds, 3)
    assert [s["label"] for s in picked] == [True, False, True]
    assert len(select_few_shot(fag.seeds, 10)) == len(fag.seeds)
    assert select_few_shot(fag.seeds, 0) == []


def test_few_shot_without_labels_keeps_file_order():
    seeds = [{"q": i} for i in range(5)]
    assert select_few_shot(seeds, 2) == [{"q": 0}, {"q": 1}]


def test_private_keys_are_stripped_from_examples(fag):
    seed = dict(fag.seeds[0], _provenance={"secret_marker": "x"})
    b = PromptBuilder(fag, few_shot=[seed])
    assert "secret_marker" not in b.static_prefix
    assert seed["conversation_id"] in b.static_prefix


def test_prefix_changes_with_seeds_and_spec(tmp_path, fag, builder):
    task = tmp_path / "fag"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))

    seeds = (task / "seeds.jsonl").read_text().splitlines()
    (task / "seeds.jsonl").write_text("\n".join(reversed(seeds)) + "\n")
    assert PromptBuilder(compile_spec(task)).prefix_hash != builder.prefix_hash

    shutil.copy(FAG_DIR / "seeds.jsonl", task / "seeds.jsonl")
    yaml_path = task / "task.yaml"
    yaml_path.write_text(yaml_path.read_text().replace("few_shot_count: 3", "few_shot_count: 2"))
    assert PromptBuilder(compile_spec(task)).prefix_hash != builder.prefix_hash


def test_no_few_shot_when_seeds_not_used_for_it(fag):
    spec = fag.spec.model_copy(
        update={"seeds": fag.spec.seeds.model_copy(update={"uses": ["gold_set"]})}
    )
    b = PromptBuilder(type(fag)(spec, fag.hooks, fag.seeds, fag.spec_version, fag.task_dir))
    assert "## Examples" not in b.static_prefix
    assert "SEED-FAG-" not in b.static_prefix
