"""The L5 and L6 sections of docs/how-it-works.md, the README reference and the
new-use-case template document the judge-independence keys (rubric.judge_context,
rubric.examples, validation.consistency, models.consistency_judge)."""

from pathlib import Path

import yaml

from sdgf.spec.schema import ConsistencyRules, RubricExample, RubricSection, ValidationSection

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
HOW = (ROOT / "docs" / "how-it-works.md").read_text(encoding="utf-8")
NEW = (ROOT / "docs" / "new-use-case.md").read_text(encoding="utf-8")


def between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    return text[i : text.index(end, i + 1)]


L5 = between(HOW, "### L5 ", "### L6 ")
L6 = between(HOW, "### L6 ", "### Sent back and dropped")


def test_l5_section_names_the_judge_only_rubric_keys():
    for key in ("rubric.judge_context", "task.description", "rubric.examples"):
        assert f"`{key}`" in L5, key
    assert "generation prompt" in L5


def test_l6_section_names_the_vote_diversity_keys_and_ballots():
    for key in (
        "validation.consistency.temperatures[i % len]",
        "models.consistency_judge",
        "models.judge",
        "consistency_k",
        "ballots",
        "layer_results[].ballots",
    ):
        assert f"`{key}`" in L6, key
    assert "stage 0 rejects" in L6


def test_the_default_temperatures_in_the_docs_match_the_schema():
    default = " / ".join(str(t) for t in ConsistencyRules().temperatures)
    assert f"(default {default})" in L6


def test_readme_documents_every_nested_key():
    rubric = between(README, "### `rubric`", "### `seeds`")
    for key in RubricExample.model_fields:
        assert f"`{key}`" in rubric, key
    validation = between(README, "### `validation`", "### `thresholds`")
    for key in ConsistencyRules.model_fields:
        assert f"`{key}`" in validation, key
    assert "`consistency_judge`" in between(README, "### `models`", "### `validation`")


def test_the_new_use_case_template_loads_the_new_keys():
    block = between(NEW, "```yaml\ntask:", "\n```").split("\n", 1)[1]
    data = yaml.safe_load(block)
    validation = ValidationSection(**data["validation"])
    assert validation.consistency.temperatures == ConsistencyRules().temperatures
    RubricSection(**data["rubric"])
    assert "# judge_context:" in block and "# examples:" in block
