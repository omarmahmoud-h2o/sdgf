import random
import re

import pytest

from sdgf.coverage.keywords import (
    ExpansionSettings,
    KeywordError,
    KeywordExpander,
    KeywordParseError,
    declared_seed_keywords,
    extraction_prompt,
    normalise_keyword,
    parse_keywords,
    seed_excerpt,
)
from sdgf.models.base import ModelResponse
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec

DESCRIPTION = "Write exam questions about fictional small-business finance."

TASK_YAML = """\
task:
  name: toykw
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Write exam questions about fictional small-business finance.
output_schema: {}
rubric:
  verdict:
    values: [pass, fail]
seeds:
  path: seeds.jsonl
  uses: [few_shot, keyword_seeding]
coverage:
  target_size: 10
  axes:
    - name: keyword
      source: keyword_expansion
    - name: label
      values: [true, false]
  params:
    keyword_expansion:
      initial_count: 3
      iterations: 1
      per_iteration: 2
      seed_keywords: [Cash Flow]
models:
  generator:
    backend: mock
    model: mock-1
  expansion:
    backend: mock
    model: mock-exp
    temperature: 0.3
    max_tokens: 111
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

SEEDS = (
    '{"id": "s1", "label": true, "keywords": ["invoice terms"], "messages": '
    '[{"turn": 1, "role": "customer", "content": "How do I read an   aged debtors report?"}]}\n'
    '{"id": "s2", "label": false, "keywords": "overdraft, gst"}\n'
)


def existing_in(prompt: str) -> list[str]:
    line = re.search(r"^Existing Keywords: (.*)$", prompt, re.M).group(1)
    return [] if line == "(none yet)" else line.split(", ")


# ── parsing ──────────────────────────────────────────────────────


def test_parse_keywords_normalises_and_dedups():
    assert parse_keywords("Keywords: Cash Flow, working-capital,  cash_flow , GST!,") == [
        "cash_flow",
        "working_capital",
        "gst",
    ]


def test_parse_keywords_accepts_newlines_and_empty():
    assert parse_keywords("liquidity\ncollateral") == ["liquidity", "collateral"]
    assert parse_keywords("") == []
    assert parse_keywords(None) == []
    assert parse_keywords(" , ; !") == []


def test_normalise_keyword():
    assert normalise_keyword("  Interest-Rate Swap ") == "interest_rate_swap"


# ── settings ─────────────────────────────────────────────────────


def test_settings_from_params_and_errors():
    s = ExpansionSettings.from_params({"iterations": 3, "directions": ["advanced"]})
    assert s.iterations == 3 and s.directions == ("advanced",)
    assert ExpansionSettings.from_params(None) == ExpansionSettings()
    with pytest.raises(KeywordError, match="unknown"):
        ExpansionSettings.from_params({"iteratons": 3})
    with pytest.raises(KeywordError, match="directions"):
        ExpansionSettings.from_params({"directions": ["sideways"]})
    with pytest.raises(KeywordError, match="per_iteration"):
        ExpansionSettings(per_iteration=0)


# ── seeds ────────────────────────────────────────────────────────


def test_declared_seed_keywords_from_settings_and_seeds():
    seeds = [{"keywords": ["Invoice Terms"]}, {"keywords": "overdraft, gst"}, {"x": 1}]
    got = declared_seed_keywords(seeds, ExpansionSettings(seed_keywords=("cash flow", "GST")))
    assert got == ["cash_flow", "gst", "invoice_terms", "overdraft"]


def test_seed_excerpt_prefers_messages_and_skips_private():
    assert seed_excerpt({"messages": [{"content": "a  b"}, {"content": "c"}]}, 100) == "a b c"
    assert seed_excerpt({"question": "What is GST?", "_provenance": "secret"}, 100) == (
        "What is GST?"
    )
    assert seed_excerpt({"question": "abcdef"}, 3) == "abc"


# ── expansion ────────────────────────────────────────────────────


def scripted(*replies):
    return MockBackend(list(replies))


def test_run_seeds_then_expands_both_directions_with_dedup():
    backend = scripted(
        "cash_flow, liquidity, collateral",  # initial
        "arithmetic, liquidity, percentages",  # prerequisite: liquidity is a dup
        "derivatives, collateral, hedging",  # advanced: collateral is a dup
    )
    settings = ExpansionSettings(initial_count=3, iterations=1, per_iteration=3)
    result = KeywordExpander(backend, DESCRIPTION, settings).run(random.Random(0))
    assert result.initial == ["cash_flow", "liquidity", "collateral"]
    assert result.keywords == [
        "cash_flow",
        "liquidity",
        "collateral",
        "arithmetic",
        "percentages",
        "derivatives",
        "hedging",
    ]
    it = result.history[1]
    assert it["added"] == {
        "prerequisite": ["arithmetic", "percentages"],
        "advanced": ["derivatives", "hedging"],
    }
    assert it["starting_count"] == 3 and it["ending_count"] == 7
    assert result.to_dict()["keywords"] == result.keywords


def test_prompts_carry_the_current_keyword_list():
    """§12.1 fix: the model sees every keyword found so far, not an empty list."""
    backend = scripted("a1, a2", "b1, b2", "c1, c2", "d1", "e1")
    settings = ExpansionSettings(initial_count=2, iterations=2, per_iteration=2, sample_size=1)
    KeywordExpander(backend, DESCRIPTION, settings).run(random.Random(1))
    prompts = [c.prompt for c in backend.calls]
    assert existing_in(prompts[0]) == []
    assert existing_in(prompts[1]) == ["a1", "a2"]
    assert existing_in(prompts[2]) == ["a1", "a2", "b1", "b2"]  # advanced sees the ↓ adds
    assert existing_in(prompts[3]) == ["a1", "a2", "b1", "b2", "c1", "c2"]
    assert existing_in(prompts[4]) == ["a1", "a2", "b1", "b2", "c1", "c2", "d1"]
    assert "BEFORE studying" in prompts[1] and "BUILD UPON" in prompts[2]
    assert all(DESCRIPTION in p for p in prompts)


def test_sample_is_shown_and_deterministic_per_seed():
    def run(seed):
        backend = scripted("k1, k2, k3, k4, k5", "p1", "a1")
        settings = ExpansionSettings(initial_count=5, iterations=1, per_iteration=1, sample_size=2)
        result = KeywordExpander(backend, DESCRIPTION, settings).run(random.Random(seed))
        return result.history[1]["sample"], backend.calls[1].prompt

    sample, prompt = run(7)
    assert len(sample) == 2 and f"Sample Keywords: {', '.join(sample)}" in prompt
    assert run(7)[0] == sample


@pytest.mark.parametrize("bad_at", [0, 1, 2])
@pytest.mark.parametrize("bad", ["", None, " , ,"])
def test_unparseable_reply_raises_instead_of_mock_fallback(bad_at, bad):
    replies = ["a1, a2", "b1", "c1"]
    replies[bad_at] = ModelResponse(text=bad) if bad is None else bad
    settings = ExpansionSettings(initial_count=2, iterations=1, per_iteration=1)
    expander = KeywordExpander(scripted(*replies), DESCRIPTION, settings)
    with pytest.raises(KeywordParseError) as err:
        expander.run(random.Random(0))
    assert err.value.stage == ["initial", "prerequisite", "advanced"][bad_at]


def test_all_duplicates_is_not_an_error():
    backend = scripted("a1, a2", "a1", "a2")
    settings = ExpansionSettings(initial_count=2, iterations=1, per_iteration=1)
    result = KeywordExpander(backend, DESCRIPTION, settings).run(random.Random(0))
    assert result.keywords == ["a1", "a2"]
    assert result.history[1]["added"] == {"prerequisite": [], "advanced": []}


def test_max_keywords_caps_and_stops_calling():
    backend = scripted("a1, a2", "b1, b2, b3", "c1")
    settings = ExpansionSettings(initial_count=2, iterations=5, per_iteration=3, max_keywords=4)
    result = KeywordExpander(backend, DESCRIPTION, settings).run(random.Random(0))
    assert result.keywords == ["a1", "a2", "b1", "b2"]
    assert len(backend.calls) == 2


def test_single_direction():
    backend = scripted("a1", "c1")
    settings = ExpansionSettings(
        initial_count=1, iterations=1, per_iteration=1, directions=("advanced",)
    )
    result = KeywordExpander(backend, DESCRIPTION, settings).run(random.Random(0))
    assert result.keywords == ["a1", "c1"] and len(backend.calls) == 2


def test_empty_description_rejected():
    with pytest.raises(KeywordError):
        KeywordExpander(scripted("a"), "  ")


def test_extraction_prompt_uses_found_keywords():
    p = extraction_prompt(DESCRIPTION, "Passage about GST.", ["gst", "bas"], ExpansionSettings())
    assert "Current Keywords: gst, bas" in p and "Passage about GST." in p


# ── from_spec ────────────────────────────────────────────────────


@pytest.fixture
def toy(tmp_path):
    (tmp_path / "task.yaml").write_text(TASK_YAML, encoding="utf-8")
    (tmp_path / "seeds.jsonl").write_text(SEEDS, encoding="utf-8")
    return compile_spec(tmp_path)


def test_from_spec_uses_params_expansion_model_and_seeds(toy):
    backend = scripted("liquidity, cash_flow, collateral", "arithmetic, ratios", "hedging")
    expander = KeywordExpander.from_spec(toy, backend)
    assert expander.settings.initial_count == 3 and expander.use_seed_excerpts
    result = expander.run(random.Random(0))
    declared = ["cash_flow", "invoice_terms", "overdraft", "gst"]
    assert result.history[0]["declared"] == declared
    assert result.initial == [*declared, "liquidity", "collateral"]
    assert result.keywords[-3:] == ["arithmetic", "ratios", "hedging"]
    first = backend.calls[0]
    assert first.max_tokens == 111 and first.temperature == 0.3
    assert existing_in(first.prompt) == declared
    assert "- How do I read an aged debtors report?" in first.prompt
    assert DESCRIPTION in first.prompt


def test_from_spec_without_keyword_seeding_shows_no_excerpts(toy):
    backend = scripted("a1", "b1", "c1")
    expander = KeywordExpander.from_spec(toy, backend, use_seed_excerpts=False)
    expander.run(random.Random(0))
    assert "Example records" not in backend.calls[0].prompt
