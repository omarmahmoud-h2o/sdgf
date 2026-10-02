"""The FAG task.yaml loads, compiles and matches the domain data in scripts/config.py."""

import sys
from pathlib import Path

import jsonschema
import pytest

from sdgf.models.mock import MockBackend
from sdgf.models.registry import build_models
from sdgf.spec.compile import compile_spec
from sdgf.tasktypes.registry import REGISTRY

SDGF_DIR = Path(__file__).resolve().parents[1]
FAG_DIR = SDGF_DIR / "tasks" / "fag"
SCRIPTS_DIR = SDGF_DIR.parent / "scripts"


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


@pytest.fixture(scope="module")
def config():
    if not SCRIPTS_DIR.is_dir():
        pytest.skip("needs the original ../scripts/ FAG generator, which is not checked out")
    # scripts/ is read-only here: import without writing bytecode into it.
    sys.path.insert(0, str(SCRIPTS_DIR))
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        import config as fag_config
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.dont_write_bytecode = dont_write
    return fag_config


def params(fag):
    return fag.spec.coverage.params


def axis(fag, name):
    return next(a for a in fag.spec.coverage.axes if a.name == name)


def test_fag_spec_compiles(fag):
    assert fag.name == "fag"
    assert len(fag.spec_version) == 64
    assert fag.spec.task.type == "classification_spans"
    assert fag.spec.task.generation_mode == "label_first"
    assert fag.spec.thresholds.unset() == []
    assert fag.spec.thresholds.governance_violations_max == 0


def test_fag_task_type_resolves(fag):
    tt = REGISTRY.resolve(fag.spec.task)
    assert tt.name == "classification_spans"


def test_fag_breach_rate_is_balanced(fag, config):
    assert config.BREACH_RATE == 0.5
    label = axis(fag, "label")
    assert dict(zip(label.values, label.weights)) == {True: 0.5, False: 0.5}
    assert fag.spec.coverage.balance["label"] == {"true": 0.5, "false": 0.5}


def test_fag_scope_and_length_weights_match_config(fag, config):
    scope = axis(fag, "product_scope")
    assert dict(zip(scope.values, scope.weights)) == config.PRODUCT_SCOPE_WEIGHTS
    length = axis(fag, "conversation_length")
    assert dict(zip(length.values, length.weights)) == config.CONVERSATION_LENGTH_WEIGHTS
    assert params(fag)["conversation_length_buckets"] == config.CONVERSATION_LENGTH_BUCKETS


def test_fag_domain_lists_match_config(fag, config):
    p = params(fag)
    assert p["corps_act_product_topics"] == config.CORPS_ACT_PRODUCT_TOPICS
    assert p["non_corps_act_topics"] == config.NON_CORPS_ACT_TOPICS
    assert p["industries"] == config.INDUSTRIES
    assert p["business_types"] == config.BUSINESS_TYPES
    assert p["jurisdictions"] == config.JURISDICTIONS
    assert p["customer_stances"] == config.CUSTOMER_STANCES
    assert set(p["advice_seeking_stances"]) == config.ADVICE_SEEKING_STANCES
    assert p["production_policy_categories"] == config.PRODUCTION_POLICY_CATEGORIES


def test_fag_tiers_and_signals_match_config(fag, config):
    p = params(fag)
    assert p["advice_tier_descriptions"] == config.ADVICE_TIER_DESCRIPTIONS
    assert p["signal_descriptions"] == config.SIGNAL_DESCRIPTIONS
    assert len(p["signal_descriptions"]) == 15
    assert p["general_advice_signals"] == config.GENERAL_ADVICE_SIGNALS
    assert p["personal_advice_signals"] == config.PERSONAL_ADVICE_SIGNALS
    assert p["domain_advice_signals"] == config.DOMAIN_ADVICE_SIGNALS
    tier_enum = fag.spec.output_schema.fields["advice_tier"].enum
    assert tier_enum == list(config.ADVICE_TIER_DESCRIPTIONS)


def test_fag_signal_groups_partition_all_signals(fag):
    p = params(fag)
    groups = p["general_advice_signals"] + p["personal_advice_signals"] + p["domain_advice_signals"]
    assert len(groups) == len(set(groups))
    assert set(groups) == set(p["signal_descriptions"])


def test_fag_rates_and_weights_match_config(fag, config):
    p = params(fag)
    assert p["domain_signal_rate"] == config.DOMAIN_SIGNAL_RATE
    assert p["contestable_rate"] == config.CONTESTABLE_RATE
    assert p["no_denial_rate"] == config.NO_DENIAL_RATE
    assert p["severity_weights_general_advice"] == config.SEVERITY_WEIGHTS_GENERAL_ADVICE
    assert p["severity_weights_personal_advice"] == config.SEVERITY_WEIGHTS_PERSONAL_ADVICE
    assert p["difficulty_weights_breach"] == config.DIFFICULTY_WEIGHTS_BREACH
    assert p["difficulty_weights_no_breach"] == config.DIFFICULTY_WEIGHTS_NO_BREACH
    assert {int(k): v for k, v in p["secondary_topic_count_weights"].items()} == (
        config.SECONDARY_TOPIC_COUNT_WEIGHTS
    )


def test_fag_has_no_compliance_status(fag):
    assert "compliance_status" not in fag.spec.output_schema.fields


def _record(**overrides):
    record = {
        "messages": [
            {"turn": 1, "role": "customer", "content": "What are the fees on the Neo card?"},
            {"turn": 2, "role": "assistant", "content": "The fictional monthly fee is $0."},
        ],
        "label": False,
        "spans": [],
        "conversation_type": "single_turn",
        "turn_count": 2,
        "industry": "retail",
        "business_type": "private_company",
        "jurisdiction": "Australia",
        "product_scope": "corps_act",
        "is_corps_question": False,
        "denial_present": False,
        "primary_topic": "neo_business_card",
        "secondary_topics": [],
        "customer_stance": "legitimate_info_seeker",
        "advice_tier": "FACTUAL_INFORMATION",
        "signal_categories": [],
        "severity": None,
        "difficulty": "EASY",
        "contestable": False,
        "customer_intent": "Find out the card's fees.",
        "problematic_turns": [],
        "reasoning_summary": "Tier 1 factual information only.",
        "expected_ai_behavior": "State the fee without opinion.",
    }
    record.update(overrides)
    return record


def test_fag_output_schema_accepts_record(fag):
    schema = REGISTRY.resolve(fag.spec.task).output_schema(fag.spec.output_schema)
    jsonschema.validate(_record(), schema)
    jsonschema.validate(_record(label=True, severity="HIGH"), schema)


@pytest.mark.parametrize(
    "overrides",
    [{"severity": "LOW"}, {"advice_tier": "OPINION"}, {"product_scope": "other"}],
)
def test_fag_output_schema_rejects_bad_enum(fag, overrides):
    schema = REGISTRY.resolve(fag.spec.task).output_schema(fag.spec.output_schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_record(**overrides), schema)


def test_fag_models_build_with_mock_overrides(fag):
    overrides = {"generator": MockBackend(["{}"]), "judge": MockBackend(["{}"])}
    models = build_models(fag.spec.models, overrides=overrides)
    assert set(models) == {"generator", "judge"}
    assert models.backend("generator") is overrides["generator"]
    assert fag.spec.models.generator.hosting == "local"
    assert fag.spec.models.judge.hosting == "local"
