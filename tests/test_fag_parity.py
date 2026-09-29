"""sdgf/tasks/fag/hooks.py agrees with scripts/policy_categories.py and scripts/scenario_sampler.py."""

import itertools
import random
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.spec.hooks import load_hooks

SDGF_DIR = Path(__file__).resolve().parents[1]
FAG_DIR = SDGF_DIR / "tasks" / "fag"
SCRIPTS_DIR = SDGF_DIR.parent / "scripts"

TIERS = ["FACTUAL_INFORMATION", "GENERAL_ADVICE", "PERSONAL_ADVICE"]
SCOPES = ["corps_act", "non_corps_act"]


def _import_scripts(*names):
    # scripts/ is read-only here: import without writing bytecode into it.
    sys.path.insert(0, str(SCRIPTS_DIR))
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return [__import__(n) for n in names]
    finally:
        sys.path.remove(str(SCRIPTS_DIR))
        sys.dont_write_bytecode = dont_write


@pytest.fixture(scope="module")
def orig():
    config, policy, sampler = _import_scripts("config", "policy_categories", "scenario_sampler")
    return config, policy, sampler


@pytest.fixture(scope="module")
def hooks():
    return load_hooks(FAG_DIR)


@pytest.fixture(scope="module")
def mod(hooks):
    # the hooks module's namespace, for its non-hook helpers
    return types.SimpleNamespace(**hooks.label_rule.__globals__)


def _signal_sets(all_signals):
    """Every subset of size <= 3, plus the full set."""
    for k in range(4):
        yield from itertools.combinations(all_signals, k)
    yield tuple(all_signals)


def test_fag_hooks_present(hooks):
    assert set(hooks.present()) == {
        "label_rule",
        "sampler_constraints",
        "extra_validators",
        "post_process",
    }


def test_fag_spec_compiles_with_hooks():
    fag = compile_spec(FAG_DIR)
    assert fag.hooks.label_rule is not None
    assert "expected_breach" in fag.hooks.source


@pytest.mark.parametrize("tier", TIERS)
@pytest.mark.parametrize("scope", SCOPES)
def test_label_rule_matches_expected_breach(hooks, orig, tier, scope):
    _, policy, _ = orig
    assert hooks.label_rule({"advice_tier": tier, "product_scope": scope}) == (
        policy.expected_breach(tier, scope)
    )


def test_label_rule_rejects_unknown_tier(hooks, orig):
    _, policy, _ = orig
    with pytest.raises(ValueError):
        policy.expected_breach("OPINION", "corps_act")
    with pytest.raises(ValueError):
        hooks.label_rule({"advice_tier": "OPINION", "product_scope": "corps_act"})


def test_policy_categories_match_across_all_combinations(hooks, mod, orig):
    config, policy, _ = orig
    n = 0
    for signals in _signal_sets(list(config.SIGNAL_DESCRIPTIONS)):
        for tier, scope, is_corps_q, denial in itertools.product(
            TIERS, SCOPES, [False, True], [False, True]
        ):
            breach = policy.expected_breach(tier, scope)
            facts = {
                "signal_categories": list(signals),
                "product_scope": scope,
                "is_corps_question": is_corps_q,
                "denial_present": denial,
                "advice_tier": tier,
            }
            old = {**facts, "financial_advice_breach": breach}
            new = {**facts, "label": breach}
            want = policy.derive_policy_categories(old)
            assert hooks.post_process(new)["policy_categories"] == want
            assert mod.unexplained_breach(new) == policy.unexplained_breach(old)
            n += 1
    assert n > 10_000


def test_post_process_does_not_mutate_input(hooks):
    record = {"label": False, "signal_categories": []}
    out = hooks.post_process(record)
    assert "policy_categories" not in record
    assert list(out["policy_categories"].values()) == [False] * 9


# ----------------------------------------------------------------- sampler


def _check_recipe(r, config, policy):
    """Invariants every scenario_sampler recipe satisfies; `label` is the breach flag."""
    breach, scope, tier = r["label"], r["product_scope"], r["advice_tier"]
    corps = scope == "corps_act"
    general, personal = set(config.GENERAL_ADVICE_SIGNALS), set(config.PERSONAL_ADVICE_SIGNALS)
    domain = set(config.DOMAIN_ADVICE_SIGNALS)
    signals = r["signal_categories"]

    assert policy.expected_breach(tier, scope) == breach
    assert (r["severity"] is not None) == breach
    if breach:
        weights = (
            config.SEVERITY_WEIGHTS_PERSONAL_ADVICE
            if tier == "PERSONAL_ADVICE"
            else config.SEVERITY_WEIGHTS_GENERAL_ADVICE
        )
        assert r["severity"] in weights
    if corps and not breach:
        assert tier == "FACTUAL_INFORMATION"
    if not corps and breach:
        assert tier == "PERSONAL_ADVICE"

    pool = config.CORPS_ACT_PRODUCT_TOPICS if corps else config.NON_CORPS_ACT_TOPICS
    assert r["primary_topic"] in pool
    assert r["primary_topic"] not in r["secondary_topics"]
    assert len(set(r["secondary_topics"])) == len(r["secondary_topics"]) <= 2
    assert set(r["secondary_topics"]) <= set(config.TOPICS)

    assert r["is_corps_question"] == (
        corps and r["customer_stance"] in config.ADVICE_SEEKING_STANCES
    )
    if not r["is_corps_question"]:
        assert r["denial_present"] is False
    elif not breach:
        assert r["denial_present"] is True

    assert len(signals) == len(set(signals))
    if tier == "FACTUAL_INFORMATION":
        assert signals == []
    else:
        core = [s for s in signals if s not in domain]
        assert len([s for s in signals if s in domain]) <= 1
        assert signals[-1] in domain or not (set(signals) & domain)
        if tier == "GENERAL_ADVICE":
            assert 1 <= len(core) <= 2 and set(core) <= general
        else:
            assert core[0] in personal and 1 <= len(core) <= 2
            assert set(core) <= general | personal

    assert r["conversation_type"] in ("single_turn", "multi_turn")
    if r["conversation_type"] == "single_turn":
        assert r["turn_count"] == 2
    assert r["turn_count"] % 2 == 0
    assert r["industry"] in config.INDUSTRIES
    assert r["business_type"] in config.BUSINESS_TYPES
    assert r["jurisdiction"] in config.JURISDICTIONS
    assert r["customer_stance"] in config.CUSTOMER_STANCES
    weights = config.DIFFICULTY_WEIGHTS_BREACH if breach else config.DIFFICULTY_WEIGHTS_NO_BREACH
    assert r["difficulty"] in weights
    assert isinstance(r["contestable"], bool)


def _cells():
    for scope, label, length in itertools.product(
        SCOPES, [True, False], ["single_turn", "short", "medium", "long"]
    ):
        yield {"product_scope": scope, "label": label, "conversation_length": length}


def test_original_sampler_satisfies_invariants(orig):
    # Validates the invariant checker against the reference implementation.
    config, policy, sampler = orig
    state = random.getstate()
    random.seed(1234)
    try:
        for _ in range(2000):
            r = sampler.sample_scenario()
            r["label"] = r.pop("financial_advice_breach")
            _check_recipe(r, config, policy)
    finally:
        random.setstate(state)


def test_hooks_sampler_satisfies_invariants_in_every_cell(hooks, orig):
    config, policy, _ = orig
    rng = random.Random(7)
    for cell in _cells():
        for _ in range(150):
            r = hooks.sampler_constraints(cell, rng)
            assert r["product_scope"] == cell["product_scope"]
            assert r["label"] is cell["label"]
            assert (
                r["turn_count"] in config.CONVERSATION_LENGTH_BUCKETS[cell["conversation_length"]]
            )
            assert r["conversation_type"] == (
                "single_turn" if cell["conversation_length"] == "single_turn" else "multi_turn"
            )
            _check_recipe(r, config, policy)


def test_hooks_sampler_fills_same_fields_as_original(hooks, orig):
    _, _, sampler = orig
    state = random.getstate()
    random.seed(0)
    try:
        want = set(sampler.sample_scenario()) - {"financial_advice_breach"} | {"label"}
    finally:
        random.setstate(state)
    got = hooks.sampler_constraints(next(_cells()), random.Random(0))
    assert set(got) - {"conversation_length"} == want


def test_hooks_sampler_is_deterministic_per_seed(hooks):
    cell = {"product_scope": "non_corps_act", "label": False, "conversation_length": "medium"}
    a = [hooks.sampler_constraints(cell, random.Random(42)) for _ in range(3)]
    assert a[0] == a[1] == a[2]
    rng1, rng2 = random.Random(5), random.Random(5)
    assert [hooks.sampler_constraints(cell, rng1) for _ in range(20)] == [
        hooks.sampler_constraints(cell, rng2) for _ in range(20)
    ]


def test_hooks_sampler_preserves_extra_cell_keys(hooks):
    cell = {
        "cell_id": "c1",
        "product_scope": "corps_act",
        "label": True,
        "conversation_length": "short",
    }
    assert hooks.sampler_constraints(cell, random.Random(1))["cell_id"] == "c1"
    assert "financial_advice_breach" not in hooks.sampler_constraints(cell, random.Random(1))


def test_hooks_sampler_fills_missing_axes(hooks, orig):
    config, policy, _ = orig
    rng = random.Random(3)
    for _ in range(200):
        _check_recipe(hooks.sampler_constraints({}, rng), config, policy)


def test_tier_distribution_matches_original_weights(hooks, orig):
    config, _, _ = orig
    rng = random.Random(11)
    corps_breach = {"product_scope": "corps_act", "label": True, "conversation_length": "short"}
    counts = Counter(
        hooks.sampler_constraints(corps_breach, rng)["advice_tier"] for _ in range(4000)
    )
    assert abs(counts["GENERAL_ADVICE"] / 4000 - 0.45) < 0.03
    non_corps_ok = {
        "product_scope": "non_corps_act",
        "label": False,
        "conversation_length": "short",
    }
    counts = Counter(
        hooks.sampler_constraints(non_corps_ok, rng)["advice_tier"] for _ in range(4000)
    )
    assert abs(counts["GENERAL_ADVICE"] / 4000 - 0.30) < 0.03
    # permitted general advice keeps its signals: the hard negative
    rec = next(
        r
        for r in (hooks.sampler_constraints(non_corps_ok, rng) for _ in range(100))
        if r["advice_tier"] == "GENERAL_ADVICE"
    )
    assert rec["label"] is False and rec["signal_categories"]


def test_denial_rate_matches_original(hooks, orig):
    config, _, _ = orig
    rng = random.Random(13)
    cell = {
        "product_scope": "corps_act",
        "label": True,
        "conversation_length": "short",
        "customer_stance": "recommendation_seeker",
    }
    denials = [hooks.sampler_constraints(cell, rng)["denial_present"] for _ in range(4000)]
    assert abs(sum(denials) / 4000 - (1 - config.NO_DENIAL_RATE)) < 0.03


@pytest.mark.parametrize(
    "cell",
    [
        # pinned tier contradicts the label
        {"product_scope": "corps_act", "label": False, "advice_tier": "GENERAL_ADVICE"},
        {"product_scope": "non_corps_act", "label": True, "advice_tier": "GENERAL_ADVICE"},
        {"product_scope": "non_corps_act", "label": False, "advice_tier": "PERSONAL_ADVICE"},
        {"product_scope": "corps_act", "label": True, "advice_tier": "OPINION"},
        # topic outside the scope's pool
        {"product_scope": "corps_act", "label": True, "primary_topic": "domestic_transfer"},
        {"product_scope": "elsewhere", "label": True},
        {"product_scope": "corps_act", "label": True, "customer_stance": "hostile"},
        {"product_scope": "corps_act", "label": True, "conversation_length": "epic"},
    ],
)
def test_hooks_sampler_rejects_invalid_cells(hooks, cell):
    assert hooks.sampler_constraints(cell, random.Random(0)) is None


def test_hooks_sampler_accepts_consistent_pinned_tier(hooks):
    cell = {
        "product_scope": "non_corps_act",
        "label": False,
        "advice_tier": "GENERAL_ADVICE",
        "primary_topic": "invoicing",
        "conversation_length": "long",
    }
    r = hooks.sampler_constraints(cell, random.Random(0))
    assert r["advice_tier"] == "GENERAL_ADVICE" and r["primary_topic"] == "invoicing"
    assert r["signal_categories"] and r["severity"] is None
