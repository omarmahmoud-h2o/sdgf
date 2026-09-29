from dataclasses import replace
from pathlib import Path

import pytest

from sdgf.governance.profile import (
    GLOBAL_PII_RULES,
    GLOBAL_PROFILE,
    GovernanceProfile,
    GovernanceProfileError,
    LooseningError,
    check_tightens,
    merge_profile,
    profile_for,
)
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import GovernanceException, GovernanceSection

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"

BASE = GovernanceProfile(entity_deny=frozenset({"Real Bank Ltd"}))


def section(**kw) -> GovernanceSection:
    return GovernanceSection(**kw)


def exc(rule: str, reason: str = "documented for tests") -> dict:
    return {"rule": rule, "reason": reason}


# ── global profile ───────────────────────────────────────────────


def test_global_profile_covers_every_scanner():
    assert set(GLOBAL_PII_RULES) <= GLOBAL_PROFILE.pii_rules
    assert {"abn", "tfn", "bsb", "account_number"} <= GLOBAL_PROFILE.pii_rules
    assert GLOBAL_PROFILE.secret_rules and GLOBAL_PROFILE.toxicity_categories
    assert GLOBAL_PROFILE.max_violations == 0
    assert GLOBAL_PROFILE.active_toxicity_categories == GLOBAL_PROFILE.toxicity_categories


def test_profile_rejects_nonzero_violations():
    with pytest.raises(GovernanceProfileError, match="max_violations"):
        GovernanceProfile(max_violations=1)


def test_profile_is_immutable():
    with pytest.raises(TypeError):
        GLOBAL_PROFILE.pii_patterns["x"] = "y"  # type: ignore[index]


def test_empty_section_gives_base_profile():
    assert merge_profile(None) == GLOBAL_PROFILE
    assert merge_profile(section(), BASE) == BASE


def test_fag_profile_is_global():
    fag = compile_spec(FAG_DIR)
    assert profile_for(fag) == GLOBAL_PROFILE


# ── tightening ───────────────────────────────────────────────────


def test_extra_pii_patterns_are_added():
    merged = merge_profile(section(extra_pii_patterns={"member_no": r"\bMBR-\d{6}\b"}), BASE)
    assert merged.pii_patterns == {"member_no": r"\bMBR-\d{6}\b"}
    assert merged.pii_rules == BASE.pii_rules


def test_entity_deny_is_unioned():
    merged = merge_profile(section(entity_deny=["Other Real Co"]), BASE)
    assert merged.entity_deny == {"Real Bank Ltd", "Other Real Co"}


def test_allowing_an_entity_not_denied_needs_no_exception():
    merged = merge_profile(section(entity_allow=["Acme Test Pty Ltd"]), BASE)
    assert "Acme Test Pty Ltd" in merged.entity_allow
    assert merged.entity_deny == BASE.entity_deny


def test_merge_can_chain_and_keeps_earlier_tightening():
    first = merge_profile(section(extra_pii_patterns={"member_no": r"MBR-\d+"}), BASE)
    second = merge_profile(section(entity_deny=["Other Real Co"]), first)
    assert second.pii_patterns == {"member_no": r"MBR-\d+"}
    check_tightens(BASE, second)


def test_to_dict_is_sorted_and_complete():
    d = merge_profile(section(entity_deny=["Zed Co", "Alpha Co"]), BASE).to_dict()
    assert d["entity_deny"] == ["Alpha Co", "Real Bank Ltd", "Zed Co"]
    assert d["max_violations"] == 0 and d["exceptions"] == []


# ── documented exceptions ────────────────────────────────────────


def test_documented_toxicity_exception_applies():
    merged = merge_profile(
        section(toxicity_exceptions=["insult"], exceptions=[exc("toxicity:insult")]), BASE
    )
    assert merged.toxicity_exceptions == {"insult"}
    assert "insult" not in merged.active_toxicity_categories
    assert merged.exceptions[0].rule == "toxicity:insult"
    check_tightens(BASE, merged)


def test_documented_entity_exception_releases_global_deny():
    merged = merge_profile(
        section(entity_allow=["Real Bank Ltd"], exceptions=[exc("entities:Real Bank Ltd")]), BASE
    )
    assert "Real Bank Ltd" not in merged.entity_deny
    assert "Real Bank Ltd" in merged.entity_allow


# ── loosening attempts ───────────────────────────────────────────


def test_undocumented_toxicity_exception_raises():
    with pytest.raises(LooseningError, match="toxicity:insult"):
        merge_profile(section(toxicity_exceptions=["insult"]), BASE)


def test_allowing_globally_denied_entity_without_exception_raises():
    with pytest.raises(LooseningError, match="global deny list"):
        merge_profile(section(entity_allow=["Real Bank Ltd"]), BASE)


@pytest.mark.parametrize("rule", ["tfn", "email", "account_number"])
def test_redefining_global_pii_rule_raises(rule):
    with pytest.raises(LooseningError, match=f"extra_pii_patterns.{rule}"):
        merge_profile(section(extra_pii_patterns={rule: r"NEVER_MATCHES"}), BASE)


def test_redefining_inherited_pii_pattern_raises():
    base = merge_profile(section(extra_pii_patterns={"member_no": r"MBR-\d+"}), BASE)
    with pytest.raises(LooseningError, match="member_no"):
        merge_profile(section(extra_pii_patterns={"member_no": r"NEVER"}), base)
    # restating the same pattern is not a change
    assert merge_profile(section(extra_pii_patterns={"member_no": r"MBR-\d+"}), base) == base


@pytest.mark.parametrize("rule", ["pii:tfn", "secrets:api_key"])
def test_pii_and_secret_rules_can_never_be_excepted(rule):
    with pytest.raises(LooseningError, match="never be excepted"):
        merge_profile(section(exceptions=[exc(rule)]), BASE)


def test_exception_without_matching_relaxation_raises():
    with pytest.raises(GovernanceProfileError, match="document no relaxation"):
        merge_profile(section(exceptions=[exc("toxicity:threat")]), BASE)


def test_malformed_and_unknown_exception_rules_raise():
    with pytest.raises(GovernanceProfileError, match="<scanner>:<name>"):
        merge_profile(section(exceptions=[exc("toxicity")]), BASE)
    with pytest.raises(GovernanceProfileError, match="unknown scanner"):
        merge_profile(section(exceptions=[exc("vibes:bad")]), BASE)


def test_duplicate_exception_raises():
    with pytest.raises(GovernanceProfileError, match="declared twice"):
        merge_profile(
            section(
                toxicity_exceptions=["insult"],
                exceptions=[exc("toxicity:insult"), exc("toxicity:insult", "again")],
            ),
            BASE,
        )


def test_unknown_toxicity_category_raises():
    with pytest.raises(GovernanceProfileError, match="unknown category"):
        merge_profile(
            section(toxicity_exceptions=["spicy"], exceptions=[exc("toxicity:spicy")]), BASE
        )


def test_contradictory_allow_and_deny_raises():
    with pytest.raises(GovernanceProfileError, match="both allowed and denied"):
        merge_profile(section(entity_allow=["X Co"], entity_deny=["X Co"]), BASE)


@pytest.mark.parametrize(
    "loosened",
    [
        {"pii_rules": BASE.pii_rules - {"tfn"}},
        {"secret_rules": frozenset()},
        {"toxicity_categories": BASE.toxicity_categories - {"threat"}},
        {"toxicity_exceptions": frozenset({"threat"})},
        {"entity_deny": frozenset()},
    ],
)
def test_check_tightens_catches_code_built_loosening(loosened):
    with pytest.raises(LooseningError, match="loosens"):
        check_tightens(BASE, replace(BASE, **loosened))


def test_check_tightens_catches_removed_pattern():
    base = replace(BASE, pii_patterns={"member_no": r"MBR-\d+"})
    with pytest.raises(LooseningError, match="member_no"):
        check_tightens(base, replace(base, pii_patterns={}))


def test_exception_model_requires_reason():
    with pytest.raises(ValueError):
        GovernanceException(rule="toxicity:insult", reason="")
