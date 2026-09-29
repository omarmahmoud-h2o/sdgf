"""FAG task hooks, ported from scripts/policy_categories.py and scripts/scenario_sampler.py.

The policy rule is deterministic code:

    FACTUAL_INFORMATION           -> no breach
    GENERAL_ADVICE on corps_act   -> breach (Tier 2 prohibited on Corps products)
    GENERAL_ADVICE on non_corps   -> no breach (permitted, monitored)
    PERSONAL_ADVICE               -> breach, always

Records carry financial_advice_breach in the task type's `label` field. Domain data
(topic pools, signal groups, weights) is read from coverage.params in the task.yaml
next to this file, so there is one source of truth.

Permitted general advice keeps its signals with no spans: a hard negative where
advisory language is present and the correct label is still "no breach".
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from sdgf.validate.base import ValidationIssue

_SPEC = yaml.safe_load((Path(__file__).with_name("task.yaml")).read_text(encoding="utf-8"))
P: dict[str, Any] = _SPEC["coverage"]["params"]
AXES: dict[str, dict[Any, float]] = {
    a["name"]: dict(zip(a["values"], a["weights"])) for a in _SPEC["coverage"]["axes"]
}

GENERAL = list(P["general_advice_signals"])
PERSONAL = list(P["personal_advice_signals"])
DOMAIN = list(P["domain_advice_signals"])
_GENERAL, _PERSONAL = set(GENERAL), set(PERSONAL)
TIERS = list(P["advice_tier_descriptions"])
TOPICS = list(P["corps_act_product_topics"]) + list(P["non_corps_act_topics"])
SIGNALS = set(P["signal_descriptions"])
ADVICE_SEEKING_STANCES = set(P["advice_seeking_stances"])


# ---------------------------------------------------------------- policy rule


def expected_breach(advice_tier: str, product_scope: str) -> bool:
    if advice_tier == "FACTUAL_INFORMATION":
        return False
    if advice_tier == "PERSONAL_ADVICE":
        return True
    if advice_tier == "GENERAL_ADVICE":
        return product_scope == "corps_act"
    raise ValueError(f"unknown advice_tier: {advice_tier}")


def label_rule(record: dict[str, Any]) -> bool:
    return expected_breach(record["advice_tier"], record["product_scope"])


# ------------------------------------------------------- policy categories


def derive_policy_categories(record: dict[str, Any]) -> dict[str, bool]:
    """Map one record onto the 9 production policy booleans. Derived, never annotated."""
    signals = set(record.get("signal_categories") or [])
    breach = bool(record.get("label"))
    corps = record.get("product_scope") == "corps_act"
    is_corps_q = bool(record.get("is_corps_question"))
    denial = bool(record.get("denial_present"))

    advice_signals = bool(signals & (_GENERAL | _PERSONAL))

    return {
        "corps_question_breach_no_denial": breach and is_corps_q and not denial,
        "corps_question_breach_general_advice": breach and is_corps_q and bool(signals & _GENERAL),
        "business_advice_corps_products": breach and corps and advice_signals,
        "business_advice_non_corps_tailored": breach and not corps and bool(signals & _PERSONAL),
        # Belongs to the groundedness guardrail; kept so the shape matches production.
        "business_advice_misleading": False,
        "tax_advice": "TAX_ADVICE" in signals,
        "investment_advice": "INVESTMENT_ADVICE" in signals,
        "legal_advice": "LEGAL_ADVICE" in signals,
        "insurance_advice": "INSURANCE_ADVICE" in signals,
    }


def unexplained_breach(record: dict[str, Any]) -> bool:
    """A breach that no policy category explains (the annotation-gap check)."""
    if not record.get("label"):
        return False
    return not any(derive_policy_categories(record).values())


def post_process(record: dict[str, Any]) -> dict[str, Any]:
    return {**record, "policy_categories": derive_policy_categories(record)}


# ---------------------------------------------------------------- validators


def extra_validators(record: dict[str, Any]) -> list[ValidationIssue]:
    """FAG rules from scripts/utils.validate_conversation that L1's schema can't express.

    Label agreement with expected_breach is L2's label_rule check, and field types,
    enums and turn order are L1's, so neither is repeated here.
    """
    issues: list[ValidationIssue] = []
    breach = record.get("label")
    signals = list(record.get("signal_categories") or [])
    spans = list(record.get("spans") or [])
    turns = {m.get("turn"): m for m in record.get("messages") or [] if isinstance(m, dict)}

    for i, sig in enumerate(signals):
        if sig not in SIGNALS:
            issues.append(
                ValidationIssue(
                    "unknown_signal",
                    f"{sig!r} is not one of the 15 FAG signal categories",
                    f"signal_categories[{i}]",
                    {"got": sig},
                )
            )

    # Severity is meaningful only for a breach.
    severity = record.get("severity")
    if breach is True and severity is None:
        issues.append(
            ValidationIssue("severity_missing", "a breach record needs a severity", "severity")
        )
    if breach is False and severity is not None:
        issues.append(
            ValidationIssue(
                "severity_on_non_breach",
                f"a non-breach record must have severity null, got {severity!r}",
                "severity",
                {"got": severity},
            )
        )

    # A Corps question is by definition about a Corps Act product.
    if record.get("is_corps_question") and record.get("product_scope") != "corps_act":
        issues.append(
            ValidationIssue(
                "corps_question_scope",
                "is_corps_question true requires product_scope corps_act",
                "is_corps_question",
                {"product_scope": record.get("product_scope")},
            )
        )

    if breach is True and not spans:
        issues.append(
            ValidationIssue(
                "breach_without_span", "a breach record needs at least one span", "spans"
            )
        )
    if breach is False and (spans or record.get("problematic_turns")):
        # Hard negatives keep their signals, but nothing is problematic.
        issues.append(
            ValidationIssue(
                "spans_on_non_breach",
                "a non-breach record must have empty spans and problematic_turns",
                "spans",
            )
        )

    span_categories = set()
    for i, span in enumerate(spans):
        path = f"spans[{i}]"
        turn_no = span.get("turn")
        msg = turns.get(turn_no)
        if msg is not None and msg.get("role") != "assistant":
            issues.append(
                ValidationIssue(
                    "span_not_assistant",
                    f"span cites turn {turn_no}, which is not an assistant turn",
                    f"{path}.turn",
                    {"turn": turn_no, "role": msg.get("role")},
                )
            )
        # Models reword spans; the quote must be exact.
        if msg is not None and str(span.get("text", "")) not in str(msg.get("content", "")):
            issues.append(
                ValidationIssue(
                    "span_not_verbatim",
                    f"span text is not a verbatim substring of turn {turn_no}: "
                    f"{span.get('text')!r}",
                    f"{path}.text",
                    {"turn": turn_no},
                )
            )
        category = span.get("category")
        span_categories.add(category)
        if category not in signals:
            issues.append(
                ValidationIssue(
                    "span_category_not_signal",
                    f"span category {category!r} is not in the record's signal_categories",
                    f"{path}.category",
                    {"got": category, "signals": signals},
                )
            )

    # Every declared signal must be evidenced by a span.
    if breach is True:
        for i, sig in enumerate(signals):
            if sig not in span_categories:
                issues.append(
                    ValidationIssue(
                        "signal_without_span",
                        f"signal_category {sig} has no annotated span",
                        f"signal_categories[{i}]",
                        {"signal": sig},
                    )
                )

    # post_process adds these after validation; a record that already carries them
    # (a seed, or a re-validated release) must agree with the derivation.
    if "policy_categories" in record and record["policy_categories"] != derive_policy_categories(
        record
    ):
        issues.append(
            ValidationIssue(
                "policy_categories_mismatch",
                "policy_categories do not match the derivation from signals and context",
                "policy_categories",
            )
        )

    if unexplained_breach(record):
        issues.append(
            ValidationIssue(
                "unexplained_breach",
                "the breach is not explained by any production policy category",
                "label",
            )
        )
    return issues


# ------------------------------------------------------------------ sampler


def _weighted_choice(rng: random.Random, options_weights: dict[Any, float]) -> Any:
    options, weights = zip(*options_weights.items())
    return rng.choices(options, weights=weights, k=1)[0]


def _sample_signals(rng: random.Random, advice_tier: str) -> list[str]:
    """Signals the response exhibits, consistent with its advice tier."""
    if advice_tier == "PERSONAL_ADVICE":
        # Personal advice must reference the customer's own circumstances.
        anchor = rng.choice(PERSONAL)
        pool = [s for s in GENERAL + PERSONAL if s != anchor]
        signals = [anchor] + rng.sample(pool, 1 if rng.random() < 0.45 else 0)
    elif advice_tier == "GENERAL_ADVICE":
        n = 2 if rng.random() < 0.35 else 1
        signals = rng.sample(GENERAL, n)
    else:  # FACTUAL_INFORMATION
        return []

    if rng.random() < P["domain_signal_rate"]:
        signals.append(rng.choice(DOMAIN))
    return signals


def _sample_tier(rng: random.Random, breach: bool, corps: bool) -> str:
    if breach:
        # On Corps Act products general advice is already a breach; off them, only
        # personal advice is.
        if corps:
            return _weighted_choice(rng, P["breach_tier_weights_corps_act"])
        return "PERSONAL_ADVICE"
    # Off Corps Act products, permitted general advice is the most useful hard negative.
    if corps:
        return "FACTUAL_INFORMATION"
    return _weighted_choice(rng, P["no_breach_tier_weights_non_corps_act"])


def _sample_denial(rng: random.Random, is_corps_question: bool, breach: bool) -> bool:
    """Whether the assistant issued the decline a Corps question requires."""
    if not is_corps_question:
        return False  # no denial required by policy
    if not breach:
        return True  # a compliant answer to a Corps question declines
    # A breaching answer usually omits the decline; sometimes it declines then advises.
    return rng.random() >= P["no_denial_rate"]


def sampler_constraints(cell: dict[str, Any], rng: random.Random) -> dict[str, Any] | None:
    """Fill a cell's recipe so every field is consistent with its scope and label.

    The axes (product_scope, label, conversation_length) come from the cell; any axis the
    cell omits is drawn from its task.yaml weights. A cell may also pin advice_tier,
    primary_topic or customer_stance; a pinned value that contradicts the cell returns
    None (invalid combination).
    """
    out = dict(cell)
    scope = out.get("product_scope")
    if scope is None:
        scope = _weighted_choice(rng, AXES["product_scope"])
    breach = out.get("label")
    if breach is None:
        breach = _weighted_choice(rng, AXES["label"])
    bucket = out.get("conversation_length")
    if bucket is None:
        bucket = _weighted_choice(rng, AXES["conversation_length"])
    if scope not in AXES["product_scope"] or bucket not in P["conversation_length_buckets"]:
        return None
    breach = bool(breach)
    corps = scope == "corps_act"

    pool = P["corps_act_product_topics"] if corps else P["non_corps_act_topics"]
    topic = out.get("primary_topic")
    if topic is None:
        topic = rng.choice(pool)
    elif topic not in pool:
        return None

    stance = out.get("customer_stance")
    if stance is None:
        stance = rng.choice(P["customer_stances"])
    elif stance not in P["customer_stances"]:
        return None
    # A "Corps question" is an advice-seeking request about a Corps Act product.
    is_corps_question = corps and stance in ADVICE_SEEKING_STANCES

    tier = out.get("advice_tier")
    if tier is None:
        tier = _sample_tier(rng, breach, corps)
    elif tier not in TIERS or expected_breach(tier, scope) != breach:
        return None

    if breach:
        severity = _weighted_choice(
            rng,
            P["severity_weights_personal_advice"]
            if tier == "PERSONAL_ADVICE"
            else P["severity_weights_general_advice"],
        )
    else:
        severity = None  # meaningless without a breach

    n_secondary = int(_weighted_choice(rng, P["secondary_topic_count_weights"]))
    remaining = [t for t in TOPICS if t != topic]

    out.update(
        {
            "conversation_type": "single_turn" if bucket == "single_turn" else "multi_turn",
            "conversation_length": bucket,
            "turn_count": rng.choice(P["conversation_length_buckets"][bucket]),
            "industry": rng.choice(P["industries"]),
            "business_type": rng.choice(P["business_types"]),
            "jurisdiction": rng.choice(P["jurisdictions"]),
            "product_scope": scope,
            "is_corps_question": is_corps_question,
            "denial_present": _sample_denial(rng, is_corps_question, breach),
            "primary_topic": topic,
            "secondary_topics": rng.sample(remaining, min(n_secondary, len(remaining))),
            "label": breach,
            "advice_tier": tier,
            "signal_categories": _sample_signals(rng, tier),
            "severity": severity,
            "difficulty": _weighted_choice(
                rng,
                P["difficulty_weights_breach"] if breach else P["difficulty_weights_no_breach"],
            ),
            "contestable": rng.random() < P["contestable_rate"],
            "customer_stance": stance,
        }
    )
    return out
