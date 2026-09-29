"""L2: rules and keywords (FRAMEWORK_DESIGN.md §6.4).

Three independent checks, all run so one repair prompt carries every error:

    1. keywords     spec validation.rules: required or forbidden keywords over the
                    record's text, optionally restricted by role, field and `when`
    2. label        the record's label agrees with the label_rule hook, i.e. the
                    code-owned facts (e.g. tier and scope) imply the fixed label
    3. extra        the task's extra_validators hook (e.g. FAG's verbatim spans)

Every failure is repairable: the errors go back to the generator as feedback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Sequence

from sdgf.spec.hooks import ExtraValidators, LabelRule
from sdgf.spec.schema import KeywordRule
from sdgf.validate.base import Layer, LayerVerdict, Record, ValidationContext, ValidationIssue

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec

LABEL_FIELD = "label"
# Hook failures on a bad record (missing or unknown fact) are record errors, not bugs.
HOOK_RECORD_ERRORS = (KeyError, TypeError, ValueError)


@dataclass(frozen=True)
class CompiledRule:
    rule: KeywordRule
    patterns: tuple[tuple[str, re.Pattern[str]], ...]

    @classmethod
    def build(cls, rule: KeywordRule) -> CompiledRule:
        flags = 0 if rule.case_sensitive else re.IGNORECASE
        patterns = []
        for keyword in rule.keywords:
            if rule.match == "regex":
                source = keyword
            elif rule.match == "word":
                source = rf"(?<!\w){re.escape(keyword)}(?!\w)"
            else:
                source = re.escape(keyword)
            patterns.append((keyword, re.compile(source, flags)))
        return cls(rule, tuple(patterns))

    def applies(self, record: Record) -> bool:
        for key, want in self.rule.when.items():
            got = record.get(key)
            if isinstance(want, list) and not isinstance(got, list):
                if got not in want:
                    return False
            elif got != want:
                return False
        return True

    def texts(self, record: Record) -> Iterator[tuple[str, str]]:
        """(path, text) for every piece of text the rule scans."""
        messages = record.get("messages")
        if isinstance(messages, list):
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict) or not isinstance(msg.get("content"), str):
                    continue
                if self.rule.roles is not None and msg.get("role") not in self.rule.roles:
                    continue
                yield f"messages[{i}].content", msg["content"]
        for name in self.rule.fields:
            value = record.get(name)
            if isinstance(value, str):
                yield name, value

    def issues(self, record: Record) -> list[ValidationIssue]:
        if not self.applies(record):
            return []
        texts = list(self.texts(record))
        rule = self.rule
        if rule.kind == "forbidden":
            out = []
            for path, text in texts:
                for keyword, pattern in self.patterns:
                    m = pattern.search(text)
                    if m:
                        out.append(
                            ValidationIssue(
                                "keyword_forbidden",
                                f"rule {rule.name!r}: forbidden keyword {keyword!r} "
                                f"found ({m.group(0)!r})",
                                path,
                                {"rule": rule.name, "keyword": keyword, "found": m.group(0)},
                            )
                        )
            return out

        found = [kw for kw, pattern in self.patterns if any(pattern.search(t) for _, t in texts)]
        missing = [kw for kw, _ in self.patterns if kw not in found]
        if rule.require == "any" and not found:
            return [
                ValidationIssue(
                    "keyword_required",
                    f"rule {rule.name!r}: none of the required keywords {list(rule.keywords)} "
                    "appear",
                    details={"rule": rule.name, "missing": missing, "require": "any"},
                )
            ]
        if rule.require == "all":
            return [
                ValidationIssue(
                    "keyword_required",
                    f"rule {rule.name!r}: required keyword {kw!r} does not appear",
                    details={"rule": rule.name, "missing": [kw], "require": "all"},
                )
                for kw in missing
            ]
        return []


def label_issues(record: Record, label_rule: LabelRule) -> list[ValidationIssue]:
    try:
        expected = label_rule(record)
    except HOOK_RECORD_ERRORS as e:
        return [
            ValidationIssue(
                "label_rule_error",
                f"label_rule could not compute the label: {type(e).__name__}: {e}",
                LABEL_FIELD,
                {"exception": type(e).__name__},
            )
        ]
    got = record.get(LABEL_FIELD)
    if got != expected or type(got) is not type(expected):
        return [
            ValidationIssue(
                "label_disagrees",
                f"label is {got!r} but the record's facts imply {expected!r}",
                LABEL_FIELD,
                {"expected": expected, "got": got},
            )
        ]
    return []


def extra_issues(record: Record, extra_validators: ExtraValidators) -> list[ValidationIssue]:
    """Hook errors may be plain strings or ValidationIssue values (to set a code/path)."""
    try:
        errors = extra_validators(record)
    except HOOK_RECORD_ERRORS as e:
        return [
            ValidationIssue(
                "extra_validator_error",
                f"extra_validators failed: {type(e).__name__}: {e}",
                details={"exception": type(e).__name__},
            )
        ]
    out = []
    for error in errors:
        if isinstance(error, ValidationIssue):
            out.append(error)
        elif isinstance(error, str):
            out.append(ValidationIssue("extra_validator", error))
        else:
            raise TypeError(
                f"extra_validators returned {type(error).__name__}; expected str or ValidationIssue"
            )
    return out


class RulesLayer(Layer):
    name = "L2"

    def __init__(
        self,
        rules: Sequence[KeywordRule] = (),
        label_rule: LabelRule | None = None,
        extra_validators: ExtraValidators | None = None,
    ):
        self.rules = tuple(CompiledRule.build(r) for r in rules)
        self.label_rule = label_rule
        self.extra_validators = extra_validators

    @classmethod
    def from_spec(cls, compiled: CompiledSpec) -> RulesLayer:
        return cls(
            compiled.spec.validation.rules,
            compiled.hooks.label_rule,
            compiled.hooks.extra_validators,
        )

    def check(self, record: Record, context: ValidationContext) -> LayerVerdict:
        issues: list[ValidationIssue] = []
        for rule in self.rules:
            issues += rule.issues(record)
        if self.label_rule is not None:
            issues += label_issues(record, self.label_rule)
        if self.extra_validators is not None:
            issues += extra_issues(record, self.extra_validators)
        return self.verdict(issues, repairable=True)
