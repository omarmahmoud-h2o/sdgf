"""Pydantic models for every task.yaml section (FRAMEWORK_DESIGN.md §4.2).

Required sections: task, output_schema, rubric, seeds, coverage, models, validation,
thresholds. Optional (with defaults): tools, governance, hitl, budget.

Release thresholds are nullable here on purpose: stage 0 (compile.py) refuses a spec
with any unset threshold, so there are no silent defaults for the gate (§6.1 step 5).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

GenerationMode = Literal["label_first", "answer_emergent"]
Hosting = Literal["local", "provider_api"]
LayerName = Literal["L1", "L2", "L3", "L4", "L5", "L6"]
ALL_LAYERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")
MAX_CHOICES = 255  # per-field choice limit of the typed judge (§7.3)

REQUIRED_SECTIONS: tuple[str, ...] = (
    "task",
    "output_schema",
    "rubric",
    "seeds",
    "coverage",
    "models",
    "validation",
    "thresholds",
)
OPTIONAL_SECTIONS: tuple[str, ...] = ("tools", "governance", "hitl", "budget")


class SpecValidationError(ValueError):
    """A task spec failed schema validation. `errors` lists (field_path, message)."""

    def __init__(self, errors: list[tuple[str, str]]):
        self.errors = errors
        lines = "\n".join(f"  {path}: {msg}" for path, msg in errors)
        super().__init__(f"invalid task spec ({len(errors)} error(s)):\n{lines}")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ── task ─────────────────────────────────────────────────────────


class TaskSection(_Section):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    type: str = Field(min_length=1)
    generation_mode: GenerationMode
    description: str = Field(min_length=1)
    # For task types with pluggable answer extractors (sft_qa): which one to use.
    answer_format: str | None = None


# ── output_schema ────────────────────────────────────────────────


class FieldSpec(_Section):
    type: Literal["string", "integer", "number", "boolean", "array", "object", "null"]
    required: bool = True
    nullable: bool = False  # the key must be present but may hold null (e.g. FAG severity)
    description: str = ""
    enum: list[Any] | None = None


class TurnStructure(_Section):
    roles: list[str] = Field(min_length=2)
    first_role: str
    alternating: bool = True
    numbered_from: int = 1

    @model_validator(mode="after")
    def _first_role_known(self) -> TurnStructure:
        if self.first_role not in self.roles:
            raise ValueError(f"first_role {self.first_role!r} is not one of roles {self.roles}")
        return self


class OutputSchemaSection(_Section):
    """Task-specific fields on top of the task type's default output schema."""

    fields: dict[str, FieldSpec] = Field(default_factory=dict)
    turns: TurnStructure | None = None
    spans: bool = False


# ── rubric ───────────────────────────────────────────────────────


class Criterion(_Section):
    name: str = Field(min_length=1)
    description: str = ""
    values: list[str] | None = None
    min: int | None = None
    max: int | None = None

    @model_validator(mode="after")
    def _one_scale(self) -> Criterion:
        has_enum = self.values is not None
        has_range = self.min is not None or self.max is not None
        if has_enum == has_range:
            raise ValueError("give exactly one scale: either values, or min and max")
        if has_enum:
            if not 2 <= len(self.values) <= MAX_CHOICES:
                raise ValueError(f"values must have 2..{MAX_CHOICES} entries")
        else:
            if self.min is None or self.max is None:
                raise ValueError("an integer scale needs both min and max")
            if self.max <= self.min:
                raise ValueError("max must be greater than min")
            if self.max - self.min + 1 > MAX_CHOICES:
                raise ValueError(f"integer scale has more than {MAX_CHOICES} values")
        return self


class Verdict(_Section):
    values: list[str] = Field(min_length=2, max_length=MAX_CHOICES)
    description: str = ""
    # Which intended label each verdict value means, for L5 fidelity (e.g. FAG
    # breach -> true). Unset means the label *is* the verdict value. A verdict value
    # left out (e.g. "unclear") agrees with no label.
    labels: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _labels_name_values(self) -> Verdict:
        if self.labels is not None:
            unknown = sorted(set(self.labels) - set(self.values))
            if unknown:
                raise ValueError(f"labels names unknown verdict values {unknown}")
        return self


class RubricExample(_Section):
    """A judge-only worked example: a record view and the verdict it should get.

    Fictional only; stage 0 scans every example for PII and toxicity like the seeds,
    and checks that `record` holds only fields the judge is allowed to see.
    """

    record: dict[str, Any] = Field(min_length=1)
    verdict: str = Field(min_length=1)
    scores: dict[str, Any] = Field(default_factory=dict)
    note: str = ""  # why this is the verdict, shown to the judge


def _criterion_accepts(c: Criterion, value: Any) -> bool:
    if c.values is not None:
        return value in c.values
    return isinstance(value, int) and not isinstance(value, bool) and c.min <= value <= c.max


class RubricSection(_Section):
    verdict: Verdict
    criteria: list[Criterion] = Field(default_factory=list)
    reason_required: Literal["never", "flagged", "always"] = "never"
    # The judge's "## Context", in place of task.description (which is written for
    # the generator). Unset means the judge reads task.description, as before.
    judge_context: str | None = None
    # Rendered in the judge's static prefix only; the generation prompt never sees them.
    examples: list[RubricExample] = Field(default_factory=list)

    @field_validator("criteria")
    @classmethod
    def _unique_names(cls, v: list[Criterion]) -> list[Criterion]:
        names = [c.name for c in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate criterion names: {dupes}")
        return v

    @model_validator(mode="after")
    def _examples_fit_rubric(self) -> RubricSection:
        criteria = {c.name: c for c in self.criteria}
        for i, ex in enumerate(self.examples):
            if ex.verdict not in self.verdict.values:
                raise ValueError(f"examples[{i}]: verdict {ex.verdict!r} is not a verdict value")
            for name, value in ex.scores.items():
                if name not in criteria:
                    raise ValueError(f"examples[{i}]: scores names unknown criterion {name!r}")
                if not _criterion_accepts(criteria[name], value):
                    raise ValueError(f"examples[{i}]: scores.{name} {value!r} not allowed")
        return self


# ── seeds ────────────────────────────────────────────────────────


class SeedsSection(_Section):
    path: str = Field(min_length=1)
    format: Literal["few_shot", "annotated"] = "few_shot"
    uses: list[Literal["few_shot", "keyword_seeding", "gold_set"]] = Field(
        default_factory=lambda: ["few_shot"], min_length=1
    )
    few_shot_count: int = Field(default=3, ge=0)


# ── coverage ─────────────────────────────────────────────────────


class Axis(_Section):
    name: str = Field(min_length=1)
    # bloom: the six Bloom levels (coverage/axes.py); values, if given, pick a subset.
    source: Literal["fixed", "keyword_expansion", "retrieval", "bloom"] = "fixed"
    values: list[Any] | None = None
    weights: list[float] | None = None

    @model_validator(mode="after")
    def _values_match_source(self) -> Axis:
        if self.source == "fixed" and not self.values:
            raise ValueError("a fixed axis needs a non-empty values list")
        if self.source == "bloom" and self.values is not None:
            from sdgf.coverage.axes import BLOOM_LEVELS

            unknown = [v for v in self.values if v not in BLOOM_LEVELS]
            if unknown or not self.values:
                raise ValueError(f"bloom values must be Bloom levels {list(BLOOM_LEVELS)}")
        if self.source in ("keyword_expansion", "retrieval") and self.weights is not None:
            raise ValueError("a keyword axis is split evenly; weights are not allowed")
        if self.weights is not None:
            if self.values is None or len(self.weights) != len(self.values):
                raise ValueError("weights must have one entry per value")
            if any(w < 0 for w in self.weights) or sum(self.weights) <= 0:
                raise ValueError("weights must be non-negative with a positive sum")
        return self


class CoverageSection(_Section):
    target_size: int = Field(gt=0)
    axes: list[Axis] = Field(min_length=1)
    balance: dict[str, dict[str, float]] = Field(default_factory=dict)
    quota_policy: Literal["even", "weighted"] = "even"
    # Task-specific domain data read by hooks (e.g. FAG topic pools per scope).
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("axes")
    @classmethod
    def _unique_axes(cls, v: list[Axis]) -> list[Axis]:
        names = [a.name for a in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate axis names: {dupes}")
        return v

    @model_validator(mode="after")
    def _balance_targets(self) -> CoverageSection:
        axis_names = {a.name for a in self.axes}
        for axis, targets in self.balance.items():
            if axis not in axis_names:
                raise ValueError(f"balance names unknown axis {axis!r}")
            if any(t < 0 for t in targets.values()) or abs(sum(targets.values()) - 1.0) > 1e-6:
                raise ValueError(f"balance targets for {axis!r} must be non-negative and sum to 1")
        return self


# ── tools ────────────────────────────────────────────────────────


class ToolUse(_Section):
    name: str = Field(min_length=1)
    max_calls_per_record: int = Field(default=5, ge=0)
    max_tokens_per_record: int | None = Field(default=None, gt=0)


# ── governance ───────────────────────────────────────────────────


class GovernanceException(_Section):
    rule: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GovernanceSection(_Section):
    """Per-task tightening beyond the global profile (§7.5)."""

    extra_pii_patterns: dict[str, str] = Field(default_factory=dict)
    entity_deny: list[str] = Field(default_factory=list)
    entity_allow: list[str] = Field(default_factory=list)
    toxicity_exceptions: list[str] = Field(default_factory=list)
    exceptions: list[GovernanceException] = Field(default_factory=list)

    @field_validator("extra_pii_patterns")
    @classmethod
    def _patterns_compile(cls, v: dict[str, str]) -> dict[str, str]:
        for name, pattern in v.items():
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"pattern {name!r} is not a valid regex: {e}") from e
        return v


# ── models ───────────────────────────────────────────────────────


class ModelConfig(_Section):
    backend: str = Field(min_length=1)
    model: str = Field(min_length=1)
    hosting: Hosting | None = None  # defaulted from the backend by the model registry
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    concurrency: int = Field(default=1, ge=1)
    # Estimated price in USD per million tokens, for cost tracking (models/usage.py).
    # Both or neither; unset means the stage is unpriced, not free.
    input_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    output_cost_per_mtok: float | None = Field(default=None, ge=0.0)
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _both_prices(self) -> ModelConfig:
        if (self.input_cost_per_mtok is None) != (self.output_cost_per_mtok is None):
            raise ValueError("set both input_cost_per_mtok and output_cost_per_mtok, or neither")
        return self


class ModelsSection(_Section):
    generator: ModelConfig
    judge: ModelConfig | None = None
    fallback_judge: ModelConfig | None = None
    # L6's voters, when they should be a different model from L5's judge; unset means
    # L6 votes with models.judge.
    consistency_judge: ModelConfig | None = None
    expansion: ModelConfig | None = None


# ── validation ───────────────────────────────────────────────────


class EscalationRules(_Section):
    low_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    on_hard_cells: bool = True
    on_contestable: bool = True
    # Escalate every record to L6, e.g. DS²-Instruct self-consistency on every question.
    always: bool = False


class CalibrationRules(_Section):
    """When a judge counts as calibrated (judge/calibration.py), beside thresholds.kappa_min.

    ece_max bounds the expected calibration error of the verdict confidence, since a
    trusted judge's confidence replaces K votes (§7.3). min_gold is the smallest gold set
    a result may pass on (§16 Q4 leaves the right size open per task).
    """

    ece_max: float = Field(default=0.10, ge=0.0, le=1.0)
    bins: int = Field(default=10, ge=1, le=100)
    min_gold: int = Field(default=30, ge=1)


class ConsistencyRules(_Section):
    """How L6 varies its K votes (§6.4, §11), beside validation.consistency_k.

    Vote i is sampled at temperatures[i % len(temperatures)], under both generation
    modes, overriding the voting stage's own temperature: K votes from one model at one
    temperature 0 would just repeat L5's verdict. The default matches DS²-Instruct.
    """

    temperatures: list[float] = Field(default_factory=lambda: [0.7, 0.8, 0.9], min_length=1)

    @field_validator("temperatures")
    @classmethod
    def _in_range(cls, v: list[float]) -> list[float]:
        bad = [t for t in v if not 0.0 <= t <= 2.0]
        if bad:
            raise ValueError(f"temperatures must be within 0..2, got {bad}")
        return v


class KeywordRule(_Section):
    """An L2 required/forbidden keyword rule over record text.

    Text scanned: message contents (only those whose role is in `roles`, if given) plus
    the top-level string fields named in `fields`. `when` restricts the rule to records
    whose fields equal the given values (a list value means "one of").
    """

    name: str = Field(min_length=1)
    kind: Literal["required", "forbidden"]
    keywords: list[str] = Field(min_length=1)
    match: Literal["substring", "word", "regex"] = "word"
    case_sensitive: bool = False
    require: Literal["any", "all"] = "any"  # required rules only
    roles: list[str] | None = None
    fields: list[str] = Field(default_factory=list)
    when: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check(self) -> KeywordRule:
        if any(not k for k in self.keywords):
            raise ValueError("keywords must be non-empty strings")
        if self.match == "regex":
            for k in self.keywords:
                try:
                    re.compile(k)
                except re.error as e:
                    raise ValueError(f"keyword {k!r} is not a valid regex: {e}") from e
        if self.kind == "forbidden" and self.require != "any":
            raise ValueError("require applies to required rules only")
        return self


class ValidationSection(_Section):
    layers: list[LayerName] = Field(default_factory=lambda: list(ALL_LAYERS), min_length=1)
    repair_tries: int = Field(default=2, ge=0)
    consistency_k: int = Field(default=5, ge=1)
    consistency: ConsistencyRules = Field(default_factory=ConsistencyRules)
    escalation: EscalationRules = Field(default_factory=EscalationRules)
    calibration: CalibrationRules = Field(default_factory=CalibrationRules)
    rules: list[KeywordRule] = Field(default_factory=list)

    @field_validator("rules")
    @classmethod
    def _unique_rules(cls, v: list[KeywordRule]) -> list[KeywordRule]:
        names = [r.name for r in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate rule names: {dupes}")
        return v

    @field_validator("layers")
    @classmethod
    def _layers_ordered(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("layers must not repeat")
        if v != sorted(v, key=ALL_LAYERS.index):
            raise ValueError("layers must run cheapest first, in order L1..L6")
        return v


# ── thresholds ───────────────────────────────────────────────────


class ThresholdsSection(_Section):
    """Release-gate thresholds (§8). None means unset; compile rejects unset values."""

    fidelity_min: float | None = Field(default=None, ge=0.0, le=1.0)
    kappa_min: float | None = Field(default=None, ge=-1.0, le=1.0)
    coverage_min_cell_fill: float | None = Field(default=None, ge=0.0, le=1.0)
    balance_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)
    distinct_n_min: float | None = Field(default=None, ge=0.0, le=1.0)
    self_bleu_max: float | None = Field(default=None, ge=0.0, le=1.0)
    semantic_diversity_min: float | None = Field(default=None, ge=0.0)
    residual_error_max: float | None = Field(default=None, ge=0.0, le=1.0)
    governance_violations_max: int | None = Field(default=0, ge=0, le=0)
    overlap_max: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_per_record_max: float | None = Field(default=None, ge=0.0)

    def unset(self) -> list[str]:
        return [name for name, value in self if value is None]


# ── hitl / budget ────────────────────────────────────────────────


class HitlSection(_Section):
    approve_coverage_plan: bool = False
    review_flagged: bool = False
    calibrate_judge: bool = False


class BudgetSection(_Section):
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_seconds: float | None = Field(default=None, gt=0)
    max_candidates: int | None = Field(default=None, gt=0)


# ── top level ────────────────────────────────────────────────────


class TaskSpec(_Section):
    task: TaskSection
    output_schema: OutputSchemaSection
    rubric: RubricSection
    seeds: SeedsSection
    coverage: CoverageSection
    models: ModelsSection
    validation: ValidationSection
    thresholds: ThresholdsSection
    tools: list[ToolUse] = Field(default_factory=list)
    governance: GovernanceSection = Field(default_factory=GovernanceSection)
    hitl: HitlSection = Field(default_factory=HitlSection)
    budget: BudgetSection = Field(default_factory=BudgetSection)

    @field_validator("tools")
    @classmethod
    def _unique_tools(cls, v: list[ToolUse]) -> list[ToolUse]:
        names = [t.name for t in v]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ValueError(f"duplicate tools: {dupes}")
        return v

    @model_validator(mode="after")
    def _judge_needed(self) -> TaskSpec:
        if "L5" in self.validation.layers and self.models.judge is None:
            raise ValueError("validation layer L5 is on but models.judge is not set")
        if self.rubric.reason_required != "never" and self.models.fallback_judge is None:
            raise ValueError("rubric requires reasons but models.fallback_judge is not set")
        return self


def _loc(loc: tuple[Any, ...]) -> str:
    parts: list[str] = []
    for p in loc:
        if isinstance(p, int):
            parts.append(f"[{p}]")
        else:
            parts.append(("." if parts else "") + str(p))
    return "".join(parts) or "<spec>"


def parse_spec(data: Any) -> TaskSpec:
    """Validate a parsed task.yaml mapping; raise SpecValidationError naming each bad field."""
    if not isinstance(data, dict):
        raise SpecValidationError([("<spec>", "task spec must be a mapping")])
    try:
        return TaskSpec.model_validate(data)
    except ValidationError as e:
        errors = [(_loc(err["loc"]), err["msg"]) for err in e.errors()]
        raise SpecValidationError(errors) from None
