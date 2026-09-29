"""Judge calibration against a human gold set (FRAMEWORK_DESIGN.md §6.5, §7.3, §8).

A judge is trusted only once it has been measured against people: Cohen's κ between
its labels and the human labels on a gold set must reach thresholds.kappa_min, and the
expected calibration error (ECE) of its verdict confidence must stay within
validation.calibration.ece_max, since a trusted judge's confidence replaces K votes in
L6. "Calibrated" is a vendor claim until this says so.

Judge verdicts are compared in label space through rubric.verdict.labels (FAG breach ->
true), exactly as L5 checks fidelity. A gold record the judge can't answer (a
JudgeParseError) counts as a wrong label in κ and accuracy, and is left out of the
reliability bins since it has no confidence.

A result is persisted per spec_version and judge model in the spec_version's shared
artefact area. A changed rubric changes spec_version and a changed judge model changes
the judge id, so either needs a fresh calibration before the judge is trusted again.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from sdgf.judge.interface import Judge, JudgeError, JudgeParseError, JudgeResult, verdict_means
from sdgf.judge.llm_judge import judge_view
from sdgf.spec.schema import ModelConfig

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec
    from sdgf.store.artefacts import ArtefactStore

LABEL_FIELD = "label"
RESULT_VERSION = 1


class CalibrationError(ValueError):
    """Calibration inputs are unusable (empty, mismatched, out of range)."""


def _key(value: Any) -> tuple[bool, Any]:
    # True == 1 in Python; keep bool labels apart from ints, as verdict_means does.
    return (isinstance(value, bool), value)


# ── metrics ──────────────────────────────────────────────────────


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float:
    """Cohen's κ between two raters' labels for the same items.

    Raises CalibrationError when κ is undefined: no items, or both raters giving every
    item the same single category (chance agreement is then 1).
    """
    if len(a) != len(b):
        raise CalibrationError(f"kappa needs paired labels, got {len(a)} and {len(b)}")
    n = len(a)
    if n == 0:
        raise CalibrationError("kappa needs at least one item")
    ka, kb = [_key(x) for x in a], [_key(y) for y in b]
    observed = sum(x == y for x, y in zip(ka, kb))
    ca, cb = Counter(ka), Counter(kb)
    chance = sum(ca[k] * cb[k] for k in ca)  # expected agreement × n²
    if chance == n * n:
        raise CalibrationError("kappa is undefined when both raters use one same category")
    return (observed * n - chance) / (n * n - chance)


@dataclass(frozen=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ReliabilityBin:
        return cls(d["lower"], d["upper"], d["count"], d["mean_confidence"], d["accuracy"])


def reliability_bins(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10
) -> tuple[ReliabilityBin, ...]:
    """Equal-width bins over [0, 1]; bin i holds [i/n, (i+1)/n), the last also holds 1."""
    if len(confidences) != len(correct):
        raise CalibrationError("reliability bins need one correctness flag per confidence")
    if n_bins < 1:
        raise CalibrationError("reliability bins need n_bins >= 1")
    members: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        if not 0.0 <= c <= 1.0:
            raise CalibrationError(f"confidence {c} is outside [0, 1]")
        members[min(int(c * n_bins), n_bins - 1)].append((c, bool(ok)))
    bins = []
    for i, items in enumerate(members):
        count = len(items)
        bins.append(
            ReliabilityBin(
                lower=i / n_bins,
                upper=(i + 1) / n_bins,
                count=count,
                mean_confidence=sum(c for c, _ in items) / count if count else None,
                accuracy=sum(ok for _, ok in items) / count if count else None,
            )
        )
    return tuple(bins)


def expected_calibration_error(bins: Iterable[ReliabilityBin]) -> float:
    """Σ (bin count / total) · |bin accuracy − bin mean confidence|."""
    bins = [b for b in bins if b.count]
    total = sum(b.count for b in bins)
    if total == 0:
        raise CalibrationError("ECE needs at least one confidence")
    return sum(b.count / total * abs(b.accuracy - b.mean_confidence) for b in bins)


# ── results ──────────────────────────────────────────────────────


def judge_id(config: ModelConfig) -> str:
    """The judge model a calibration belongs to: backend and model name."""
    return f"{config.backend}:{config.model}"


def spec_judge_id(compiled: CompiledSpec, stage: str = "judge") -> str:
    config = getattr(compiled.spec.models, stage)
    if config is None:
        raise JudgeError(f"models.{stage} is not set, so there is no judge to calibrate")
    return judge_id(config)


@dataclass(frozen=True)
class CalibrationResult:
    spec_version: str
    judge_id: str
    n: int
    unparseable: int
    accuracy: float
    kappa: float | None  # None when undefined on this gold set
    ece: float | None  # None when no gold record got a parseable answer
    bins: tuple[ReliabilityBin, ...]
    kappa_min: float
    ece_max: float
    min_gold: int
    problems: tuple[str, ...] = ()  # why it didn't pass; empty means passed
    confusion: Mapping[str, Mapping[str, int]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.problems

    def trusts(self, spec_version: str, judge: str) -> bool:
        """Passed, and measured for this very spec_version and judge model."""
        return self.passed and self.spec_version == spec_version and self.judge_id == judge

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": RESULT_VERSION,
            "spec_version": self.spec_version,
            "judge_id": self.judge_id,
            "n": self.n,
            "unparseable": self.unparseable,
            "accuracy": self.accuracy,
            "kappa": self.kappa,
            "ece": self.ece,
            "bins": [b.to_dict() for b in self.bins],
            "kappa_min": self.kappa_min,
            "ece_max": self.ece_max,
            "min_gold": self.min_gold,
            "passed": self.passed,
            "problems": list(self.problems),
            "confusion": {j: dict(h) for j, h in self.confusion.items()},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CalibrationResult:
        if d.get("version") != RESULT_VERSION:
            raise CalibrationError(f"unknown calibration result version {d.get('version')!r}")
        return cls(
            spec_version=d["spec_version"],
            judge_id=d["judge_id"],
            n=d["n"],
            unparseable=d["unparseable"],
            accuracy=d["accuracy"],
            kappa=d["kappa"],
            ece=d["ece"],
            bins=tuple(ReliabilityBin.from_dict(b) for b in d["bins"]),
            kappa_min=d["kappa_min"],
            ece_max=d["ece_max"],
            min_gold=d["min_gold"],
            problems=tuple(d["problems"]),
            confusion={j: dict(h) for j, h in d["confusion"].items()},
        )


UNPARSEABLE = "<unparseable>"
UNMAPPED = "<unmapped>"


def calibrate(
    results: Sequence[JudgeResult | None],
    human: Sequence[Any],
    *,
    spec_version: str,
    judge: str,
    kappa_min: float,
    labels: Mapping[str, Any] | None = None,
    ece_max: float = 0.10,
    n_bins: int = 10,
    min_gold: int = 30,
) -> CalibrationResult:
    """Score judge results (None = unparseable) against human labels for the same items.

    `labels` is rubric.verdict.labels; without it a verdict means itself.
    """
    if len(results) != len(human):
        raise CalibrationError(
            f"{len(results)} judge results for {len(human)} human labels; they must pair up"
        )
    n = len(results)
    if n == 0:
        raise CalibrationError("the gold set is empty")

    judge_labels: list[Any] = []
    correct: list[bool] = []
    confidences: list[float] = []
    confident_correct: list[bool] = []
    confusion: dict[str, Counter[str]] = {}
    for result, truth in zip(results, human):
        if result is None:
            judged: Any = UNPARSEABLE
            ok = False
        else:
            mapping = labels if labels is not None else {result.verdict: result.verdict}
            judged = mapping.get(result.verdict, UNMAPPED)
            ok = verdict_means(mapping, result.verdict, truth)
            confidences.append(result.verdict_confidence)
            confident_correct.append(ok)
        judge_labels.append(judged)
        correct.append(ok)
        confusion.setdefault(repr(judged), Counter())[repr(truth)] += 1

    problems: list[str] = []
    if n < min_gold:
        problems.append(f"gold set has {n} items, fewer than min_gold {min_gold}")
    try:
        kappa: float | None = cohen_kappa(judge_labels, list(human))
    except CalibrationError as e:
        kappa = None
        problems.append(str(e))
    if kappa is not None and kappa < kappa_min:
        problems.append(f"kappa {kappa:.3f} is below kappa_min {kappa_min}")

    bins = reliability_bins(confidences, confident_correct, n_bins)
    ece = expected_calibration_error(bins) if confidences else None
    if ece is None:
        problems.append("no gold item got a parseable judge answer, so ECE can't be measured")
    elif ece > ece_max:
        problems.append(f"expected calibration error {ece:.3f} is above ece_max {ece_max}")

    return CalibrationResult(
        spec_version=spec_version,
        judge_id=judge,
        n=n,
        unparseable=sum(r is None for r in results),
        accuracy=sum(correct) / n,
        kappa=kappa,
        ece=ece,
        bins=bins,
        kappa_min=kappa_min,
        ece_max=ece_max,
        min_gold=min_gold,
        problems=tuple(problems),
        confusion={j: dict(h) for j, h in confusion.items()},
    )


@dataclass(frozen=True)
class GoldItem:
    """A record with the label a person gave it."""

    record: Mapping[str, Any]
    label: Any


def gold_from_records(
    records: Iterable[Mapping[str, Any]], label_field: str = LABEL_FIELD
) -> list[GoldItem]:
    """Gold items from hand-labelled records that carry their human label in a field."""
    items = []
    for i, record in enumerate(records):
        if label_field not in record:
            raise CalibrationError(f"gold record {i} has no {label_field!r} field")
        items.append(GoldItem(record, record[label_field]))
    return items


def run_calibration(
    compiled: CompiledSpec,
    judge: Judge,
    gold: Sequence[GoldItem],
    *,
    judge_name: str | None = None,
    fields: Iterable[str] | None = None,
) -> CalibrationResult:
    """Judge every gold record blind, then calibrate against the human labels.

    The judge sees only judge_view(record, fields), fields defaulting to the task type's
    judge_fields() without the label, exactly as in L5.
    """
    from sdgf.tasktypes.registry import REGISTRY

    spec = compiled.spec
    if spec.thresholds.kappa_min is None:
        raise CalibrationError("thresholds.kappa_min is unset")
    if fields is None:
        fields = REGISTRY.resolve(spec.task).judge_fields()
    fields = tuple(f for f in fields if f != LABEL_FIELD)
    results: list[JudgeResult | None] = []
    for item in gold:
        try:
            results.append(judge.judge(judge_view(item.record, fields)))
        except JudgeParseError:
            results.append(None)
    rules = spec.validation.calibration
    return calibrate(
        results,
        [item.label for item in gold],
        spec_version=compiled.spec_version,
        judge=judge_name or spec_judge_id(compiled),
        kappa_min=spec.thresholds.kappa_min,
        labels=spec.rubric.verdict.labels,
        ece_max=rules.ece_max,
        n_bins=rules.bins,
        min_gold=rules.min_gold,
    )


def trust_for(
    compiled: CompiledSpec, calibration: CalibrationResult | None, judge: str | None = None
) -> bool:
    """Whether a calibration result makes this spec's judge trusted (None: not trusted)."""
    if calibration is None:
        return False
    return calibration.trusts(compiled.spec_version, judge or spec_judge_id(compiled))


# ── persistence ──────────────────────────────────────────────────


def _stage_name(judge: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", judge).strip("-.")[:48] or "judge"
    digest = hashlib.sha256(judge.encode("utf-8")).hexdigest()[:12]
    return f"calibration-{slug}-{digest}"


class CalibrationStore:
    """Calibration results in the artefact store, one per spec_version and judge model."""

    def __init__(self, store: ArtefactStore):
        self.store = store

    def save(self, result: CalibrationResult):
        return self.store.write_shared(
            result.spec_version, _stage_name(result.judge_id), result.to_dict()
        )

    def load(self, spec_version: str, judge: str) -> CalibrationResult | None:
        stage = _stage_name(judge)
        if not self.store.has_shared(spec_version, stage):
            return None
        result = CalibrationResult.from_dict(self.store.read_shared(spec_version, stage))
        if result.judge_id != judge or result.spec_version != spec_version:
            raise CalibrationError(f"calibration artefact {stage} belongs to another judge")
        return result

    def load_for(self, compiled: CompiledSpec, stage: str = "judge") -> CalibrationResult | None:
        return self.load(compiled.spec_version, spec_judge_id(compiled, stage))

    def trusted(self, spec_version: str, judge: str) -> bool:
        result = self.load(spec_version, judge)
        return result is not None and result.trusts(spec_version, judge)
