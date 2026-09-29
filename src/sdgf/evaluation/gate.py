"""Stage 5 release gate (FRAMEWORK_DESIGN.md §6.6, §8): metrics against spec.thresholds.

Each threshold is checked on the overall metrics; the ones that make sense for a single
cell (fidelity, residual error, balance, governance, overlap, coverage fill) are also
checked per cell, so a failure names the cells that fall short. Diversity, kappa and
cost are overall only: a small cell's self-BLEU says little, and kappa and cost aren't
measured per cell.

    pass           every threshold met, overall and per cell
    fail           some metric misses its threshold, or a thresholded metric is
                   unmeasured (None) and wasn't waived
    hard fail      governance violations above zero, or governance unmeasured; this
                   can't be waived and always fails (§3 principle 5)

Comparisons follow the layers: *_min passes at value >= threshold, *_max at value <=
threshold (L4 drops only above overlap_max). Overlap is gated on the shingle scores for
seeds and held-out items; embedding scores aren't on overlap_max's scale.

short_cells lists the cells below coverage_min_cell_fill with the records each still
needs to reach its quota, which is what the scheduler refills.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from sdgf.evaluation.metrics import OVERLAP_SOURCES, Metrics, MetricsReport

if TYPE_CHECKING:
    from sdgf.spec.schema import ThresholdsSection

DEFAULT_DISTINCT_N = 2
GOVERNANCE = "governance_violations_max"

Reason = Literal["below_min", "above_max", "not_measured", "governance"]


class GateError(ValueError):
    """The gate was asked to do something it must not, e.g. waive governance."""


@dataclass(frozen=True)
class GateFailure:
    metric: str  # the threshold name, e.g. fidelity_min
    reason: Reason
    value: float | None
    threshold: float
    cell: str | None = None  # None for the overall metrics


@dataclass(frozen=True)
class GateResult:
    passed: bool
    hard_fail: bool
    failures: tuple[GateFailure, ...] = ()
    short_cells: dict[str, int] = field(default_factory=dict)  # cell -> records missing
    waived: tuple[str, ...] = ()
    spec_version: str | None = None

    @property
    def failing_metrics(self) -> list[str]:
        return sorted({f.metric for f in self.failures})

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["failures"] = [asdict(f) for f in self.failures]
        d["waived"] = list(self.waived)
        d["failing_metrics"] = self.failing_metrics
        return d


def _distinct(m: Metrics, n: int) -> float | None:
    return None if m.diversity is None else m.diversity.distinct.get(n)


def _overlap(m: Metrics) -> float | None:
    scores = [v for k, v in m.overlap.items() if k in OVERLAP_SOURCES]
    return max(scores) if scores else None


def _values(m: Metrics, distinct_n: int) -> dict[str, float | None]:
    """Threshold name -> the metric it gates."""
    return {
        "fidelity_min": m.fidelity,
        "kappa_min": m.kappa,
        "coverage_min_cell_fill": m.min_cell_fill,
        "balance_tolerance": m.balance_max_deviation,
        "distinct_n_min": _distinct(m, distinct_n),
        "self_bleu_max": None if m.diversity is None else m.diversity.self_bleu,
        "semantic_diversity_min": None if m.diversity is None else m.diversity.cluster_entropy,
        "residual_error_max": m.residual_error,
        GOVERNANCE: None if m.governance_violations is None else float(m.governance_violations),
        "overlap_max": _overlap(m),
        "cost_per_record_max": m.cost_per_record,
    }


def threshold_values(m: Metrics, distinct_n: int = DEFAULT_DISTINCT_N) -> dict[str, float | None]:
    """Threshold name -> the value of the metric it gates, for reports."""
    return _values(m, distinct_n)


# Thresholds where the value must stay at or under the threshold; the rest are minimums.
_UPPER = frozenset(
    {
        "balance_tolerance",
        "self_bleu_max",
        "residual_error_max",
        GOVERNANCE,
        "overlap_max",
        "cost_per_record_max",
    }
)
_PER_CELL = frozenset(
    {
        "fidelity_min",
        "coverage_min_cell_fill",
        "balance_tolerance",
        "residual_error_max",
        GOVERNANCE,
        "overlap_max",
    }
)


def _check(
    name: str, value: float | None, threshold: float, cell: str | None
) -> GateFailure | None:
    if name == GOVERNANCE:
        if value is None or value > threshold:
            return GateFailure(name, "governance", value, threshold, cell)
        return None
    if value is None:
        return GateFailure(name, "not_measured", None, threshold, cell)
    if name in _UPPER:
        return GateFailure(name, "above_max", value, threshold, cell) if value > threshold else None
    return GateFailure(name, "below_min", value, threshold, cell) if value < threshold else None


def evaluate_gate(
    report: MetricsReport,
    thresholds: ThresholdsSection | Mapping[str, float | None],
    *,
    distinct_n: int = DEFAULT_DISTINCT_N,
    waive: Iterable[str] = (),
) -> GateResult:
    """Compare a metrics report with the release thresholds.

    Unset thresholds (None) aren't checked, except governance, which is always zero;
    stage 0 already refuses a spec with any unset. waive names thresholds whose metric may be unmeasured without failing (e.g.
    kappa_min before any gold set exists); a waived metric that was measured is still
    checked. Governance can't be waived."""
    limits = dict(thresholds)  # a ThresholdsSection iterates as (name, value) pairs
    limits[GOVERNANCE] = 0  # §8: zero, hard, whatever the spec says
    waived = tuple(sorted(set(waive)))
    unknown = [w for w in waived if w not in _values(report.overall, distinct_n)]
    if unknown:
        raise GateError(f"unknown thresholds to waive: {unknown}")
    if GOVERNANCE in waived:
        raise GateError("governance violations always gate the release and can't be waived")

    failures: list[GateFailure] = []

    def run(m: Metrics, names: Iterable[str], cell: str | None) -> None:
        values = _values(m, distinct_n)
        for name in names:
            threshold = limits.get(name)
            if threshold is None:
                continue
            value = values[name]
            # Per cell, an unmeasured metric (e.g. no records yet) is left to the overall
            # check and to coverage, except governance, which must always be measured.
            if value is None and (name in waived or (cell is not None and name != GOVERNANCE)):
                continue
            if (failure := _check(name, value, float(threshold), cell)) is not None:
                failures.append(failure)

    run(report.overall, _values(report.overall, distinct_n), None)
    for cell_id, m in report.per_cell.items():
        run(m, sorted(_PER_CELL), cell_id)

    fill_min = limits.get("coverage_min_cell_fill")
    short: dict[str, int] = {}
    if fill_min is not None:
        for cell_id, m in report.per_cell.items():
            if m.fill is not None and m.fill < fill_min and m.quota is not None:
                short[cell_id] = max(m.quota - m.accepted, 0)

    hard = any(f.reason == "governance" for f in failures)
    return GateResult(
        passed=not failures,
        hard_fail=hard,
        failures=tuple(failures),
        short_cells=short,
        waived=waived,
        spec_version=report.spec_version,
    )


__all__ = [
    "DEFAULT_DISTINCT_N",
    "GateError",
    "GateFailure",
    "GateResult",
    "evaluate_gate",
    "threshold_values",
]
