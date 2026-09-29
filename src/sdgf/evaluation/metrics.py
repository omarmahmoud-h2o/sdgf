"""Stage 4 dataset metrics (FRAMEWORK_DESIGN.md §6.5, §8), overall and per cell.

    fidelity          accepted records whose final L5 verdict passed (the judge agreed
                      with the intended label) ÷ accepted records L5 judged
    kappa             Cohen's κ of the judge against the gold set, from calibration
    coverage          accepted ÷ quota per cell; share of cells at quota; minimum fill
    balance           accepted share of each value of a balanced axis against its target;
                      max_deviation is the largest absolute gap in points
    diversity         distinct-n, self-BLEU, optional cluster entropy (diversity.py)
    error rate        rejections at each layer ÷ candidates generated (every try counts,
                      repairs included), from accepted provenance and the drop log
    residual error    estimated wrong labels left in the accepted set: for each accepted
                      record, the share of gold items where the judge gave the record's
                      label but the human didn't (1 - per-label precision), averaged
    governance        released records the governance scanners still flag (L3 re-run)
    overlap           maximum similarity of any accepted record to a seed or held-out item
    cost              tokens, dollars and seconds ÷ accepted records, overall and per model
                      stage (overall only), from the run's usage ledger (usage.json)
    yield             accepted ÷ candidates generated

Metrics nothing measured are None, never 0, so the gate can tell "not measured" from
"measured and fine". The gold set and run usage aren't per cell, so per-cell kappa and
cost are None; per-cell residual error uses the cell's own labels.

Candidate counts: an accepted record cost repair_count + 1 tries; a drop cost its
attempts. A try with no failed layer result before the final attempt is a generation
failure (no parseable record), counted under "generate". Drops written before the drop
log kept a per-try history count every try at the drop's final layer.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sdgf.coverage.axes import value_id
from sdgf.evaluation.diversity import (
    UNKNOWN_CELL,
    DiversityScores,
    Embedder,
    cell_of,
    score_texts,
)
from sdgf.generate.scheduler import Cell
from sdgf.judge.calibration import CalibrationResult
from sdgf.models.usage import USAGE_STAGE
from sdgf.store.provenance import PROVENANCE_KEY
from sdgf.validate.base import Layer, ValidationContext
from sdgf.validate.l3_governance import TOOL_TRACE_KEY
from sdgf.validate.l4_overlap import OverlapLayer, record_text
from sdgf.validate.repair import GENERATE_STAGE, Drop

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec
    from sdgf.store.artefacts import RunDir

JUDGE_LAYER = "L5"
LABEL_AXIS = "label"
OVERLAP_SOURCES = ("seed", "held_out")


class MetricsError(ValueError):
    """Metrics were asked for with inputs that don't fit together."""


def _bare(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != PROVENANCE_KEY}


def _prov(record: Mapping[str, Any]) -> Mapping[str, Any]:
    p = record.get(PROVENANCE_KEY)
    return p if isinstance(p, Mapping) else {}


def _ratio(num: float, den: float) -> float | None:
    return num / den if den else None


# ── per-record facts ─────────────────────────────────────────────


def record_tries(record: Mapping[str, Any]) -> list[str | None]:
    """The layer each try of an accepted record failed at, None for the final pass."""
    prov = _prov(record)
    final = int(prov.get("repair_count", 0))
    failed: dict[int, str] = {}
    for r in prov.get("layer_results", ()):
        if r.get("outcome") != "pass":
            failed.setdefault(int(r.get("attempt", 0)), r["layer"])
    return [failed.get(a, GENERATE_STAGE) for a in range(final)] + [None]


def judge_agreed(record: Mapping[str, Any]) -> bool | None:
    """The final L5 outcome of an accepted record; None when L5 didn't judge it."""
    prov = _prov(record)
    final = int(prov.get("repair_count", 0))
    outcomes = [
        r.get("outcome")
        for r in prov.get("layer_results", ())
        if r.get("layer") == JUDGE_LAYER and int(r.get("attempt", 0)) == final
    ]
    return outcomes[-1] == "pass" if outcomes else None


def _drop_dict(drop: Drop | Mapping[str, Any]) -> Mapping[str, Any]:
    return drop.to_dict() if isinstance(drop, Drop) else drop


def drop_tries(drop: Drop | Mapping[str, Any]) -> list[str]:
    """The layer each try of a dropped slot failed at."""
    d = _drop_dict(drop)
    attempts = int(d.get("attempts", 0))
    history = [h[0] for h in d.get("history") or ()]
    if len(history) == attempts:
        return history
    return [d["layer"]] * attempts


def residual_by_label(calibration: CalibrationResult) -> dict[str, float]:
    """1 - precision of the judge per judged label (keys are repr(label), as in confusion)."""
    out: dict[str, float] = {}
    for judged, humans in calibration.confusion.items():
        total = sum(humans.values())
        if total:
            out[judged] = 1.0 - humans.get(judged, 0) / total
    return out


# ── report ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class AxisBalance:
    target: dict[str, float]
    actual: dict[str, float]
    max_deviation: float | None  # None when there were no records to measure


@dataclass(frozen=True)
class Metrics:
    accepted: int
    quota: int | None = None
    fidelity: float | None = None
    judged: int = 0
    kappa: float | None = None
    fill: float | None = None  # accepted ÷ quota
    cells_at_quota: float | None = None  # overall only: share of cells at or above quota
    min_cell_fill: float | None = None
    short_cells: dict[str, int] = field(default_factory=dict)  # cell -> records missing
    balance: dict[str, AxisBalance] = field(default_factory=dict)
    balance_max_deviation: float | None = None
    diversity: DiversityScores | None = None
    candidates: int = 0  # tries generated, repairs included
    rejections: dict[str, int] = field(default_factory=dict)
    error_rates: dict[str, float] = field(default_factory=dict)
    residual_error: float | None = None
    governance_violations: int | None = None
    governance_codes: dict[str, int] = field(default_factory=dict)
    overlap: dict[str, float] = field(default_factory=dict)  # source -> max similarity
    overlap_max: float | None = None
    tokens_per_record: float | None = None
    cost_per_record: float | None = None
    seconds_per_record: float | None = None
    # stage -> calls, tokens, cost_usd (None if unpriced), plus the per-record figures
    usage_by_stage: dict[str, dict[str, Any]] = field(default_factory=dict)
    yield_: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["diversity"] = None if self.diversity is None else self.diversity.to_dict()
        d["yield"] = d.pop("yield_")
        return d


@dataclass(frozen=True)
class MetricsReport:
    spec_version: str | None
    overall: Metrics
    per_cell: dict[str, Metrics]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "overall": self.overall.to_dict(),
            "per_cell": {c: m.to_dict() for c, m in self.per_cell.items()},
        }


# ── computation ──────────────────────────────────────────────────


def _balance(
    records: Sequence[Mapping[str, Any]],
    targets: Mapping[str, Mapping[str, float]],
    params: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, AxisBalance], float | None]:
    out: dict[str, AxisBalance] = {}
    for axis, target in targets.items():
        values: list[str] = []
        for r in records:
            # The record's own label wins over the cell's, so a human relabel counts.
            if axis == LABEL_AXIS and LABEL_AXIS in r:
                values.append(value_id(r[LABEL_AXIS]))
            elif axis in (cell := params.get(cell_of(r), {})):
                values.append(value_id(cell[axis]))
        counts = Counter(values)
        n = len(values)
        actual = {v: counts[v] / n for v in sorted(set(target) | set(counts))} if n else {}
        dev = max(abs(actual.get(v, 0.0) - target.get(v, 0.0)) for v in actual) if actual else None
        out[axis] = AxisBalance(dict(target), actual, dev)
    devs = [b.max_deviation for b in out.values() if b.max_deviation is not None]
    return out, max(devs) if devs else None


def _cell_targets(cell: Cell, axes: Iterable[str]) -> dict[str, dict[str, float]]:
    return {a: {value_id(cell.params[a]): 1.0} for a in axes if a in cell.params}


def _planned_targets(cells: Sequence[Cell], axis: str) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for c in cells:
        if axis in c.params:
            counts[value_id(c.params[axis])] += c.quota
    total = sum(counts.values())
    return {v: n / total for v, n in counts.items()} if total else {}


def _governance(layer: Layer, records: Sequence[Mapping[str, Any]]) -> list[tuple[str, ...]]:
    """Per record: the issue codes the governance layer raises; empty means clean."""
    out: list[tuple[str, ...]] = []
    for r in records:
        prov = _prov(r)
        context = ValidationContext(
            cell_id=prov.get("cell_id"),
            extra={TOOL_TRACE_KEY: list(prov.get("tool_trace", ()))},
        )
        verdict = layer.check(_bare(r), context)
        out.append(() if verdict.passed else verdict.codes or ("governance",))
    return out


def _overlap(layer: OverlapLayer, records: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
    """Per record: source -> max similarity over every engine (seed and held-out only)."""
    out: list[dict[str, float]] = []
    for r in records:
        best: dict[str, float] = {}
        for (engine, source), match in layer.scores(_bare(r)).items():
            if source in OVERLAP_SOURCES:
                key = source if engine == "shingle" else f"{engine}:{source}"
                best[key] = max(best.get(key, 0.0), match.score)
        out.append(best)
    return out


def _max_overlap(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        for k, v in row.items():
            best[k] = max(best.get(k, 0.0), v)
    return dict(sorted(best.items()))


def compute_metrics(
    accepted: Sequence[Mapping[str, Any]],
    drops: Iterable[Drop | Mapping[str, Any]] = (),
    *,
    cells: Sequence[Cell] | None = None,
    balance: Mapping[str, Mapping[str, float]] | None = None,
    calibration: CalibrationResult | None = None,
    governance: Layer | None = None,
    overlap: OverlapLayer | None = None,
    usage: Mapping[str, float] | None = None,
    usage_by_stage: Mapping[str, Mapping[str, Any]] | None = None,
    embed: Embedder | None = None,
    k: int = 8,
    seed: int = 0,
    max_hypotheses: int | None = None,
    spec_version: str | None = None,
) -> MetricsReport:
    """Every §8 metric over accepted records (with provenance) and the drop log.

    cells are the run's cells and quotas; balance the target shares per axis (without
    it, the planned label shares from the quotas, when cells carry a label). governance
    and overlap are the L3 and L4 layers to re-run on the released set; usage is the
    run's usage totals (tokens, cost_usd, seconds) and usage_by_stage the same per model
    stage (models/usage.StageUsage.to_dict())."""
    drops = [_drop_dict(d) for d in drops]
    cells = list(cells or ())
    quotas = {c.id: c.quota for c in cells}
    by_id = {c.id: c for c in cells}
    params = {c.id: c.params for c in cells}
    if balance is None:
        planned = _planned_targets(cells, LABEL_AXIS)
        balance = {LABEL_AXIS: planned} if planned else {}

    by_cell: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(accepted):
        by_cell[cell_of(r)].append(i)
    unknown = sorted(c for c in by_cell if c not in quotas) if cells else []
    if unknown:
        raise MetricsError(f"accepted records name cells the plan doesn't have: {unknown}")

    agreed = [judge_agreed(r) for r in accepted]
    tries = [record_tries(r) for r in accepted]
    drop_cells = [d.get("cell_id") or UNKNOWN_CELL for d in drops]
    d_tries = [drop_tries(d) for d in drops]
    residual = residual_by_label(calibration) if calibration is not None else None
    fallback_residual = 1.0 - calibration.accuracy if calibration is not None else None
    gov = _governance(governance, accepted) if governance is not None else []
    overlaps = _overlap(overlap, accepted) if overlap is not None else []
    texts = [record_text(r) for r in accepted]
    vectors = [list(v) for v in embed(texts)] if embed is not None and texts else None

    def metrics(idx: Sequence[int], drop_idx: Sequence[int], label: str) -> Metrics:
        recs = [accepted[i] for i in idx]
        judged = [agreed[i] for i in idx if agreed[i] is not None]
        failed = Counter(layer for i in idx for layer in tries[i] if layer is not None) + Counter(
            layer for j in drop_idx for layer in d_tries[j]
        )
        n_tries = sum(len(tries[i]) for i in idx) + sum(len(d_tries[j]) for j in drop_idx)
        res = None
        if residual is not None and recs:
            per = [residual.get(repr(r.get(LABEL_AXIS)), fallback_residual) for r in recs]
            res = sum(per) / len(per)
        gov_codes = Counter(code for i in idx for code in gov[i]) if gov else Counter()
        over = _max_overlap(overlaps[i] for i in idx) if overlap is not None else {}
        return Metrics(
            accepted=len(recs),
            fidelity=_ratio(sum(judged), len(judged)),
            judged=len(judged),
            diversity=score_texts(
                [texts[i] for i in idx],
                max_hypotheses=max_hypotheses,
                rng=random.Random(f"{seed}:{label}"),
                vectors=None if vectors is None else [vectors[i] for i in idx],
                k=k,
                seed=seed,
            ),
            candidates=n_tries,
            rejections=dict(sorted(failed.items())),
            error_rates={layer: n / n_tries for layer, n in sorted(failed.items())},
            residual_error=res,
            governance_violations=sum(bool(gov[i]) for i in idx)
            if governance is not None
            else None,
            governance_codes=dict(sorted(gov_codes.items())),
            overlap=over,
            overlap_max=max(over.values()) if over else None,
            yield_=_ratio(len(recs), n_tries),
        )

    cell_ids = list(quotas) if cells else sorted(set(by_cell) | set(drop_cells))
    per_cell: dict[str, Metrics] = {}
    for cid in cell_ids:
        idx = by_cell.get(cid, [])
        drop_idx = [j for j, c in enumerate(drop_cells) if c == cid]
        m = metrics(idx, drop_idx, cid)
        recs = [accepted[i] for i in idx]
        if cid in quotas:
            bal, dev = _balance(recs, _cell_targets(by_id[cid], balance), params)
            fill = _ratio(len(idx), quotas[cid]) if quotas[cid] else 1.0
            short = {cid: quotas[cid] - len(idx)} if len(idx) < quotas[cid] else {}
            m = replace(
                m,
                quota=quotas[cid],
                fill=fill,
                min_cell_fill=fill,
                short_cells=short,
                balance=bal,
                balance_max_deviation=dev,
            )
        per_cell[cid] = m

    overall = metrics(range(len(accepted)), range(len(drops)), "overall")
    bal, dev = _balance(accepted, balance, params)
    usage = usage or {}
    n = len(accepted)
    updates: dict[str, Any] = {
        "kappa": calibration.kappa if calibration is not None else None,
        "balance": bal,
        "balance_max_deviation": dev,
        "tokens_per_record": _per(usage.get("tokens"), n),
        "cost_per_record": _per(usage.get("cost_usd"), n),
        "seconds_per_record": _per(usage.get("seconds"), n),
        "usage_by_stage": {
            stage: dict(u)
            | {
                "tokens_per_record": _per(u.get("tokens"), n),
                "cost_per_record": _per(u.get("cost_usd"), n),
            }
            for stage, u in sorted((usage_by_stage or {}).items())
        },
    }
    if cells:
        fills = [per_cell[c].fill for c in quotas]
        updates |= {
            "quota": sum(quotas.values()),
            "fill": _ratio(n, sum(quotas.values())),
            "cells_at_quota": sum(f is not None and f >= 1.0 for f in fills) / len(fills),
            "min_cell_fill": min(f for f in fills if f is not None),
            "short_cells": {c: s for m in per_cell.values() for c, s in m.short_cells.items()},
        }
    return MetricsReport(spec_version, replace(overall, **updates), per_cell)


def _per(total: float | None, n: int) -> float | None:
    return None if total is None or not n else total / n


# ── from a run directory ─────────────────────────────────────────


def metrics_for_run(
    compiled: CompiledSpec,
    run: RunDir,
    *,
    calibration: CalibrationResult | None = None,
    held_out_paths: Iterable[str | Path] | None = None,
    usage: Mapping[str, float] | None = None,
    embed: Embedder | None = None,
    k: int = 8,
    seed: int = 0,
    max_hypotheses: int | None = None,
) -> MetricsReport:
    """compute_metrics over a pipeline run's artefacts: accepted.jsonl, drops.jsonl,
    cells.json and, for usage unless passed, the usage ledger (usage.json, every
    invocation of the run) or else summary.json. L3 and L4 are rebuilt from the spec; the
    held-out check runs only when held_out_paths is passed."""
    from sdgf.validate.l3_governance import GovernanceLayer

    cells = [Cell(c["id"], dict(c["params"]), int(c["quota"])) for c in run.read_stage("cells")]
    usage_by_stage = None
    if usage is None and run.has_stage(USAGE_STAGE):
        ledger = run.read_stage(USAGE_STAGE)
        usage, usage_by_stage = ledger["total"], ledger["stages"]
    elif usage is None and run.has_stage("summary"):
        usage = run.read_stage("summary").get("usage")
    overlap = (
        OverlapLayer.from_spec(compiled, held_out_paths=held_out_paths)
        if compiled.spec.thresholds.overlap_max is not None
        else None
    )
    return compute_metrics(
        run.read_jsonl("accepted"),
        run.read_jsonl("drops"),
        cells=cells,
        balance=compiled.spec.coverage.balance or None,
        calibration=calibration,
        governance=GovernanceLayer.from_spec(compiled),
        overlap=overlap,
        usage=usage,
        usage_by_stage=usage_by_stage,
        embed=embed,
        k=k,
        seed=seed,
        max_hypotheses=max_hypotheses,
        spec_version=compiled.spec_version,
    )


__all__ = [
    "AxisBalance",
    "Metrics",
    "MetricsError",
    "MetricsReport",
    "compute_metrics",
    "drop_tries",
    "judge_agreed",
    "metrics_for_run",
    "record_tries",
    "residual_by_label",
]
