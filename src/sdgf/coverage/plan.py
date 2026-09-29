"""Stage 1 coverage plan (FRAMEWORK_DESIGN.md §6.2): keywords × axes ─► cells × quotas.

    1 keywords     only when an axis has a keyword source: keywords.py expansion, then
                   retrieval.py extraction on top when an axis asks for retrieval
    2 crossing     every axis resolved to values (axes.py) and crossed into combinations
    3 constraints  sampler_constraints drops invalid combinations
    4 quotas       shares from the axis weights, fitted to coverage.balance, apportioned
                   to the target size

A task whose axes are all fixed (FAG) makes no model call and needs no backend.

Constraint checks run sampler_constraints once per combination with an rng seeded from
the plan seed and the cell id, so the same spec and seed always drop the same cells. A
hook whose None depends on its random draws, not on the pinned axis values, would make
that check a single sample; hooks should return None only for contradictory values.

Balance: shares are rescaled (iterative proportional fitting) until each balanced axis's
marginal matches its targets. The first balanced axis is then apportioned first, so its
per-value counts are within one record of target × size; each value's count is split
across its cells by share. A balanced-axis value the targets leave out gets quota 0.

The plan is written to the spec_version's shared artefact area as coverage_plan.json and
reused by every run until the spec changes; a target or seed other than the spec's
default gets its own file, so an override never overwrites the default plan.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from sdgf.coverage.axes import (
    Combo,
    ResolvedAxis,
    cross,
    keyword_sources,
    resolve_axes,
    value_id,
)
from sdgf.coverage.keywords import KeywordExpander
from sdgf.coverage.retrieval import RetrievalExtractor
from sdgf.generate.scheduler import Cell
from sdgf.models.base import ModelBackend
from sdgf.spec.compile import CompiledSpec
from sdgf.store.artefacts import ArtefactStore

PLAN_STAGE = "coverage_plan"
PLAN_FORMAT = 1
DEFAULT_SEED = 0
_FIT_ITERATIONS = 200
_FIT_TOLERANCE = 1e-12


class PlanError(ValueError):
    """The coverage plan can't be built, or a stored plan doesn't match the request."""


# ── quotas ───────────────────────────────────────────────────────


def apportion(weights: Sequence[float], total: int) -> list[int]:
    """Largest-remainder split of `total` in proportion to `weights`; ties go to plan order."""
    wsum = sum(weights)
    if total < 0:
        raise PlanError(f"cannot apportion a negative total {total}")
    if wsum <= 0:
        if total:
            raise PlanError("cannot apportion records over weights that sum to zero")
        return [0] * len(weights)
    exact = [w / wsum * total for w in weights]
    quotas = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - quotas[i]), i))
    for i in order[: total - sum(quotas)]:
        quotas[i] += 1
    return quotas


def _balance_groups(
    combos: Sequence[Combo], axis: str, targets: Mapping[str, float], values: Sequence[Any]
) -> dict[str, list[int]]:
    known = [value_id(v) for v in values]
    unknown = sorted(set(targets) - set(known))
    if unknown:
        raise PlanError(f"coverage.balance.{axis} names values not on the axis: {unknown}")
    groups: dict[str, list[int]] = {k: [] for k in known}
    for i, c in enumerate(combos):
        groups[value_id(c.params[axis])].append(i)
    return groups


def fit_shares(
    combos: Sequence[Combo],
    axes: Sequence[ResolvedAxis],
    balance: Mapping[str, Mapping[str, float]],
) -> list[float]:
    """Rescale combination shares so every balanced axis's marginal matches its targets."""
    shares = [c.share for c in combos]
    if not balance:
        return shares
    by_name = {a.name: a for a in axes}
    grouped = {
        axis: _balance_groups(combos, axis, targets, by_name[axis].values)
        for axis, targets in balance.items()
    }
    for axis, targets in balance.items():
        for v, t in targets.items():
            if t > 0 and sum(shares[i] for i in grouped[axis][v]) <= 0:
                raise PlanError(
                    f"coverage.balance.{axis}: value {v!r} has target {t} "
                    "but no valid cell to fill it"
                )
    for _ in range(_FIT_ITERATIONS):
        worst = 0.0
        for axis, targets in balance.items():
            total = sum(shares)
            if total <= 0:
                raise PlanError(f"coverage.balance.{axis}: the targets leave no valid cell")
            for v, members in grouped[axis].items():
                target = targets.get(v, 0.0)
                mass = sum(shares[i] for i in members) / total
                worst = max(worst, abs(mass - target))
                factor = 0.0 if mass <= 0 else target / mass
                for i in members:
                    shares[i] *= factor
        if worst < _FIT_TOLERANCE:
            break
    return shares


def assign_quotas(
    combos: Sequence[Combo],
    axes: Sequence[ResolvedAxis],
    balance: Mapping[str, Mapping[str, float]],
    target_size: int,
) -> list[int]:
    shares = fit_shares(combos, axes, balance)
    if sum(shares) <= 0:
        raise PlanError("no valid cell has a positive share")
    if not balance:
        return apportion(shares, target_size)
    first, targets = next(iter(balance.items()))
    by_name = {a.name: a for a in axes}
    groups = _balance_groups(combos, first, targets, by_name[first].values)
    names = list(groups)
    totals = apportion([targets.get(v, 0.0) for v in names], target_size)
    quotas = [0] * len(combos)
    for v, total in zip(names, totals):
        members = groups[v]
        for i, q in zip(members, apportion([shares[i] for i in members], total)):
            quotas[i] = q
    return quotas


# ── constraints ──────────────────────────────────────────────────


def cell_rng(seed: int, cell_id: str) -> random.Random:
    digest = hashlib.sha256(f"plan:{seed}:{cell_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def filter_valid(
    combos: Sequence[Combo],
    constraints: Callable[[dict[str, Any], random.Random], dict[str, Any] | None] | None,
    seed: int,
) -> tuple[list[Combo], list[dict[str, Any]]]:
    """(valid combinations, dropped ones with reasons) under the sampler_constraints hook."""
    if constraints is None:
        return list(combos), []
    valid, dropped = [], []
    for c in combos:
        if constraints(dict(c.params), cell_rng(seed, c.id)) is None:
            dropped.append({"id": c.id, "params": dict(c.params), "reason": "sampler_constraints"})
        else:
            valid.append(c)
    return valid, dropped


# ── keywords ─────────────────────────────────────────────────────


def gather_keywords(
    compiled: CompiledSpec,
    backend: ModelBackend | None,
    store: ArtefactStore | None,
    rng: random.Random,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Keyword lists per source the axes use, and the expansion history; ({}, {}) if none."""
    sources = keyword_sources(compiled.spec.coverage)
    if not sources:
        return {}, {}
    if backend is None:
        raise PlanError(f"axes with sources {sources} need an expansion backend")
    expanded = KeywordExpander.from_spec(compiled, backend).run(rng)
    keywords = {"keyword_expansion": list(expanded.keywords)}
    history: dict[str, Any] = {"keyword_expansion": expanded.to_dict()}
    if "retrieval" in sources:
        if store is None:
            raise PlanError("a retrieval axis needs an artefact store for its index")
        retrieved = RetrievalExtractor.from_spec(compiled, backend, store).run(
            expanded.keywords, rng
        )
        keywords["retrieval"] = list(retrieved.keywords)
        history["retrieval"] = retrieved.to_dict()
    return keywords, history


# ── plan ─────────────────────────────────────────────────────────


@dataclass
class CoveragePlan:
    spec_version: str
    target_size: int
    seed: int
    axes: list[ResolvedAxis]
    cells: list[Cell]
    keywords: dict[str, list[str]] = field(default_factory=dict)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    keyword_history: dict[str, Any] = field(default_factory=dict)

    def label_counts(self, axis: str) -> dict[str, int]:
        """Planned records per value of `axis`."""
        counts: dict[str, int] = {}
        for c in self.cells:
            k = value_id(c.params[axis])
            counts[k] = counts.get(k, 0) + c.quota
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PLAN_FORMAT,
            "spec_version": self.spec_version,
            "target_size": self.target_size,
            "seed": self.seed,
            "axes": [a.to_dict() for a in self.axes],
            "keywords": {k: list(v) for k, v in self.keywords.items()},
            "cells": [{"id": c.id, "params": dict(c.params), "quota": c.quota} for c in self.cells],
            "dropped": [dict(d) for d in self.dropped],
            "keyword_history": dict(self.keyword_history),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoveragePlan:
        try:
            if data["format"] != PLAN_FORMAT:
                raise PlanError(f"unsupported coverage plan format {data['format']!r}")
            return cls(
                spec_version=data["spec_version"],
                target_size=int(data["target_size"]),
                seed=int(data["seed"]),
                axes=[
                    ResolvedAxis(a["name"], a["source"], tuple(a["values"]), tuple(a["weights"]))
                    for a in data["axes"]
                ],
                cells=[Cell(c["id"], dict(c["params"]), int(c["quota"])) for c in data["cells"]],
                keywords={k: list(v) for k, v in data.get("keywords", {}).items()},
                dropped=[dict(d) for d in data.get("dropped", [])],
                keyword_history=dict(data.get("keyword_history", {})),
            )
        except (KeyError, TypeError) as e:
            raise PlanError(f"corrupt coverage plan: {e!r}") from None


def build_plan(
    compiled: CompiledSpec,
    *,
    target_size: int | None = None,
    seed: int = DEFAULT_SEED,
    backend: ModelBackend | None = None,
    store: ArtefactStore | None = None,
) -> CoveragePlan:
    coverage = compiled.spec.coverage
    target = coverage.target_size if target_size is None else target_size
    if target <= 0:
        raise PlanError(f"target size must be > 0, got {target}")
    keywords, history = gather_keywords(compiled, backend, store, random.Random(seed))
    axes = resolve_axes(coverage, keywords)
    valid, dropped = filter_valid(cross(axes), compiled.hooks.sampler_constraints, seed)
    if not valid:
        raise PlanError("sampler_constraints dropped every combination of the axes")
    quotas = assign_quotas(valid, axes, coverage.balance, target)
    return CoveragePlan(
        spec_version=compiled.spec_version,
        target_size=target,
        seed=seed,
        axes=axes,
        cells=[Cell(c.id, dict(c.params), q) for c, q in zip(valid, quotas)],
        keywords=keywords,
        dropped=dropped,
        keyword_history=history,
    )


def plan_stage_name(compiled: CompiledSpec, target_size: int | None, seed: int) -> str:
    target = compiled.spec.coverage.target_size if target_size is None else target_size
    if target == compiled.spec.coverage.target_size and seed == DEFAULT_SEED:
        return PLAN_STAGE
    return f"{PLAN_STAGE}-t{target}-s{seed}"


def load_or_build_plan(
    store: ArtefactStore,
    compiled: CompiledSpec,
    *,
    target_size: int | None = None,
    seed: int = DEFAULT_SEED,
    backend: ModelBackend | None = None,
) -> tuple[CoveragePlan, bool]:
    """The cached plan for this spec_version, target and seed, building it only if absent.

    Returns (plan, built). A cached plan is reused without any model call.
    """
    stage = plan_stage_name(compiled, target_size, seed)
    if store.has_shared(compiled.spec_version, stage):
        plan = CoveragePlan.from_dict(store.read_shared(compiled.spec_version, stage))
        target = compiled.spec.coverage.target_size if target_size is None else target_size
        if (plan.spec_version, plan.target_size, plan.seed) != (
            compiled.spec_version,
            target,
            seed,
        ):
            raise PlanError(f"stored plan {stage} does not match this spec, target and seed")
        return plan, False
    plan = build_plan(compiled, target_size=target_size, seed=seed, backend=backend, store=store)
    store.write_shared(compiled.spec_version, stage, plan.to_dict())
    return plan, True
