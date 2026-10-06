"""Quota scheduler (FRAMEWORK_DESIGN.md §6.3, §7.6).

The scheduler owns *which cell* the next candidate is generated for. It hands out the
cell furthest below its quota, stops a cell once its quota is met, keeps a failed
candidate's retry in the same cell, and stops the run when spec.budget is exhausted.

Usage is a reserve/settle loop, so the same object works for sequential and (later)
concurrent generation:

    while (cell := scheduler.next_cell()) is not None:
        ...generate and validate...
        scheduler.accept(cell.id)   # or scheduler.reject(cell.id, reason)

next_cell() reserves one slot in the cell; accept() fills it, reject() frees it so the
cell is picked again. Because a cell never hands out more slots than it has left and
only accepts count, the final per-cell counts equal the quotas whatever gets dropped.

"Furthest below quota" is the lowest fill fraction (accepted + in flight) / quota, ties
broken by the larger absolute deficit and then by plan order. Fill fraction rather than
raw deficit keeps partially filled runs proportional to the plan, so a budget stop
leaves the distribution as close to the target as possible.

Token, cost and time are charged when a candidate settles, so a caller that reserves
many slots at once (a wave) asks wave_allowance() first: how many slots the remaining
budget covers at the average usage per settled candidate so far, and 1 before any has
settled. The allowance depends only on charged usage, not on how many workers run it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from sdgf.spec.schema import BudgetSection


class SchedulerError(ValueError):
    """Bad cells, or an accept/reject that doesn't match a reservation."""


@dataclass(frozen=True)
class Cell:
    id: str
    params: Mapping[str, Any]
    quota: int

    def __post_init__(self) -> None:
        if not self.id:
            raise SchedulerError("cell id must be non-empty")
        if self.quota < 0:
            raise SchedulerError(f"cell {self.id!r}: quota must be >= 0, got {self.quota}")


@dataclass
class CellState:
    cell: Cell
    accepted: int = 0
    in_flight: int = 0
    attempts: int = 0
    rejected: int = 0
    stalled: bool = False  # hit max_attempts_per_cell before filling
    reject_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return self.cell.quota - self.accepted

    @property
    def full(self) -> bool:
        return self.accepted >= self.cell.quota

    @property
    def open_slots(self) -> int:
        return self.cell.quota - self.accepted - self.in_flight


@dataclass
class Usage:
    candidates: int = 0
    tokens: int = 0
    cost_usd: float = 0.0


class Scheduler:
    def __init__(
        self,
        cells: Iterable[Cell],
        budget: BudgetSection | None = None,
        *,
        max_attempts_per_cell: int | None = None,
        attempts_per_quota: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._order: list[str] = []
        self._states: dict[str, CellState] = {}
        for cell in cells:
            if cell.id in self._states:
                raise SchedulerError(f"duplicate cell id {cell.id!r}")
            self._order.append(cell.id)
            self._states[cell.id] = CellState(cell)
        if not self._order:
            raise SchedulerError("scheduler needs at least one cell")
        if max_attempts_per_cell is not None and max_attempts_per_cell < 1:
            raise SchedulerError("max_attempts_per_cell must be >= 1")
        if attempts_per_quota is not None and attempts_per_quota < 1:
            raise SchedulerError("attempts_per_quota must be >= 1")
        # A cell's attempt cap: max_attempts_per_cell if given, else attempts_per_quota x
        # its quota, else none (only the budget stops a cell that never fills).
        self.attempts_per_quota = attempts_per_quota
        self.budget = budget or BudgetSection()
        self.max_attempts_per_cell = max_attempts_per_cell
        self.usage = Usage()
        self._clock = clock
        self._started = clock()
        self._created = self._started
        self.stop_reason: str | None = None
        # usage charged since this scheduler was created, for per-candidate averages
        self._charged = Usage()
        self._settled = 0

    # ── picking ──────────────────────────────────────────────────

    def next_cell(self) -> Cell | None:
        """Reserve a slot in the cell furthest below quota; None when done or out of budget."""
        if self.stop_reason is not None:
            return None
        over = self._budget_exceeded()
        if over:
            self.stop_reason = over
            return None
        best: tuple[float, int, int] | None = None
        best_id: str | None = None
        for pos, cid in enumerate(self._order):
            st = self._states[cid]
            if st.stalled or st.open_slots <= 0 or self._out_of_attempts(st):
                continue
            key = ((st.accepted + st.in_flight) / st.cell.quota, -st.open_slots, pos)
            if best is None or key < best:
                best, best_id = key, cid
        if best_id is None:
            if self.in_flight == 0:
                for st in self._states.values():
                    if not st.full and self._out_of_attempts(st):
                        st.stalled = True
                self.stop_reason = "stalled" if self.stalled_cells() else "complete"
            return None
        st = self._states[best_id]
        st.in_flight += 1
        st.attempts += 1
        self.usage.candidates += 1
        return st.cell

    # ── settling ─────────────────────────────────────────────────

    def accept(self, cell_id: str) -> None:
        st = self._settle(cell_id)
        st.accepted += 1

    def reject(self, cell_id: str, reason: str = "unspecified") -> None:
        """A dropped candidate; the slot goes back to the same cell."""
        st = self._settle(cell_id)
        st.rejected += 1
        st.reject_reasons[reason] = st.reject_reasons.get(reason, 0) + 1
        if self._out_of_attempts(st) and st.in_flight == 0 and not st.full:
            st.stalled = True

    def _out_of_attempts(self, st: CellState) -> bool:
        cap = self.max_attempts_per_cell
        if cap is None and self.attempts_per_quota is not None:
            cap = self.attempts_per_quota * st.cell.quota
        return cap is not None and st.attempts >= cap

    def _settle(self, cell_id: str) -> CellState:
        st = self._states.get(cell_id)
        if st is None:
            raise SchedulerError(f"unknown cell {cell_id!r}")
        if st.in_flight <= 0:
            raise SchedulerError(f"cell {cell_id!r} has no reservation to settle")
        st.in_flight -= 1
        self._settled += 1
        return st

    # ── budget ───────────────────────────────────────────────────

    def charge(self, *, tokens: int = 0, cost_usd: float = 0.0) -> None:
        if tokens < 0 or cost_usd < 0:
            raise SchedulerError("charges must be non-negative")
        self.usage.tokens += tokens
        self.usage.cost_usd += cost_usd
        self._charged.tokens += tokens
        self._charged.cost_usd += cost_usd

    def wave_allowance(self) -> int | None:
        """Slots the remaining token, cost and time budget covers; None if it sets none."""
        b = self.budget
        per = self._settled
        spent = [
            (b.max_tokens, self.usage.tokens, self._charged.tokens),
            (b.max_cost_usd, self.usage.cost_usd, self._charged.cost_usd),
            (b.max_seconds, self.elapsed(), self._clock() - self._created),
        ]
        spent = [(limit, used, charged) for limit, used, charged in spent if limit is not None]
        if not spent:
            return None
        if not per:
            return 1  # nothing to average yet: probe with one candidate
        allowance: int | None = None
        for limit, used, charged in spent:
            if charged <= 0:
                continue  # e.g. a free model: this budget can't run out
            n = int((limit - used) / (charged / per))
            allowance = n if allowance is None else min(allowance, n)
        return None if allowance is None else max(1, allowance)

    def elapsed(self) -> float:
        return self._clock() - self._started

    def _budget_exceeded(self) -> str | None:
        b = self.budget
        if b.max_candidates is not None and self.usage.candidates >= b.max_candidates:
            return "budget:max_candidates"
        if b.max_tokens is not None and self.usage.tokens >= b.max_tokens:
            return "budget:max_tokens"
        if b.max_cost_usd is not None and self.usage.cost_usd >= b.max_cost_usd:
            return "budget:max_cost_usd"
        if b.max_seconds is not None and self.elapsed() >= b.max_seconds:
            return "budget:max_seconds"
        return None

    # ── reporting ────────────────────────────────────────────────

    @property
    def in_flight(self) -> int:
        return sum(st.in_flight for st in self._states.values())

    @property
    def done(self) -> bool:
        return all(st.full for st in self._states.values())

    def state(self, cell_id: str) -> CellState:
        return self._states[cell_id]

    def counts(self) -> dict[str, int]:
        return {cid: self._states[cid].accepted for cid in self._order}

    def short_cells(self) -> dict[str, int]:
        """Cells below quota -> how many records they still need."""
        return {
            cid: self._states[cid].remaining for cid in self._order if not self._states[cid].full
        }

    def stalled_cells(self) -> list[str]:
        return [cid for cid in self._order if self._states[cid].stalled]

    def snapshot(self) -> dict[str, Any]:
        return {
            "stop_reason": self.stop_reason,
            "stalled_cells": self.stalled_cells(),
            "usage": {
                "candidates": self.usage.candidates,
                "tokens": self.usage.tokens,
                "cost_usd": self.usage.cost_usd,
                "seconds": self.elapsed(),
            },
            "cells": {
                cid: {
                    "quota": st.cell.quota,
                    "accepted": st.accepted,
                    "attempts": st.attempts,
                    "rejected": st.rejected,
                    "stalled": st.stalled,
                    "reject_reasons": dict(st.reject_reasons),
                }
                for cid, st in ((c, self._states[c]) for c in self._order)
            },
        }

    def restore_usage(
        self, *, candidates: int = 0, tokens: int = 0, cost_usd: float = 0.0, seconds: float = 0.0
    ) -> None:
        """Carry usage over from earlier rounds, so the budget covers the whole run."""
        if min(candidates, tokens, cost_usd, seconds) < 0:
            raise SchedulerError("restored usage must be non-negative")
        self.usage = Usage(candidates, tokens, cost_usd)
        self._started = self._clock() - seconds

    def restore_accepted(self, counts: Mapping[str, int]) -> None:
        """Resume: seed accepted counts (e.g. from an existing accepted stream)."""
        for cid, n in counts.items():
            st = self._states.get(cid)
            if st is None:
                raise SchedulerError(f"unknown cell {cid!r} in resume counts")
            if n < 0 or n > st.cell.quota:
                raise SchedulerError(f"cell {cid!r}: resume count {n} outside 0..{st.cell.quota}")
            st.accepted = n
