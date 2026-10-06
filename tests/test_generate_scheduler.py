import random

import pytest

from sdgf.generate.scheduler import Cell, Scheduler, SchedulerError
from sdgf.spec.schema import BudgetSection

CELLS = [
    Cell("a", {"label": True}, 5),
    Cell("b", {"label": False}, 3),
    Cell("c", {"label": False, "scope": "x"}, 8),
]
QUOTAS = {c.id: c.quota for c in CELLS}


def run(scheduler: Scheduler, accept) -> list[str]:
    """Drive the loop; accept(cell, n) decides each candidate. Returns cells handed out."""
    handed = []
    while (cell := scheduler.next_cell()) is not None:
        handed.append(cell.id)
        if accept(cell, len(handed)):
            scheduler.accept(cell.id)
        else:
            scheduler.reject(cell.id, "L2")
    return handed


def test_quotas_fill_exactly_with_no_failures():
    s = Scheduler(CELLS)
    handed = run(s, lambda c, n: True)
    assert s.counts() == QUOTAS
    assert len(handed) == sum(QUOTAS.values())
    assert s.done and s.stop_reason == "complete"
    assert s.short_cells() == {}


@pytest.mark.parametrize("seed", range(20))
def test_drops_cannot_change_final_distribution(seed):
    rng = random.Random(seed)
    # Drop rates differ per cell, the pattern that skews naive rejection sampling.
    drop = {"a": 0.8, "b": 0.1, "c": 0.5}
    s = Scheduler(CELLS)
    run(s, lambda c, n: rng.random() >= drop[c.id])
    assert s.counts() == QUOTAS
    for cid in QUOTAS:
        st = s.state(cid)
        assert st.attempts == st.accepted + st.rejected


def test_failed_record_is_regenerated_in_same_cell():
    s = Scheduler([Cell("a", {}, 1), Cell("b", {}, 1)])
    first = s.next_cell()
    s.reject(first.id, "L1")
    # a failure leaves its cell furthest below quota, so it comes straight back
    assert s.next_cell().id == first.id
    assert s.state(first.id).reject_reasons == {"L1": 1}


def test_picks_cell_furthest_below_quota():
    s = Scheduler([Cell("small", {}, 2), Cell("big", {}, 10)])
    s.restore_accepted({"small": 1, "big": 1})
    # big is at 10% fill, small at 50%
    assert s.next_cell().id == "big"


def test_full_cell_is_never_handed_out_again():
    s = Scheduler([Cell("a", {}, 1), Cell("b", {}, 4)])
    handed = run(s, lambda c, n: True)
    assert handed.count("a") == 1


def test_reservations_do_not_overfill():
    # With several candidates in flight, a cell never hands out more than it needs.
    s = Scheduler([Cell("a", {}, 2), Cell("b", {}, 1)])
    got = [s.next_cell() for _ in range(4)]
    assert [c.id if c else None for c in got].count("a") == 2
    assert got[-1] is None and s.stop_reason is None  # still in flight, not complete
    for c in got[:3]:
        s.accept(c.id)
    assert s.next_cell() is None and s.stop_reason == "complete"


def test_zero_quota_cell_is_skipped():
    s = Scheduler([Cell("empty", {}, 0), Cell("a", {}, 2)])
    run(s, lambda c, n: True)
    assert s.counts() == {"empty": 0, "a": 2}


def test_order_is_deterministic():
    a = run(Scheduler(CELLS), lambda c, n: n % 3 != 0)
    b = run(Scheduler(CELLS), lambda c, n: n % 3 != 0)
    assert a == b


def test_budget_max_candidates_stops_run():
    s = Scheduler(CELLS, BudgetSection(max_candidates=4))
    handed = run(s, lambda c, n: True)
    assert len(handed) == 4
    assert s.stop_reason == "budget:max_candidates"
    assert sum(s.short_cells().values()) == sum(QUOTAS.values()) - 4


def test_budget_stop_keeps_distribution_proportional():
    cells = [Cell("a", {}, 10), Cell("b", {}, 30)]
    s = Scheduler(cells, BudgetSection(max_candidates=20))
    run(s, lambda c, n: True)
    assert s.counts() == {"a": 5, "b": 15}


def test_budget_tokens_and_cost():
    s = Scheduler(CELLS, BudgetSection(max_tokens=100))
    s.next_cell()
    s.charge(tokens=100)
    assert s.next_cell() is None and s.stop_reason == "budget:max_tokens"

    s = Scheduler(CELLS, BudgetSection(max_cost_usd=0.5))
    s.next_cell()
    s.charge(cost_usd=0.25)
    assert s.next_cell() is not None
    s.charge(cost_usd=0.25)
    assert s.next_cell() is None and s.stop_reason == "budget:max_cost_usd"


def test_budget_seconds_uses_injected_clock():
    now = [0.0]
    s = Scheduler(CELLS, BudgetSection(max_seconds=10), clock=lambda: now[0])
    assert s.next_cell() is not None
    now[0] = 10.0
    assert s.next_cell() is None and s.stop_reason == "budget:max_seconds"


def test_max_attempts_stalls_an_impossible_cell():
    s = Scheduler([Cell("bad", {}, 2), Cell("ok", {}, 2)], max_attempts_per_cell=3)
    handed = run(s, lambda c, n: c.id == "ok")
    assert handed.count("bad") == 3
    assert s.counts() == {"bad": 0, "ok": 2}
    assert s.stalled_cells() == ["bad"]
    assert s.stop_reason == "stalled"
    assert s.short_cells() == {"bad": 2}


def test_attempts_per_quota_stalls_an_impossible_cell_in_proportion_to_its_quota():
    # quota 2 x 3 attempts per quota slot = 6 tries, then the cell stalls by name
    s = Scheduler([Cell("bad", {}, 2), Cell("ok", {}, 2)], attempts_per_quota=3)
    handed = run(s, lambda c, n: c.id == "ok")
    assert handed.count("bad") == 6
    assert s.stalled_cells() == ["bad"]
    assert s.stop_reason == "stalled"


def test_an_explicit_attempt_cap_wins_over_attempts_per_quota():
    s = Scheduler([Cell("bad", {}, 2)], max_attempts_per_cell=4, attempts_per_quota=3)
    assert run(s, lambda c, n: False).count("bad") == 4


def test_settle_errors():
    s = Scheduler(CELLS)
    with pytest.raises(SchedulerError, match="no reservation"):
        s.accept("a")
    with pytest.raises(SchedulerError, match="unknown cell"):
        s.reject("zzz")


def test_bad_cells():
    with pytest.raises(SchedulerError, match="duplicate"):
        Scheduler([Cell("a", {}, 1), Cell("a", {}, 1)])
    with pytest.raises(SchedulerError, match="at least one"):
        Scheduler([])
    with pytest.raises(SchedulerError, match="quota"):
        Cell("a", {}, -1)
    with pytest.raises(SchedulerError, match="resume count"):
        Scheduler([Cell("a", {}, 1)]).restore_accepted({"a": 2})


def test_resume_only_fills_remaining():
    s = Scheduler(CELLS)
    s.restore_accepted({"a": 5, "b": 1})
    handed = run(s, lambda c, n: True)
    assert "a" not in handed
    assert handed.count("b") == 2
    assert s.counts() == QUOTAS


def test_snapshot_is_plain_data():
    s = Scheduler(CELLS, BudgetSection(max_candidates=2), clock=lambda: 0.0)
    run(s, lambda c, n: n == 1)
    snap = s.snapshot()
    assert snap["stop_reason"] == "budget:max_candidates"
    assert snap["usage"]["candidates"] == 2
    assert sum(c["rejected"] for c in snap["cells"].values()) == 1
