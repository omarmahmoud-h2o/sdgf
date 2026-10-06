"""The stress tool classifies how a run ends: COMPLETED, CRASHED at a location, HUNG.
Mock backends only."""

from pathlib import Path

from sdgf.stress import SCENARIOS, describe, report, run_isolated, run_scenario

TESTS = Path(__file__).resolve().parent
FAG = str(TESTS.parent / "tasks" / "fag")
WORLD = f"{TESTS / 'cli_backends.py'}:fag_world"
BY_KEY = {s.key: s for s in SCENARIOS}


def test_a_run_that_absorbs_its_faults_is_completed_and_ok(tmp_path):
    r = run_scenario("2", FAG, WORLD, str(tmp_path))
    assert r["result"] == "COMPLETED"
    assert r["kept"] == 8
    assert BY_KEY["2"].ok(r)


def test_a_run_that_raises_is_crashed_at_the_sdgf_line_that_raised(tmp_path):
    r = run_scenario("11a", FAG, WORLD, str(tmp_path))
    assert r["result"] == "CRASHED"
    assert r["error_type"] == "AttributeError"
    assert r["where"].startswith("l2_rules.py:")
    assert not BY_KEY["11a"].ok(r)


def test_a_run_still_going_at_the_timeout_is_hung():
    r = run_isolated("2", FAG, WORLD, timeout=0.05)
    assert r["result"] == "HUNG"
    assert describe(r, 0.05) == "HUNG > 0.05s"


def test_the_report_marks_a_bug_with_what_should_have_happened():
    crashed = {
        "result": "CRASHED",
        "error_type": "HTTPError",
        "where": "generator.py:212",
        "message": "HTTP Error 429",
        "extra": {},
    }
    table, md = report("tasks/fag", [(BY_KEY["5"], crashed)], 60)
    assert "1 scenarios, 1 BUG" in table
    assert "CRASHED HTTPError at generator.py:212" in table
    assert "should retry a transient rate limit" in table
    assert "HTTP Error 429" in md
