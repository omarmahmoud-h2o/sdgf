"""sdgf.jobs: run, resume, evaluate and release as background processes, with status,
cancel, timeout and restart. Real subprocesses with MockBackends from tests/cli_backends.py
(fag_slow keeps a job running long enough to stop it); no model or API calls."""

import json
import os
import signal
import time
from pathlib import Path

import pytest

from sdgf.jobs import (
    CANCELLED,
    FAILED,
    FINISHED,
    LOST,
    RUNNING,
    TIMED_OUT,
    JobManager,
    child_env,
)
from sdgf.service import RunOptions, Service, ServiceError, TaskRegistry

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
TARGET = 20
ENV = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}
NO_JUDGE = RunOptions(target_size=TARGET, layers=("L1", "L2", "L3", "L4"))
WAIVE = ("semantic_diversity_min", "kappa_min", "residual_error_max")
WAIT = 120  # seconds; a generous ceiling, jobs here take a few seconds


def manager(tmp_path, backends="cli_backends:fag_world", **kwargs):
    """An agent-facing manager: tasks by name only, releases and backends fixed here."""
    service = Service(tmp_path / "store", tasks=TaskRegistry([TASKS]), allow_paths=False)
    return JobManager(
        tmp_path / "store",
        service=service,
        releases=tmp_path / "releases",
        backends=backends,
        env=ENV,
        **kwargs,
    )


def slow(tmp_path, **kwargs):
    return manager(tmp_path, backends="cli_backends:fag_slow", **kwargs)


def accepted_at_least(jobs, job_id, n):
    deadline = time.monotonic() + WAIT
    while time.monotonic() < deadline:
        status = jobs.status(job_id)
        assert status["state"] == RUNNING, status
        if (status["progress"] or {}).get("accepted", 0) >= n:
            return status
        time.sleep(0.05)
    pytest.fail(f"job {job_id} did not reach {n} accepted records")


def finished(jobs, job_id):
    status = jobs.wait(job_id, timeout=WAIT)
    assert status["state"] == FINISHED, status
    return status


# ── jobs that finish ─────────────────────────────────────────────


def test_a_run_job_reports_progress_and_the_run(tmp_path):
    jobs = manager(tmp_path)
    started = jobs.start("run", "fag", options=NO_JUDGE)
    assert started["state"] in (RUNNING, FINISHED) and started["run_id"]
    done = finished(jobs, started["job_id"])
    assert done["status"] == "ok" and done["data"]["accepted"] == TARGET
    assert done["run_id"] == started["run_id"] == done["data"]["run_id"]
    assert done["progress"]["accepted"] == done["progress"]["target"] == TARGET
    assert [j["job_id"] for j in jobs.list()] == [started["job_id"]]


def test_evaluate_and_release_jobs(tmp_path):
    jobs = manager(tmp_path)
    run = finished(jobs, jobs.start("run", "fag", options=RunOptions(target_size=TARGET))["job_id"])

    evaluated = finished(jobs, jobs.start("evaluate", "fag")["job_id"])  # the latest run
    assert evaluated["run_id"] == run["run_id"]
    assert evaluated["status"] == "incomplete" and not evaluated["data"]["passed"]

    released = finished(
        jobs, jobs.start("release", "fag", run_id=run["run_id"], waive=WAIVE)["job_id"]
    )
    assert released["status"] == "ok" and released["data"]["released"]
    path = Path(released["data"]["path"])
    assert path.is_relative_to(tmp_path / "releases") and (path / "dataset.jsonl").is_file()


def test_a_step_that_fails_ends_the_job_with_its_error(tmp_path):
    failed = manager(tmp_path).start("evaluate", "fag")  # no runs yet
    status = manager(tmp_path).wait(failed["job_id"], timeout=WAIT)
    assert status["state"] == FAILED and status["error"]["code"] == "unknown_run"
    assert "log_tail" in status


# ── stopping and continuing ──────────────────────────────────────


def test_cancel_then_restart_continues_the_run(tmp_path):
    jobs = slow(tmp_path)
    job_id = jobs.start("run", "fag", options=NO_JUDGE)["job_id"]
    accepted_at_least(jobs, job_id, 3)
    cancelled = jobs.cancel(job_id)
    assert cancelled["state"] == CANCELLED and cancelled["error"]["hint"]
    before = cancelled["progress"]["accepted"]
    assert 3 <= before < TARGET
    assert jobs.cancel(job_id)["state"] == CANCELLED  # cancelling again changes nothing

    restarted = jobs.restart(job_id)
    assert restarted["restarted_from"] == job_id and restarted["run_id"] == cancelled["run_id"]
    done = finished(jobs, restarted["job_id"])
    assert done["data"]["accepted"] == TARGET
    assert done["data"]["accepted_this_invocation"] == TARGET - before  # nothing redone


def test_a_job_over_its_time_limit_times_out_and_can_continue(tmp_path):
    job_id = slow(tmp_path, timeout_s=1.0).start("run", "fag", options=NO_JUDGE)["job_id"]
    jobs = slow(tmp_path)  # restart takes this manager's limit
    timed_out = jobs.wait(job_id, timeout=WAIT)
    assert timed_out["state"] == TIMED_OUT and timed_out["error"]["code"] == "timed_out"
    assert timed_out["progress"]["accepted"] < TARGET
    assert finished(jobs, jobs.restart(job_id)["job_id"])["data"]["accepted"] == TARGET


def test_a_killed_worker_is_lost_and_can_continue(tmp_path):
    jobs = slow(tmp_path)
    job_id = jobs.start("run", "fag", options=NO_JUDGE)["job_id"]
    accepted_at_least(jobs, job_id, 2)
    pid = json.loads((jobs.root / job_id / "job.json").read_text())["pid"]
    os.killpg(pid, signal.SIGKILL)  # as if the machine went down
    lost = jobs.wait(job_id, timeout=WAIT)
    assert lost["state"] == LOST and lost["error"]["code"] == "worker_lost"
    assert finished(jobs, jobs.restart(job_id)["job_id"])["data"]["accepted"] == TARGET


def test_a_new_manager_sees_a_running_job(tmp_path):
    job_id = slow(tmp_path).start("run", "fag", options=NO_JUDGE)["job_id"]
    other = slow(tmp_path)  # e.g. the server restarted
    assert other.status(job_id)["state"] == RUNNING
    with pytest.raises(ServiceError) as e:
        other.restart(job_id)
    assert e.value.code == "job_running"
    assert finished(other, job_id)["data"]["accepted"] == TARGET


# ── limits and bad requests ──────────────────────────────────────


def test_running_jobs_are_limited(tmp_path):
    jobs = slow(tmp_path, max_running=1)
    first = jobs.start("run", "fag", options=NO_JUDGE)["job_id"]
    with pytest.raises(ServiceError) as e:
        jobs.start("run", "fag", options=NO_JUDGE)
    assert e.value.code == "too_many_jobs" and first in e.value.hint
    jobs.cancel(first)


def test_a_target_over_the_limit_fails_the_job(tmp_path):
    jobs = manager(tmp_path, max_target_size=5)
    status = jobs.wait(jobs.start("run", "fag", options=NO_JUDGE)["job_id"], timeout=WAIT)
    assert status["state"] == FAILED and status["error"]["code"] == "limit_exceeded"
    assert status["progress"] is None  # refused before any run directory was made


@pytest.mark.parametrize(
    "call, code",
    [
        (lambda jobs: jobs.start("run", "../tasks/fag"), "unknown_task"),
        (lambda jobs: jobs.start("build", "fag"), "bad_kind"),
        (lambda jobs: jobs.status("job-nope"), "unknown_job"),
        (lambda jobs: jobs.status("../store"), "unknown_job"),
    ],
)
def test_bad_requests_are_named(tmp_path, call, code):
    with pytest.raises(ServiceError) as e:
        call(manager(tmp_path))
    assert e.value.code == code


def test_a_release_needs_a_configured_release_directory(tmp_path):
    jobs = JobManager(tmp_path / "store", backends="cli_backends:fag_world", env=ENV)
    with pytest.raises(ServiceError) as e:
        jobs.start("release", TASKS / "fag")
    assert e.value.code == "no_releases"


def test_held_out_files_reach_the_worker_but_are_never_written(tmp_path):
    held = tmp_path / "secret-held-out-items.jsonl"
    held.write_text("".join((TASKS / "fag" / "seeds.jsonl").read_text().splitlines(True)[:3]))
    jobs = manager(tmp_path, held_out=[held])
    finished(jobs, jobs.start("run", "fag", options=NO_JUDGE)["job_id"])
    evaluated = finished(jobs, jobs.start("evaluate", "fag")["job_id"])
    assert "held_out" in evaluated["data"]["metrics"]["overall"]["overlap"]
    for path in (tmp_path / "store").rglob("*"):
        if path.is_file():
            assert held.name not in path.read_text(errors="ignore"), path


def test_a_worker_without_the_callers_environment_keeps_its_pythonpath(tmp_path, monkeypatch):
    # As from `python -m sdgf.mcp --dry-run-backends cli_backends:...`: no env passed, so
    # the backends module is found only through the operator's PYTHONPATH.
    monkeypatch.setenv("PYTHONPATH", ENV["PYTHONPATH"])
    monkeypatch.setenv("SDGF_TEST_API_KEY", "s3cret")
    env = child_env(inherit=False)
    assert "SDGF_TEST_API_KEY" not in env and str(ROOT / "tests") in env["PYTHONPATH"]
    jobs = JobManager(tmp_path / "store", backends="cli_backends:fag_world", inherit_env=False)
    done = finished(jobs, jobs.start("run", TASKS / "fag", options=NO_JUDGE)["job_id"])
    assert done["data"]["accepted"] == TARGET
