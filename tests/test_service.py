"""sdgf.service in-process: tasks by name, Outcome statuses and ServiceError codes, with
MockBackends from tests/cli_backends.py. No model or API calls. The CLI's own behaviour,
which runs on top of the service, stays covered by test_cli.py."""

import os
import shutil
from pathlib import Path

import pytest

from cli_backends import fag_unsure, fag_world
from sdgf.service import OPTIONS_STAGE, RunOptions, Service, ServiceError, TaskRegistry
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
TARGET = 20
NO_JUDGE = RunOptions(target_size=TARGET, layers=("L1", "L2", "L3", "L4"))
WAIVE = ("semantic_diversity_min", "kappa_min", "residual_error_max")


@pytest.fixture(scope="module")
def fag():
    return compile_spec(TASKS / "fag")


def agent_service(store=None, roots=(TASKS,)):
    """How an agent-facing adapter builds the service: names only, no paths."""
    return Service(store, tasks=TaskRegistry(roots), allow_paths=False)


# ── task registry ────────────────────────────────────────────────


def test_registry_names_the_task_directories():
    names = TaskRegistry([TASKS]).names()
    assert {"fag", "cfa"} <= set(names)
    assert names == sorted(names)


@pytest.mark.parametrize(
    "name", ["../tasks/fag", "fag/../fag", str(TASKS / "fag"), ".", "", "nope"]
)
def test_an_agent_cannot_reach_a_task_by_path(name):
    with pytest.raises(ServiceError) as e:
        agent_service().validate_spec(name)
    assert e.value.code == "unknown_task" and e.value.hint


def test_a_symlink_out_of_the_root_is_not_a_task(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    os.symlink(TASKS / "fag", root / "escape")
    assert TaskRegistry([root]).find("escape") is None


def test_the_first_root_wins(tmp_path):
    shutil.copytree(TASKS / "fag", tmp_path / "fag")
    assert TaskRegistry([tmp_path, TASKS]).find("fag") == (tmp_path / "fag").resolve()


def test_list_tasks_reports_a_broken_spec_without_hiding_the_rest(tmp_path):
    shutil.copytree(TASKS / "fag", tmp_path / "fag")
    shutil.copytree(TASKS / "fag", tmp_path / "broken")
    task = tmp_path / "broken" / "task.yaml"
    task.write_text(task.read_text().replace("thresholds:", "thresholdz:", 1))
    listed = {t["name"]: t for t in agent_service(roots=[tmp_path]).list_tasks().data["tasks"]}
    assert listed["fag"]["task_type"] == "classification_spans"
    assert "thresholds" in listed["broken"]["error"]


def test_list_tasks_needs_a_registry():
    with pytest.raises(ServiceError) as e:
        Service().list_tasks()
    assert e.value.code == "no_registry"


# ── describing a task ────────────────────────────────────────────


def test_validate_and_describe_by_name(fag):
    service = agent_service()
    assert service.validate_spec("fag").data["spec_version"] == fag.spec_version
    described = service.describe_task("fag")
    assert described.status == "ok"
    data = described.data
    assert data["spec_version"] == fag.spec_version
    assert data["description"] == fag.spec.task.description
    assert [a["name"] for a in data["axes"]] == [a.name for a in fag.spec.coverage.axes]
    assert data["models"]["generator"]["model"] == fag.spec.models.generator.model
    assert data["thresholds"]["governance_violations_max"] == 0


def test_calls_that_need_a_store_say_so():
    with pytest.raises(ServiceError) as e:
        agent_service().plan("fag", target_size=TARGET)
    assert e.value.code == "no_store"


# ── runs ─────────────────────────────────────────────────────────


def test_run_evaluate_release_by_name(fag, tmp_path):
    service = agent_service(tmp_path / "store")
    run = service.run("fag", "r1", options=RunOptions(target_size=TARGET), backends=fag_world)
    assert run.status == "ok" and run.data["accepted"] == TARGET and run.data["complete"]
    saved = ArtefactStore(tmp_path / "store").open_run(fag.spec_version, "r1")
    assert saved.read_stage(OPTIONS_STAGE)["target_size"] == TARGET

    evaluated = service.evaluate("fag", backends=fag_world)  # the latest run
    assert evaluated.data["run_id"] == "r1"
    assert evaluated.status == "incomplete" and not evaluated.data["passed"]
    assert service.evaluate("fag", backends=fag_world, waive=WAIVE).status == "ok"

    released = service.release("fag", tmp_path / "releases", "r1", backends=fag_world, waive=WAIVE)
    assert released.status == "ok" and released.data["released"]
    assert (Path(released.data["path"]) / "dataset.jsonl").is_file()


def test_saved_options_cannot_change(tmp_path):
    service = agent_service(tmp_path)
    service.run("fag", "r1", options=NO_JUDGE, backends=fag_world)
    with pytest.raises(ServiceError) as e:
        service.resume("fag", "r1", options=RunOptions(seed=7), backends=fag_world)
    assert e.value.code == "options_clash" and "seed=0" in str(e.value)


def test_unknown_runs_are_named(tmp_path):
    service = agent_service(tmp_path)
    with pytest.raises(ServiceError) as e:
        service.resume("fag", backends=fag_world)
    assert e.value.code == "unknown_run"
    service.run("fag", "r1", options=NO_JUDGE, backends=fag_world)
    with pytest.raises(ServiceError) as e:
        service.evaluate("fag", "r2", backends=fag_world)
    assert e.value.code == "unknown_run"


def test_a_plan_awaiting_approval_is_a_status_not_an_error(tmp_path):
    shutil.copytree(TASKS / "fag", tmp_path / "tasks" / "fag")
    task = tmp_path / "tasks" / "fag" / "task.yaml"
    text = task.read_text()
    task.write_text(text.replace("approve_coverage_plan: false", "approve_coverage_plan: true"))
    service = agent_service(tmp_path / "store", roots=[tmp_path / "tasks"])
    options = RunOptions(target_size=TARGET, max_attempts_per_cell=20)

    pending = service.run("fag", "h1", options=options, backends=fag_unsure)
    assert pending.status == "approval_pending"
    assert "awaiting approval" in pending.data["message"]

    planned = service.plan("fag", target_size=TARGET, approve_as="A. Reviewer")
    assert planned.data["approved"] is True
    assert service.run("fag", "h1", options=options, backends=fag_unsure).status == "ok"


def test_a_backends_factory_must_return_backends(tmp_path):
    with pytest.raises(ServiceError, match="non-backends"):
        agent_service(tmp_path).run(
            "fag", "r1", options=NO_JUDGE, backends=lambda compiled: {"generator": object()}
        )
    assert not any(tmp_path.rglob("run.json"))  # no stale run directory
