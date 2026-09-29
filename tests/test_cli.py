"""The sdgf command line, run as a subprocess on the FAG spec with MockBackends supplied by
tests/cli_backends.py through --backends. No model or API calls."""

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from sdgf.cli import EXIT_APPROVAL, EXIT_ERROR, EXIT_INCOMPLETE, EXIT_OK, OPTIONS_STAGE
from sdgf.evaluation.reports import verify_release
from sdgf.judge.calibration import CalibrationStore
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore
from test_evaluation_gate import calibration

ROOT = Path(__file__).resolve().parents[1]
FAG_DIR = ROOT / "tasks" / "fag"
TARGET = "20"
NO_JUDGE = ["--layers", "L1", "L2", "L3", "L4"]
WAIVE = ["--waive", "semantic_diversity_min"]  # no embedder in tests
REVIEWER = "Test Reviewer"


def sdgf(*args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])
    proc = subprocess.run(
        [sys.executable, "-m", "sdgf.cli", *map(str, args)],
        capture_output=True,
        text=True,
        env=env,
        cwd=cwd or ROOT,
        timeout=300,
    )
    out = json.loads(proc.stdout) if proc.stdout.strip() else None
    return proc.returncode, out, proc.stderr


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def trust(fag, store):
    # The judge built from cli_backends' MockBackend is mock:mock.
    cal = dataclasses.replace(calibration(fag), judge_id="mock:mock")
    CalibrationStore(ArtefactStore(store)).save(cal)


# ── packaging ────────────────────────────────────────────────────


def test_console_script_is_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["scripts"] == {"sdgf": "sdgf.cli:main"}


def test_help_lists_every_subcommand():
    proc = subprocess.run(
        [sys.executable, "-m", "sdgf.cli", "--help"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert proc.returncode == 0
    for name in ("validate-spec", "plan", "run", "resume", "evaluate", "review", "release"):
        assert name in proc.stdout


# ── validate-spec and plan ───────────────────────────────────────


def test_validate_spec_reports_the_compiled_spec(fag):
    code, out, _ = sdgf("validate-spec", FAG_DIR)
    assert code == EXIT_OK
    assert out["spec_version"] == fag.spec_version
    assert out["task_type"] == "classification_spans"
    assert out["generation_mode"] == "label_first" and out["seeds"] == len(fag.seeds)


def test_validate_spec_fails_clearly_on_a_broken_spec(tmp_path):
    shutil.copytree(FAG_DIR, tmp_path / "fag")
    task = tmp_path / "fag" / "task.yaml"
    task.write_text(task.read_text().replace("thresholds:", "thresholdz:", 1))
    code, out, err = sdgf("validate-spec", tmp_path / "fag")
    assert code == EXIT_ERROR and out is None
    assert err.startswith("sdgf: error:") and "thresholds" in err


def test_plan_builds_once_then_reuses(fag, tmp_path):
    code, out, _ = sdgf("plan", FAG_DIR, "--store", tmp_path, "--target-size", TARGET)
    assert code == EXIT_OK and out["built"] is True
    assert sum(c["quota"] for c in out["cells"]) == int(TARGET)
    assert Path(out["path"]).is_file() and out["approval_required"] is False
    code, again, _ = sdgf("plan", FAG_DIR, "--store", tmp_path, "--target-size", TARGET)
    assert code == EXIT_OK and again["built"] is False and again["cells"] == out["cells"]


# ── run, resume, evaluate, release ───────────────────────────────


def test_run_fills_every_cell_and_saves_its_options(fag, tmp_path):
    code, out, _ = sdgf(
        "run", FAG_DIR, "--store", tmp_path, "--run-id", "r1", "--target-size", TARGET,
        *NO_JUDGE, "--backends", "cli_backends:fag_world",
    )  # fmt: skip
    assert code == EXIT_OK, out
    assert out["complete"] and out["accepted"] == int(TARGET) and out["run_id"] == "r1"
    assert out["layers"] == NO_JUDGE[1:]  # the plugin's judge is ignored, not an error
    # usage and cost per accepted record; FAG prices its local generator at 0
    assert out["usage"]["calls"] >= int(TARGET) and out["usage"]["tokens"] > 0
    assert out["cost_per_accepted"] == 0.0
    run = ArtefactStore(tmp_path).open_run(fag.spec_version, "r1")
    assert run.read_stage(OPTIONS_STAGE)["target_size"] == int(TARGET)
    assert [m["stage"] for m in run.read_stage("spec")["models"]] == ["generator"]


def test_resume_finishes_a_stalled_run_with_its_saved_options(fag, tmp_path):
    code, out, _ = sdgf(
        "run", FAG_DIR, "--store", tmp_path, "--run-id", "r1", "--target-size", TARGET,
        *NO_JUDGE, "--max-attempts-per-cell", "2", "--backends", "cli_backends:fag_broken",
    )  # fmt: skip
    assert code == EXIT_INCOMPLETE and out["stop_reason"] == "stalled"
    assert out["accepted"] == int(TARGET) - 2
    code, out, err = sdgf(
        "resume", FAG_DIR, "--store", tmp_path, "--seed", "7",
        "--backends", "cli_backends:fag_world",
    )  # fmt: skip
    assert code == EXIT_ERROR and "seed=0" in err  # saved options can't change
    code, out, _ = sdgf(
        "resume", FAG_DIR, "--store", tmp_path, "--backends", "cli_backends:fag_world"
    )
    assert code == EXIT_OK and out["run_id"] == "r1"  # the latest run by default
    assert out["complete"] and out["accepted"] == int(TARGET)
    assert out["accepted_this_invocation"] == 2


def test_resume_without_a_run_is_an_error(tmp_path):
    code, _, err = sdgf("resume", FAG_DIR, "--store", tmp_path)
    assert code == EXIT_ERROR and "no runs" in err


@pytest.fixture(scope="module")
def judged(fag, tmp_path_factory):
    store = tmp_path_factory.mktemp("cli") / "store"
    trust(fag, store)
    code, out, err = sdgf(
        "run", FAG_DIR, "--store", store, "--run-id", "j1", "--target-size", TARGET,
        "--backends", "cli_backends:fag_world",
    )  # fmt: skip
    assert code == EXIT_OK, err
    return store


def test_evaluate_gates_a_run_without_writing(fag, judged):
    run_dir = ArtefactStore(judged).open_run(fag.spec_version, "j1").path
    before = sorted(p.name for p in run_dir.iterdir())
    args = ("evaluate", FAG_DIR, "--store", judged, "--run-id", "j1")
    code, out, _ = sdgf(*args, "--backends", "cli_backends:fag_world", *WAIVE)
    assert code == EXIT_OK and out["passed"], out["gate"]["failures"]
    assert out["metrics"]["overall"]["kappa"] == 0.9  # the stored calibration was used
    code, out, _ = sdgf(*args, "--backends", "cli_backends:fag_world")
    assert code == EXIT_INCOMPLETE and "semantic_diversity_min" in out["gate"]["failing_metrics"]
    assert sorted(p.name for p in run_dir.iterdir()) == before


def test_release_writes_a_verified_release(fag, tmp_path):
    trust(fag, tmp_path / "store")
    code, out, err = sdgf(
        "release", FAG_DIR, "--store", tmp_path / "store", "--releases", tmp_path / "rel",
        "--run-id", "rel1", "--target-size", TARGET, "--backends", "cli_backends:fag_world",
        *WAIVE,
    )  # fmt: skip
    assert code == EXIT_OK, err
    assert out["released"] and out["stop_reason"] == "released"
    manifest = verify_release(out["path"])
    assert manifest["records"] == int(TARGET) and manifest["spec_version"] == fag.spec_version


def test_release_that_fails_the_gate_writes_a_shortfall(fag, tmp_path):
    # no calibration and nothing waived: kappa and diversity can't be measured
    code, out, _ = sdgf(
        "release", FAG_DIR, "--store", tmp_path / "store", "--releases", tmp_path / "rel",
        "--run-id", "rel1", "--target-size", TARGET, "--backends", "cli_backends:fag_world",
    )  # fmt: skip
    assert code == EXIT_INCOMPLETE and not out["released"]
    assert out["stop_reason"] == "no_short_cells" and "kappa_min" in out["failing_metrics"]
    assert Path(out["path"]).is_file() and not (tmp_path / "rel").exists()


# ── HITL: plan approval and review ───────────────────────────────


@pytest.fixture
def hitl_task(tmp_path):
    shutil.copytree(FAG_DIR, tmp_path / "fag")
    task = tmp_path / "fag" / "task.yaml"
    text = task.read_text()
    for flag in ("approve_coverage_plan", "review_flagged"):
        assert f"{flag}: false" in text
        text = text.replace(f"{flag}: false", f"{flag}: true")
    task.write_text(text)
    return tmp_path / "fag"


def test_plan_approval_then_review_resolution(hitl_task, tmp_path):
    store = tmp_path / "store"
    run = (
        "run", hitl_task, "--store", store, "--run-id", "h1", "--target-size", TARGET,
        "--max-attempts-per-cell", "20", "--backends", "cli_backends:fag_unsure",
    )  # fmt: skip
    code, out, err = sdgf(*run)
    assert code == EXIT_APPROVAL and out is None and "awaiting approval" in err

    code, _, err = sdgf("plan", hitl_task, "--store", store, "--target-size", TARGET, "--approve")
    assert code == EXIT_ERROR and "--reviewer" in err
    code, out, _ = sdgf(
        "plan", hitl_task, "--store", store, "--target-size", TARGET,
        "--approve", "--reviewer", REVIEWER,
    )  # fmt: skip
    assert code == EXIT_OK and out["approved"] is True

    code, out, err = sdgf(*run)
    assert code == EXIT_OK, err
    assert out["drops_by_layer"].get("L5", 0) > 0  # low-confidence verdicts went to review

    code, listed, _ = sdgf("review", hitl_task, "--store", store, "list")
    pending = listed["pending"]
    assert listed["run_id"] == "h1" and len(pending) == out["drops_by_layer"]["L5"]
    assert {p["code"] for p in pending} == {"low_confidence"}

    first, second = pending[0]["id"], pending[1]["id"]
    resolve = ("review", hitl_task, "--store", store, "resolve")
    code, res, _ = sdgf(*resolve, first, "--action", "accept", "--reviewer", REVIEWER)
    assert code == EXIT_OK and res["decision"]["action"] == "accept"
    assert res["label"] == pending[0]["intended_label"]
    flipped = json.dumps(not pending[1]["intended_label"])
    code, res, _ = sdgf(
        *resolve, second, "--action", "relabel", "--reviewer", REVIEWER, "--new-label", flipped
    )
    assert code == EXIT_OK and res["label"] is (not pending[1]["intended_label"])
    gold = [json.loads(line) for line in Path(res["gold"]).read_text().splitlines()]
    assert [g["_gold"]["action"] for g in gold] == ["accept", "relabel"]

    code, _, err = sdgf(*resolve, first, "--action", "reject", "--reviewer", REVIEWER)
    assert code == EXIT_ERROR and "already resolved" in err
    code, listed, _ = sdgf("review", hitl_task, "--store", store, "list")
    assert len(listed["pending"]) == len(pending) - 2 and listed["resolved"] == 2


# ── bad --backends ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "ref, message",
    [
        ("no_colon", "expected MODULE:ATTR"),
        ("cli_backends:missing", "is not a callable"),
        ("cli_backends:not_callable", "is not a callable"),
        ("no_such_module_xyz:f", "no_such_module_xyz"),
        ("missing_file.py:f", "no file"),
    ],
)
def test_bad_backends_reference_is_an_error(tmp_path, ref, message):
    code, _, err = sdgf(
        "run", FAG_DIR, "--store", tmp_path, "--target-size", TARGET, *NO_JUDGE, "--backends", ref
    )
    assert code == EXIT_ERROR and message in err
    assert not tmp_path.exists() or not any(tmp_path.rglob("run.json"))  # no stale run dir


def test_backends_from_a_file_path(fag, tmp_path):
    code, out, err = sdgf(
        "run", FAG_DIR, "--store", tmp_path, "--target-size", TARGET, *NO_JUDGE,
        "--backends", ROOT / "tests" / "cli_backends.py:fag_world",
    )  # fmt: skip
    assert code == EXIT_OK, err
    assert out["complete"]
