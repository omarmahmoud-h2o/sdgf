"""Stage 5 outputs: the versioned release on pass and the shortfall report on fail.
FAG runs use mock backends only; no model or API calls."""

import json
from datetime import datetime, timezone

import pytest

from sdgf.evaluation import reports
from sdgf.evaluation.gate import GateFailure, GateResult, evaluate_gate
from sdgf.evaluation.metrics import metrics_for_run
from sdgf.evaluation.reports import (
    ReportError,
    data_endpoints,
    publish,
    record_sha256,
    verify_release,
    write_release,
    write_shortfall,
)
from sdgf.models.mock import MockBackend
from sdgf.pipeline import Pipeline
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import PROVENANCE_KEY
from test_evaluation_gate import THRESHOLDS, calibration, cell, report
from test_m4_checkpoint import World
from test_pipeline import fag, fag_reply, recipe_from_prompt  # noqa: F401

NOW = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
FILES = {
    "dataset.jsonl",
    "provenance.jsonl",
    "dataset_card.md",
    "governance_report.json",
    "metrics.json",
    "manifest.json",
}


@pytest.fixture(scope="module")
def judged(fag, tmp_path_factory):  # noqa: F811
    """A judged FAG run whose judge is an external (provider_api) endpoint, gated to pass."""
    world = World()
    backends = world.backends()
    backends["judge"] = MockBackend(world.judge, model="judge-api", hosting="provider_api")
    pipe = Pipeline(fag, tmp_path_factory.mktemp("store"), model_overrides=backends, target_size=20)
    run = pipe.run("rel").run
    metrics = metrics_for_run(fag, run, calibration=calibration(fag), usage={"cost_usd": 0.1})
    gate = evaluate_gate(metrics, fag.spec.thresholds, waive=["semantic_diversity_min"])
    assert gate.passed, gate.failures
    return run, metrics, gate


@pytest.fixture(scope="module")
def release(fag, judged, tmp_path_factory):  # noqa: F811
    run, metrics, gate = judged
    return write_release(tmp_path_factory.mktemp("releases"), fag, run, metrics, gate, now=NOW)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def read_json(path):
    return json.loads(path.read_text())


def failed_gate(**kw):
    base = dict(
        passed=False,
        hard_fail=False,
        failures=(
            GateFailure("coverage_min_cell_fill", "below_min", 0.5, 0.9, "a"),
            GateFailure("kappa_min", "not_measured", None, 0.7),
        ),
        short_cells={"a": 5},
        spec_version="sv",
    )
    return GateResult(**(base | kw))


# ── release ──────────────────────────────────────────────────────


def test_release_directory_is_versioned_and_complete(fag, judged, release):  # noqa: F811
    run, _, _ = judged
    assert {p.name for p in release.iterdir()} == FILES
    assert release.parent.name == fag.spec.task.name
    assert release.name == f"{fag.spec.task.version}-{fag.spec_version[:12]}-{run.run_id}"
    manifest = verify_release(release)
    assert manifest["spec_version"] == fag.spec_version and manifest["run_id"] == run.run_id
    assert manifest["records"] == 20 and manifest["created_at"] == NOW.isoformat()
    assert set(manifest["files"]) == FILES - {"manifest.json"}


def test_dataset_holds_bare_records_and_provenance_lines_up(judged, release):
    run, _, _ = judged
    accepted = run.read_jsonl("accepted")
    data = read_jsonl(release / "dataset.jsonl")
    prov = read_jsonl(release / "provenance.jsonl")
    assert len(data) == len(prov) == len(accepted) == 20
    assert all(PROVENANCE_KEY not in r for r in data)
    for i, (record, line, original) in enumerate(zip(data, prov, accepted)):
        assert record == {k: v for k, v in original.items() if k != PROVENANCE_KEY}
        assert line["index"] == i and line["record_sha256"] == record_sha256(record)
        assert line["provenance"] == original[PROVENANCE_KEY]


def test_governance_report_lists_external_endpoints(fag, release):  # noqa: F811
    gov = read_json(release / "governance_report.json")
    assert gov["spec_version"] == fag.spec_version
    assert gov["violations"] == 0 and gov["governance_gate"] == "pass"
    stages = {e["stage"]: e for e in gov["endpoints"]}
    assert stages["generator"]["hosting"] == "local"
    assert stages["judge"] == {
        "stage": "judge",
        "backend": "mock",
        "model": "judge-api",
        "hosting": "provider_api",
    }
    assert gov["external_endpoints"] == [stages["judge"]]
    assert gov["held_out_check"] is False
    assert "pii_rules" in gov["profile"]


def test_metrics_report_carries_metrics_and_gate(fag, judged, release):  # noqa: F811
    _, metrics, gate = judged
    m = read_json(release / "metrics.json")
    assert m["metrics"] == json.loads(json.dumps(metrics.to_dict()))
    assert m["gate"]["passed"] is True and m["gate"]["waived"] == ["semantic_diversity_min"]


def test_dataset_card_describes_task_models_and_gate(fag, release):  # noqa: F811
    card = (release / "dataset_card.md").read_text()
    assert card.startswith(f"# {fag.spec.task.name} ")
    assert fag.spec_version in card and "Records: 20" in card
    assert "| judge | mock | judge-api | provider_api |" in card
    assert "External endpoints that received data: 1" in card
    assert "| fidelity_min |" in card and "| semantic_diversity_min |" in card
    assert "waived" in card and "**label**" in card


def test_release_is_never_overwritten(fag, judged, release):  # noqa: F811
    run, metrics, gate = judged
    before = verify_release(release)
    with pytest.raises(ReportError, match="already exists"):
        write_release(release.parent.parent, fag, run, metrics, gate, now=NOW)
    assert verify_release(release) == before


def test_explicit_version_and_tamper_detection(fag, judged, tmp_path):  # noqa: F811
    run, metrics, gate = judged
    path = write_release(tmp_path, fag, run, metrics, gate, version="v1")
    assert path == tmp_path / fag.spec.task.name / "v1"
    (path / "dataset.jsonl").write_text("{}\n")
    with pytest.raises(ReportError, match="altered"):
        verify_release(path)
    with pytest.raises(ReportError, match="invalid release version"):
        write_release(tmp_path, fag, run, metrics, gate, version="../escape")


def test_failed_gate_never_releases(fag, judged, tmp_path):  # noqa: F811
    run, metrics, _ = judged
    for gate in (failed_gate(spec_version=fag.spec_version), failed_gate(hard_fail=True)):
        with pytest.raises(ReportError, match="did not pass"):
            write_release(tmp_path, fag, run, metrics, gate)
    assert not any(tmp_path.iterdir())


def test_mismatched_spec_version_is_refused(fag, judged, tmp_path):  # noqa: F811
    run, metrics, gate = judged
    other = GateResult(passed=True, hard_fail=False, spec_version="other")
    with pytest.raises(ReportError, match="different spec_version"):
        write_release(tmp_path, fag, run, metrics, other)


def test_a_failure_mid_write_leaves_no_release(fag, judged, tmp_path, monkeypatch):  # noqa: F811
    run, metrics, gate = judged

    def boom(*a, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(reports, "dataset_card", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        write_release(tmp_path, fag, run, metrics, gate)
    assert list((tmp_path / fag.spec.task.name).iterdir()) == []


def test_data_endpoints_merge_run_models_with_provenance():
    run_models = [{"stage": "generator", "backend": "vllm", "model": "g", "hosting": "local"}]
    records = [
        {
            PROVENANCE_KEY: {
                "models": [
                    {"stage": "generator", "backend": "vllm", "model": "g", "hosting": "local"},
                    {
                        "stage": "fallback_judge",
                        "backend": "anthropic",
                        "model": "r",
                        "hosting": "provider_api",
                        "version": None,
                    },
                ]
            }
        },
        {"no": "provenance"},
    ]
    assert data_endpoints(run_models, records) == [
        {
            "stage": "fallback_judge",
            "backend": "anthropic",
            "model": "r",
            "hosting": "provider_api",
        },
        {"stage": "generator", "backend": "vllm", "model": "g", "hosting": "local"},
    ]


# ── shortfall ────────────────────────────────────────────────────


def test_shortfall_names_failing_metrics_and_short_cells(tmp_path):
    run = ArtefactStore(tmp_path).open_run("sv", "r1")
    metrics = report(a=cell(accepted=5), b=cell())
    path = write_shortfall(run, metrics, failed_gate())
    assert path == run.path / "shortfall.json"
    data = run.read_stage("shortfall")
    assert data["passed"] is False and data["hard_fail"] is False
    assert data["failing_metrics"] == ["coverage_min_cell_fill", "kappa_min"]
    assert data["short_cells"] == {"a": 5} and data["records_missing"] == 5
    assert data["accepted"] == 15 and data["quota"] == 20 and data["run_id"] == "r1"
    assert {(f["metric"], f["reason"], f["cell"]) for f in data["failures"]} == {
        ("coverage_min_cell_fill", "below_min", "a"),
        ("kappa_min", "not_measured", None),
    }
    md = (run.path / "shortfall.md").read_text()
    assert "| coverage_min_cell_fill | below_min | `a` | 0.5 | 0.9 |" in md
    assert "| kappa_min | not_measured | overall | not measured | 0.7 |" in md
    assert "| `a` | 5 |" in md and "Hard fail" not in md


def test_shortfall_flags_hard_fail(tmp_path):
    run = ArtefactStore(tmp_path).open_run("sv", "r1")
    gate = failed_gate(
        hard_fail=True,
        failures=(GateFailure("governance_violations_max", "governance", 2.0, 0.0),),
        short_cells={},
    )
    write_shortfall(run, report(), gate)
    assert run.read_stage("shortfall")["hard_fail"] is True
    md = (run.path / "shortfall.md").read_text()
    assert "Hard fail" in md and "none" in md


def test_shortfall_refuses_a_passed_gate(tmp_path):
    run = ArtefactStore(tmp_path).open_run("sv", "r1")
    passed = evaluate_gate(report(), THRESHOLDS)
    with pytest.raises(ReportError, match="no shortfall"):
        write_shortfall(run, report(), passed)


# ── publish ──────────────────────────────────────────────────────


def test_publish_releases_on_pass_and_reports_shortfall_on_fail(fag, judged, tmp_path):  # noqa: F811
    run, metrics, gate = judged
    released = publish(tmp_path / "rel", fag, run, metrics, gate, version="pub")
    assert released == tmp_path / "rel" / fag.spec.task.name / "pub"
    verify_release(released)

    def reply(call):
        return json.dumps(fag_reply(recipe_from_prompt(call.prompt)))

    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": MockBackend(reply)},
        target_size=20,
        layers=("L1", "L2", "L3", "L4"),
    )
    unjudged = pipe.run("nojudge").run
    m = metrics_for_run(fag, unjudged)
    failed = evaluate_gate(m, fag.spec.thresholds)
    path = publish(tmp_path / "rel2", fag, unjudged, m, failed)
    assert path == unjudged.path / "shortfall.json"
    assert "fidelity_min" in unjudged.read_stage("shortfall")["failing_metrics"]
    assert not (tmp_path / "rel2").exists()
