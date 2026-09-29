"""M7 checkpoint: evaluation and release end to end on FAG. A judged run with repairs and
one stalled cell is measured, gated, refilled and released, and the release is checked
against the run it came from; a run that skipped L3 can't release. MockBackends only."""

import json
from datetime import datetime, timezone

import pytest
from test_m4_checkpoint import World
from test_pipeline import fag  # noqa: F401
from test_pipeline_release import FLAKY_ID, WAIVE, FlakyWorld, trusted_calibration

from sdgf.evaluation.gate import evaluate_gate
from sdgf.evaluation.metrics import metrics_for_run
from sdgf.evaluation.reports import (
    CARD,
    DATASET,
    GOVERNANCE,
    METRICS,
    PROVENANCE,
    record_sha256,
    verify_release,
)
from sdgf.pipeline import Pipeline

TARGET = 40
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
TFN = "000 000 000"  # fictional
# The attempt cap is a total per cell, so it's set to the largest quota at TARGET (5).
# FLAKY's cell (quota 3) drops FLAKY_DROPS slots and runs out of attempts one record short.
MAX_ATTEMPTS = 5
FLAKY_DROPS = 3


def flaky_world(fag):  # noqa: F811
    return FlakyWorld(failures=(fag.spec.validation.repair_tries + 1) * FLAKY_DROPS)


def release(fag, root, world, run_id, **kw):  # noqa: F811
    pipe = Pipeline(
        fag,
        root / "store",
        model_overrides=world.backends(),
        target_size=TARGET,
        calibration=trusted_calibration(fag),
        max_attempts_per_cell=MAX_ATTEMPTS,
        **kw,
    )
    return pipe, pipe.release(root / "releases", run_id, waive=WAIVE, now=NOW)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture(scope="module")
def released(fag, tmp_path_factory):  # noqa: F811
    world = flaky_world(fag)
    pipe, result = release(fag, tmp_path_factory.mktemp("m7"), world, "m7")
    return world, pipe, result


def test_fag_releases_after_repairs_and_one_refill_round(released):
    world, _, result = released
    assert world.failures == 0 and world.advised > 0  # L5 had advice wording to catch and repair
    assert result.released and result.stop_reason == "released"
    first, second = result.rounds
    assert first["outcome"] == "refill" and first["short_cells"] == {FLAKY_ID: 1}
    assert second["cells"] == [FLAKY_ID] and second["outcome"] == "released"
    manifest = verify_release(result.path)
    assert manifest["records"] == TARGET


def test_release_matches_the_run_it_came_from(fag, released):  # noqa: F811
    _, _, result = released
    accepted = result.run.read_jsonl("accepted")
    dataset = read_jsonl(result.path / DATASET)
    provenance = read_jsonl(result.path / PROVENANCE)
    assert len(dataset) == len(provenance) == TARGET
    assert not any(k.startswith("_") for r in dataset for k in r)
    assert [{k: v for k, v in r.items() if k != "_provenance"} for r in accepted] == dataset
    for record, prov, acc in zip(dataset, provenance, accepted, strict=True):
        assert prov["record_sha256"] == record_sha256(record)
        assert prov["provenance"] == acc["_provenance"]
        assert prov["provenance"]["spec_version"] == fag.spec_version
    # every label still comes from the cell and agrees with the policy rule
    rule = fag.hooks.label_rule
    assert all(r["label"] == rule(r) for r in dataset)
    assert sum(r["label"] for r in dataset) == TARGET // 2  # BREACH_RATE 0.5
    # no advice wording reached a non-breach record
    assert not any(
        "ideal for your business" in m["content"]
        for r in dataset
        if not r["label"]
        for m in r["messages"]
    )


def test_released_metrics_reproduce_from_the_run(fag, released):  # noqa: F811
    _, pipe, result = released
    stored = json.loads((result.path / METRICS).read_text())
    again = metrics_for_run(fag, result.run, calibration=pipe.calibration, seed=pipe.seed)
    assert stored["metrics"]["overall"] == json.loads(json.dumps(again.overall.to_dict()))
    assert stored["gate"]["passed"] and not stored["gate"]["hard_fail"]
    assert evaluate_gate(again, fag.spec.thresholds, waive=WAIVE).passed
    overall = again.overall
    assert overall.fidelity == 1.0 and overall.governance_violations == 0
    assert set(stored["gate"]["waived"]) == set(WAIVE)


def test_release_card_and_governance_report(fag, released):  # noqa: F811
    _, _, result = released
    card = (result.path / CARD).read_text()
    assert fag.spec_version in card and FLAKY_ID in card
    gov = json.loads((result.path / GOVERNANCE).read_text())
    assert gov["violations"] == 0
    endpoints = {(e["stage"], e["hosting"]) for e in gov["endpoints"]}
    assert {s for s, _ in endpoints} >= {"generator", "judge"}


def test_release_is_deterministic_per_seed(fag, released, tmp_path):  # noqa: F811
    _, again = release(fag, tmp_path, flaky_world(fag), "m7")
    _, _, first = released
    for name in (DATASET, PROVENANCE):
        assert (again.path / name).read_bytes() == (first.path / name).read_bytes()


def test_a_run_that_skipped_l3_cannot_release(fag, tmp_path):  # noqa: F811
    class LeakyWorld(World):
        def generate(self, call):
            r = json.loads(super().generate(call))
            r["messages"][0]["content"] += f" My TFN is {TFN}."
            return json.dumps(r)

    _, result = release(fag, tmp_path, LeakyWorld(), "leaky", layers=("L1", "L2", "L4", "L5", "L6"))
    # the release set is re-scanned at stage 4, so skipping L3 at accept time doesn't help
    assert not result.released and result.stop_reason == "hard_fail"
    assert result.gate.hard_fail and len(result.rounds) == 1
    assert "governance_violations_max" in result.gate.failing_metrics
    overall = result.metrics.overall
    assert overall.accepted == TARGET and overall.governance_violations == TARGET
    assert result.path.name == "shortfall.json"
    assert TFN not in result.path.read_text()
    assert not (tmp_path / "releases").exists()
