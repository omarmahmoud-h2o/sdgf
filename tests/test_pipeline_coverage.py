"""Stage 1 wired into the pipeline: the scheduler's cells and quotas come from the coverage plan."""

from collections import Counter

import pytest

from sdgf.coverage.plan import CoveragePlan, plan_stage_name
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ACCEPTED_STREAM, Pipeline, PipelineError
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import split
from test_coverage_plan import expansion_backend, make_task
from test_pipeline import NO_JUDGE, TARGET, fag, run, valid_backend  # noqa: F401

BREACH_RATE = 0.5


def stages(endpoints):
    return [e["stage"] for e in endpoints]


# ── FAG: fixed topics, no expansion ─────────────────────────────


@pytest.mark.parametrize("target", [1000, 20, 7, 3])
def test_fag_plan_label_balance_matches_breach_rate(fag, tmp_path, target):  # noqa: F811
    pipe = Pipeline(
        fag,
        tmp_path,
        model_overrides={"generator": valid_backend()},
        target_size=target,
        layers=NO_JUDGE,
    )
    assert "expansion" not in pipe.used_stages  # fixed axes: no keyword model call
    plan = pipe.plan()
    assert sum(c.quota for c in plan.cells) == target
    counts = plan.label_counts("label")
    assert abs(counts["true"] - target * BREACH_RATE) <= 1
    assert abs(counts["false"] - target * (1 - BREACH_RATE)) <= 1


def test_fag_run_takes_cells_and_quotas_from_the_plan(fag, tmp_path):  # noqa: F811
    pipe, result = run(fag, tmp_path, valid_backend())
    assert result.complete

    stage = plan_stage_name(fag, TARGET, 0)
    assert pipe.store.has_shared(fag.spec_version, stage)
    plan = CoveragePlan.from_dict(pipe.store.read_shared(fag.spec_version, stage))
    quotas = {c.id: c.quota for c in plan.cells}
    assert {c["id"]: c["quota"] for c in result.run.read_stage("cells")} == quotas
    assert result.counts == quotas

    accepted = result.run.read_jsonl(ACCEPTED_STREAM)
    labels = Counter(str(split(r)[0]["label"]).lower() for r in accepted)
    assert labels == plan.label_counts("label")
    assert stages(result.run.read_stage("spec")["models"]) == ["generator"]


def test_resume_keeps_the_runs_cells_even_if_the_shared_plan_changes(fag, tmp_path):  # noqa: F811
    pipe, first = run(fag, tmp_path, valid_backend())
    cells = first.run.read_stage("cells")
    stage = plan_stage_name(fag, TARGET, 0)
    tampered = pipe.store.read_shared(fag.spec_version, stage)
    tampered["cells"][0]["quota"] += 5
    pipe.store.write_shared(fag.spec_version, stage, tampered)

    _, resumed = run(fag, tmp_path, valid_backend())
    assert resumed.run.resumed
    assert resumed.run.read_stage("cells") == cells
    assert resumed.accepted == []  # already complete: nothing regenerated


def test_plan_seed_selects_its_own_plan_file(fag, tmp_path):  # noqa: F811
    pipe = Pipeline(
        fag,
        tmp_path,
        model_overrides={"generator": valid_backend()},
        target_size=TARGET,
        plan_seed=3,
        layers=NO_JUDGE,
    )
    pipe.plan()
    assert pipe.store.has_shared(fag.spec_version, plan_stage_name(fag, TARGET, 3))
    assert not pipe.store.has_shared(fag.spec_version, plan_stage_name(fag, TARGET, 0))


# ── keyword axes: the expansion model is built only to make a plan ──


def keyword_pipeline(compiled, store, **overrides):
    models = {"generator": MockBackend(["{}"]), **overrides}
    return Pipeline(compiled, store, model_overrides=models, layers=["L1", "L2"])


def test_keyword_axes_build_the_plan_with_the_expansion_model(tmp_path):
    compiled = make_task(tmp_path)
    store = ArtefactStore(tmp_path / "store")
    backend = expansion_backend()
    pipe = keyword_pipeline(compiled, store, expansion=backend)
    assert "expansion" in pipe.used_stages
    assert stages(pipe.models.endpoints()) == ["generator", "expansion"]

    plan = pipe.plan()
    assert len(backend.calls) == 2
    assert plan.keywords == {"keyword_expansion": ["cash_flow", "gst", "working_capital"]}
    assert sum(c.quota for c in plan.cells) == 24
    assert plan.label_counts("label") == {"true": 12, "false": 12}
    assert store.has_shared(compiled.spec_version, plan_stage_name(compiled, None, 0))


def test_cached_plan_needs_no_expansion_model(tmp_path):
    compiled = make_task(tmp_path)
    store = ArtefactStore(tmp_path / "store")
    built = keyword_pipeline(compiled, store, expansion=expansion_backend()).plan()

    again = keyword_pipeline(compiled, store)
    assert "expansion" not in again.used_stages
    assert stages(again.models.endpoints()) == ["generator"]  # D12: it gets no data this run
    assert again.plan().to_dict() == built.to_dict()

    with pytest.raises(PipelineError, match="expansion"):
        keyword_pipeline(compiled, store, expansion=expansion_backend())
