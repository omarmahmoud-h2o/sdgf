"""M5 checkpoint: stage 1 coverage planning end to end, for FAG (fixed axes) and a keyword task."""

import json
from collections import Counter

from sdgf.coverage.axes import BLOOM_LEVELS
from sdgf.coverage.plan import build_plan, load_or_build_plan
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ACCEPTED_STREAM, Pipeline
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import split
from test_coverage_plan import KEYWORD_COVERAGE, make_task
from test_pipeline import NO_JUDGE, fag, run, valid_backend  # noqa: F401

BREACH_RATE = 0.5


def test_fag_default_plan_is_balanced_and_needs_no_model(fag):  # noqa: F811
    plan = build_plan(fag)  # no backend: fixed axes make no keyword call
    target = fag.spec.coverage.target_size
    assert sum(c.quota for c in plan.cells) == target
    assert plan.keywords == {} and plan.keyword_history == {}
    counts = plan.label_counts("label")
    assert abs(counts["true"] - target * BREACH_RATE) <= 1
    assert abs(counts["false"] - target * (1 - BREACH_RATE)) <= 1


def test_fag_plan_is_deterministic_per_seed(fag):  # noqa: F811
    assert build_plan(fag, target_size=20).to_dict() == build_plan(fag, target_size=20).to_dict()


def test_fag_run_fills_the_plan_with_breach_balance(fag, tmp_path):  # noqa: F811
    pipe, result = run(fag, tmp_path, valid_backend(), target_size=40)
    assert result.complete
    labels = Counter(str(split(r)[0]["label"]).lower() for r in result.accepted)
    assert labels == {"true": 20, "false": 20}
    assert labels == pipe.plan().label_counts("label")
    assert len(result.run.read_jsonl(ACCEPTED_STREAM)) == 40


def test_keyword_task_plan_expands_retrieves_crosses_bloom_and_caches(tmp_path):
    corpus = {"text": "A fictional bakery tracks invoices, GST and working capital each quarter."}
    coverage = KEYWORD_COVERAGE.replace("source: keyword_expansion", "source: retrieval") + (
        "    retrieval:\n      corpus: [corpus.jsonl]\n      iterations: 1\n      top_k: 1\n"
    )
    compiled = make_task(tmp_path, coverage=coverage, extra={"corpus.jsonl": json.dumps(corpus)})
    store = ArtefactStore(tmp_path / "store")
    backend = MockBackend(["cash_flow, gst", "working_capital", "invoice_ageing"])

    plan, built = load_or_build_plan(store, compiled, backend=backend)
    assert built
    keywords = plan.keywords["retrieval"]
    assert "invoice_ageing" in keywords
    assert {c.params["keyword"] for c in plan.cells} == set(keywords)
    assert {c.params["bloom_level"] for c in plan.cells} <= set(BLOOM_LEVELS)
    assert not any(
        c.params["bloom_level"] == "Create" and c.params["label"] is False for c in plan.cells
    )
    assert plan.dropped  # sampler_constraints removed the contradictory combinations
    assert plan.label_counts("label") == {"true": 12, "false": 12}

    calls = len(backend.calls)
    again, built = load_or_build_plan(store, compiled, backend=backend)
    assert not built and len(backend.calls) == calls
    assert again.to_dict() == plan.to_dict()


def test_pipeline_with_cached_keyword_plan_builds_no_expansion_model(tmp_path):
    compiled = make_task(tmp_path)
    store = ArtefactStore(tmp_path / "store")
    load_or_build_plan(store, compiled, backend=MockBackend(["cash_flow, gst", "working_capital"]))
    pipe = Pipeline(
        compiled, store, model_overrides={"generator": MockBackend(["{}"])}, layers=["L1", "L2"]
    )
    assert "expansion" not in pipe.used_stages
    assert [e["stage"] for e in pipe.models.endpoints()] == ["generator"]
