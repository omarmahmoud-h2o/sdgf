import json
from pathlib import Path

import pytest
import yaml

from sdgf.coverage.axes import (
    BLOOM_LEVELS,
    AxisError,
    ResolvedAxis,
    bloom_description,
    cross,
    keyword_sources,
    resolve_axis,
)
from sdgf.coverage.plan import (
    PLAN_STAGE,
    CoveragePlan,
    PlanError,
    apportion,
    assign_quotas,
    build_plan,
    filter_valid,
    load_or_build_plan,
    plan_stage_name,
)
from sdgf.models.mock import MockBackend
from sdgf.pipeline import fixed_axis_cells
from sdgf.spec.compile import compile_spec
from sdgf.spec.schema import Axis, parse_spec
from sdgf.store.artefacts import ArtefactStore

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"

THRESHOLDS = """\
thresholds:
  fidelity_min: 0.95
  kappa_min: 0.8
  coverage_min_cell_fill: 0.9
  balance_tolerance: 0.05
  distinct_n_min: 0.3
  self_bleu_max: 0.6
  semantic_diversity_min: 1.0
  residual_error_max: 0.05
  overlap_max: 0.8
  cost_per_record_max: 0.05
"""


def toy_yaml(coverage: str, target: int = 24) -> str:
    return f"""\
task:
  name: toyplan
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Write exam questions about fictional small-business finance.
output_schema: {{}}
rubric:
  verdict:
    values: [pass, fail]
seeds:
  path: seeds.jsonl
coverage:
  target_size: {target}
{coverage}
models:
  generator:
    backend: mock
    model: mock-1
  expansion:
    backend: mock
    model: mock-exp
validation:
  layers: [L1, L2]
{THRESHOLDS}"""


KEYWORD_COVERAGE = """\
  quota_policy: weighted
  axes:
    - name: keyword
      source: keyword_expansion
    - name: bloom_level
      source: bloom
      values: [Remember, Apply, Create]
    - name: label
      values: [true, false]
      weights: [0.8, 0.2]
  balance:
    label: {"true": 0.5, "false": 0.5}
  params:
    keyword_expansion:
      initial_count: 2
      iterations: 1
      per_iteration: 1
      directions: [advanced]
"""

SEEDS = '{"id": "s1", "label": true}\n{"id": "s2", "label": false}\n'

# Drops (Create, false): a "Create" question with a negative label is contradictory here.
HOOKS = """\
def sampler_constraints(cell, rng):
    if cell.get("bloom_level") == "Create" and cell.get("label") is False:
        return None
    return dict(cell)
"""


def make_task(tmp_path, coverage=KEYWORD_COVERAGE, target=24, hooks=HOOKS, extra=None):
    task = tmp_path / "task"
    task.mkdir(exist_ok=True)
    (task / "task.yaml").write_text(toy_yaml(coverage, target))
    (task / "seeds.jsonl").write_text(SEEDS)
    if hooks is not None:
        (task / "hooks.py").write_text(hooks)
    for name, text in (extra or {}).items():
        (task / name).write_text(text)
    return compile_spec(task)


def expansion_backend():
    return MockBackend(["cash_flow, gst", "working_capital"])


# ── Bloom axis ───────────────────────────────────────────────────


def test_bloom_levels_ported_in_order():
    assert list(BLOOM_LEVELS) == [
        "Remember",
        "Understand",
        "Apply",
        "Analyze",
        "Evaluate",
        "Create",
    ]
    assert bloom_description("Apply").startswith("Formulate instructions that demand practical use")
    with pytest.raises(AxisError, match="unknown Bloom level"):
        bloom_description("Memorise")


def test_bloom_axis_defaults_to_all_six_or_a_subset():
    full = resolve_axis(Axis(name="bloom", source="bloom"))
    assert full.values == tuple(BLOOM_LEVELS) and full.weights == (1.0,) * 6
    sub = resolve_axis(Axis(name="bloom", source="bloom", values=["Analyze", "Remember"]))
    assert sub.values == ("Analyze", "Remember")


def test_schema_rejects_bad_bloom_values_and_keyword_weights():
    with pytest.raises(ValueError, match="Bloom levels"):
        Axis(name="bloom", source="bloom", values=["Memorise"])
    with pytest.raises(ValueError, match="split evenly"):
        Axis(name="kw", source="keyword_expansion", values=["a", "b"], weights=[1, 2])


# ── axes ─────────────────────────────────────────────────────────


def test_keyword_axis_needs_its_list_and_puts_spec_values_first():
    axis = Axis(name="kw", source="keyword_expansion", values=["gst"])
    with pytest.raises(AxisError, match="needs the keyword_expansion keyword list"):
        resolve_axis(axis)
    got = resolve_axis(axis, {"keyword_expansion": ["cash_flow", "gst"]})
    assert got.values == ("gst", "cash_flow") and got.weights == (1.0, 1.0)


def test_weights_count_only_when_weighted():
    axis = Axis(name="label", values=[True, False], weights=[3, 1])
    assert resolve_axis(axis).weights == (1.0, 1.0)
    assert resolve_axis(axis, weighted=True).weights == (3, 1)


def test_axis_values_may_not_contain_the_id_separator():
    with pytest.raises(AxisError, match="may not contain"):
        resolve_axis(Axis(name="kw", values=["a|b"]))


def test_cross_ids_params_and_shares():
    axes = [
        ResolvedAxis("scope", "fixed", ("x", "y"), (0.6, 0.4)),
        ResolvedAxis("label", "fixed", (True, False), (0.5, 0.5)),
    ]
    combos = cross(axes)
    assert [c.id for c in combos] == ["x|true", "x|false", "y|true", "y|false"]
    assert combos[1].params == {"scope": "x", "label": False}
    assert combos[2].share == pytest.approx(0.2)


def test_keyword_sources():
    spec = parse_spec(yaml.safe_load(toy_yaml(KEYWORD_COVERAGE)))
    assert keyword_sources(spec.coverage) == ["keyword_expansion"]


# ── quotas ───────────────────────────────────────────────────────


def test_apportion_largest_remainder():
    assert apportion([1, 1, 1], 10) == [4, 3, 3]
    assert apportion([0.65, 0.35], 7) == [5, 2]
    assert apportion([0, 0], 0) == [0, 0]
    with pytest.raises(PlanError):
        apportion([0, 0], 3)


def test_balance_overrides_skewed_weights_within_one_record():
    axes = [
        ResolvedAxis("topic", "fixed", ("a", "b", "c"), (1.0, 1.0, 1.0)),
        ResolvedAxis("label", "fixed", (True, False), (0.9, 0.1)),
    ]
    combos = cross(axes)
    for target in (7, 11, 100):
        quotas = assign_quotas(combos, axes, {"label": {"true": 0.5, "false": 0.5}}, target)
        trues = sum(q for c, q in zip(combos, quotas) if c.params["label"])
        assert sum(quotas) == target
        assert abs(trues - target / 2) <= 0.5


def test_balance_errors():
    axes = [ResolvedAxis("label", "fixed", (True, False), (1.0, 1.0))]
    combos = cross(axes)
    with pytest.raises(PlanError, match="not on the axis"):
        assign_quotas(combos, axes, {"label": {"maybe": 1.0}}, 10)
    with pytest.raises(PlanError, match="no valid cell"):
        assign_quotas(combos[:1], axes, {"label": {"true": 0.5, "false": 0.5}}, 10)


def test_filter_valid_is_deterministic_per_seed_and_cell():
    axes = [ResolvedAxis("n", "fixed", tuple(range(20)), (1.0,) * 20)]
    combos = cross(axes)

    def coin(cell, rng):
        return None if rng.random() < 0.5 else cell

    first = filter_valid(combos, coin, seed=3)
    assert filter_valid(combos, coin, seed=3) == first
    assert first[1] and first[0]
    assert all(d["reason"] == "sampler_constraints" for d in first[1])


# ── FAG: fixed axes, no expansion ────────────────────────────────


def test_fag_plan_skips_expansion_and_balances_labels():
    compiled = compile_spec(FAG_DIR)
    for target in (1000, 20, 7):
        plan = build_plan(compiled, target_size=target)  # no backend: no model call
        assert not plan.keywords and not plan.dropped
        assert sum(c.quota for c in plan.cells) == target
        counts = plan.label_counts("label")
        assert abs(counts["true"] - target * 0.5) <= 1
        # Same grid and quotas as the pipeline's fixed-axis stand-in.
        stand_in = fixed_axis_cells(compiled.spec.coverage, target)
        assert [(c.id, c.quota) for c in plan.cells] == [(c.id, c.quota) for c in stand_in]


# ── keyword plan ─────────────────────────────────────────────────


def test_keyword_plan_crosses_keywords_bloom_and_label(tmp_path):
    compiled = make_task(tmp_path)
    backend = expansion_backend()
    plan = build_plan(compiled, backend=backend, seed=1)
    assert plan.keywords == {"keyword_expansion": ["cash_flow", "gst", "working_capital"]}
    assert len(backend.calls) == 2
    # 3 keywords x 3 Bloom levels x 2 labels, less the 3 dropped (Create, false) cells.
    assert len(plan.cells) == 15
    assert sorted(d["id"] for d in plan.dropped) == [
        "cash_flow|Create|false",
        "gst|Create|false",
        "working_capital|Create|false",
    ]
    assert sum(c.quota for c in plan.cells) == 24
    assert plan.label_counts("label") == {"true": 12, "false": 12}
    assert plan.cells[0].params == {
        "keyword": "cash_flow",
        "bloom_level": "Remember",
        "label": True,
    }


def test_keyword_plan_without_backend_raises(tmp_path):
    with pytest.raises(PlanError, match="need an expansion backend"):
        build_plan(make_task(tmp_path))


def test_constraints_dropping_everything_raises(tmp_path):
    compiled = make_task(tmp_path, hooks="def sampler_constraints(cell, rng):\n    return None\n")
    with pytest.raises(PlanError, match="dropped every combination"):
        build_plan(compiled, backend=expansion_backend())


def test_retrieval_axis_extends_the_expanded_keywords(tmp_path):
    corpus = {
        "text": "Cash flow forecasting for a fictional bakery covers invoices, GST and "
        "working capital buffers over a trading quarter."
    }
    coverage = KEYWORD_COVERAGE.replace("source: keyword_expansion", "source: retrieval") + (
        "    retrieval:\n      corpus: [corpus.jsonl]\n      iterations: 1\n      top_k: 1\n"
    )
    compiled = make_task(tmp_path, coverage=coverage, extra={"corpus.jsonl": json.dumps(corpus)})
    backend = MockBackend(["cash_flow, gst", "working_capital", "invoice_ageing"])
    store = ArtefactStore(tmp_path / "store")
    plan = build_plan(compiled, backend=backend, store=store)
    assert plan.keywords["keyword_expansion"] == ["cash_flow", "gst", "working_capital"]
    assert plan.keywords["retrieval"][-1] == "invoice_ageing"
    assert {c.params["keyword"] for c in plan.cells} == set(plan.keywords["retrieval"])
    with pytest.raises(PlanError, match="artefact store"):
        build_plan(compiled, backend=MockBackend(["a, b", "c", "d"]))


# ── caching ──────────────────────────────────────────────────────


def test_plan_cached_by_spec_version_and_reused_without_model_calls(tmp_path):
    compiled = make_task(tmp_path)
    store = ArtefactStore(tmp_path / "store")
    plan, built = load_or_build_plan(store, compiled, backend=expansion_backend())
    assert built
    path = store.root / compiled.spec_version / "shared" / "coverage_plan.json"
    assert path.is_file()

    idle = MockBackend(["unused"])
    again, built = load_or_build_plan(store, compiled, backend=idle)
    assert not built and idle.calls == []
    assert again.to_dict() == plan.to_dict()


def test_target_or_seed_override_gets_its_own_plan(tmp_path):
    compiled = make_task(tmp_path)
    assert plan_stage_name(compiled, None, 0) == PLAN_STAGE
    assert plan_stage_name(compiled, 24, 0) == PLAN_STAGE
    assert plan_stage_name(compiled, 10, 2) == "coverage_plan-t10-s2"
    store = ArtefactStore(tmp_path / "store")
    small, built = load_or_build_plan(store, compiled, target_size=10, backend=expansion_backend())
    assert built and sum(c.quota for c in small.cells) == 10
    assert not store.has_shared(compiled.spec_version, PLAN_STAGE)


def test_changed_spec_builds_a_new_plan(tmp_path):
    store = ArtefactStore(tmp_path / "store")
    first = make_task(tmp_path)
    load_or_build_plan(store, first, backend=expansion_backend())
    second = make_task(tmp_path, target=30)
    assert second.spec_version != first.spec_version
    plan, built = load_or_build_plan(store, second, backend=expansion_backend())
    assert built and sum(c.quota for c in plan.cells) == 30


def test_stored_plan_mismatch_raises(tmp_path):
    compiled = make_task(tmp_path)
    store = ArtefactStore(tmp_path / "store")
    plan = build_plan(compiled, backend=expansion_backend(), seed=5)
    store.write_shared(compiled.spec_version, PLAN_STAGE, plan.to_dict())
    with pytest.raises(PlanError, match="does not match"):
        load_or_build_plan(store, compiled)


def test_plan_round_trips_through_json(tmp_path):
    plan = build_plan(make_task(tmp_path), backend=expansion_backend())
    data = json.loads(json.dumps(plan.to_dict()))
    assert CoveragePlan.from_dict(data).to_dict() == plan.to_dict()
    with pytest.raises(PlanError, match="corrupt"):
        CoveragePlan.from_dict({"format": 1})


def test_plan_is_deterministic_per_seed(tmp_path):
    compiled = make_task(tmp_path)
    a = build_plan(compiled, backend=expansion_backend(), seed=4)
    b = build_plan(compiled, backend=expansion_backend(), seed=4)
    assert a.to_dict() == b.to_dict()
