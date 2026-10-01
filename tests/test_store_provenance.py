import json

import pytest

from sdgf.models.mock import MockBackend
from sdgf.models.registry import build_models
from sdgf.spec.schema import ModelConfig, ModelsSection
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import (
    PROVENANCE_KEY,
    HumanDecision,
    LayerResult,
    ModelRef,
    Provenance,
    ProvenanceBuilder,
    ProvenanceError,
    ToolTraceEntry,
    attach,
    prompt_hash,
    split,
)

GEN = ModelRef(stage="generator", backend="mock", model="gen-mock", hosting="local")
JUDGE = ModelRef(stage="judge", backend="mock", model="judge-mock", hosting="provider_api")
RECORD = {
    "messages": [{"turn": 1, "role": "customer", "content": "Hi, Acme Test Pty Ltd here."}],
    "label": False,
    "spans": [],
}


def builder(**kw):
    args = dict(spec_version="abc123", cell_id="cell-0001", seed=7, models=[GEN, JUDGE])
    args.update(kw)
    return ProvenanceBuilder(**args)


def full_provenance():
    b = builder(run_id="run-1")
    b.set_prompt("generate a conversation")
    b.add_tool_call(
        ToolTraceEntry(
            tool="calculator",
            arguments={"expr": "1+1"},
            result=2,
            sensitivity="public",
            cached=True,
        )
    )
    b.add_layer_result("L1", "pass")
    b.add_layer_result("L2", "fail_repairable", ["span not found verbatim in turn 2"])
    b.start_repair()
    b.add_layer_result("L1", "pass")
    b.add_layer_result("L2", "pass")
    b.add_human_decision(
        HumanDecision(
            action="relabel",
            reviewer="reviewer-a",
            new_label=True,
            decided_at="2026-01-01T00:00:00Z",
        )
    )
    return b.build()


def test_prompt_hash_is_stable_and_distinct():
    assert prompt_hash("a") == prompt_hash("a")
    assert prompt_hash("a") != prompt_hash("b")
    assert prompt_hash("a").startswith("sha256:")


def test_builder_collects_every_field():
    p = full_provenance()
    assert p.spec_version == "abc123"
    assert p.cell_id == "cell-0001"
    assert p.seed == 7
    assert p.run_id == "run-1"
    assert p.prompt_hash == prompt_hash("generate a conversation")
    assert p.model("generator") == GEN
    assert p.model("judge").hosting == "provider_api"
    assert p.model("expansion") is None
    assert p.repair_count == 1
    assert [(r.layer, r.outcome, r.attempt) for r in p.layer_results] == [
        ("L1", "pass", 0),
        ("L2", "fail_repairable", 0),
        ("L1", "pass", 1),
        ("L2", "pass", 1),
    ]
    assert p.layer_results[1].errors == ("span not found verbatim in turn 2",)
    assert p.tool_trace[0].cached and p.tool_trace[0].sensitivity == "public"
    assert p.human_decisions[0].new_label is True


def test_round_trip_through_json():
    p = full_provenance()
    restored = Provenance.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored == p


def test_from_dict_rejects_unknown_format_and_malformed():
    d = full_provenance().to_dict()
    with pytest.raises(ProvenanceError, match="format"):
        Provenance.from_dict({**d, "format": 99})
    bad = dict(d)
    del bad["cell_id"]
    with pytest.raises(ProvenanceError, match="malformed"):
        Provenance.from_dict(bad)


def test_build_requires_prompt():
    with pytest.raises(ProvenanceError, match="prompt_hash"):
        builder().build()


@pytest.mark.parametrize("field", ["spec_version", "cell_id"])
def test_required_identifiers(field):
    b = builder(**{field: ""})
    b.set_prompt("p")
    with pytest.raises(ProvenanceError, match=field):
        b.build()


def test_generator_model_required_and_stages_unique():
    b = builder(models=[JUDGE])
    b.set_prompt("p")
    with pytest.raises(ProvenanceError, match="generator"):
        b.build()
    b = builder(models=[GEN, GEN])
    b.set_prompt("p")
    with pytest.raises(ProvenanceError, match="repeats"):
        b.build()


def test_models_from_registry_endpoints_record_hosting():
    models = ModelsSection(
        generator=ModelConfig(backend="mock", model="gen-mock", hosting="local"),
        judge=ModelConfig(backend="mock", model="judge-mock", hosting="provider_api"),
    )
    stages = build_models(models, overrides={"generator": MockBackend(["x"], model="gen-mock")})
    b = builder(models=stages.endpoints())
    b.set_prompt("p")
    p = b.build()
    assert p.model("generator") == ModelRef("generator", "mock", "gen-mock", "local")
    assert p.model("judge").hosting == "provider_api"


def test_invalid_values_rejected():
    with pytest.raises(ProvenanceError, match="hosting"):
        ModelRef(stage="generator", backend="mock", model="m", hosting="cloud")
    with pytest.raises(ProvenanceError, match="outcome"):
        LayerResult(layer="L1", outcome="ok")
    with pytest.raises(ProvenanceError, match="attempt"):
        LayerResult(layer="L1", outcome="pass", attempt=-1)
    with pytest.raises(ProvenanceError, match="action"):
        HumanDecision(action="approve", reviewer="r")


def test_relabel_requires_new_label_and_only_relabel():
    with pytest.raises(ProvenanceError, match="new_label"):
        HumanDecision(action="relabel", reviewer="r")
    with pytest.raises(ProvenanceError, match="new_label"):
        HumanDecision(action="accept", reviewer="r", new_label=True)
    assert HumanDecision(action="relabel", reviewer="r", new_label=False).new_label is False


def test_check_accepted_uses_final_attempt():
    full_provenance().check_accepted(["L1", "L2"])


def test_check_accepted_rejects_missing_or_failed_layer():
    p = full_provenance()
    with pytest.raises(ProvenanceError, match=r"missing layers \['L3'\]"):
        p.check_accepted(["L1", "L2", "L3"])
    b = builder()
    b.set_prompt("p")
    b.add_layer_result("L1", "pass")
    b.add_layer_result("L2", "fail_hard", ["tfn found"])
    with pytest.raises(ProvenanceError, match=r"failed layers \['L2'\]"):
        b.build().check_accepted(["L1", "L2"])


def test_check_accepted_rejects_repair_count_mismatch():
    b = builder()
    b.set_prompt("p")
    b.add_layer_result("L1", "pass")
    b.start_repair()  # a repair started but its layers were never recorded
    with pytest.raises(ProvenanceError, match="repair_count"):
        b.build().check_accepted(["L1"])


def test_attach_and_split_leave_record_untouched():
    p = full_provenance()
    attached = attach(RECORD, p)
    assert PROVENANCE_KEY not in RECORD
    assert attached[PROVENANCE_KEY]["spec_version"] == "abc123"
    bare, restored = split(json.loads(json.dumps(attached)))
    assert bare == RECORD
    assert restored == p


def test_attach_refuses_to_overwrite_and_split_requires_key():
    p = full_provenance()
    with pytest.raises(ProvenanceError, match="already"):
        attach(attach(RECORD, p), p)
    with pytest.raises(ProvenanceError, match="no"):
        split(RECORD)


def test_accepted_stream_round_trip(tmp_path):
    p = full_provenance()
    run = ArtefactStore(tmp_path).open_run("abc123", "run-1")
    with run.jsonl("accepted") as w:
        w.write(attach(RECORD, p))
    [row] = run.read_jsonl("accepted")
    assert split(row) == (RECORD, p)


def test_layer_results_carry_ballots_and_older_provenance_still_parses():
    from sdgf.store.provenance import Ballot

    b = builder()
    b.set_prompt("p")
    b.add_layer_result("L1", "pass")
    b.add_layer_result(
        "L6",
        "pass",
        ballots=[
            {"stage": "judge", "model": "judge-mock", "temperature": 0.7, "vote": "breach"},
            Ballot(vote=None, stage="judge", model="judge-mock", temperature=0.8),
        ],
    )
    p = b.build()
    assert p.layer_results[0].ballots == ()
    assert p.layer_results[1].ballots[0] == Ballot("breach", "judge", "judge-mock", 0.7)
    d = json.loads(json.dumps(p.to_dict()))
    assert Provenance.from_dict(d) == p
    # Provenance written before ballots existed has none and still reads back.
    for r in d["layer_results"]:
        del r["ballots"]
    assert all(r.ballots == () for r in Provenance.from_dict(d).layer_results)
