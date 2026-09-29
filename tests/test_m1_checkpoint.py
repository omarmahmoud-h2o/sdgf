"""M1 checkpoint: the FAG spec compiles and every M1 module composes end to end.

Stands in for the M2 pipeline by hand: compile -> build mock models -> sample a recipe ->
treat each seed as a generated record -> attach provenance -> write, resume and read back.
"""

import random
from pathlib import Path

import jsonschema
import pytest

from sdgf.models.mock import MockBackend
from sdgf.models.registry import build_models
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore
from sdgf.store.provenance import ProvenanceBuilder, attach, split
from sdgf.tasktypes.classification_spans import turn_structure_errors
from sdgf.tasktypes.registry import REGISTRY

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
LAYERS = ("L1", "L2")


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def test_fag_compiles_with_hooks_seeds_and_task_type(fag):
    assert fag.name == "fag"
    assert len(fag.spec_version) == 64
    assert compile_spec(FAG_DIR).spec_version == fag.spec_version
    assert len(fag.seeds) == 6
    tt = REGISTRY.resolve(fag.spec.task)
    assert tt.name == "classification_spans"
    assert fag.spec.task.generation_mode == "label_first"
    for hook in ("label_rule", "sampler_constraints", "post_process"):
        assert getattr(fag.hooks, hook) is not None, hook


def test_sampled_recipes_agree_with_label_rule(fag):
    rng = random.Random(0)
    for scope in ("corps_act", "non_corps_act"):
        for label in (True, False):
            for _ in range(25):
                recipe = fag.hooks.sampler_constraints(
                    {"product_scope": scope, "label": label}, rng
                )
                assert recipe is not None
                assert fag.hooks.label_rule(recipe) is label


def test_seeds_flow_through_provenance_and_store(fag, tmp_path):
    models = build_models(
        fag.spec.models,
        overrides={"generator": MockBackend(["{}"]), "judge": MockBackend(["{}"])},
    )
    schema = REGISTRY.resolve(fag.spec.task).output_schema(fag.spec.output_schema)
    run = ArtefactStore(tmp_path).open_run(fag.spec_version, run_id="m1")

    written = []
    with run.jsonl("accepted") as out:
        for i, seed in enumerate(fag.seeds):
            record = fag.hooks.post_process(seed)
            jsonschema.validate(record, schema)
            assert turn_structure_errors(record, fag.spec.output_schema.turns) == []
            b = ProvenanceBuilder(
                fag.spec_version, f"cell-{i:04d}", seed=i, models=models.endpoints(), run_id="m1"
            )
            b.set_prompt(f"prompt {i}")
            for layer in LAYERS:
                b.add_layer_result(layer, "pass")
            prov = b.build()
            prov.check_accepted(LAYERS)
            out.write(attach(record, prov))
            written.append(record)

    resumed = ArtefactStore(tmp_path).open_run(fag.spec_version, run_id="m1")
    rows = resumed.read_jsonl("accepted")
    assert len(rows) == 6
    for row, record in zip(rows, written, strict=True):
        body, prov = split(row)
        assert body == record
        assert prov.spec_version == fag.spec_version
        assert prov.model("generator").hosting == "local"
