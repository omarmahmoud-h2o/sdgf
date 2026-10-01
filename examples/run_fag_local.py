"""Run the FAG use case through all six layers against a local OpenAI-compatible server.

    cd sdgf && PYTHONPATH=src python examples/run_fag_local.py

The generator and the judge are the same model. task.yaml is left untouched: its
backends are replaced by model_overrides pointing at API_BASE.
"""

import json
import logging
from collections import Counter

from sdgf.models.openai_compat import OpenAICompatBackend
from sdgf.pipeline import Pipeline
from sdgf.evaluation.gate import evaluate_gate
from sdgf.evaluation.metrics import metrics_for_run

API_BASE = "http://127.0.0.1:8082/v1"
MODEL = "Qwen/Qwen3.5-4B"  # must be an id listed by GET {API_BASE}/models
STORE = "/tmp/sdgf-local/store"
RELEASES = "/tmp/sdgf-local/releases"
TARGET_SIZE = 20
RUN_ID = "local-smoke"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def backend(stop=None):
    # No api_key_env: a local server needs no key. Sampling (temperature, max_tokens)
    # still comes from models.<stage> in task.yaml.
    params = {"api_base": API_BASE, "timeout": 600}
    if stop:
        params["stop"] = stop
    return OpenAICompatBackend(MODEL, hosting="local", params=params)


pipe = Pipeline(
    "tasks/fag",
    STORE,
    model_overrides={
        "generator": backend(stop=["<|im_end|>", "<|endoftext|>"]),
        "judge": backend(),
    },
    target_size=TARGET_SIZE,
    seed=0,
    layers=["L1", "L2", "L3", "L4", "L5", "L6"],
)
print("layers run:", pipe.layers, "skipped:", pipe.skipped_layers)

# Stages 0-3: plan, generate, and run every candidate through L1 -> L6.
result = pipe.run(RUN_ID)
print("\nstop_reason:", result.stop_reason)
print("accepted this invocation:", len(result.accepted))
print("accepted per cell:", result.counts)
print("drops by layer:", result.drops.by_layer())
print("drops by code:", result.drops.by_code())
# Every failed try, repairs included, so a layer that rejects and then gets repaired shows up.
print(
    "failed tries by layer:", Counter(layer for d in result.drops.drops for layer, _ in d.history)
)
print("run dir:", result.run.path if hasattr(result.run, "path") else result.run)

if result.accepted:
    print("\nfirst accepted record:")
    print(json.dumps(result.accepted[0], indent=2, ensure_ascii=False)[:4000])

# Stages 4-5 without releasing: metrics and the gate.
metrics = metrics_for_run(pipe.compiled, result.run, calibration=pipe.calibration, seed=pipe.seed)
gate = evaluate_gate(
    metrics,
    pipe.compiled.spec.thresholds,
    waive=["kappa_min", "residual_error_max", "semantic_diversity_min"],
)
print("\ngate passed:", gate.passed)
print(json.dumps(gate.to_dict(), indent=2))

# Uncomment to release (refills short cells, up to max_rounds, then writes the release):
# rel = pipe.release(RELEASES, RUN_ID, max_rounds=3,
#                    waive=["kappa_min", "residual_error_max", "semantic_diversity_min"])
# print(rel.released, rel.stop_reason, rel.path)
