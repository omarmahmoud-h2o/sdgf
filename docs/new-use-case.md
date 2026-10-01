# Setting up a new use case

What you need to write to generate your own dataset with sdgf. Read
[how-it-works.md](how-it-works.md) first for the run and the six checks. Every
`task.yaml` key is listed in [../README.md](../README.md).

## Checklist

1. **Pick a task type.**
   - `classification_spans`: numbered, alternating conversation turns, a label set by code, and spans quoted exactly from the text. Uses the label-set-by-code mode (`label_first`). Example: FAG.
   - `sft_qa`: question, response and answer, with a pluggable `answer_format`. Uses the answer-found-by-the-model mode (`answer_emergent`). Example: CFA.
   - If neither fits, write your own (see [Writing a task type](#writing-a-task-type)).
2. **Write `tasks/<name>/task.yaml`.** Copy `tasks/fag/task.yaml` if code sets the label, or `tasks/cfa/task.yaml` if the model finds the answer.
3. **Write `hooks.py`** if any fact about a record is decided by code. A pure answer-found-by-the-model task doesn't need one.
4. **Write `seeds.jsonl`.** Hand-written, fictional examples that pass the task type's schema. Never copy them from evaluation data.
5. **Check the spec:** `sdgf validate-spec tasks/<name>`.
6. **Do a mock run** before spending money: pass `--backends module:factory` returning `MockBackend`s for `generator` and `judge`, with `--target-size 20`. `tests/cli_backends.py` shows how.
7. **Add tests** under `tests/` that use only `MockBackend`.
8. **Run for real**, then `sdgf evaluate`, then `sdgf release`. Until you have a calibrated judge and an embedder, add `--waive kappa_min residual_error_max semantic_diversity_min`.

## `task.yaml`

The required sections are `task`, `output_schema`, `rubric`, `seeds`, `coverage`,
`models`, `validation` and `thresholds`. The optional ones are `tools`, `governance`,
`hitl` and `budget`. Any key the schema doesn't know is an error.

```yaml
task:
  name: mytask
  version: "1.0"
  type: classification_spans        # or sft_qa
  generation_mode: label_first       # or answer_emergent
  description: >-                    # goes into the shared part of every prompt
    What the model is writing and for whom.

output_schema:
  fields:                            # added to the task type's own fields
    product_scope: {type: string, enum: [a, b]}
    severity:      {type: string, nullable: true, enum: [low, high]}
  turns: {roles: [customer, assistant], first_role: customer}
  spans: true

rubric:                              # what the blind judge returns; never contains the label
  verdict:
    values: [positive, negative]
    labels: {positive: true, negative: false}
  criteria:
    - {name: realism, min: 1, max: 5}
  reason_required: never
  # judge_context: >-               # optional: the judge's context instead of task.description
  #   Definitions the judge needs, without the writer's instructions.
  # examples: []                    # optional judge-only worked examples (fictional only)

seeds: {path: seeds.jsonl, format: annotated, uses: [few_shot], few_shot_count: 3}

coverage:
  target_size: 200
  axes:
    - {name: product_scope, source: fixed, values: [a, b]}
    - {name: label,         source: fixed, values: [true, false]}
  balance: {label: {"true": 0.5, "false": 0.5}}
  quota_policy: even
  params: {}                         # domain data your hooks read

models:
  generator: {backend: openai_compat, model: ..., hosting: local, temperature: 0.8, max_tokens: 4096,
              input_cost_per_mtok: 0.0, output_cost_per_mtok: 0.0, params: {api_base: http://...}}
  judge:     {backend: anthropic, model: claude-opus-5-5, temperature: 0.0, max_tokens: 512,
              input_cost_per_mtok: 4.0, output_cost_per_mtok: 20.0, params: {api_key_env: ANTHROPIC_API_KEY}}

validation:
  layers: [L1, L2, L3, L4, L5, L6]
  repair_tries: 2                    # re-prompts before a candidate is dropped
  consistency_k: 5                   # extra votes at L6
  consistency: {temperatures: [0.7, 0.8, 0.9]}   # vote i at temperatures[i % len]
  escalation: {low_confidence: 0.7, on_hard_cells: true, on_contestable: true}
  calibration: {ece_max: 0.10, bins: 10, min_gold: 30}

thresholds:                          # every key is required
  fidelity_min: 0.95
  kappa_min: 0.7
  coverage_min_cell_fill: 0.9
  balance_tolerance: 0.05
  distinct_n_min: 0.6
  self_bleu_max: 0.4
  semantic_diversity_min: 0.5
  residual_error_max: 0.05
  governance_violations_max: 0
  overlap_max: 0.6
  cost_per_record_max: 0.10
```

Four things decide whether the dataset can be relied on:

- **Every axis value is a record field.** A cell's values are written into the record as fixed facts and override the model. Declare each axis field in `output_schema.fields`, unless the task type already owns it (like `label`).
- **The label must follow from facts.** With a label set by code, `label_rule(record)` should compute the label from other fixed facts. The `label` axis then only sets the target, and `sampler_constraints` picks facts that agree with it. Without a `label_rule`, L2 can't check the label.
- **The judge sees only its judge fields.** Anything that gives the label away (spans, a tier field) must not be a judge field. To change what the judge sees, override `judge_fields` in a task type, not in the spec.
- **Prices and hosting are part of the spec.** The audit trail and the governance report name every model that received data. `hosting: provider_api` marks an external endpoint. A stage without prices blocks `budget.max_cost_usd`.

## `hooks.py`

Module-level functions; write any subset. Stage 0 checks their signatures.

```python
import random

def label_rule(record) -> bool | str | int:
    """The correct label, computed from the record's facts. Checked at L2."""

def sampler_constraints(cell: dict, rng: random.Random) -> dict | None:
    """Fill the fields the cell leaves open so they agree with its values.
    Return None if the combination is contradictory. Use rng for every draw."""

def extra_validators(record) -> list[str | ValidationIssue]:
    """Task rules the L1 schema can't express. An empty list means pass. Sent back at L2."""

def post_process(record) -> dict:
    """Fields derived after a record is kept. A record that already has them
    (a seed, a re-checked release) must still agree."""
```

- **One source of truth.** Read domain data from `coverage.params` in the YAML. FAG loads its own `task.yaml` with `yaml.safe_load` at import.
- **`sampler_constraints` runs once per combination**, at plan time, with a seeded rng. Return `None` only when the values contradict each other, never because of a random draw. Otherwise the plan keeps or drops the combination based on a single sample.

## `seeds.jsonl`

One JSON object per line, valid against the task type's schema plus your fields.

- **Scanned at intake.** Seeds are checked for personal data and toxicity, and the spec is rejected if one fails.
- **Copied by the model.** The generator sees seeds verbatim as examples and will imitate every field in them.
- **No derived fields.** Leave out any field that `post_process` derives. The model would guess it, and an `extra_validators` check on that field would reject the guess.
- **What to keep:** the fields the model is meant to write, plus the fixed facts.

## Writing a task type

Subclass `TaskType` and register it:

```python
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import REGISTRY

class MyType(TaskType):
    name = "my_type"
    generation_modes = ("label_first",)             # and/or "answer_emergent"

    def base_schema(self): ...                       # JSON schema of the fields this type owns
    def default_validators(self): return [...]       # record -> list[str]; run at L1
    def judge_fields(self): return ("messages",)     # what the blind judge may see
    def label_field(self): return "label"            # what L5 compares the verdict with
    def answer_extractor(self): ...                  # required for answer_emergent
    def derive_fields(self, record): return record   # fill fields readable from the output

REGISTRY.register(MyType())
```

`check_definition()` runs at registration. A type that declares `answer_emergent` must
have an answer extractor, and `base_schema` must be an object schema. `sft_qa.py` shows a
type with per-spec options: `configure(task)` reads `task.answer_format`.

## What starts a new run, and what continues one

- **Spec version.** It changes when `task.yaml`, `hooks.py` or `seeds.jsonl` change. Plans, runs and calibrations under the old version stay on disk but aren't reused. Keep provider variants as separate files (`task.<provider>.yaml`) next to one `hooks.py`.
- **Resume.** The same `run_id` continues from `accepted.jsonl`. A new id starts a new run under the same plan.
- **Budget.** `budget.max_tokens`, `max_cost_usd`, `max_seconds` and `max_candidates` each stop a run cleanly. `sdgf resume` picks it up, and `usage.json` keeps the total across invocations.
- **Calibration.** It applies to one spec version and one judge model. Until a calibration passes, the judge isn't trusted: L6 votes on every flagged candidate, and `kappa_min` and `residual_error_max` stay unmeasured.
