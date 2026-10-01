# sdgf — Synthetic Data Generation Framework

A spec-driven, governed pipeline for generating synthetic training and evaluation data.
A use case is a `task.yaml` (what to generate, how to judge it, what "good enough"
means), an optional `hooks.py` (the task's checkable rules in code) and a `seeds.jsonl`.
The framework handles coverage planning, generation, the six-layer validation cascade,
evaluation, the release gate and provenance. The design is in
[../FRAMEWORK_DESIGN.md](../FRAMEWORK_DESIGN.md); section numbers (§) below refer to it.
Start with [docs/overview.md](docs/overview.md) for what sdgf produces and what it can't
guarantee. [docs/how-it-works.md](docs/how-it-works.md) walks through a run and each of the
six checks. [docs/new-use-case.md](docs/new-use-case.md) explains how to set up a new
dataset. Terms are defined in [CONTEXT.md](CONTEXT.md).
[docs/sdgf-guide.html](docs/sdgf-guide.html) is all three in one interactive page that works
offline: open it in a browser. It is generated from the markdown, so edit the `.md` files
and rerun `python docs/build_guide.py` (needs the `docs` extra).

The first use case is the Financial Advice Guardrail (FAG), in [tasks/fag/](tasks/fag/),
a `label_first` task: code fixes the label and the model writes matching text. The second,
in [tasks/cfa/](tasks/cfa/), is DS²-Instruct's CFA exam questions, an `answer_emergent`
task: the model writes the question and its answer, a blind judge and K fresh answers
check it, and a record is kept only when its answer is the majority's.

## Install

Python 3.11+. The core depends only on `pydantic`, `pyyaml`, `jsonschema` and `numpy`.

```bash
cd sdgf
pip install -e .            # core, plus the `sdgf` console script
pip install -e '.[dev]'     # + pytest, ruff
python -m pytest -q         # the suite uses only the mock backend: no model or API calls
```

## Pipeline

| Stage | What happens | Code |
|---|---|---|
| 0 | Compile `task.yaml` + `hooks.py` + seeds into a frozen `CompiledSpec`; check hook signatures, re-scan seeds for PII and toxicity, check tools exist, require every release threshold | `spec/` |
| 1 | Coverage plan: cross the axes, drop invalid cells via `sampler_constraints`, assign quotas; cached as `coverage_plan.json` | `coverage/` |
| 2 | Generate: the scheduler picks the cell furthest below quota; labels come from the cell, the model writes prose | `generate/` |
| 3 | Validate: L1 schema → L2 rules → L3 governance → L4 overlap → L5 judge → L6 consistency, stopping at the first failure; repairable failures are re-prompted in the same cell | `validate/` |
| 4 | Evaluate: fidelity, kappa, coverage fill, balance, diversity, per-layer error rates, residual error, governance, overlap, cost, yield | `evaluation/metrics.py` |
| 5 | Gate and release: compare with `thresholds`; on a fail, refill only the short cells, up to `--max-rounds` | `evaluation/gate.py`, `evaluation/reports.py` |

Every run lives under `<store>/<spec_version>/runs/<run_id>/`. `spec_version` is a sha256
over the spec, the hooks source and the seeds, so editing any of the three starts a new
version rather than resuming an old run.

## Running the FAG example with the mock backend

`tasks/fag/task.yaml` points at a local vLLM server. To run it without one, pass
`--backends MODULE:ATTR` (or `path/to/file.py:ATTR`): a callable that takes the
`CompiledSpec` and returns `{stage: ModelBackend}`, replacing those stages. The test
suite's `tests/cli_backends.py` provides mock "worlds" that write valid FAG records
and judge them correctly:

```bash
cd sdgf
export PYTHONPATH=src:tests       # cli_backends lives in tests/
S=/tmp/sdgf-demo

python -m sdgf.cli validate-spec tasks/fag
python -m sdgf.cli plan    tasks/fag --store $S/store --target-size 20
python -m sdgf.cli run     tasks/fag --store $S/store --target-size 20 --run-id demo \
                           --backends cli_backends:fag_world
python -m sdgf.cli evaluate tasks/fag --store $S/store --run-id demo \
                           --backends cli_backends:fag_world
python -m sdgf.cli release tasks/fag --store $S/store --releases $S/releases --run-id demo \
                           --backends cli_backends:fag_world \
                           --waive kappa_min residual_error_max semantic_diversity_min
```

Each command prints one JSON object. Exit codes: `0` done, `1` finished but incomplete
(a run stopped short, a failed gate, no release), `2` error, `3` the coverage plan awaits
approval.

The `--waive` flags on `release` are needed in this demo, and they only let through a
metric that could not be measured at all:

- `kappa_min` and `residual_error_max` need a judge calibration: a `CalibrationResult`
  for the judge model and `spec_version`, built with `judge.calibration.calibrate` from
  a human gold set and saved with `CalibrationStore`. No CLI command builds one yet.
  Without one, these metrics are unmeasured.
- `semantic_diversity_min` needs an embedder, e.g.
  `validate.l4_overlap.sentence_transformers_embedder`, passed as
  `Pipeline.release(..., embed=...)`.

`governance_violations_max` can never be waived. Any governance finding fails the gate.

A passing release writes `<releases>/<task>/<version>/` containing `dataset.jsonl`,
`dataset_card.md`, `provenance.jsonl`, `governance_report.json` (the external endpoints
that received data, per D12), `metrics.json` and `manifest.json`. A failing gate writes
`shortfall.json` and `shortfall.md` into the run directory instead.

Other commands:

```bash
python -m sdgf.cli resume tasks/fag --store $S/store --backends cli_backends:fag_world
python -m sdgf.cli plan   tasks/fag --store $S/store --target-size 20 --approve --reviewer "A. Reviewer"
python -m sdgf.cli review tasks/fag --store $S/store list
python -m sdgf.cli review tasks/fag --store $S/store resolve ITEM_ID \
                          --action relabel --new-label true --reviewer "A. Reviewer"
```

A run's options (`--seed`, `--target-size`, `--plan-seed`, `--layers`,
`--max-attempts-per-cell`) are saved as `cli_options.json` on first use and reused by
`resume`, `evaluate` and `release`. `--held-out PATH` enables the L4 held-out overlap
check. It is never saved and never reaches a model, so pass it again on each command.

From Python:

```python
from sdgf.pipeline import Pipeline
from sdgf.models.mock import MockBackend

pipe = Pipeline("tasks/fag", "/tmp/sdgf-demo/store",
                model_overrides={"generator": MockBackend(my_fn), "judge": MockBackend(judge_fn)},
                target_size=20, seed=0)
result = pipe.run()
```

`MockBackend` takes a list of scripted replies (`cycle=True` repeats them) or a callable
`MockCall -> reply`. A reply may be text, a `ToolCall` or list of them (one agent round),
or a `ModelResponse` carrying token counts. Use a callable when you compare concurrent
runs, because a script is consumed in whatever order the calls arrive.

## `task.yaml` reference

These sections are required: `task`, `output_schema`, `rubric`, `seeds`, `coverage`,
`models`, `validation`, `thresholds`. These are optional: `tools`, `governance`, `hitl`,
`budget`. Unknown keys are errors, and every error names the field that failed. The
schema is in [src/sdgf/spec/schema.py](src/sdgf/spec/schema.py).

### `task`

| Key | Meaning |
|---|---|
| `name`, `version` | identify the task; used in release paths |
| `type` | a registered task type: `classification_spans` or `sft_qa` |
| `generation_mode` | `label_first` (the label is fixed by the cell, and the model writes text that matches it) or `answer_emergent` (the model writes the answer, and L6 majority voting decides it) |
| `description` | the task description; goes into the static prompt prefix |
| `answer_format` | optional; for `sft_qa`, how answers are read out of text: `multiple_choice` (default), `yes_no_maybe`, `numeric` or `boxed_math` |

### `output_schema`

Adds fields on top of the task type's own schema.

- `fields.<name>`: `type` (`string|integer|number|boolean|array|object|null`),
  `required` (default true), `nullable` (the key must be present but may be null),
  `enum`, `description`.
- `turns`: `roles`, `first_role`, `alternating` (default true), `numbered_from`
  (default 1). L1 enforces these.
- `spans`: whether records carry span annotations.

### `rubric`

This section is what the judge scores. It compiles into a typed output schema (§7.3).

- `verdict`: `values` holds 2 to 255 verdict values, with an optional `description`.
  `verdict.labels` maps each verdict to the intended label it means (FAG:
  `{breach: true, no_breach: false}`), which L5 uses to compute fidelity. If you leave it
  unset, each verdict value is itself the label.
- `criteria`: each has a `name` and exactly one scale, either `values` (2 to 255) or
  `min` and `max` (at most 255 integers).
- `reason_required`: `never`, `flagged` or `always`. Any value other than `never` needs
  `models.fallback_judge`.
- `judge_context`: optional text the judge reads as its `## Context` in place of
  `task.description`, which is written for the generator. Use it for domain definitions
  the judge needs without the writer's instructions. Unset, the judge reads
  `task.description`. The generation prompt never contains it.

The judge prompt never contains the intended label.

### `seeds`

`path` (relative to the task directory), `format` (`few_shot|annotated`), `uses`
(any of `few_shot`, `keyword_seeding`, `gold_set`) and `few_shot_count` (default 3).
Stage 0 re-scans every seed and refuses to compile if any seed fails.

### `coverage`

- `target_size`: the number of records to accept.
- `axes`: each has a `name` and a `source`:
  - `fixed`: needs `values`, with optional `weights`.
  - `keyword_expansion`: bi-directional keyword expansion driven by `models.expansion`.
  - `retrieval`: BM25 keywords from a corpus.
  - `bloom`: the six Bloom levels, or a subset listed in `values`.

  Keyword axes are split evenly, so they take no `weights`.
- `balance`: `{axis: {value: share}}`. The shares sum to 1 and the gate checks them.
- `quota_policy`: `even` or `weighted`.
- `params`: task-specific domain data that the hooks read. For FAG this holds the topic
  pools, tier descriptions, the 15 signals and their groupings, and the sampler weights.

### `models`

The stages are `generator` (required), `judge` (required when L5 is on),
`fallback_judge`, `consistency_judge` (L6's voters, when they should come from a
different model than L5's judge; unset means L6 votes with `judge`) and `expansion`.
Each stage takes these keys:

| Key | Meaning |
|---|---|
| `backend` | `openai_compat`, `anthropic`, `vllm`, `mlx`, `mock` or `jev` (a stub, see below) |
| `model` | model id |
| `hosting` | `local` or `provider_api`. It defaults from the backend, except that `openai_compat` must declare it. It is recorded in provenance and in the governance report (D12) |
| `temperature`, `max_tokens` | sampling |
| `concurrency` | per-stage call limit. The pool size is the largest across stages. Output is the same for a given seed at any concurrency |
| `input_cost_per_mtok`, `output_cost_per_mtok` | USD per million tokens. Set both or neither. Unset means *unpriced*, which is not the same as free |
| `params` | backend options. `openai_compat`: `api_base`, `api_key_env`, `top_p`, `stop`, `timeout`, `check_model`. `anthropic`: `api_key_env`, `top_p`, `thinking_headroom`, `max_tokens_cap`. `vllm`: `engine`, `top_p`, `stop`. `mlx`: `top_p`, `enable_thinking`. `mock`: `responses`, `cycle` |

API keys are read from the environment variable that `api_key_env` names. They never
go in the spec.

### `validation`

- `layers`: a subset of `L1`…`L6`, in that order (cheapest first).
- `repair_tries` (default 2): the number of re-prompts with the validator's errors
  before the record is dropped.
- `consistency_k` (default 5): the number of L6 votes.
- `consistency`: `temperatures` (default `[0.7, 0.8, 0.9]`). Vote i is sampled at
  `temperatures[i % len]`, overriding the voting stage's own temperature, so K votes
  from a temperature-0 judge don't just repeat L5's verdict. With L6 on and
  `consistency_k` above 1, stage 0 rejects a list whose every temperature is 0: one
  model at temperature 0 casts the same vote K times, at K times the cost. Use
  `consistency_k: 1` for a single vote. Each vote is kept with its stage, model and
  temperature, in L6's `ballots` detail and in the record's provenance
  (`layer_results[].ballots`), so vote agreement can be analysed after a run.
- `escalation`: `low_confidence` (default 0.7), `on_hard_cells`, `on_contestable`, and
  `always` (default false; escalate every record, as CFA does for DS²-Instruct
  self-consistency). Escalated records go to L6. Under `answer_emergent`, L6's K answers
  come from the voting stage's model, shown only the question and the answer format.
- `calibration`: `ece_max`, `bins` and `min_gold`. These, together with
  `thresholds.kappa_min`, decide when the judge's confidence can stand in for K votes.
- `rules`: L2 keyword rules. Each rule has these keys:
  - `name`
  - `kind`: `required` or `forbidden`
  - `keywords`
  - `match`: `substring`, `word` or `regex`
  - `case_sensitive`
  - `require`: `any` or `all`
  - `roles`
  - `fields`
  - `when`: `{field: value}` conditions

L1, L2 and L5 failures can be repaired. L3 (governance) and L4 (overlap) failures are
hard drops.

### `thresholds`

These set the release gate (§8). Every threshold must be set, because stage 0 rejects
unset values.
`fidelity_min`, `kappa_min`, `coverage_min_cell_fill`, `balance_tolerance`,
`distinct_n_min`, `self_bleu_max`, `semantic_diversity_min`, `residual_error_max`,
`governance_violations_max` (must be 0), `overlap_max`, `cost_per_record_max`.

### `tools` (optional)

This lists the registered tools the generator may call. Each entry has `name`,
`max_calls_per_record` (default 5) and `max_tokens_per_record`. Every call goes through
the gateway, which does the following:

- enforces the allowlist and the per-record budget;
- attaches the tool's sensitivity label to the result;
- caches the result by tool name and canonical arguments;
- appends the call to the record's tool trace.

There are two built-in tools: `product_catalogue_lookup` (a fictional local JSON
catalogue) and `calculator`.

### `governance` (optional)

This section can only tighten the global profile, never loosen it (§7.5). Its keys are:

- `extra_pii_patterns`: `{name: regex}`.
- `entity_deny` and `entity_allow`.
- `toxicity_exceptions`.
- `exceptions`: each has a `rule` and a `reason`. Every exception must be declared here.

### `hitl` (optional)

- `approve_coverage_plan`: the pipeline stops with exit 3 until you run `sdgf plan --approve`.
- `review_flagged`: low-confidence L5 results go to the review queue.
- `calibrate_judge`.

When you accept or relabel a review item, it is added to the task's gold set.

### `budget` (optional)

`max_tokens`, `max_cost_usd`, `max_seconds`, `max_candidates`. When the budget runs out,
the run stops cleanly, and you can pick it up again with `resume`. The budget applies
afresh to each invocation. `usage.json` in the run directory records the spend across
all invocations. `max_cost_usd` is an error if any stage in use is unpriced.

## Adding a use case

1. **Pick a task type.** `classification_spans` handles label-first classification with
   numbered alternating turns, a label and verbatim spans. For a new shape, subclass
   `tasktypes.base.TaskType`, which covers the output schema, the generation modes, the
   default axes, the default validators and an optional answer extractor. Then register
   it with `tasktypes.registry.register_task_type`.
2. **Write `tasks/<name>/task.yaml`.** Use the reference above, and copy
   [tasks/fag/task.yaml](tasks/fag/task.yaml) as a worked example. Put domain data in
   `coverage.params`, not in code.
3. **Write `tasks/<name>/hooks.py` (optional).** It holds module-level functions, and
   you can define any subset of them. Stage 0 checks their signatures.

   ```python
   def label_rule(record) -> label                   # the correct label from checkable facts (L2)
   def sampler_constraints(cell, rng) -> cell | None # make a cell consistent; None = invalid
   def extra_validators(record) -> list[str]         # task-specific L2 errors; empty = pass
   def post_process(record) -> record                # derived fields, added after validation
   ```

   `rng` is a seeded `random.Random`. Use it for every random draw. Put the policy in
   these functions so that the corpus is correct by construction, not by trusting the
   model. For example, FAG's `label_rule` is `expected_breach(tier, scope)`.
4. **Write `tasks/<name>/seeds.jsonl`.** Hand-write fictional examples that pass the
   task type's schema. Seeds are scanned for PII and toxicity at compile time. Never
   derive them from evaluation data.
5. **Check and run the spec.** Run `sdgf validate-spec tasks/<name>`, then a
   small mock run: `--target-size 20 --backends your_module:factory`. Then add tests
   under `tests/` that use `MockBackend` only.

## Optional adapters

Heavy engines are imported lazily, only when you build their adapter. The core and the
test suite never need them.

| Extra | Package | Adapter |
|---|---|---|
| `pii` | `presidio-analyzer`, `presidio-anonymizer` | `governance.pii.PresidioPIIScanner` (the default is a regex scanner for email, phone, ABN, TFN, BSB and account numbers, plus per-task patterns) |
| `toxicity` | `detoxify` | `governance.toxicity.DetoxifyToxicityScanner` (the default is a keyword list) |
| `embeddings` | `sentence-transformers` | `validate.l4_overlap.sentence_transformers_embedder`, for L4 embedding overlap and the `semantic_diversity_min` cluster entropy (the default is character-shingle Jaccard) |
| `docs` | `markdown` | not an adapter: `docs/build_guide.py` uses it to build `docs/sdgf-guide.html` |
| `retrieval` | `rank-bm25` | not required: `coverage/retrieval.py` is a pure-Python BM25 that scores the same as `rank_bm25.BM25Okapi` |

Install an extra with `pip install -e '.[pii]'`. Model SDKs work the same way:
`openai_compat` needs only the standard library, `anthropic` uses `anthropic`, `vllm`
uses `vllm`, and `mlx` uses `mlx` and `mlx-lm`. Each SDK is imported only at the
backend's `setup()`.

**Jev (TypeSafe System One)** is registered as the `jev` backend, but it is an
interface stub. Selecting it raises `NotImplementedError` until its API is documented
(§7.3, §16 Q1).

## Layout

```
src/sdgf/
  spec/        schema, loader, hooks, compile (stage 0)
  tasktypes/   TaskType interface, registry, classification_spans
  models/      backend interface, registry, mock and real backends, usage metering
  coverage/    keywords, retrieval, axes, plan (stage 1)
  generate/    scheduler, prompts, generator + agent loop (stage 2)
  tools/       registry, gateway, cache, builtin tools
  governance/  profile, PII, toxicity, secrets, entities
  validate/    cascade, L1–L6, repair (stage 3)
  judge/       typed interface, LLM judge, Jev stub, calibration
  evaluation/  metrics, diversity, gate, reports (stages 4–5)
  store/       artefact store, provenance
  hitl/        review queue, plan approval
  pipeline.py  cli.py
tasks/fag/     task.yaml hooks.py seeds.jsonl
tasks/cfa/     task.yaml seeds.jsonl
tests/
```
