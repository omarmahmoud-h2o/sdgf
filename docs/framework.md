# sdgf framework guide

How the framework is put together, what each validation layer does, and what a new
use case needs. Install, CLI usage and the field-by-field `task.yaml` reference are in
[../README.md](../README.md); design rationale is in
[../../FRAMEWORK_DESIGN.md](../../FRAMEWORK_DESIGN.md).

## 1. What it does

sdgf generates a dataset from a declarative spec. A use case is a directory with:

| File | Role |
|---|---|
| `task.yaml` | what to generate, how to cover the space, which models to use, how to judge, what "good enough" means |
| `hooks.py` (optional) | the task's rules as code: the correct label from facts, cell consistency, extra checks, derived fields |
| `seeds.jsonl` | hand-written examples used as few-shot prompts and, optionally, as keyword seeds or a gold set |

The framework turns that into accepted records with per-record provenance, a metrics
report, and a versioned release that only exists if it passed the gate.

The rule that shapes everything: **code decides, the model writes.** Every fact that
carries a label or a metric (the coverage cell, the label under `label_first`, derived
fields) comes from the spec and hooks. The model's output is text that must match those
facts, and six layers check that it does. Nothing about a record is taken on the
model's word.

### Terms

- **Cell** — one combination of coverage-axis values (e.g. `corps_act|true|short`) with a quota of records to accept.
- **Recipe** — the cell's fixed fields for one candidate, after `sampler_constraints` fills in the conditioned ones. The recipe always overrides what the model returns.
- **Candidate** — one model output for a recipe. A candidate becomes a **record** when it passes the cascade.
- **spec_version** — sha256 over the canonical spec, the `hooks.py` source and the seed bytes. Editing any of them starts a new version; plans, runs and calibrations are keyed by it.
- **Run** — one invocation of stages 2–3 against a plan, resumable by `run_id`.
- **Release** — an immutable directory written only when the gate passes.

### Generation modes

| | `label_first` | `answer_emergent` |
|---|---|---|
| Who fixes the label | code, before generation (the cell + `label_rule`) | the model, while writing (the answer it reaches) |
| L5 checks | a blind judge's verdict means the fixed label | a blind judge, shown only the question, reaches the same answer |
| L6 checks | K judge votes; the majority must mean the fixed label | K fresh answers; the record's answer must be the majority's |
| Example | FAG (`tasks/fag`) | CFA (`tasks/cfa`) |

## 2. Architecture and pipeline

```
0 INTAKE     task.yaml + hooks.py + seeds ─► CompiledSpec, spec_version
1 PLAN       axes (× keyword expansion) ─► sampler_constraints ─► cells × quotas      ◆ approve
2 GENERATE   scheduler ─► prompt(static prefix + cell) ─► model [◄─► tool gateway] ─► candidate
3 VALIDATE   L1 ─► L2 ─► L3 ─► L4 ─► L5 ─► L6      fail_repairable ─► re-prompt with errors (≤ repair_tries)
                                                 fail_hard ─► drop                    ◆ review queue
4 EVALUATE   fidelity · kappa · coverage · balance · diversity · error rates · residual · governance · overlap · cost · yield
5 GATE       every threshold met? ─► release dir      else ─► shortfall report ─► refill short cells (≤ max_rounds)
```

### Stage 0 — intake (`spec/`)

`compile_spec(path)` loads the three files, validates the YAML against the pydantic
schema (unknown keys are errors), checks each hook's signature, and runs three gates
before any model is called: every seed is re-scanned for PII and toxicity, every listed
tool must exist in the tool registry, and every release threshold must be set. All
problems are reported together. The output is a frozen `CompiledSpec` and its
`spec_version`.

### Stage 1 — coverage plan (`coverage/`)

1. Resolve each axis to values. Sources: `fixed` (declared values and weights), `bloom` (the six Bloom levels), `keyword_expansion` (bi-directional expansion by the `expansion` model), `retrieval` (BM25 extraction on top).
2. Cross the axes into combinations.
3. Call `sampler_constraints(cell, rng)` once per combination with an rng seeded from the plan seed and the cell id; `None` drops the combination.
4. Assign quotas: `even`, or `weighted` from axis weights, fitted to `coverage.balance` by iterative proportional fitting and apportioned to `target_size`.

The plan is cached as `<store>/<spec_version>/shared/coverage_plan.json` and reused by
every run of that version; a non-default target or seed gets its own file. With
`hitl.approve_coverage_plan`, the pipeline stops (exit 3) until `sdgf plan --approve`.

### Stage 2 — generate (`generate/`, `tools/`)

The scheduler hands out slots in the cell furthest below quota. Per slot, the
generator builds a recipe from the cell and its conditioned draws, renders the prompt,
and runs the agent loop.

The prompt has two parts in fixed order so provider prefix caches hit across the run:
a **static prefix** (task description, rubric, output schema, turn and span rules,
few-shot seeds chosen once per spec with labels interleaved, output-format
instructions) and a **cell section** (`## Fixed parameters for this record`). The
prefix hash is stored in provenance.

If the task lists tools, every call goes through the gateway, which enforces the
allowlist, per-record call and token budgets, argument schemas, a shared cache keyed by
tool and canonical arguments, and attaches the tool's sensitivity label. Every call,
allowed or denied, lands in the record's tool trace.

The reply is parsed as JSON; the recipe is laid over it (`merge`), so a model can't
change a fixed field; the task type's `derive_fields` fills anything readable from the
output (e.g. the answer from the response). Unparseable output is a generation failure
and is fed back like a validation error.

Concurrency is per stage (`models.<stage>.concurrency`); one pool sized to the largest
value runs whole candidates in parallel, each stage's backend capped at its own limit.
Candidate seeds derive from the run seed, cell id and attempt index, so output is the
same at any concurrency.

### Stage 3 — validate (`validate/`)

The cascade and layers are in section 3. Every accepted record carries a
`_provenance` object: spec_version, each stage's model with backend and hosting, prompt
hash, seed, cell id, tool trace, each layer's outcome per attempt, repair count, human
decisions.

### Stage 4 — evaluate (`evaluation/metrics.py`)

Computed overall and per cell from the accepted stream, the drop log, usage and (if
present) a calibration:

| Metric | Definition |
|---|---|
| fidelity | accepted records whose final L5 verdict agreed ÷ accepted records L5 judged |
| kappa | Cohen's κ of the judge against the human gold set (from calibration) |
| coverage | accepted ÷ quota per cell; share of cells at quota; minimum fill |
| balance | actual share of each value on a balanced axis vs target; max deviation |
| diversity | distinct-n, self-BLEU, cluster entropy (needs an embedder) |
| error rates | rejections per layer ÷ candidates generated, repairs included |
| residual error | estimated wrong labels left in the accepted set, from per-label judge precision on gold |
| governance | accepted records the scanners still flag when L3 is re-run |
| overlap | maximum shingle similarity of any accepted record to a seed or held-out item |
| cost | tokens, USD and seconds ÷ accepted record; per model stage |
| yield | accepted ÷ candidates |

Anything not measured is `None`, never 0.

### Stage 5 — gate and release (`evaluation/gate.py`, `reports.py`)

`*_min` passes at value ≥ threshold, `*_max` at value ≤ threshold, checked overall and
(where meaningful) per cell. An unmeasured thresholded metric fails unless it is
waived. `governance_violations_max` can't be waived and any violation fails. On a fail,
`shortfall.json`/`.md` go into the run directory and `release` refills only the short
cells, up to `--max-rounds`. On a pass the release directory is built under a temporary
name and renamed into place:

```
<releases>/<task>/<version>/
  dataset.jsonl  provenance.jsonl  dataset_card.md  governance_report.json  metrics.json  manifest.json
```

`governance_report.json` lists every model endpoint that received data, external ones
apart.

### Shared components

| Component | Where | What it fixes |
|---|---|---|
| Task-type registry | `tasktypes/` | the record shape, default validators, judge view, answer extractor for a *kind* of dataset (`classification_spans`, `sft_qa`) |
| Model registry | `models/` | one `call(prompt, max_tokens, temperature, tools)` behind `openai_compat`, `anthropic`, `vllm`, `mlx`, `mock`; API keys only via `api_key_env`; usage metering and pricing per stage |
| Tool registry + gateway | `tools/` | allowlist, budgets, cache, sensitivity labels, trace |
| Governance profile | `governance/` | global PII, toxicity, secrets and entity rules; a task may only tighten them, with documented exceptions for toxicity categories and named entities only |
| Artefact store + provenance | `store/` | `<store>/<spec_version>/shared/` and `runs/<run_id>/` (`spec.json`, `cells.json`, `accepted.jsonl`, `drops.jsonl`, `usage.json`, `cli_options.json`) |
| HITL | `hitl/` | plan approval, review queue, resolutions feeding the gold set |
| Judge calibration | `judge/calibration.py` | κ and ECE against gold; a judge is *trusted* only when both pass for this spec_version and judge model |

### On-disk layout of a run

```
<store>/<spec_version>/
  shared/                            coverage_plan.json, calibration results, tool_cache.jsonl
  runs/<run_id>/
    run.json  spec.json  cells.json  run manifest; stage 0 summary; this run's cells and quotas
    accepted.jsonl  drops.jsonl      records with provenance; every drop with layer, codes and per-try history
    summary.json  usage.json         scheduler snapshot; tokens, cost and seconds summed over every invocation
    cli_options.json                 options the CLI reuses for resume / evaluate / release
    review.jsonl  review_decisions.jsonl   when hitl.review_flagged is on
    rounds.json  shortfall.json/.md  release rounds; the last failed gate
```

## 3. The validation layers

### The cascade

Layers run in the order `L1 … L6` and stop at the first failure, so paid judge calls
never see a record a free check already rejected. Each layer returns `pass`,
`fail_repairable` or `fail_hard`:

- **fail_repairable** — the generator is re-prompted with the *original* prompt plus the specific errors (`## Your previous attempt was rejected`, one line per issue), same cell, same recipe, up to `validation.repair_tries` times (default 2). Then the candidate is dropped and the slot re-queued in the same cell.
- **fail_hard** — dropped at once. Used where feeding the error back would teach the model to hide the problem rather than avoid it.

Every layer's verdict, with its issue codes, is written to provenance for accepted
records and to `drops.jsonl` for dropped ones. Later layers can read earlier verdicts
(L6 reads L5's). A layer that raises is a pipeline bug and propagates; it is never
recorded as a bad record.

| Layer | Checks | Failure | Cost | Needs |
|---|---|---|---|---|
| L1 schema | record shape, turn structure, task-type structural validators | repairable | free | — |
| L2 rules | keyword rules, `label_rule` agreement, `extra_validators` | repairable | free | hooks |
| L3 governance | PII, toxicity, secrets, denied entities, tool-data leaks | **hard** | free | — |
| L4 overlap | similarity to seeds, a held-out set, and records accepted this run | **hard** | free | `thresholds.overlap_max` |
| L5 judge | a blind judge's verdict against the intended label | repairable / hard / review | 1 judge call | `models.judge`, `rubric` |
| L6 consistency | K votes on escalated records only | repairable / hard | ≤ K calls | `models.judge`, `consistency_k` |

### L1 — schema (`l1_schema.py`)

Three checks, each run only if the previous is clean, since a malformed record makes
the later ones report noise:

1. **JSON schema** — the task type's base schema plus `output_schema.fields` (required fields, types, enums, nullability).
2. **Turn structure** — if `output_schema.turns` is declared: numbering from `numbered_from`, known roles, alternation from `first_role`.
3. **Task-type validators** — e.g. `classification_spans` checks turn numbers are consecutive and each span cites an existing turn.

Issue codes: `schema_<keyword>`, `turn_numbering`, `unknown_role`, `first_role`,
`role_alternation`, and the validator's function name.

### L2 — rules (`l2_rules.py`)

Three independent checks, all run so one repair prompt carries every error:

1. **Keyword rules** — `validation.rules`: `required` or `forbidden` keywords over message text and named fields, `substring`/`word`/`regex` matching, restricted by `roles`, `fields` and `when: {field: value}`.
2. **Label rule** — `label` must equal `label_rule(record)`, value and type. This is where the corpus becomes correct by construction: for FAG, `expected_breach(advice_tier, product_scope)`.
3. **Extra validators** — `extra_validators(record)` returns strings or `ValidationIssue`s (to set a code and path). FAG uses it for verbatim spans, every signal having a span, severity only on breaches, and derived-field agreement.

A hook that raises `KeyError`/`TypeError`/`ValueError` on a record is reported as
`label_rule_error`/`extra_validator_error` (a record problem), not as a crash.

### L3 — governance (`l3_governance.py`)

Runs every scanner of the task's effective governance profile over all text in the
record (keys starting with `_` skipped): PII (regex by default: email, phone, ABN, TFN,
BSB, account number, plus `governance.extra_pii_patterns`; Presidio with the `pii`
extra), toxicity (keyword list by default; Detoxify with the `toxicity` extra),
secrets (API keys, AWS keys, private keys, bearer tokens, JWTs, password assignments),
and the entity deny list.

When the record's tool trace carries a result labelled anything but `public`, two more
checks run: `tool_data_leak` (a value from that result appears verbatim in the record,
case- and whitespace-insensitive, ≥ 6 characters) and `sensitive_identifier` (any
6+-digit run the PII rules didn't already flag). An unknown sensitivity label counts as
sensitive.

Every finding is `fail_hard`. Issues carry the rule, path and character offsets, never
the matched text, because drop logs are written to disk.

### L4 — overlap (`l4_overlap.py`)

Compares the record's text with three sources and drops it if any exceeds
`thresholds.overlap_max`:

| Source | Issue code | Purpose |
|---|---|---|
| seeds | `seed_overlap` | the model copied a few-shot example |
| held-out | `held_out_overlap` | contamination of an evaluation set |
| corpus | `near_duplicate` | a near-copy of a record accepted this run |

Similarity is Jaccard over 5-character shingles of normalised text (casefolded,
punctuation and whitespace collapsed). An embedding engine (cosine over
sentence-transformers vectors, or any `embed` callable) can run alongside with its own
thresholds, since the two scores aren't on one scale.

The held-out check is on only when a path is passed at run time (`--held-out` or
`Pipeline(held_out_paths=...)`). The spec has no field for it, the path is never saved,
and held-out documents are reduced to shingle sets on load so their text can't reach a
prompt. Records join the corpus only after acceptance; a batch under concurrency is
settled in a fixed order so the result equals a sequential run. Failures are
`fail_hard`.

### L5 — decision judge (`l5_judge.py`, `judge/`)

A separate judge scores the record **blind to its intended label**. It sees only the
task type's `judge_fields` — for `classification_spans`, just `messages`; for `sft_qa`,
just `question` — never the label, the spans that justify it, or `_` keys. The
`rubric` section compiles into a typed output schema: a `verdict` field with the
declared values, one field per criterion (enumerated values or an integer range), a
confidence per field, and a `reason` when `reason_required` asks for one (written by
`models.fallback_judge`). The judge prompt never contains the intended label.

Agreement is computed in label space through `rubric.verdict.labels`
(FAG: `{breach: true, no_breach: false}`). Under `answer_emergent` the "intended label"
is the answer the record reached, and the judge answers the question afresh.

| Judge result | Outcome |
|---|---|
| agrees, confident | `pass` |
| disagrees, confident | `fail_repairable` `judge_disagrees` — the message names the verdict given and the one(s) the label needs, plus the reason if one was written |
| confidence < `escalation.low_confidence`, `hitl.review_flagged` on | `fail_hard` `sent_to_review` — queued for a person; leaves the automatic path |
| low confidence, review off | judged as above, and marked for escalation to L6 |
| output not schema-valid | `sent_to_review` if review is on, else `fail_hard` `judge_error` — regenerating the record doesn't fix a broken judge, and the drop log makes it visible |

Every verdict's details record the judge result, `agrees`, `low_confidence`,
`escalate` and `trusted` for L6, metrics and provenance. `escalate` is true when
confidence is low, the cell's `difficulty` is `hard` and `escalation.on_hard_cells` is
set, the record is `contestable` and `on_contestable` is set, or `escalation.always`.
`trusted` is true only when a calibration for this spec_version and judge model passed
`thresholds.kappa_min` and `calibration.ece_max`.

### L6 — adaptive consistency (`l6_consistency.py`)

Runs only on records L5 escalated (or, without an L5 verdict, on hard/contestable
cells per `validation.escalation`). Everything else passes with `method: skipped`, so
the K paid votes are spent only where the single verdict isn't enough.

| Mode | Behaviour |
|---|---|
| `label_first`, trusted judge, L5 confident and agreeing | `pass` on confidence, no votes |
| `label_first`, otherwise | K judge votes; more than half of the votes cast must mean the fixed label, else `fail_repairable` `consistency_disagrees` |
| `answer_emergent` | K fresh answers to the question (the judge-stage model, temperatures cycling 0.7/0.8/0.9); the majority answer must equal the record's answer, else `fail_repairable` `consistency_answer_mismatch`; no majority → `consistency_no_majority` |

An unparseable vote is an abstention and counts in neither numerator nor denominator
(DS²-Instruct divided by K, which pulled good records under the threshold). If no vote
is readable the judge is broken, not the record: `fail_hard` `consistency_no_votes`.
The winning voter's text is kept in details only; it never replaces the record's
response, since the cheaper layers never checked it.

### Repair and drops (`repair.py`)

```
attempt 0 ─► cascade ─► pass ──────────────► accepted
                    ├► fail_hard ──────────► drop
                    └► fail_repairable ──► attempt 1 = original prompt + errors ─► … ─► drop after repair_tries
```

Generation failures (`no_json`, `no_text`, `tool_calls` when no tools are allowed) are
fed back the same way and dropped under layer `generate`. A drop records the cell, the
final layer and codes, the attempt count, and the `(layer, codes)` history of every
try, which is what the per-layer error rates are computed from. When a task has tools,
one tool session spans all tries of a slot, so per-record budgets cover repairs.

## 4. Running a new use case

### Checklist

1. **Pick a task type.** `classification_spans` (numbered alternating turns + a code-fixed label + verbatim spans, `label_first`) or `sft_qa` (question/response/answer with a pluggable `answer_format`, `answer_emergent`). If neither fits, subclass `tasktypes.base.TaskType` (see below).
2. **Write `tasks/<name>/task.yaml`.** Required sections: `task`, `output_schema`, `rubric`, `seeds`, `coverage`, `models`, `validation`, `thresholds`. Optional: `tools`, `governance`, `hitl`, `budget`. Copy `tasks/fag/task.yaml` for a label-first task or `tasks/cfa/task.yaml` for an answer-emergent one.
3. **Write `hooks.py`** if any fact about a record is decided by code. Skip it for a pure answer-emergent task.
4. **Write `seeds.jsonl`**: fictional, hand-written, passing the task type's schema. Never derived from evaluation data.
5. **Check the spec:** `sdgf validate-spec tasks/<name>`.
6. **Do a mock run** before spending anything: a `--backends module:factory` that returns `MockBackend`s for `generator` and `judge`, `--target-size 20`. `tests/cli_backends.py` shows the shape.
7. **Add tests** under `tests/` using `MockBackend` only.
8. **Run for real**, then `evaluate`, then `release`. Expect to `--waive kappa_min residual_error_max semantic_diversity_min` until you have a calibration and an embedder.

### What goes in `task.yaml`

Minimal skeleton; the README has every key.

```yaml
task:
  name: mytask
  version: "1.0"
  type: classification_spans        # or sft_qa
  generation_mode: label_first       # or answer_emergent
  description: >-                    # goes into the static prompt prefix
    What the model is writing and for whom.

output_schema:
  fields:                            # on top of the task type's own fields
    product_scope: {type: string, enum: [a, b]}
    severity:      {type: string, nullable: true, enum: [low, high]}
  turns: {roles: [customer, assistant], first_role: customer}
  spans: true

rubric:                              # what the judge scores; never contains the label
  verdict:
    values: [positive, negative]
    labels: {positive: true, negative: false}
  criteria:
    - {name: realism, min: 1, max: 5}
  reason_required: never

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
  repair_tries: 2
  consistency_k: 5
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

Points that decide whether the corpus is trustworthy:

- **Every axis value is a record field.** The cell's values are written into the record as the recipe and override the model. Declare each axis field in `output_schema.fields` (or it's one the task type owns, like `label`).
- **The label must come from facts.** Under `label_first`, `label_rule(record)` should compute the label from other recipe fields; the `label` axis then only sets the target, and `sampler_constraints` picks facts consistent with it. Without a `label_rule`, L2 can't check the label at all.
- **The judge sees only `judge_fields`.** Anything that encodes the label (spans, a tier field) must not be a judge field. If you add fields the judge should see, override `judge_fields` in a task type, not in the spec.
- **Prices and hosting are part of the spec.** Provenance and the governance report name the models that received data; `hosting: provider_api` marks an external endpoint. Unpriced stages block `budget.max_cost_usd`.

### What goes in `hooks.py`

Module-level functions, any subset. Stage 0 checks the signatures.

```python
import random

def label_rule(record) -> bool | str | int:
    """The correct label from the record's checkable facts. Checked at L2."""

def sampler_constraints(cell: dict, rng: random.Random) -> dict | None:
    """Fill the fields the cell leaves open so they're consistent with its values;
    return None if the combination is contradictory. Every draw uses rng."""

def extra_validators(record) -> list[str | ValidationIssue]:
    """Task rules L1's schema can't express. Empty list = pass. Repairable at L2."""

def post_process(record) -> dict:
    """Derived fields, added to an accepted record after validation. A record that
    already carries them (a seed, a re-validated release) should still agree."""
```

Read domain data from `coverage.params` in the YAML (FAG loads its own `task.yaml`
with `yaml.safe_load` at import) so there is one source of truth. `sampler_constraints`
is called once per combination at plan time with a seeded rng; return `None` only for
contradictory values, not on a random draw, or the plan will be a single sample of it.

### What goes in `seeds.jsonl`

One JSON object per line, valid against the task type's schema plus your fields. Seeds
are scanned for PII and toxicity at compile time and rejected if they fail. They are
shown to the generator verbatim as few-shot examples, so **the model will imitate every
field present**. A field that `post_process` derives should not appear in seeds: the
model will guess it, and an `extra_validators` check on it will reject the guess. Keep
seeds to fields the model is meant to write plus the recipe fields.

### Writing a task type

Subclass `TaskType` and register it:

```python
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import REGISTRY

class MyType(TaskType):
    name = "my_type"
    generation_modes = ("label_first",)             # and/or "answer_emergent"

    def base_schema(self): ...                       # JSON schema of the fields the type owns
    def default_validators(self): return [...]       # record -> list[str]; run at L1
    def judge_fields(self): return ("messages",)     # what a blind judge may see
    def label_field(self): return "label"            # what L5 compares the verdict with
    def answer_extractor(self): ...                  # required for answer_emergent
    def derive_fields(self, record): return record   # fill fields readable from the output

REGISTRY.register(MyType())
```

`check_definition()` runs at registration: a type that declares `answer_emergent` must
have an extractor, and `base_schema` must be an object schema. `sft_qa.py` shows a type
with per-spec options (`configure(task)` reads `task.answer_format`).

### What changes a run

- **`spec_version`** changes when `task.yaml`, `hooks.py` or `seeds.jsonl` change. Plans, runs and calibrations under the old version stay but aren't reused. Keep provider variants as separate files (`task.<provider>.yaml`) beside one `hooks.py`.
- **Resume** with the same `run_id` continues from `accepted.jsonl`; a new id starts fresh under the same plan.
- **Budget** (`budget.max_tokens|max_cost_usd|max_seconds|max_candidates`) stops a run cleanly; `resume` picks it up and `usage.json` carries the total across invocations.
- **Calibration** is per spec_version and judge model. Until one passes, the judge is untrusted: L6 always votes on escalated records and `kappa_min`/`residual_error_max` are unmeasured.
