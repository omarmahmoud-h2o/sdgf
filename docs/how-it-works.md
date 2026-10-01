# How sdgf works

This page follows one run from task spec to release, then goes through the six checks
one at a time. The plain names used here are defined in [../CONTEXT.md](../CONTEXT.md);
each code name is given once in backticks. For setting up a new dataset, see
[new-use-case.md](new-use-case.md). For every `task.yaml` key, see
[../README.md](../README.md).

## The run

![Run flow](diagrams/run-flow.svg)

[Interactive version](diagrams/run-flow.html)

| Stage | What happens | Code |
|---|---|---|
| 0 Intake | Load the three task spec files, check the YAML against the schema, check hook signatures, scan seeds for personal data and toxicity, confirm every listed tool exists and every release target is set. All problems are reported together. The output is the compiled spec and its spec version. | `spec/` |
| 1 Plan | Turn each axis into values, cross them into cells, let `sampler_constraints` fill in or reject each cell, and give each cell a quota. The plan is saved per spec version and reused. If `hitl.approve_coverage_plan` is on, the run stops (exit 3) until someone runs `sdgf plan --approve`. | `coverage/` |
| 2 Generate | The scheduler picks the cell furthest below its quota. Code builds the fixed facts for one record, the prompt is rendered, and the model writes a candidate. | `generate/` |
| 3 Check | The candidate goes through L1 to L6. It is kept, sent back, or dropped. | `validate/` |
| 4 Measure | Agreement rate, coverage, balance, diversity, copying, safety, cost and yield are computed over the whole run. Anything not measured is recorded as missing, never as 0. | `evaluation/metrics.py` |
| 5 Release check | Each measurement is compared with its target in `task.yaml`. If all are met, the release directory is written. If not, a shortfall report is written and only the short cells are generated again, up to `--max-rounds` times. | `evaluation/gate.py` |

### The prompt

The prompt has two parts, always in this order:

1. A part that is the same for every record: task description, judging criteria, output
   format, turn and span rules, and the few-shot seeds (chosen once per spec version).
2. A part for this record only: `## Fixed parameters for this record`, listing the
   fixed facts.

Keeping the first part identical lets provider prompt caches reuse it for the whole run.
Its hash goes into the audit trail.

### What the model can and can't change

The model's reply is parsed as JSON, then the fixed facts are written over it (`merge`).
A model that returns `"advice_tier": "FACTUAL_INFORMATION"` when the fixed facts say
`GENERAL_ADVICE` ends up with `GENERAL_ADVICE` in the record. If its text doesn't match,
the checks catch it. A reply that isn't JSON is treated like a failed check and sent back.

### Same seed, same output

Each candidate's random seed comes from the run seed, the cell and the attempt number,
so a run gives the same result with 1 worker or 8. Concurrency is set per model stage
(`models.<stage>.concurrency`).

## The six checks

![The six checks](diagrams/checks-flow.svg)

[Interactive version](diagrams/checks-flow.html)

The checks (`cascade`) run in order and stop at the first failure. Each one returns one
of three results:

- **pass:** go to the next check.
- **sent back** (`fail_repairable`): the model is re-prompted with the original prompt,
  the same fixed facts, and a section headed `## Your previous attempt was rejected`
  listing one error per line. This happens up to `validation.repair_tries` times
  (default 2). After that the candidate is dropped and the cell gets a new attempt.
- **dropped** (`fail_hard`): discarded with no re-prompt.

Every result and its issue codes are written to the audit trail of a kept record, or to
`drops.jsonl` for a dropped one. If a check itself crashes, that's treated as a bug in
sdgf and the run stops. It's never recorded as a bad record.

The two examples next to each check are:

- **FAG:** a conversation between a business-banking customer and the bank's assistant.
  The label says whether the assistant breached the advice policy.
- **CFA:** a multiple-choice exam question with a worked answer. The label is the
  correct option.

### L1 Shape

Checks that the candidate has the right structure. Three steps, each run only if the
previous one passed, because a broken record makes later steps report noise:

1. Fields: the task type's fields plus `output_schema.fields` are present, with the right types and allowed values.
2. Turns: numbered from 1, only known roles, alternating, starting with the customer.
3. Task type rules: for conversations, each span cites a turn that exists. For Q&A, the response contains an answer that can be read.

| | FAG | CFA |
|---|---|---|
| Fails when | The model returns the conversation but leaves out `severity`, `signal_categories` and four other required fields. | The `answer` field says C but the response ends `Answer: B`, or the question lists only one option. |
| Issue codes | `schema_required`, `turn_numbering`, `role_alternation`, `first_role` | `response_answer_errors`, `option_errors` |
| Result | sent back | sent back |

### L2 Rules

Checks the task's own rules, written in `task.yaml` and `hooks.py`. All three parts run
every time, so a single re-prompt lists every error:

1. Keyword rules (`validation.rules`): words that must or must not appear, optionally only in some roles or fields.
2. Label rule: the record's label must equal what `label_rule(record)` computes from its facts. For FAG, Tier 1 never breaches, Tier 3 always does, and Tier 2 breaches only on a Corporations Act product.
3. Extra checks (`extra_validators`): any other rule written as code.

| | FAG | CFA |
|---|---|---|
| Fails when | A span quotes the assistant as "you should go with the fixed rate", but the turn says "I'd go with the fixed rate". A span must be an exact substring of its turn. | Doesn't apply: CFA has no `hooks.py` and no keyword rules, so L2 has nothing to check and passes. |
| Issue codes | `span_not_verbatim`, `signal_without_span`, `spans_on_non_breach`, `severity_missing`, `policy_categories_mismatch` | none |
| Result | sent back | pass |

Because the label is a fixed fact, the label rule rarely fails in practice. Most L2
failures in FAG runs come from spans: they get reworded, they're missing for a declared
signal, or they appear on a no-breach record.

### L3 Safety scan

Checks all text in the candidate (any key starting with `_` is skipped) for:

- personal data: email, phone, ABN, TFN, BSB, account number, plus any patterns in `governance.extra_pii_patterns`;
- toxic language;
- secrets: API keys, private keys, tokens, password assignments;
- names on the deny list.

If a tool returned data marked anything other than `public`, the scan also looks for that
data copied into the text.

| | FAG | CFA |
|---|---|---|
| Fails when | The assistant writes "send it to BSB 062-000, account 1234 5678". | The question includes "email j.smith@example.com for the answer key". |
| Issue codes | `pii_bsb`, `pii_account_number` | `pii_email` |
| Result | dropped | dropped |

Issues record the rule name, path and character position, never the matched text,
because the drop log is written to disk.

### L4 Copy check

Compares the candidate with three sources, using Jaccard similarity over 5-character
pieces of the normalised text. It's dropped if any score is above
`thresholds.overlap_max` (0.80 for both tasks):

| Compared with | Issue code | Catches |
|---|---|---|
| seeds | `seed_overlap` | the model copied a few-shot example |
| held-out test set | `held_out_overlap` | leakage into an evaluation set |
| records kept this run | `near_duplicate` | the same record written twice |

| | FAG | CFA |
|---|---|---|
| Fails when | The conversation is the seed's term-deposit example with only the customer's name changed. | A second "bonds, Understand" question asks again whether a bond's price falls when rates rise. |
| Issue codes | `seed_overlap` | `near_duplicate` |
| Result | dropped | dropped |

The held-out check runs only when a path is passed at run time (`--held-out`). The path
is never saved, and the file is reduced to fingerprints on load, so its text can't
reach a prompt.

### L5 Blind judge

A second model labels the candidate without seeing the intended label. It sees only the
task type's judge fields: the conversation for FAG, the question for CFA. It never sees
the label, the spans that justify the label, or the response. It returns a verdict, a
score for each criterion in `rubric`, and a confidence for each.

The judge's `## Context` is `rubric.judge_context` if set, else `task.description`.
`task.description` is written for the generator, so a task whose description carries
writer's instructions (annotate every claim, the fixed parameters) should set
`judge_context` to just the definitions the judge needs. `rubric.examples` adds
judge-only worked examples (a record view, the expected verdict, optional scores and a
note) under `## Worked examples`. Neither ever reaches the generation prompt, and stage 0
scans the examples for PII and toxicity like the seeds.

| Judge result | Outcome |
|---|---|
| agrees, confident | pass |
| disagrees, confident | sent back (`judge_disagrees`); the re-prompt names the verdict given and the one needed |
| unsure (confidence below `escalation.low_confidence`) and `hitl.review_flagged` on | dropped from the automatic path and queued for a person (`sent_to_review`) |
| unsure, review off | passes L5, flagged for extra votes |
| output unusable | to review if it's on, else dropped (`judge_error`): a broken judge isn't fixed by rewriting the record |

A candidate is also flagged for extra votes if its cell is marked hard
(`escalation.on_hard_cells`) or contestable (`on_contestable`), or if
`escalation.always` is set.

| | FAG | CFA |
|---|---|---|
| Fails when | The fixed facts say Tier 2 general advice on a Corporations Act product, so the label is breach. The assistant only explains how the product works, so the judge says no breach with high confidence. | The record's answer is B. The judge, shown only the question, answers C because two options are both defensible. |
| Issue code | `judge_disagrees` | `judge_disagrees` |
| Result | sent back | sent back |

Cost: one judge call per candidate that reached L5.

### L6 Extra votes

Runs only on flagged candidates. Every other candidate passes without a call
(`method: skipped`), so the K calls go only where one verdict isn't enough.

| Mode | What happens |
|---|---|
| label set by code (FAG), judge trusted, L5 confident and agreeing | pass, no votes |
| label set by code (FAG), otherwise | K judge votes. More than half of the readable votes must agree with the label, else sent back (`consistency_disagrees`). |
| answer found by the model (CFA) | K fresh answers to the question. The majority answer must equal the record's answer, else sent back (`consistency_answer_mismatch`). No majority: sent back (`consistency_no_majority`). |

In both modes vote i is sampled at `validation.consistency.temperatures[i % len]`
(default 0.7 / 0.8 / 0.9), overriding the voting stage's own temperature. Without this,
a temperature-0 judge would cast L5's verdict K times. The votes come from
`models.consistency_judge` if set, so they can come from a different model than L5;
otherwise from `models.judge`. With L6 on and `consistency_k` above 1, stage 0 rejects a
temperature list that is all 0, since the K votes would be identical.

A vote that can't be read doesn't count either way. If none can be read, the judge is
broken, not the record: dropped (`consistency_no_votes`).

Each vote is kept with its stage, model and temperature, in L6's `ballots` detail and in
the record's provenance (`layer_results[].ballots`), so vote agreement can be analysed
after a run.

| | FAG | CFA |
|---|---|---|
| Flagged because | The judge agreed but with confidence 0.6, below 0.7. | Every CFA record is flagged (`escalation.always: true`). |
| Fails when | 5 votes: 2 breach, 3 no breach. | 5 answers: C, C, B, C, unreadable. The majority is C; the record says B. |
| Issue code | `consistency_disagrees` | `consistency_answer_mismatch` |
| Result | sent back | sent back |

Cost: up to K calls (`consistency_k`, 5 for both tasks), only for flagged candidates.
A judge counts as trusted only after a calibration for this spec version and judge model
has passed `thresholds.kappa_min` and `calibration.ece_max`. Until then, every flagged
FAG candidate gets votes.

### Sent back and dropped

```
attempt 0 ─► checks ─► pass ──────────────► kept
                    ├► dropped ───────────► drops.jsonl
                    └► sent back ──► attempt 1 = original prompt + errors ─► … ─► dropped after 2 re-prompts
```

A drop records the cell, the last check and its codes, the number of attempts, and the
check and codes of every attempt. The error rate for each check is computed from these
records. When a task uses tools, one tool session covers every attempt for a record, so
the per-record tool budget includes re-prompts.

A real FAG drop from a 15-record local run went through three attempts:

| Attempt | Failed at | Codes |
|---|---|---|
| 0 | L1 | `schema_required` ×6 |
| 1 | L2 | `spans_on_non_breach` |
| 2 | L5 | `judge_disagrees` |

Each re-prompt fixed the previous error, then the record failed a later check. After 2
re-prompts it was dropped.

## The measurements

Computed overall and per cell, from the kept records, the drop log, the usage log, and a
calibration if there is one:

| Measurement | Definition | Target key |
|---|---|---|
| judge agreement rate (`fidelity`) | kept records the blind judge agreed with ÷ kept records it judged | `fidelity_min` |
| kappa | Cohen's κ of the judge against the human gold set; needs a calibration | `kappa_min` |
| coverage | kept ÷ quota per cell, and the lowest fill of any cell | `coverage_min_cell_fill` |
| balance | the share of each value on a balanced axis compared with its target | `balance_tolerance` |
| diversity | distinct-n, self-BLEU, and cluster entropy (needs an embedder) | `distinct_n_min`, `self_bleu_max`, `semantic_diversity_min` |
| error rate per check | failures at that check ÷ candidates, re-prompts included | — |
| remaining label errors | estimated wrong labels left in the kept records, from judge precision on gold | `residual_error_max` |
| safety | kept records the safety scan still flags when re-run | `governance_violations_max` |
| copying | highest similarity of any kept record to a seed or held-out item | `overlap_max` |
| cost | tokens, USD and seconds per kept record, per model stage | `cost_per_record_max` |
| yield | kept ÷ candidates | — |

`*_min` targets pass at value ≥ target, `*_max` at value ≤ target. A missing measurement
fails its target unless you waive it with `--waive`. `governance_violations_max` can't be
waived.

## Files on disk

```
<store>/<spec_version>/
  shared/                coverage_plan.json, calibration results, tool_cache.jsonl
  runs/<run_id>/
    run.json spec.json cells.json     run manifest, intake summary, this run's cells and quotas
    accepted.jsonl drops.jsonl        kept records with audit trail; every drop with its history
    summary.json usage.json           scheduler snapshot; tokens, cost and seconds over every invocation
    cli_options.json                  options reused by resume / evaluate / release
    review.jsonl review_decisions.jsonl   when hitl.review_flagged is on
    rounds.json shortfall.json/.md    release rounds; the last failed release check

<releases>/<task>/<version>/
  dataset.jsonl provenance.jsonl dataset_card.md governance_report.json metrics.json manifest.json
```

The release directory is built under a temporary name and renamed into place, so it
either exists complete or not at all.

## Shared parts

| Part | Where | What it does |
|---|---|---|
| Task types | `tasktypes/` | the record shape, the L1 rules, the judge fields, and the answer reader for one kind of dataset (`classification_spans`, `sft_qa`) |
| Models | `models/` | one `call(prompt, max_tokens, temperature, tools)` for `openai_compat`, `anthropic`, `vllm`, `mlx` and `mock`; API keys are read only from the variable named in `api_key_env`; usage and price per stage |
| Tool gateway | `tools/` | allowed tools, per-record call and token budgets, a shared cache, sensitivity labels, and a trace of every call |
| Safety profile | `governance/` | global PII, toxicity, secret and name rules; a task can only make them stricter |
| Store | `store/` | the run and shared directories above |
| Review | `hitl/` | plan approval, the review queue, and decisions that feed the gold set |
| Judge calibration | `judge/calibration.py` | κ and ECE against the gold set; decides whether the judge is trusted |
