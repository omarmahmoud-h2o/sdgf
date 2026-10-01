# sdgf

sdgf turns a written task spec into a synthetic dataset. Code fixes the facts of each record, a model writes the text, and a fixed series of checks decides which records are kept.

## The spec

**Task spec**:
The three files that define one dataset: `task.yaml`, an optional `hooks.py`, and `seeds.jsonl`.
_Avoid_: config, use-case definition

**Spec version** (`spec_version`):
A hash of the three task spec files. Editing any of them starts a new version.
_Avoid_: task version, revision

**Seed**:
A hand-written example record in `seeds.jsonl`, shown to the model as an example to imitate.
_Avoid_: gold example, template

## Planning

**Axis**:
One dimension the dataset must cover, such as product scope or label, with a fixed set of values.
_Avoid_: dimension, facet

**Cell**:
One combination of axis values, with a quota of records to keep.
_Avoid_: bucket, stratum, slot

**Fixed facts** (`recipe`):
The field values code chooses for one record before the model writes. The model's output can't change them.
_Avoid_: recipe, scenario, parameters

**Label set by code** (`label_first`):
The mode where code chooses the label before generation and the checks confirm that the text matches it.
_Avoid_: label-first

**Answer found by the model** (`answer_emergent`):
The mode where the model's answer becomes the label and the checks confirm that independent answers agree with it.
_Avoid_: answer-emergent

## Checking

**Candidate**:
One model output for one set of fixed facts, before it has passed the checks.
_Avoid_: draft, sample

**Record**:
A candidate that passed every check and was kept.
_Avoid_: example, sample, row

**The checks** (`cascade`):
The six layers L1 to L6 that every candidate goes through in order, stopping at the first failure.
_Avoid_: cascade, pipeline, validators

**Sent back** (`fail_repairable`):
A check result that re-prompts the model with the original prompt plus the errors found.
_Avoid_: repairable failure, retry

**Dropped** (`fail_hard`):
A check result that discards the candidate without re-prompting.
_Avoid_: hard failure, rejected

**Safety scan** (L3, `governance`):
The check for personal data, secrets, toxic language and banned names in a candidate's text.
_Avoid_: governance, compliance check

**Copy check** (L4, `overlap`):
The check that a candidate isn't a near-copy of a seed, a held-out item, or a record already kept.
_Avoid_: overlap, dedup

**Blind judge** (L5):
A second model that labels the candidate without being shown the intended label.
_Avoid_: decision judge, evaluator

**Flagged for extra votes** (`escalate`):
A candidate the blind judge was unsure about, or one from a cell marked hard, that goes on to extra votes.
_Avoid_: escalation

**Extra votes** (L6, `consistency`):
K more judge calls on a flagged candidate. The majority must agree with the record's label.
_Avoid_: adaptive consistency, self-consistency

**Trusted judge**:
A judge whose agreement with human labels has been measured for this spec version and judge model, and passed.
_Avoid_: calibrated judge

## Release

**Judge agreement rate** (`fidelity`):
The share of kept records whose blind-judge label matched the intended label.
_Avoid_: fidelity, accuracy

**Release check** (`gate`):
The comparison of the run's measurements with the spec's thresholds. A dataset is released only if every threshold is met.
_Avoid_: gate, quality gate

**Release**:
A directory that can't be changed after it is written. It holds the dataset, its audit trail and its report, and exists only if the release check passed.
_Avoid_: export, snapshot

**Audit trail** (`provenance`):
The per-record log of the spec version, models, prompt hash, seed, cell and every check result.
_Avoid_: provenance, lineage
