# sdgf overview

sdgf builds synthetic training and evaluation datasets. You describe the dataset in three
files. sdgf plans what to generate, has a model write each record, checks every record
six times, and releases the dataset only if it meets the targets you set.

This page is for anyone deciding whether to use it. [how-it-works.md](how-it-works.md)
has the details, and [new-use-case.md](new-use-case.md) explains how to set up your own
dataset. Terms are defined in [../CONTEXT.md](../CONTEXT.md).

![sdgf architecture](diagrams/architecture.svg)

[Interactive version](diagrams/architecture.html)

## What goes in

A **task spec** is one directory with three files:

| File | What you write in it |
|---|---|
| `task.yaml` | what each record looks like, what the dataset must cover, which models to use, how to judge a record, and the targets for release |
| `hooks.py` | the rules that code decides, such as "which label is correct for these facts" (optional) |
| `seeds.jsonl` | a few hand-written example records for the model to imitate |

## What comes out

A release directory that can't be changed after it is written:

| File | Contents |
|---|---|
| `dataset.jsonl` | the records |
| `provenance.jsonl` | for each record: the spec version, the models used, the prompt, and the result of every check |
| `metrics.json` | the measurements the release was judged on |
| `governance_report.json` | every model endpoint that received data, with external providers listed separately |
| `dataset_card.md`, `manifest.json` | a readable summary and a file list with hashes |

If the dataset misses a target, there is no release. You get a shortfall report that
names the targets it missed and the parts of the plan that are short.

## Code decides, the model writes

Before the model writes anything, code chooses the facts of each record: which part of
the plan it fills and, where the task has one, its label. The model only writes text that
fits those facts. If the model's output tries to change a fact, the fact wins.

Two use cases ship with the framework:

| | FAG (financial advice guardrail) | CFA (exam questions) |
|---|---|---|
| A record is | a bank customer/assistant conversation | a multiple-choice question with a worked answer |
| The label | does the assistant breach the advice policy? (yes/no) | the correct option, A–D |
| Who decides the label | code, from the advice tier and product scope | the model; independent answers must agree with it |
| Plan covers | product scope × breach × length | concept keyword × Bloom level |

## The six checks

Every record goes through these in order and stops at the first one it fails. The four
free checks run first, so the paid judge never sees a record that already failed.

| | Check | Question it answers | If it fails | Cost |
|---|---|---|---|---|
| L1 | Shape | Are all fields present, with the right types and turn order? | sent back | free |
| L2 | Rules | Does the record follow the task's rules, including the label rule? | sent back | free |
| L3 | Safety scan | Does the text contain personal data, secrets, toxic language or banned names? | dropped | free |
| L4 | Copy check | Is it a near-copy of a seed, a held-out test item, or a record already kept? | dropped | free |
| L5 | Blind judge | Does a second model, not shown the label, reach the same label? | sent back | 1 call |
| L6 | Extra votes | For records the judge was unsure about: does a majority of K votes agree? | sent back | K calls |

"Sent back" means the model gets the same prompt plus a list of what was wrong, up to 2
more times. After that the record is dropped. L3 and L4 drop at once, because telling the
model what it leaked teaches it to hide the leak rather than avoid it.

## What release requires

`task.yaml` sets a target for each measurement, and every target must be met:

- **Judge agreement:** the share of kept records where the blind judge agreed with the label.
- **Coverage:** every part of the plan filled to its quota, and each label at its target share.
- **Diversity:** records that don't repeat each other's wording.
- **Copying:** no record too close to a seed or a held-out test item.
- **Safety:** zero safety-scan findings. This target can't be waived.
- **Cost:** money and tokens spent per kept record.

## What it can't guarantee

- **The label is only as good as the rule.** The checks confirm that each record matches
  the rule in `hooks.py`. If the rule encodes the policy wrongly, every record will be
  wrong in the same way, and every check will pass.
- **The judge isn't trusted until it has been measured against people.** Agreement
  between the judge and the label shows that two models agree, not that they are right.
  Until a human-labelled gold set has been used to calibrate the judge, the "agrees with
  people" (kappa) and "remaining label errors" measurements are missing, and a release
  has to waive them explicitly.
- **The default safety scan only finds what is on its lists.** It uses regular
  expressions for personal data (email, phone, ABN, TFN, BSB, account numbers) and a
  keyword list for toxicity. Anything they don't match gets through. Installing the
  `pii` (Presidio) and `toxicity` (Detoxify) extras swaps in model-based scanners.
