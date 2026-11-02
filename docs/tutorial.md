# sdgf tutorial: from a task spec to a released dataset, by hand and by agents

This tutorial walks through everything sdgf does, in the order you'd meet it:

| Part | You will | Section |
|---|---|---|
| 1 | Run the pipeline from the command line: validate, plan, generate, check, release | [The pipeline](#part-1-the-pipeline-from-the-command-line) |
| 2 | Do the same from Python | [Python](#part-2-the-same-from-python) |
| 3 | Run long steps in the background, cancel and continue them | [Jobs](#part-3-long-steps-as-background-jobs) |
| 4 | Let an agent run tasks through MCP | [Agents run tasks](#part-4-letting-an-agent-run-tasks) |
| 5 | Let an agent write a new task, which a person approves | [Agents write tasks](#part-5-letting-an-agent-write-a-task) |
| 6 | Share one server between several teams | [Shared server](#part-6-a-shared-server-for-several-teams) |

Every step runs offline on mock models, so it costs nothing and needs no API keys. The
outputs shown are from real runs, trimmed with `jq` to the fields that matter. Terms
such as *cell*, *record* and *release check* are defined in [../CONTEXT.md](../CONTEXT.md);
[overview.md](overview.md) explains what sdgf checks and why.

## Setup

```bash
cd sdgf
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,mcp]'          # mcp is needed from part 4 on
export PYTHONPATH=src:tests          # the mock models live in tests/cli_backends.py
S=/tmp/sdgf-tutorial                 # where this tutorial writes
```

`--backends cli_backends:fag_world` swaps every model in the task for a mock that writes
valid records and judges them correctly. Other mocks in that file misbehave on purpose:
`fag_broken` can never fill one cell, `fag_unsure` gives low-confidence verdicts, and
`fag_slow` takes time, so a job is still running when you act on it.

## Part 1: the pipeline from the command line

### 1.1 The task

A task is a directory of three files. `tasks/fag` is the reference task: bank
customer/assistant conversations labelled by whether the assistant gave financial advice
it shouldn't.

| File | Holds |
|---|---|
| `task.yaml` | what a record looks like, what the dataset must cover, the models, how to judge, the release targets |
| `hooks.py` | the rules code decides, e.g. which label is correct for a record's facts (optional) |
| `seeds.jsonl` | a few hand-written example records |

[new-use-case.md](new-use-case.md) explains how to write your own.

### 1.2 Validate the spec (stage 0)

```bash
python -m sdgf.cli validate-spec tasks/fag
```
```json
{"ok": true, "task": "fag", "task_type": "classification_spans", "generation_mode": "label_first",
 "seeds": 6, "layers": ["L1", "L2", "L3", "L4", "L5", "L6"], "spec_version": "568e6c76…"}
```

`spec_version` is a hash of the three files. Changing any of them makes a new version,
and every run, plan and release is filed under it.

### 1.3 Plan the coverage (stage 1)

```bash
python -m sdgf.cli plan tasks/fag --store $S/store --target-size 20
```
```json
{"built": true, "target_size": 20, "cells": [{"id": "corps_act|true|single_turn", "quota": 2}, …],
 "approval_required": false, "approved": false}
```

The plan splits the 20 records over 16 *cells* (product scope × label × conversation
length). Code fixes each record's cell and label before any model writes.

### 1.4 Generate and check (stages 2 and 3)

```bash
python -m sdgf.cli run tasks/fag --store $S/store --target-size 20 --run-id demo \
    --backends cli_backends:fag_world
```
```json
{"accepted": 20, "complete": true, "stop_reason": "complete", "drops_by_layer": {},
 "usage": {"calls": 73, "tokens": 144091, "cost_usd": 0.0}}
```

Each candidate went through the six checks L1 to L6. You will also see a warning that L6
votes were identical on every escalated record: the mock judge never varies its vote,
which is exactly what that warning exists to catch with real models.

### 1.5 Evaluate (stages 4 and 5, read only)

```bash
python -m sdgf.cli evaluate tasks/fag --store $S/store --run-id demo --backends cli_backends:fag_world
echo $?    # 1
```
```json
{"passed": false, "gate": {"failing_metrics": ["kappa_min", "residual_error_max", "semantic_diversity_min"], …}}
```

The release check fails because three targets couldn't be measured: agreement with
people needs a judge calibrated against a human gold set, and semantic diversity needs an
embedder. Exit code 1 means "finished, but not complete".

### 1.6 Release

```bash
python -m sdgf.cli release tasks/fag --store $S/store --releases $S/releases --run-id demo \
    --backends cli_backends:fag_world --waive kappa_min residual_error_max semantic_diversity_min
ls $S/releases/fag/*/
```
```
dataset.jsonl  dataset_card.md  governance_report.json  manifest.json  metrics.json  provenance.jsonl
```

`--waive` only lets through a target that couldn't be measured at all, and never the
governance (safety) target. A release directory is never changed after it is written.

### 1.7 When a run stops short

```bash
python -m sdgf.cli run tasks/fag --store $S/store --target-size 20 --run-id stalled \
    --max-attempts-per-cell 2 --backends cli_backends:fag_broken
```
```json
{"accepted": 18, "complete": false, "stop_reason": "stalled"}
```
```bash
python -m sdgf.cli resume tasks/fag --store $S/store --run-id stalled --backends cli_backends:fag_world
```
```json
{"accepted": 20, "accepted_this_invocation": 2, "complete": true}
```

A run keeps every record it accepted, even if it is killed. `resume` reuses the options
the run started with and generates only what is missing.

### 1.8 People in the loop: plan approval and the review queue

Set `hitl.approve_coverage_plan: true` and `hitl.review_flagged: true` in a copy of the
task (here `$S/tasks/fag`). A run now waits for a person to approve the plan:

```bash
python -m sdgf.cli run $S/tasks/fag --store $S/store2 --target-size 20 --run-id h1 \
    --max-attempts-per-cell 20 --backends cli_backends:fag_unsure
# sdgf: coverage plan coverage_plan-t20-s0 is awaiting approval; review it and approve it (…), then run again
echo $?    # 3
python -m sdgf.cli plan $S/tasks/fag --store $S/store2 --target-size 20 --approve --reviewer "A. Reviewer"
python -m sdgf.cli run  $S/tasks/fag --store $S/store2 --target-size 20 --run-id h1 \
    --max-attempts-per-cell 20 --backends cli_backends:fag_unsure
```
```json
{"accepted": 20, "drops_by_layer": {"L5": 7}}
```

The judge was unsure about 7 candidates, so they went to people:

```bash
python -m sdgf.cli review $S/tasks/fag --store $S/store2 list
python -m sdgf.cli review $S/tasks/fag --store $S/store2 resolve ITEM_ID --action accept --reviewer "A. Reviewer"
```

`--action` is `accept`, `reject` or `relabel --new-label …` (a relabel must change the
label). Decisions build the task's gold set, which is what a judge is later calibrated on.

### 1.9 Exit codes and where things go

| Exit | Meaning |
|---|---|
| 0 | done |
| 1 | finished but not complete: a stalled run, a failed release check, no release |
| 2 | an error (the message says what) |
| 3 | the coverage plan awaits a person's approval |

```
$S/store/<spec_version>/
  shared/coverage_plan-t20-s0.json      the plan, shared by every run of this spec_version
  runs/<run_id>/                        spec.json cells.json cli_options.json summary.json usage.json
                                        accepted.jsonl drops.jsonl review.jsonl rounds.json jobs.json
$S/releases/<task>/<version>/           the release (1.6)
```

### 1.10 Real models

`tasks/fag/task.yaml` points at a local vLLM server; `task.local.yaml`, `task.openai.yaml`
and `task.anthropic.yaml` are the same task on other models. Drop `--backends`, set the
API key variables the spec names ([Environment](#environment-variables)), and every
command above works the same. Set `models.<stage>.input_cost_per_mtok` and
`output_cost_per_mtok` to see real costs in `usage`.

## Part 2: the same from Python

`sdgf.service.Service` is what the CLI runs on: each step returns an `Outcome` with a
`status` (`ok`, `incomplete` or `approval_pending`) and the same data the CLI prints.

```python
from cli_backends import fag_world
from sdgf.service import RunOptions, Service, ServiceError, TaskRegistry

service = Service("/tmp/sdgf-tutorial/py-store", tasks=TaskRegistry(["tasks"]))
outcome = service.run("fag", "py1", options=RunOptions(target_size=20), backends=fag_world)
print(outcome.status, outcome.data["accepted"])          # ok 20

page = service.records("fag", "py1", limit=2).data      # a page of accepted records
print(page["total"], page["next_offset"])                # 20 2

try:
    service.resume("fag", "py1", options=RunOptions(seed=7))
except ServiceError as e:
    print(e.code, e.hint)                                # options_clash …
```

With a `TaskRegistry`, tasks are named (`"fag"`). `Service(..., allow_paths=False)` refuses
paths, `..` and symlinks out of the task roots, which is how the agent-facing parts use it.

## Part 3: long steps as background jobs

A real run takes hours. `sdgf.jobs.JobManager` runs `run`, `resume`, `evaluate` and
`release` each in its own process and returns at once:

```python
from sdgf.jobs import JobManager
from sdgf.service import RunOptions

jobs = JobManager("/tmp/sdgf-tutorial/py-store", service=service,
                  releases="/tmp/sdgf-tutorial/py-releases", backends="cli_backends:fag_slow")
job = jobs.start("run", "fag", options=RunOptions(target_size=20))
jobs.status(job["job_id"])["progress"]      # {'accepted': 3, 'target': 20, 'usage': {…}}
jobs.cancel(job["job_id"])                  # state 'cancelled'; accepted records are kept
again = jobs.restart(job["job_id"])         # a new job on the same run
done = jobs.wait(again["job_id"])
done["data"]["accepted"], done["data"]["accepted_this_invocation"]    # (20, 15)
```

| State | Meaning |
|---|---|
| `running` | the worker is alive |
| `finished` | the step returned; `status` is `ok`, `incomplete` or `approval_pending`, `data` its result |
| `failed` | `error` has a `code` and a `hint`; `log_tail` the worker's last lines |
| `timed_out` | over the manager's `timeout_s`; the worker stops itself even with no manager running |
| `cancelled` | stopped by `cancel` |
| `lost` | the worker died without a result (killed, machine restarted) |

Any stopped job continues with `restart`. Job state lives in `<store>/jobs/<id>/`, so a
new `JobManager` (after a server restart, say) sees every job, and a running job outlives
the manager that started it. Whoever creates the manager fixes where releases go, which
models jobs use, the time limit, how many jobs run at once and the largest target size.
Jobs need macOS or Linux.

## Part 4: letting an agent run tasks

`python -m sdgf.mcp` serves your tasks to MCP clients such as Claude Code:

```bash
python -m sdgf.mcp --store $S/mcp-store --tasks tasks/ --releases $S/mcp-releases \
    --backends cli_backends:fag_world \
    --waivable kappa_min residual_error_max semantic_diversity_min
```

In Claude Code:

```bash
claude mcp add sdgf -- env PYTHONPATH=src:tests python -m sdgf.mcp --store /tmp/sdgf-tutorial/mcp-store \
    --tasks tasks/ --releases /tmp/sdgf-tutorial/mcp-releases --backends cli_backends:fag_world \
    --waivable kappa_min residual_error_max semantic_diversity_min
```

Then ask, for example, *"Release a 20-record FAG dataset and show me three records."* The
agent will call `list_tasks`, `describe_task`, `start_release` (which returns a job id at
once), `wait_job` until the job leaves `running`, and `get_records`:

```json
{"state": "finished", "status": "ok", "data": {"released": true, "stop_reason": "released", …}, "omitted": ["rounds"]}
```

Results are brief by default (`omitted` lists what `detail=True` adds) and records come at
most 100 a page, so replies fit an agent's context. Every tool is in
[the reference](#mcp-tools).

What an agent can't do, on purpose:

- **Pass paths.** It names tasks; the operator chose the store, task roots and release directory.
- **Approve or review.** Plan approval and the review queue stay with people (1.8); `plan`
  returns the command a person runs as a `hint`.
- **Waive more than you allow.** `--waivable` lists what agents may waive; governance never.
- **Choose models.** `--backends` is the operator's, for every job.

Errors reach the agent as JSON it can act on:

```json
{"code": "unknown_task", "message": "unknown task '../tasks/fag'", "hint": "list_tasks names the tasks available"}
```

The quick tools (`describe_task`, `plan`, …) import the task's `hooks.py` in the server
process, so serve only tasks you trust. Agent-written tasks go through part 5 instead.

## Part 5: letting an agent write a task

Add a drafts workspace and test models for dry runs:

```bash
python -m sdgf.mcp --store $S/mcp-store --tasks tasks/ --releases $S/mcp-releases \
    --backends cli_backends:fag_world --waivable kappa_min residual_error_max semantic_diversity_min \
    --drafts $S/drafts --dry-run-backends cli_backends:fag_world
```

The agent drafts, checks and submits:

| Agent calls | Result |
|---|---|
| `create_draft(name="fag-v2", from_task="fag")` | `state: editing`, a copy of the task's three files |
| `write_draft_file`, `append_seeds`, `delete_draft_file` | edits; each one withdraws an earlier submission |
| `validate_draft(name="fag-v2")` | `ok: true` and the spec, or what's wrong; `warnings` if seed text would go to an external model |
| `dry_run_draft(name="fag-v2")` | a job (`task: draft:fag-v2`) on the test models: L1–L4 only, 10 records, its own store |
| `submit_draft(name="fag-v2")` | `state: submitted`, `hint: waiting for a person to approve or reject it` |

A person decides:

```bash
python -m sdgf.cli drafts --drafts $S/drafts list
python -m sdgf.cli drafts --drafts $S/drafts show fag-v2 --into tasks/      # files, diff against tasks/fag-v2, validation
python -m sdgf.cli drafts --drafts $S/drafts approve fag-v2 --reviewer "A. Reviewer" --into tasks/
python -m sdgf.cli drafts --drafts $S/drafts reject fag-v2 --reviewer "A. Reviewer" --reason "tier 2 is ignored"
```

`approve` copies the draft into `tasks/fag-v2`, where `start_run` can now use it (it
refuses to overwrite an existing task without `--replace`). `reject` sends the reason
back: the agent sees it in `read_draft` and edits again.

Approval binds to the exact bytes submitted, so you only ever approve what you saw. A
draft's `hooks.py` is code an agent wrote: sdgf imports it only in a subprocess with a
timeout and none of the server's environment beyond `PATH`, `PYTHONPATH`, `HOME`, locale
and temp settings, so it can't read your API keys. That isn't a sandbox (it can still
reach the network and the file system), so `--hooks deny` stops agents writing
`hooks.py` at all; it is the default over `--http`. A draft copied from a task keeps that
task's approved `hooks.py`.

## Part 6: a shared server for several teams

Give each team or harness a token:

```bash
python -m sdgf.clients new team-a
```
```json
{"token": "sdgf_…", "entry": {"id": "team-a", "token_sha256": "…"}}
```

Give the token to team-a and put only the entry, with its limits, in a clients file:

```json
{"clients": [{"id": "team-a", "token_sha256": "…", "max_running": 1, "max_tokens": 1000000,
              "max_cost_usd": 50, "allow_hooks": false}]}
```

```bash
python -m sdgf.mcp --store $S/shared-store --tasks tasks/ --releases $S/shared-releases \
    --backends cli_backends:fag_world --drafts $S/shared-drafts \
    --http --host 0.0.0.0 --port 8765 --clients clients.json --public-url https://sdgf.example
```

From a Python harness:

```python
from sdgf.client import SdgfClient

async with SdgfClient("https://sdgf.example/mcp", token=TOKEN) as sdgf:
    job = await sdgf.call("start_run", task="fag", target_size=20)
    done = await sdgf.wait(job["job_id"])                # however long it takes
    usage = await sdgf.call("my_usage")
# done["data"]["accepted"] == 20
# usage == {"client": "team-a", "running": 0, "runs": 1, "tokens": 144091, "cost_usd": 0.0, …}
```

`sdgf.client.call(url, tool, token=…, **args)` makes one blocking call. A harness on the
same machine as the store can use `Service` and `JobManager` (parts 2 and 3) directly.

What the server enforces:

- **Who:** every call needs a known bearer token; anything else gets HTTP 401. Without
  `--clients`, `--http` binds only to a loopback address.
- **Whose:** a client sees and controls only its own jobs (another's job id is
  `unknown_job`) and drafts, which go under `<drafts>/<client id>/` (so a person reviews
  team-a's with `sdgf drafts --drafts $S/shared-drafts/team-a …`). Approved tasks are shared.
- **How much:** `max_running`, `max_tokens` and `max_cost_usd` are checked before every
  job starts or restarts, against what the client's runs have used (`my_usage`). Runs on
  unpriced models count tokens but not cost. A running job isn't stopped part way; the
  task's own `budget` still caps each run.
- **Record:** every call goes to `<store>/audit.jsonl` with the client, tool, arguments
  (long text and lists by size only) and outcome; each run's `jobs.json` names the jobs
  and clients that touched it.

Tokens travel in the clear over plain HTTP: put TLS in front of the server.
`--held-out FILE …` turns on the L4 held-out check for every real run without writing the
paths anywhere, and `--trace langsmith` traces every run and release job, tagged with the
job and client.

## Reference

These tables are generated from the code by `python docs/tutorial_reference.py`.

### Command line

<!-- AUTO-GENERATED:cli-commands -->
| Command | What it does |
|---|---|
| `sdgf validate-spec` | compile the spec (stage 0) and report it |
| `sdgf plan` | build or show the coverage plan (stage 1) |
| `sdgf run` | generate and validate a run (stages 0 to 3) |
| `sdgf resume` | continue a run with its saved options (default: the latest) |
| `sdgf evaluate` | compute metrics and gate a run without writing (stages 4, 5) |
| `sdgf release` | run, evaluate, gate and release, refilling short cells |
| `sdgf review` | list or resolve a run's HITL review queue |
| `sdgf review … list` | pending review items |
| `sdgf review … resolve` | accept, reject or relabel one item |
| `sdgf drafts` | review task specs agents wrote: list, show, approve or reject |
| `sdgf drafts … list` | every draft and its state |
| `sdgf drafts … show` | a draft's files, diff and validation, for review |
| `sdgf drafts … approve` | copy a submitted draft into a task directory |
| `sdgf drafts … reject` | send a submitted draft back with a reason |
<!-- /AUTO-GENERATED:cli-commands -->

Other entry points: `python -m sdgf.mcp` (below), `python -m sdgf.clients new ID`, and
`python -m sdgf.stress TASK`, which runs a task through 17 fault scenarios (see the README).

### MCP server options

<!-- AUTO-GENERATED:mcp-options -->
| Option | What it sets |
|---|---|
| `--store STORE` | artefact store root |
| `--tasks TASKS` | directories of task specs |
| `--releases RELEASES` | release root; without it, no start_release |
| `--backends MODULE:ATTR` | model overrides for every job |
| `--max-running MAX_RUNNING` | jobs running at once (default 2) |
| `--timeout TIMEOUT` | per job, seconds (default 3600.0) |
| `--max-target-size MAX_TARGET_SIZE` | the largest target size a job may ask for |
| `--waivable THRESHOLD` | thresholds agents may waive (governance can never be waived) |
| `--held-out PATH` | L4 held-out files |
| `--trace {langsmith}` | trace every run and release job |
| `--drafts DRAFTS` | drafts workspace; enables the authoring tools |
| `--hooks {allow,deny}` | may agents write a draft's hooks.py (default: allow over stdio, deny over --http) |
| `--dry-run-backends MODULE:ATTR` | test models for dry runs |
| `--dry-run-max-target DRY_RUN_MAX_TARGET` | the most records a dry run makes (default 10) |
| `--http` | streamable HTTP instead of stdio |
| `--host HOST` | the address --http binds to (default 127.0.0.1) |
| `--port PORT` | the port --http listens on (default 8765) |
| `--clients CLIENTS` | clients file (sdgf.clients); HTTP only |
| `--public-url PUBLIC_URL` | the URL clients reach this server at (with --clients) |
<!-- /AUTO-GENERATED:mcp-options -->

### MCP tools

The draft tools exist only with `--drafts`.

<!-- AUTO-GENERATED:mcp-tools -->
| Tool | What it does |
|---|---|
| `list_tasks()` | The task specs this server can run, with their type and description. |
| `describe_task(task)` | A task's coverage axes, target size, models, release thresholds and budget. |
| `validate_spec(task)` | Compile a task spec (task.yaml, hooks.py, seeds) and report its spec_version. |
| `plan(task, target_size)` | Build or show the coverage plan: the cells and how many records each needs. If approval_required and not approved, generation waits for a person. |
| `start_run(task, target_size, seed, layers, max_attempts_per_cell)` | Start generating and checking records (stages 0 to 3) as a job; returns its status at once. Calls models, so it costs what the task's models cost. |
| `start_release(task, run_id, target_size, seed, waive)` | Start a job that runs, evaluates and releases in rounds, refilling short cells, and writes a release only if every threshold passes. run_id continues that run. waive lists thresholds that may stay unmeasured, among those this server allows. |
| `start_evaluate(task, run_id, waive)` | Start a job computing a run's metrics and release check (the latest run by default), without writing a release. |
| `job_status(job_id, detail)` | A job's state (running, finished, failed, timed_out, cancelled, lost), progress and result. detail=True includes the full result (e.g. every metric). |
| `wait_job(job_id, timeout_s, detail)` | Wait up to timeout_s seconds (at most 120) for a job to stop running, then return its status as job_status does. |
| `list_jobs()` | Your jobs, oldest first, with their state and progress; dry_run marks a draft's. |
| `cancel_job(job_id)` | Stop a running job. Records it already accepted are kept; restart_job continues. |
| `restart_job(job_id)` | Continue a stopped job's run (cancelled, timed out, lost or failed) in a new job; nothing already accepted is generated again. |
| `my_usage()` | What your jobs have used (tokens, cost, runs, running jobs) and your limits. |
| `get_records(task, run_id, offset, limit)` | One page of a run's accepted records (the latest run by default), at most 100 at a time, without their audit trail; next_offset is null on the last page. |
| `list_reviews(task, run_id)` | Records the judge flagged for a person to review; a person resolves them. |
| `list_drafts()` | Your task drafts and their state: editing, submitted, rejected or approved. |
| `create_draft(name, from_task)` | Start a task draft, empty or a copy of an existing task's spec files. |
| `read_draft(name)` | A draft's task.yaml and hooks.py, a preview of its seeds, and its state (with a person's reason if they rejected it). |
| `write_draft_file(name, file, content)` | Replace one spec file of a draft: task.yaml, seeds.jsonl or hooks.py (if this server allows agents to write hooks). Any edit needs submitting again. |
| `append_seeds(name, records)` | Add example records to a draft's seeds.jsonl (at most 1000 per call). |
| `delete_draft_file(name, file)` | Remove one spec file from a draft (e.g. hooks.py, which is optional). |
| `validate_draft(name)` | Compile a draft (in a separate process) and describe it, or say what is wrong. warnings include where seed text would be sent. |
| `dry_run_draft(name, target_size)` | Start a small run of a draft on this server's test models, checks L1-L4 only, as a job (see job_status). Shows whether the spec produces records that pass. |
| `submit_draft(name)` | Validate a draft and submit exactly this content for a person to approve. Edits afterwards withdraw it. |
<!-- /AUTO-GENERATED:mcp-tools -->

### Optional extras

Install with `pip install -e '.[name,…]'`; the README's Optional adapters section says
what each enables.

<!-- AUTO-GENERATED:extras -->
| Extra | Installs |
|---|---|
| `pii` | `presidio-analyzer`, `presidio-anonymizer` |
| `toxicity` | `detoxify` |
| `embeddings` | `sentence-transformers` |
| `retrieval` | `rank-bm25` |
| `dev` | `pytest`, `ruff` |
| `docs` | `markdown>=3.5` |
| `tracing` | `langsmith>=0.2` |
| `mcp` | `mcp>=2.3,<3` |
<!-- /AUTO-GENERATED:extras -->

### Environment variables

<!-- AUTO-GENERATED:environment -->
| Variable | Needed for |
|---|---|
| `ANTHROPIC_API_KEY` | the API key of models in `tasks/fag/task.anthropic.yaml`, `tasks/fag/task.openai.yaml` |
| `OPENAI_API_KEY` | the API key of models in `tasks/fag/task.local.yaml`, `tasks/fag/task.openai.yaml` |
| `OPENROUTER_API_KEY` | the API key of models in `tasks/fag/task.anthropic.yaml` |
| `VLLM_API_KEY` | the API key of models in `tasks/cfa/task.yaml`, `tasks/fag/task.yaml` |
| `LANGSMITH_API_KEY` | read by `src/sdgf/tracing.py` |
| `LANGSMITH_ENDPOINT` | read by `src/sdgf/tracing.py` |
<!-- /AUTO-GENERATED:environment -->

## Troubleshooting

| You see | Because | Do |
|---|---|---|
| exit 3, `awaiting approval` | the task requires plan approval | `sdgf plan … --approve --reviewer NAME` |
| `options_clash` (`run … was started with seed=0`) | a run's options are fixed once it starts | drop the option, or start a new run id |
| exit 1, `stop_reason: stalled` | a cell couldn't be filled within its attempts | look at `drops.jsonl`, fix the cause, `resume` |
| `ModuleNotFoundError` for a backends module | it isn't importable by the worker | install it, or put its directory on `PYTHONPATH` before starting sdgf |
| job `lost` | the worker died without a result | `restart` it; nothing accepted is lost |
| `too_many_jobs` / `budget_exhausted` | a server or client limit | wait, cancel a job, or raise the limit |
| `waiver_not_allowed` | the threshold isn't in `--waivable` | ask a person to release, or calibrate the judge |
| `hooks_not_allowed` | `--hooks deny` or the client's `allow_hooks: false` | use the task type's built-in rules, or have a person write the hooks |
| `not_submitted` on approve | the draft changed since it was submitted | the agent submits again |
| HTTP 401 | a missing or unknown token | check the client's token and the clients file |
