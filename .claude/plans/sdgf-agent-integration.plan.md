# Plan: sdgf as an agent-facing component of the system harness

**Created**: 2026-10-31
**Complexity**: Large (phases 1-3 give a usable run-only integration)
**Status**: Phase 1 done; Phase 2 next

## Summary

Make sdgf a component that agents call, alongside people using the CLI. A
harness-neutral service layer (`sdgf.service`) becomes the single source of truth; the
CLI, an MCP server and a Python harness are thin adapters over it. Agents can run
existing tasks and draft new task specs, which a person approves (bound to
`spec_version`) before any paid run. Quick operations are synchronous; runs and
releases are async jobs, one subprocess each.

## Decisions

| Question | Decision |
|---|---|
| Harness | Undecided between MCP-capable agents and an in-house Python orchestrator. Build a harness-neutral core; MCP adapter in phase 3, Python import / `sdgf.client` in phase 5 |
| Agent role | Run and author. Authored specs need human approval before a paid run |
| Runtime | Both: synchronous for validate / plan / describe / results / small dry runs; async jobs for run / resume / evaluate / release |
| Deployment | Same machine or shared service: MCP over stdio (local) and streamable HTTP (shared). HTTP without auth binds to localhost only |
| Approvers (open) | Default: CLI only (`sdgf drafts approve`, `sdgf plan --approve`) with a named reviewer. Agents see that approval is pending, never grant it |
| Hooks sandbox (open) | Default: subprocess with a timeout. `authoring.allow_hooks` off on a shared server, on locally. Sandbox behind one function so containers can replace it |

Principles:

1. `cli.py` calls the service, so adapters can't drift apart.
2. Jobs and draft validation run in a subprocess, never in the server process:
   agent-written `hooks.py` is untrusted code (mirrors `stress.py` isolation).
3. Jobs reuse resume: cancel = kill, restart = resume the same run id. Job state is a
   JSON file in the store, so a restarted server reattaches.
4. Approvals bind to `spec_version`; any edit invalidates them.
5. Credentials stay on the server. Agents name tasks and options, never keys, paths or
   `--backends` (unless an allowlist permits it).

## Architecture

```
adapters (thin)   sdgf.mcp (stdio + HTTP)  |  Python import / sdgf.client  |  sdgf.cli
                                  |
sdgf.service      typed requests/responses, ServiceError(code, message, hint)
                  tasks: list / describe / validate / plan / approve_plan
                  jobs: start(run|resume|evaluate|release) / status / cancel / list
                  results: summary, metrics, shortfall, records (paged), release files
                  review: list / resolve
                  drafts: create / write / validate / dry_run / submit / approve (human)
                                  |
sdgf.jobs         one subprocess per job, state in <store>/jobs/<id>/job.json
                                  |
Pipeline / ArtefactStore (unchanged)
```

## Phases

### Phase 1: Service core, no behaviour change (Medium)
- `src/sdgf/service/`: operations ported out of `cli.py`'s `cmd_*` functions, taking
  plain arguments instead of `argparse.Namespace` and returning JSON-ready dicts instead
  of printing.
- `ServiceError` with a code and hint; exit codes map to statuses (complete, incomplete,
  approval_pending, error).
- Task registry: named task roots, name -> directory, with path-traversal checks.
- `cli.py` becomes a thin wrapper.
- Validate: existing `tests/test_cli.py` passes unchanged; new `tests/test_service.py`
  drives the service in-process with the mock worlds.

### Phase 2: Jobs (Medium)
- `src/sdgf/jobs.py`: `JobManager.start/status/cancel/list/reattach`; each job is a
  subprocess; status read from the run directory (`summary.json`, `usage.json`).
- Per-job wall-clock timeout, max target size, spec budget; per-store concurrency limit.
- Validate: start -> poll -> complete, cancel -> resume, restart -> reattach; two new
  `sdgf.stress` scenarios (job killed mid-wave, hung job subprocess).

### Phase 3: MCP adapter, run-only (Medium)
- `src/sdgf/mcp/server.py` behind an optional `mcp` extra; stdio and streamable HTTP.
- Tools: list_tasks, describe_task, validate_spec, plan, start_run, start_release,
  job_status, cancel_job, get_results, get_records (paged), list_reviews.
- Human-only operations (approve_plan, waivers, resolving reviews) are not agent tools.
- Agent-sized outputs: summaries by default, detail on request, paged records.
- Validate: in-process MCP client test; manual run from Claude Code over the FAG mock flow.

### Phase 4: Authoring (High)
- Drafts workspace `<workspace>/drafts/<name>/`, separate from approved tasks.
- Tools: create_draft (template or copy), write_task_yaml, append_seeds, write_hooks
  (if allowed), read_draft, validate_draft (stage 0 in a subprocess), dry_run_draft
  (mock backend or L1-L4, no cost), submit_draft.
- Human: `sdgf drafts approve <name> --reviewer ...` promotes the draft into the task
  registry; paid runs accept approved specs only; approval shows the hooks diff and
  `seeds_sent_to`.
- Validate: scripted agent drafts, fails validation, fixes, dry-runs; edit after
  approval invalidates it; promotion refused without approval.

### Phase 5: Python adapter and hardening (Low-Medium)
- `sdgf.client` over HTTP when sdgf runs remotely; `sdgf.service` import when local.
- Shared mode: bearer tokens per client, per-client isolation of jobs and drafts,
  per-client job and spend limits, names only (no filesystem paths from agents).
- Traces tagged with job_id / agent_id; agent recorded in provenance; `<store>/audit.jsonl`.
- Docs: CONTEXT.md terms (Job, Draft, Approved spec), README section, how-it-works note.

## Patterns to Mirror

| Category | Source | Pattern |
|---|---|---|
| Errors | `src/sdgf/cli.py` `CLIError`, `src/sdgf/pipeline.py` `PipelineError` | One `ValueError` subclass per layer with a plain message |
| Output | `src/sdgf/cli.py` `_print`, `cmd_validate_spec` | Flat JSON dicts, sorted keys |
| Isolation | `src/sdgf/stress.py` | One process per unit of work, with a timeout |
| Approval | `approve_plan`, `EXIT_APPROVAL`, `src/sdgf/hitl/queue.py` | Human gates keyed on spec_version and a named reviewer |
| Tests | `tests/test_cli.py`, `tests/cli_backends.py` | Mock worlds, no model calls |

## Risks

| Risk | Level | Mitigation |
|---|---|---|
| Agent-written hooks.py runs arbitrary code | HIGH | Subprocess only, approval before paid runs, hooks diff to approver, `authoring.allow_hooks` |
| Agents spend money | HIGH | Per-job and per-client caps, no paid run on an unapproved spec |
| Agents waive gates to force a release | HIGH | Waivers human-only; governance unwaivable |
| CLI refactor changes behaviour | MEDIUM | Phase 1 gated on the existing CLI tests |
| Agent-written seeds reach an external generator | MEDIUM | Surface `seeds_sent_to` in validate_draft and at approval |
| Outputs too large for agent context | MEDIUM | Summaries by default, paged records |

## Acceptance
- [x] Phase 1: service layer, CLI on top, tests green (2026-10-31)
- [ ] Phase 2: jobs
- [ ] Phase 3: MCP adapter
- [ ] Phase 4: authoring
- [ ] Phase 5: hardening and Python client
