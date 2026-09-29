"""The run pipeline (FRAMEWORK_DESIGN.md §5, §6): stages 0 to 5.

    0 intake     compile task.yaml + hooks.py + seeds, resolve the task type
    1 coverage   coverage/plan.py: keywords × axes ─► cells × quotas, cached per spec_version
    2 generate   scheduler ─► sampler_constraints ─► prompt ─► generator backend
    3 validate   cascade L1-L6 ─► repair ─► accepted + provenance | drop log
    4 evaluate   evaluation/metrics.py over the run's artefacts
    5 gate       evaluation/gate.py ─► release directory | shortfall + refill round

Pipeline.run() is stages 0 to 3; Pipeline.release() runs them, then stages 4 and 5 in
rounds: a failed gate with short cells sends the scheduler back to fill only those cells
(with a fresh per-cell attempt allowance, and the budget carried over), until the gate
passes, max_rounds is reached, the budget runs out, or nothing is left to refill.

Run artefacts live in an ArtefactStore run directory keyed by spec_version:

    spec.json      stage 0 summary (spec_version, task type, layers run and skipped)
    cells.json     this run's cells and quotas, copied from the coverage plan; reused on
                   resume so quotas can't shift even if the shared plan is rebuilt

The coverage plan itself is shared by every run of a spec_version (coverage_plan.json,
or a per target and plan seed file). The expansion model is built, and so named in the
D12 endpoint record, only when an axis needs keywords and no plan is cached yet.
    accepted.jsonl accepted records, post_processed, with provenance under _provenance
    drops.jsonl    every dropped candidate with cell, layer, codes and reason
    summary.json   the scheduler snapshot and drop counts at the end of the run
    review.jsonl   records L5 queued for people, when hitl.review_flagged and no sink is given;
                   hitl/queue.ReviewQueue.for_run() resolves them into review_decisions.jsonl
    rounds.json    Pipeline.release(): per round, the cells run, the gate outcome, short cells
    shortfall.json the last failed gate, when release() stops without releasing

When the task lists tools, the generator runs as an agent behind one ToolGateway built
from the tool registry (tools/registry.REGISTRY unless one is passed), with the
spec_version's shared tool cache (shared/tool_cache.jsonl). Each record's tool trace goes
to L3 and into its provenance.

Reopening an existing run id resumes it: accepted counts are restored per cell from
accepted.jsonl, and per-candidate seeds continue from where the run stopped, so a
resumed run never regenerates a candidate it already tried. The L4 near-duplicate corpus
is rebuilt from accepted.jsonl too, so a resumed run can't accept a copy of an earlier record.

L3 (governance) and L4 (overlap) failures are hard drops, never repaired. The L4 held-out
check runs only when held_out_paths is passed to the Pipeline; the path is a run-time
argument, not recorded in any run artefact, and held-out text never reaches a prompt.

L5 (judge) and L6 (consistency) build the judge stage, and the fallback judge only when
the rubric may ask for reasons, so the D12 endpoint record names only models that get
data. The judge counts as trusted when a calibration result for this spec_version and
judge model is in the store (judge/calibration.py), or one is passed in. Layers the run
leaves out are reported as skipped in spec.json and summary.json, never silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Collection, Iterable, Mapping, Sequence

from sdgf.coverage.axes import keyword_sources
from sdgf.coverage.plan import DEFAULT_SEED, CoveragePlan, load_or_build_plan, plan_stage_name
from sdgf.evaluation.diversity import Embedder
from sdgf.evaluation.gate import GateResult, evaluate_gate
from sdgf.evaluation.metrics import MetricsReport, metrics_for_run
from sdgf.evaluation.reports import write_release, write_shortfall
from sdgf.generate.generator import Generator
from sdgf.hitl.queue import require_plan_approval
from sdgf.judge.calibration import CalibrationResult, CalibrationStore
from sdgf.judge.interface import Judge
from sdgf.judge.llm_judge import LLMJudge
from sdgf.generate.scheduler import Cell, Scheduler
from sdgf.models.base import ModelBackend
from sdgf.models.registry import StageModels, build_models
from sdgf.spec.compile import CompiledSpec, compile_spec
from sdgf.store.artefacts import ArtefactStore, JsonlWriter, RunDir
from sdgf.store.provenance import PROVENANCE_KEY, ProvenanceBuilder, attach
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import REGISTRY as TASK_TYPES
from sdgf.tools.cache import ToolCache
from sdgf.tools.gateway import ToolGateway
from sdgf.tools.registry import REGISTRY as TOOLS
from sdgf.tools.registry import ToolRegistry
from sdgf.validate.base import Layer
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.l3_governance import GovernanceLayer
from sdgf.validate.l4_overlap import OverlapLayer
from sdgf.validate.l5_judge import JudgeLayer, ReviewItem, ReviewSink
from sdgf.validate.l6_consistency import Answerer, ConsistencyLayer
from sdgf.validate.repair import GENERATE_STAGE, Drop, DropLog, RepairLoop

log = logging.getLogger(__name__)

IMPLEMENTED_LAYERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")
JUDGE_LAYERS: tuple[str, ...] = ("L5", "L6")

ACCEPTED_STREAM = "accepted"
DROPS_STREAM = "drops"
REVIEW_STREAM = "review"
ROUNDS_STAGE = "rounds"
DEFAULT_MAX_ROUNDS = 3


class PipelineError(ValueError):
    """The spec or run options can't be run by the pipeline as built so far."""


# ── cells ────────────────────────────────────────────────────────


def cells_to_json(cells: Sequence[Cell]) -> list[dict[str, Any]]:
    return [{"id": c.id, "params": dict(c.params), "quota": c.quota} for c in cells]


def cells_from_json(data: Sequence[Mapping[str, Any]]) -> list[Cell]:
    return [Cell(d["id"], dict(d["params"]), int(d["quota"])) for d in data]


# ── seeds ────────────────────────────────────────────────────────


def candidate_seed(run_seed: int, cell_id: str, index: int) -> int:
    """A stable per-candidate seed: the same run seed, cell and index give the same draws."""
    digest = hashlib.sha256(f"{run_seed}:{cell_id}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


# ── pipeline ─────────────────────────────────────────────────────


def _bare(record: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != PROVENANCE_KEY}


class RunReviewSink:
    """Writes L5 review items to the open run's review stream, which ReviewQueue.for_run reads."""

    def __init__(self) -> None:
        self.writer: JsonlWriter | None = None

    def submit(self, item: ReviewItem) -> None:
        if self.writer is None:
            raise PipelineError("review item submitted outside a run")
        self.writer.write(item.to_dict())


@dataclass
class RunResult:
    run: RunDir
    accepted: list[dict[str, Any]]  # this invocation's accepted records, with provenance
    drops: DropLog
    counts: dict[str, int]  # accepted per cell, including records restored on resume
    stop_reason: str | None
    layers: tuple[str, ...]
    skipped_layers: tuple[str, ...]
    snapshot: dict[str, Any] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        return self.stop_reason == "complete"


@dataclass
class ReleaseResult:
    run: RunDir
    released: bool
    path: Path  # the release directory, or shortfall.json when not released
    metrics: MetricsReport
    gate: GateResult
    rounds: list[dict[str, Any]]  # this invocation's rounds, as in rounds.json
    stop_reason: str  # released | hard_fail | no_short_cells | max_rounds | budget:...


class Pipeline:
    def __init__(
        self,
        spec: CompiledSpec | str | Path,
        store: ArtefactStore | str | Path,
        *,
        model_overrides: Mapping[str, ModelBackend] | None = None,
        seed: int = 0,
        target_size: int | None = None,
        plan_seed: int = DEFAULT_SEED,
        layers: Sequence[str] | None = None,
        max_attempts_per_cell: int | None = None,
        held_out_paths: Iterable[str | Path] | None = None,
        calibration: CalibrationResult | None = None,
        review: ReviewSink | None = None,
        answerer: Answerer | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        # Stage 0: spec schema and hook signatures are checked on load; the task type
        # and its generation mode are resolved here, before any model is built.
        tool_registry = tool_registry if tool_registry is not None else TOOLS
        self.compiled = (
            spec
            if isinstance(spec, CompiledSpec)
            else compile_spec(spec, tool_registry=tool_registry)
        )
        self.task_type: TaskType = TASK_TYPES.resolve(self.compiled.spec.task)
        self.store = store if isinstance(store, ArtefactStore) else ArtefactStore(store)
        self.seed = seed
        self.target_size = target_size
        self.plan_seed = plan_seed
        self.max_attempts_per_cell = max_attempts_per_cell
        self.held_out_paths = list(held_out_paths) if held_out_paths is not None else None

        configured = self.compiled.spec.validation.layers
        if layers is None:
            self.layers = tuple(n for n in configured if n in IMPLEMENTED_LAYERS)
        else:
            unknown = [n for n in layers if n not in IMPLEMENTED_LAYERS]
            if unknown:
                raise PipelineError(
                    f"layers {unknown} are not implemented yet; have {list(IMPLEMENTED_LAYERS)}"
                )
            self.layers = tuple(layers)
        if not self.layers:
            raise PipelineError(f"no implemented layer enabled; have {list(IMPLEMENTED_LAYERS)}")
        self.skipped_layers = tuple(n for n in configured if n not in self.layers)
        if self.skipped_layers:
            log.warning("validation layers %s are enabled but not run", self.skipped_layers)

        # Only the stages this pipeline uses are built, so provenance and the D12
        # endpoint record name exactly the models that received data.
        self.used_stages = self._used_stages()
        overrides = dict(model_overrides or {})
        unused = sorted(set(overrides) - set(self.used_stages))
        if unused:
            raise PipelineError(
                f"model overrides for stages {unused} that this run doesn't use; "
                f"it uses {list(self.used_stages)}"
            )
        spec_models = self.compiled.spec.models
        missing = [
            st
            for st in self.used_stages
            if getattr(spec_models, st) is None and st not in overrides
        ]
        if missing:
            raise PipelineError(f"layers {list(self.layers)} need models {missing}; none is set")
        spec_models = spec_models.model_copy(
            update={
                st: None
                for st in ("judge", "fallback_judge", "expansion")
                if st not in self.used_stages
            }
        )
        self.models: StageModels = build_models(spec_models, overrides)
        # Tools: the task's allowlist behind one gateway, with the spec_version's shared
        # response cache, so every run of the spec reuses (and can replay) tool results.
        self.tool_cache: ToolCache | None = None
        gateway = None
        if self.compiled.spec.tools:
            self.tool_cache = ToolCache.for_store(self.store, self.compiled.spec_version)
            gateway = ToolGateway(tool_registry.for_task(self.compiled.spec), self.tool_cache)
        self.generator = Generator(
            self.compiled,
            self.models.backend("generator"),
            task_type=self.task_type,
            gateway=gateway,
        )
        self.review_sink: RunReviewSink | None = None
        if review is None and self.compiled.spec.hitl.review_flagged and "L5" in self.layers:
            self.review_sink = RunReviewSink()
            review = self.review_sink
        self.judge: Judge | None = None
        self.trusted = False
        self.calibration: CalibrationResult | None = None
        implementations = self._layer_implementations(calibration, review, answerer)
        overlap = implementations.get("L4")
        assert overlap is None or isinstance(overlap, OverlapLayer)
        self.overlap: OverlapLayer | None = overlap
        self.cascade = Cascade.from_config(self.layers, implementations)

    def _plan_cached(self) -> bool:
        stage = plan_stage_name(self.compiled, self.target_size, self.plan_seed)
        return self.store.has_shared(self.compiled.spec_version, stage)

    def _used_stages(self) -> tuple[str, ...]:
        stages = ["generator"]
        # Stage 1 calls the expansion model only for keyword axes, and only to build a plan.
        if keyword_sources(self.compiled.spec.coverage) and not self._plan_cached():
            stages.append("expansion")
        if any(n in JUDGE_LAYERS for n in self.layers):
            stages.append("judge")
        # The fallback judge only writes reasons, so it gets data only if the rubric asks.
        spec = self.compiled.spec
        if (
            "L5" in self.layers
            and spec.models.fallback_judge is not None
            and spec.rubric.reason_required != "never"
        ):
            stages.append("fallback_judge")
        return tuple(stages)

    def _judge_id(self) -> str:
        """The judge model a calibration must belong to: the backend actually built."""
        backend = self.models.backend("judge")
        return f"{backend.name}:{backend.model}"

    def _layer_implementations(
        self,
        calibration: CalibrationResult | None,
        review: ReviewSink | None,
        answerer: Answerer | None,
    ) -> dict[str, Layer]:
        """Only the enabled layers are built, so e.g. L4 doesn't need overlap_max without L4."""
        if self.held_out_paths is not None and "L4" not in self.layers:
            raise PipelineError("held_out_paths needs L4 enabled")
        judge_name = None
        if "judge" in self.used_stages:
            self.judge = LLMJudge.from_spec(self.compiled, self.models.backend("judge"))
            judge_name = self._judge_id()
            if calibration is None:
                calibration = CalibrationStore(self.store).load(
                    self.compiled.spec_version, judge_name
                )
            self.trusted = calibration is not None and calibration.trusts(
                self.compiled.spec_version, judge_name
            )
            # Stage 4 reads kappa and residual error only from this judge's calibration.
            if (
                calibration is not None
                and calibration.spec_version == self.compiled.spec_version
                and calibration.judge_id == judge_name
            ):
                self.calibration = calibration
        fallback = None
        if "fallback_judge" in self.used_stages:
            fallback = LLMJudge.from_spec(
                self.compiled, self.models.backend("fallback_judge"), stage="fallback_judge"
            )
        build = {
            "L1": lambda: SchemaLayer.from_spec(self.compiled, self.task_type),
            "L2": lambda: RulesLayer.from_spec(self.compiled),
            "L3": lambda: GovernanceLayer.from_spec(self.compiled),
            "L4": lambda: OverlapLayer.from_spec(self.compiled, held_out_paths=self.held_out_paths),
            "L5": lambda: JudgeLayer.from_spec(
                self.compiled,
                self.judge,
                review=review,
                fallback_judge=fallback,
                calibration=calibration,
                judge_name=judge_name,
            ),
            "L6": lambda: ConsistencyLayer.from_spec(
                self.compiled,
                judge=self.judge,
                answerer=answerer,
                calibration=calibration,
                judge_name=judge_name,
            ),
        }
        return {name: build[name]() for name in self.layers}

    def _intake_summary(self) -> dict[str, Any]:
        spec = self.compiled.spec
        return {
            "spec_version": self.compiled.spec_version,
            "task": spec.task.name,
            "task_version": spec.task.version,
            "task_type": self.task_type.name,
            "generation_mode": spec.task.generation_mode,
            "seeds": len(self.compiled.seeds),
            "layers": list(self.layers),
            "skipped_layers": list(self.skipped_layers),
            "held_out_check": self.held_out_paths is not None,
            "judge_trusted": self.trusted,
            "models": self.models.endpoints(),
        }

    def plan(self) -> CoveragePlan:
        """Stage 1: the shared coverage plan for this spec_version, built once if absent."""
        backend = self.models.backend("expansion") if "expansion" in self.used_stages else None
        plan, _ = load_or_build_plan(
            self.store,
            self.compiled,
            target_size=self.target_size,
            seed=self.plan_seed,
            backend=backend,
        )
        return plan

    def approved_plan(self) -> CoveragePlan:
        """Stage 1 plus its HITL checkpoint: with hitl.approve_coverage_plan, raise
        ApprovalPending until a person has approved this plan (hitl/queue.approve_plan).
        A resumed run that already copied its cells doesn't ask again."""
        plan = self.plan()
        if self.compiled.spec.hitl.approve_coverage_plan:
            require_plan_approval(self.store, self.compiled.spec_version, self.plan_stage)
        return plan

    @property
    def plan_stage(self) -> str:
        return plan_stage_name(self.compiled, self.target_size, self.plan_seed)

    def run(
        self,
        run_id: str | None = None,
        *,
        only: Collection[str] | None = None,
        prior_usage: Mapping[str, float] | None = None,
    ) -> RunResult:
        """Stages 0 to 3. only restricts generation to those cells (a refill round);
        prior_usage is the usage of earlier rounds, which counts against the budget."""
        run = self.store.open_run(self.compiled.spec_version, run_id)
        run.stage("spec", self._intake_summary)
        target = self.target_size
        cells = cells_from_json(
            run.stage("cells", lambda: cells_to_json(self.approved_plan().cells))
        )
        if target is not None and sum(c.quota for c in cells) != target:
            raise PipelineError(
                f"run {run.run_id!r} was planned for {sum(c.quota for c in cells)} records, "
                f"not {target}; start a new run to change the target size"
            )

        if only is not None:
            unknown = sorted(set(only) - {c.id for c in cells})
            if unknown:
                raise PipelineError(f"cells {unknown} are not in run {run.run_id!r}")
        scheduled = [c for c in cells if only is None or c.id in only]
        scheduler = Scheduler(
            scheduled, self.compiled.spec.budget, max_attempts_per_cell=self.max_attempts_per_cell
        )
        if prior_usage is not None:
            scheduler.restore_usage(
                candidates=int(prior_usage.get("candidates", 0)),
                tokens=int(prior_usage.get("tokens", 0)),
                cost_usd=float(prior_usage.get("cost_usd", 0.0)),
                seconds=float(prior_usage.get("seconds", 0.0)),
            )
        prior = run.read_jsonl(ACCEPTED_STREAM)
        prior_accepted = Counter(r[PROVENANCE_KEY]["cell_id"] for r in prior)
        if self.overlap is not None:
            for r in prior:
                self.overlap.remember(_bare(r))
        prior_drops = Counter(d["cell_id"] for d in run.read_jsonl(DROPS_STREAM))
        scheduler.restore_accepted({c.id: prior_accepted[c.id] for c in scheduled})
        tried = {c.id: prior_accepted[c.id] + prior_drops[c.id] for c in cells}

        accepted: list[dict[str, Any]] = []
        with ExitStack() as streams:
            if self.tool_cache is not None:
                streams.callback(self.tool_cache.close)
            out = streams.enter_context(run.jsonl(ACCEPTED_STREAM))
            drops = DropLog(streams.enter_context(run.jsonl(DROPS_STREAM)))
            if self.review_sink is not None:
                self.review_sink.writer = streams.enter_context(run.jsonl(REVIEW_STREAM))
                streams.callback(setattr, self.review_sink, "writer", None)
            loop = RepairLoop(self.generator, self.cascade, drop_log=drops)
            while (cell := scheduler.next_cell()) is not None:
                seed = candidate_seed(self.seed, cell.id, tried[cell.id])
                tried[cell.id] += 1
                record, reason = self._candidate(loop, drops, run.run_id, cell, seed)
                if record is None:
                    scheduler.reject(cell.id, reason)
                    continue
                out.write(record)
                accepted.append(record)
                if self.overlap is not None:
                    self.overlap.remember(_bare(record))
                scheduler.accept(cell.id)

        snapshot = scheduler.snapshot()
        snapshot["drops"] = {"by_layer": drops.by_layer(), "by_code": drops.by_code()}
        snapshot["skipped_layers"] = list(self.skipped_layers)
        snapshot["only"] = None if only is None else [c.id for c in scheduled]
        run.write_stage("summary", snapshot)
        return RunResult(
            run=run,
            accepted=accepted,
            drops=drops,
            counts={c.id: prior_accepted[c.id] for c in cells} | scheduler.counts(),
            stop_reason=scheduler.stop_reason,
            layers=self.layers,
            skipped_layers=self.skipped_layers,
            snapshot=snapshot,
        )

    def release(
        self,
        release_root: str | Path,
        run_id: str | None = None,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        waive: Iterable[str] = (),
        embed: Embedder | None = None,
        version: str | None = None,
        now: datetime | None = None,
    ) -> ReleaseResult:
        """Stages 0 to 5 in rounds: generate, evaluate, gate; on a failed gate refill only
        the short cells and gate again, at most max_rounds times in this invocation.

        Releases under release_root on a pass; otherwise writes the shortfall report into
        the run directory. waive names thresholds whose metric may be unmeasured (e.g.
        semantic_diversity_min without an embedder); governance can't be waived."""
        if max_rounds < 1:
            raise PipelineError("max_rounds must be >= 1")
        waive = tuple(waive)
        only: list[str] | None = None
        usage: Mapping[str, float] | None = None
        rounds: list[dict[str, Any]] = []
        history: list[dict[str, Any]] | None = None
        while True:
            result = self.run(run_id, only=only, prior_usage=usage)
            run, run_id = result.run, result.run.run_id
            usage = result.snapshot["usage"]
            # Stage 4: every §8 metric over the whole run, not just this round's records.
            metrics = metrics_for_run(
                self.compiled,
                run,
                calibration=self.calibration,
                held_out_paths=self.held_out_paths,
                usage=usage,
                embed=embed,
                seed=self.seed,
            )
            # Stage 5
            gate = evaluate_gate(metrics, self.compiled.spec.thresholds, waive=waive)
            stop = self._round_stop(gate, result, len(rounds) + 1, max_rounds)
            rounds.append(
                {
                    "round": len(rounds) + 1,
                    "cells": only,
                    "accepted": len(result.accepted),
                    "scheduler_stop": result.stop_reason,
                    "passed": gate.passed,
                    "hard_fail": gate.hard_fail,
                    "failing_metrics": gate.failing_metrics,
                    "short_cells": dict(gate.short_cells),
                    "outcome": stop or "refill",
                }
            )
            if history is None:  # rounds of earlier invocations of a resumed run
                history = run.read_stage(ROUNDS_STAGE) if run.has_stage(ROUNDS_STAGE) else []
            run.write_stage(ROUNDS_STAGE, history + rounds)
            if stop is not None:
                break
            only = sorted(gate.short_cells)
            log.info("gate failed %s; refilling %d short cells", gate.failing_metrics, len(only))

        if stop == "released":
            path = write_release(
                release_root, self.compiled, run, metrics, gate, version=version, now=now
            )
        else:
            path = write_shortfall(run, metrics, gate)
        return ReleaseResult(run, stop == "released", path, metrics, gate, rounds, stop)

    @staticmethod
    def _round_stop(gate: GateResult, result: RunResult, n: int, max_rounds: int) -> str | None:
        """Why the release loop stops after this round, or None to refill the short cells."""
        if gate.passed:
            return "released"
        if gate.hard_fail:
            return "hard_fail"  # governance: more records can't fix the ones accepted
        if not gate.short_cells:
            return "no_short_cells"  # the failures aren't about coverage
        if result.stop_reason and result.stop_reason.startswith("budget:"):
            return result.stop_reason
        if n >= max_rounds:
            return "max_rounds"
        return None

    def _candidate(
        self, loop: RepairLoop, drops: DropLog, run_id: str, cell: Cell, seed: int
    ) -> tuple[dict[str, Any] | None, str]:
        """One scheduler slot: an accepted record with provenance, or None and a reason."""
        recipe = self.generator.recipe(cell.params, random.Random(seed))
        if recipe is None:
            drops.add(
                Drop(
                    cell_id=cell.id,
                    layer=GENERATE_STAGE,
                    codes=("invalid_cell",),
                    reason="sampler_constraints rejected the cell",
                    attempts=0,
                )
            )
            return None, f"{GENERATE_STAGE}:invalid_cell"

        prompt = self.generator.prompts.build(recipe)
        prov = ProvenanceBuilder(
            self.compiled.spec_version, cell.id, seed, self.models.endpoints(), run_id
        )
        outcome = loop.run(cell.id, recipe, prompt, prov)
        if outcome.record is None:
            drop = outcome.drop
            assert drop is not None
            return None, f"{drop.layer}:{drop.codes[0] if drop.codes else 'unknown'}"

        record = outcome.record
        post = self.compiled.hooks.post_process
        if post is not None:
            record = post(record)
        provenance = prov.build()
        provenance.check_accepted(self.cascade.names)
        # A JSON round trip makes the returned record identical to the stored line.
        return json.loads(json.dumps(attach(record, provenance))), ""
