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
    summary.json   the scheduler snapshot, this invocation's usage per stage and drop counts
    usage.json     the usage ledger: calls, tokens and estimated cost per model stage and
                   seconds, summed over every invocation of the run; rewritten after each
                   wave, so a killed run keeps its count; stage 4 cost per record reads it
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

Generation runs in batches ("waves"): each wave reserves every slot the scheduler will
hand out, runs the slots (generate, validate, repair) on a thread pool of
max(models.<stage>.concurrency) workers over the generator and judge stages, each backend
limited to its own stage's concurrency, and then settles them in reservation order.
Each slot's drops and review items are held until it settles, and L4 checks a wave
against the corpus as it stood when the wave began, then settle re-checks each record
against the records settled before it in the same wave (OverlapLayer.check_staged). So
what a run accepts, and every artefact it writes, depends on the seed and not on the
number of workers or the order calls finish in. Two limits: a scripted MockBackend
answers in call order, so it is only order-independent as a callable of the prompt; and
with tools, a trace entry's cached flag records whether a call was served from the cache,
which depends on timing.

Budgets and cost: every stage's backend is metered (models/usage.py), tokens from the
backend response or estimated from text length, cost from models.<stage> prices. Each
slot's usage is charged to the scheduler as it settles, and a wave holds only as many
slots as the remaining token, cost and time budget covers at the average usage per slot
so far (Scheduler.wave_allowance, one slot before any has settled), so spec.budget stops
the run cleanly between waves, overshooting by at most what that estimate missed. A budget stop is resumable: rerunning the run id continues
from what was accepted, with the budget applying afresh to that invocation, while
usage.json keeps the run's total. Release rounds within one invocation share a budget.

L6 votes on models.consistency_judge when it is set, else on the judge stage's backend,
cycling validation.consistency.temperatures per vote: under label_first one blind judge
per temperature (not L5's judge reused K times), under answer_emergent, unless an
answerer is passed, a BackendAnswerer shown only the question plus the task type's answer
suffix, so the votes are independent of the generator.

L5 (judge) and L6 (consistency) build the judge stage (L6 only without a consistency_judge), and the fallback judge only when
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
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Any, Collection, Iterable, Iterator, Mapping, Sequence

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
from sdgf.models.base import BoundedBackend, ModelBackend
from sdgf.models.registry import StageModels, build_models
from sdgf.models.usage import USAGE_STAGE, MeteredBackend, Pricing, UsageLedger, UsageMeter
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
from sdgf.validate.l6_consistency import (
    Answerer,
    ConsistencyError,
    ConsistencyLayer,
    vote_stage,
)
from sdgf.validate.repair import GENERATE_STAGE, Drop, DropLog, RepairLoop

log = logging.getLogger(__name__)

IMPLEMENTED_LAYERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6")
# Stages whose calls run inside a slot, so on the thread pool; expansion runs before it.
POOL_STAGES: tuple[str, ...] = ("generator", "judge", "fallback_judge", "consistency_judge")

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


class SlotReviews:
    """Holds the review items a slot's L5 submits (per thread) until the slot settles, so
    they reach the real sink in reservation order. Outside a slot items pass straight on."""

    def __init__(self, target: ReviewSink) -> None:
        self.target = target
        self._local = threading.local()

    def submit(self, item: ReviewItem) -> None:
        held = getattr(self._local, "items", None)
        if held is None:
            self.target.submit(item)
        else:
            held.append(item)

    @contextmanager
    def capture(self) -> Iterator[list[ReviewItem]]:
        self._local.items = held = []
        try:
            yield held
        finally:
            self._local.items = None


@dataclass
class Slot:
    """One reserved scheduler slot after it ran: an accepted record or None, plus what it
    logged, which settle writes in reservation order."""

    cell: Cell
    record: dict[str, Any] | None
    reason: str
    drops: list[Drop]
    reviews: list[ReviewItem]
    usage: UsageLedger = field(default_factory=UsageLedger)
    attempts: int = 0
    history: tuple[tuple[str, tuple[str, ...]], ...] = ()


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
    usage: dict[str, Any] = field(default_factory=dict)  # the run's usage ledger, as in usage.json

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
        ignore_unused_overrides: bool = False,
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
        if unused and ignore_unused_overrides:
            # e.g. a CLI backends plugin that supplies every stage; unused ones get no data
            overrides = {st: b for st, b in overrides.items() if st in self.used_stages}
        elif unused:
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
                for st in ("judge", "fallback_judge", "consistency_judge", "expansion")
                if st not in self.used_stages
            }
        )
        self.models: StageModels = build_models(spec_models, overrides)
        # Every stage is metered; stage prices come from models.<stage>, overridden or not.
        pricing = {st: Pricing.from_config(getattr(spec_models, st)) for st in self.used_stages}
        if self.compiled.spec.budget.max_cost_usd is not None:
            unpriced = [st for st, p in pricing.items() if p is None]
            if unpriced:
                raise PipelineError(
                    f"budget.max_cost_usd is set but models {unpriced} have no price; set "
                    "input_cost_per_mtok and output_cost_per_mtok (0 for a free model)"
                )
        self.meter = UsageMeter()
        self._metered: dict[str, ModelBackend] = {
            st: MeteredBackend(self.models.backend(st), st, self.meter, pricing[st])
            for st in self.used_stages
        }
        # Concurrency: one pool sized for the busiest stage, each stage's backend capped
        # at its own models.<stage>.concurrency. One worker runs the slots inline.
        limits = {
            st: (cfg.concurrency if (cfg := getattr(spec_models, st)) is not None else 1)
            for st in self.used_stages
            if st in POOL_STAGES
        }
        self.workers = max(limits.values())
        self._bounded: dict[str, ModelBackend] = {
            st: BoundedBackend(self._metered[st], n) if self.workers > 1 else self._metered[st]
            for st, n in limits.items()
        }
        # Tools: the task's allowlist behind one gateway, with the spec_version's shared
        # response cache, so every run of the spec reuses (and can replay) tool results.
        self.tool_cache: ToolCache | None = None
        gateway = None
        if self.compiled.spec.tools:
            self.tool_cache = ToolCache.for_store(self.store, self.compiled.spec_version)
            gateway = ToolGateway(tool_registry.for_task(self.compiled.spec), self.tool_cache)
        self.generator = Generator(
            self.compiled,
            self._bounded["generator"],
            task_type=self.task_type,
            gateway=gateway,
        )
        self.review_sink: RunReviewSink | None = None
        if review is None and self.compiled.spec.hitl.review_flagged and "L5" in self.layers:
            self.review_sink = RunReviewSink()
            review = self.review_sink
        self.reviews = SlotReviews(review) if review is not None else None
        review = self.reviews
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
        spec = self.compiled.spec
        # L6 votes on models.consistency_judge when set, so then only L5 needs the judge.
        own_voters = "L6" in self.layers and spec.models.consistency_judge is not None
        if "L5" in self.layers or ("L6" in self.layers and not own_voters):
            stages.append("judge")
        if own_voters:
            stages.append("consistency_judge")
        # The fallback judge only writes reasons, so it gets data only if the rubric asks.
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
            self.judge = LLMJudge.from_spec(self.compiled, self._bounded["judge"])
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
                self.compiled, self._bounded["fallback_judge"], stage="fallback_judge"
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
                backend=self._bounded[vote_stage(self.compiled)],
                answerer=answerer,
                calibration=calibration,
                judge_name=judge_name,
            ),
        }
        try:
            return {name: build[name]() for name in self.layers}
        except ConsistencyError as e:
            raise PipelineError(str(e)) from e

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
            "workers": self.workers,
            "models": self.models.endpoints(),
        }

    def plan(self) -> CoveragePlan:
        """Stage 1: the shared coverage plan for this spec_version, built once if absent."""
        backend = self._metered.get("expansion")
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
        started = time.monotonic()
        base = run.read_stage(USAGE_STAGE) if run.has_stage(USAGE_STAGE) else None
        ledger = UsageLedger()
        # usage outside any slot, e.g. stage 1 keyword expansion building the plan
        self._charge(scheduler, ledger, self.meter.take_loose())
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
            pool = None
            if self.workers > 1:
                pool = streams.enter_context(ThreadPoolExecutor(self.workers))
                # an error in one slot shouldn't wait for the rest of the wave to run
                streams.callback(pool.shutdown, cancel_futures=True)
            while True:
                wave: list[tuple[Cell, int]] = []
                limit = scheduler.wave_allowance()
                while (limit is None or len(wave) < limit) and (
                    cell := scheduler.next_cell()
                ) is not None:
                    wave.append((cell, candidate_seed(self.seed, cell.id, tried[cell.id])))
                    tried[cell.id] += 1
                if not wave:
                    break

                def one(slot: tuple[Cell, int], run_id: str = run.run_id) -> Slot:
                    return self._candidate(run_id, *slot)

                # map keeps reservation order; without a pool each slot runs as it settles
                for slot in pool.map(one, wave) if pool is not None else map(one, wave):
                    self._charge(scheduler, ledger, slot.usage)
                    record = self._settle(slot, scheduler, drops)
                    if record is not None:
                        out.write(record)
                        accepted.append(record)
                if self.overlap is not None:
                    self.overlap.commit_staged()
                write_usage(run, base, ledger, time.monotonic() - started)

        usage = write_usage(run, base, ledger, time.monotonic() - started)
        snapshot = scheduler.snapshot()
        snapshot["usage_by_stage"] = ledger.to_dict()
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
            usage=usage,
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
            # Stage 4: every §8 metric over the whole run, not just this round's records;
            # cost per record from the run's usage ledger, every invocation included.
            metrics = metrics_for_run(
                self.compiled,
                run,
                calibration=self.calibration,
                held_out_paths=self.held_out_paths,
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

    @staticmethod
    def _charge(scheduler: Scheduler, ledger: UsageLedger, usage: UsageLedger) -> None:
        scheduler.charge(tokens=usage.tokens, cost_usd=usage.cost_usd or 0.0)
        ledger.add(usage)

    def _settle(self, slot: Slot, scheduler: Scheduler, drops: DropLog) -> dict[str, Any] | None:
        """Settle one slot in reservation order; returns the record if it is accepted."""
        record = slot.record
        if record is not None and self.overlap is not None:
            # a near-duplicate of a record settled earlier in the same wave
            verdict = self.overlap.check_staged(_bare(record))
            if not verdict.passed:
                codes = verdict.codes
                slot.drops.append(
                    Drop(
                        cell_id=slot.cell.id,
                        layer=verdict.layer,
                        codes=codes,
                        reason=f"{verdict.layer} {', '.join(codes)} (hard failure, not repaired)",
                        attempts=slot.attempts,
                        hard=True,
                        errors=tuple(e.to_dict() for e in verdict.errors),
                        history=slot.history + ((verdict.layer, codes),),
                    )
                )
                slot.reviews.clear()  # checked in order, L4 would have stopped it before L5
                record, slot.reason = None, f"{verdict.layer}:{codes[0]}"
        for drop in slot.drops:
            drops.add(drop)
        if self.reviews is not None:
            for item in slot.reviews:
                self.reviews.target.submit(item)
        if record is None:
            scheduler.reject(slot.cell.id, slot.reason)
            return None
        if self.overlap is not None:
            self.overlap.stage(_bare(record))
        scheduler.accept(slot.cell.id)
        return record

    def _candidate(self, run_id: str, cell: Cell, seed: int) -> Slot:
        """Run one scheduler slot; thread-safe, and writes nothing until it settles."""
        local = _HeldDrops()
        with ExitStack() as stack:
            reviews = stack.enter_context(self.reviews.capture()) if self.reviews else []
            usage = stack.enter_context(self.meter.capture())
            slot = Slot(cell, None, "", local.drops, reviews, usage)
            recipe = self.generator.recipe(cell.params, random.Random(seed))
            if recipe is None:
                local.drops.append(
                    Drop(
                        cell_id=cell.id,
                        layer=GENERATE_STAGE,
                        codes=("invalid_cell",),
                        reason="sampler_constraints rejected the cell",
                        attempts=0,
                    )
                )
                slot.reason = f"{GENERATE_STAGE}:invalid_cell"
                return slot

            prompt = self.generator.prompts.build(recipe)
            prov = ProvenanceBuilder(
                self.compiled.spec_version, cell.id, seed, self.models.endpoints(), run_id
            )
            loop = RepairLoop(self.generator, self.cascade, drop_log=local)
            outcome = loop.run(cell.id, recipe, prompt, prov)
            slot.attempts, slot.history = outcome.attempts, tuple(outcome.history)
            if outcome.record is None:
                drop = outcome.drop
                assert drop is not None
                slot.reason = f"{drop.layer}:{drop.codes[0] if drop.codes else 'unknown'}"
                return slot

            record = outcome.record
            post = self.compiled.hooks.post_process
            if post is not None:
                record = post(record)
            provenance = prov.build()
            provenance.check_accepted(self.cascade.names)
            # A JSON round trip makes the returned record identical to the stored line.
            slot.record = json.loads(json.dumps(attach(record, provenance)))
            return slot


def write_usage(
    run: RunDir, base: Mapping[str, Any] | None, ledger: UsageLedger, seconds: float
) -> dict[str, Any]:
    """Write the run's usage ledger: base (earlier invocations) plus this invocation."""
    stages = UsageLedger.from_dict(base["stages"]) if base else UsageLedger()
    stages.add(ledger)
    seconds += float(base["seconds"]) if base else 0.0
    total = stages.total()
    data = {
        "invocations": (int(base["invocations"]) if base else 0) + 1,
        "seconds": seconds,
        "stages": stages.to_dict(),
        "total": {
            "calls": total.calls,
            "input_tokens": total.input_tokens,
            "output_tokens": total.output_tokens,
            "tokens": total.tokens,
            "estimated_calls": total.estimated_calls,
            "cost_usd": total.cost_usd,
            "seconds": seconds,
        },
    }
    run.write_stage(USAGE_STAGE, data)
    return data


class _HeldDrops(DropLog):
    """A slot's drop log: holds drops for settle to log and write, in reservation order."""

    def add(self, drop: Drop) -> None:
        self.drops.append(drop)
