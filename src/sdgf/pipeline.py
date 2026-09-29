"""The run pipeline (FRAMEWORK_DESIGN.md §5, §6): stages 0, 2 and 3 so far.

    0 intake     compile task.yaml + hooks.py + seeds, resolve the task type
    - cells      fixed axes crossed into cells with quotas (stand-in until stage 1, M5)
    2 generate   scheduler ─► sampler_constraints ─► prompt ─► generator backend
    3 validate   cascade (L1, L2 for now) ─► repair ─► accepted + provenance | drop log

Run artefacts live in an ArtefactStore run directory keyed by spec_version:

    spec.json      stage 0 summary (spec_version, task type, layers run and skipped)
    cells.json     the cell grid with quotas; reused on resume so quotas can't shift
    accepted.jsonl accepted records, post_processed, with provenance under _provenance
    drops.jsonl    every dropped candidate with cell, layer, codes and reason
    summary.json   the scheduler snapshot and drop counts at the end of the run

Reopening an existing run id resumes it: accepted counts are restored per cell from
accepted.jsonl, and per-candidate seeds continue from where the run stopped, so a
resumed run never regenerates a candidate it already tried.

Only L1 and L2 are implemented; enabled layers without an implementation are reported
as skipped in spec.json and summary.json rather than silently ignored.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from sdgf.generate.generator import Generator
from sdgf.generate.scheduler import Cell, Scheduler
from sdgf.models.base import ModelBackend
from sdgf.models.registry import StageModels, build_models
from sdgf.spec.compile import CompiledSpec, compile_spec
from sdgf.spec.schema import CoverageSection
from sdgf.store.artefacts import ArtefactStore, RunDir
from sdgf.store.provenance import PROVENANCE_KEY, ProvenanceBuilder, attach
from sdgf.tasktypes.base import TaskType
from sdgf.tasktypes.registry import REGISTRY as TASK_TYPES
from sdgf.validate.base import Layer
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.repair import GENERATE_STAGE, Drop, DropLog, RepairLoop

log = logging.getLogger(__name__)

IMPLEMENTED_LAYERS: tuple[str, ...] = ("L1", "L2")
USED_STAGES: tuple[str, ...] = ("generator",)  # model stages run so far; judge arrives with L5

ACCEPTED_STREAM = "accepted"
DROPS_STREAM = "drops"


class PipelineError(ValueError):
    """The spec or run options can't be run by the pipeline as built so far."""


# ── cells (stand-in for the stage 1 coverage plan) ───────────────


def _value_id(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value)


def _apportion(weights: Sequence[float], total: int) -> list[int]:
    """Largest-remainder split of `total` in proportion to `weights`; ties go to plan order."""
    wsum = sum(weights)
    exact = [w / wsum * total for w in weights]
    quotas = [int(x) for x in exact]
    order = sorted(range(len(exact)), key=lambda i: (-(exact[i] - quotas[i]), i))
    for i in order[: total - sum(quotas)]:
        quotas[i] += 1
    return quotas


def fixed_axis_cells(coverage: CoverageSection, target_size: int | None = None) -> list[Cell]:
    """Cross the fixed axes into cells with quotas summing to the target size.

    quota_policy "weighted" makes each cell's share the product of its axis weights;
    "even" (or an axis without weights) gives every value the same share. Cell ids are
    the axis values joined with "|", in axis order.
    """
    target = coverage.target_size if target_size is None else target_size
    if target <= 0:
        raise PipelineError(f"target size must be > 0, got {target}")
    not_fixed = [a.name for a in coverage.axes if a.source != "fixed"]
    if not_fixed:
        raise PipelineError(
            f"axes {not_fixed} need keyword expansion or retrieval; the stage 1 coverage "
            "plan is not built yet, so only fixed axes can be run"
        )
    weighted = coverage.quota_policy == "weighted"
    per_axis = []
    for axis in coverage.axes:
        assert axis.values is not None
        ws = axis.weights if weighted and axis.weights is not None else [1.0] * len(axis.values)
        per_axis.append([(axis.name, v, w) for v, w in zip(axis.values, ws)])

    combos = list(itertools.product(*per_axis))
    shares = []
    for combo in combos:
        share = 1.0
        for _, _, w in combo:
            share *= w
        shares.append(share)
    quotas = _apportion(shares, target)
    return [
        Cell(
            id="|".join(_value_id(v) for _, v, _ in combo),
            params={name: v for name, v, _ in combo},
            quota=q,
        )
        for combo, q in zip(combos, quotas)
    ]


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


class Pipeline:
    def __init__(
        self,
        spec: CompiledSpec | str | Path,
        store: ArtefactStore | str | Path,
        *,
        model_overrides: Mapping[str, ModelBackend] | None = None,
        seed: int = 0,
        target_size: int | None = None,
        layers: Sequence[str] | None = None,
        max_attempts_per_cell: int | None = None,
    ):
        # Stage 0: spec schema and hook signatures are checked on load; the task type
        # and its generation mode are resolved here, before any model is built.
        self.compiled = spec if isinstance(spec, CompiledSpec) else compile_spec(spec)
        self.task_type: TaskType = TASK_TYPES.resolve(self.compiled.spec.task)
        self.store = store if isinstance(store, ArtefactStore) else ArtefactStore(store)
        self.seed = seed
        self.target_size = target_size
        self.max_attempts_per_cell = max_attempts_per_cell

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
        overrides = dict(model_overrides or {})
        unused = sorted(set(overrides) - set(USED_STAGES))
        if unused:
            raise PipelineError(f"model overrides for stages {unused} that are not run yet")
        spec_models = self.compiled.spec.models.model_copy(
            update={"judge": None, "fallback_judge": None, "expansion": None}
        )
        self.models: StageModels = build_models(spec_models, overrides)
        self.generator = Generator(
            self.compiled, self.models.backend("generator"), task_type=self.task_type
        )
        self.cascade = Cascade.from_config(self.layers, self._layer_implementations())

    def _layer_implementations(self) -> dict[str, Layer]:
        return {
            "L1": SchemaLayer.from_spec(self.compiled, self.task_type),
            "L2": RulesLayer.from_spec(self.compiled),
        }

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
            "models": self.models.endpoints(),
        }

    def run(self, run_id: str | None = None) -> RunResult:
        run = self.store.open_run(self.compiled.spec_version, run_id)
        run.stage("spec", self._intake_summary)
        target = self.target_size
        cells = cells_from_json(
            run.stage(
                "cells",
                lambda: cells_to_json(fixed_axis_cells(self.compiled.spec.coverage, target)),
            )
        )
        if target is not None and sum(c.quota for c in cells) != target:
            raise PipelineError(
                f"run {run.run_id!r} was planned for {sum(c.quota for c in cells)} records, "
                f"not {target}; start a new run to change the target size"
            )

        scheduler = Scheduler(
            cells, self.compiled.spec.budget, max_attempts_per_cell=self.max_attempts_per_cell
        )
        prior_accepted = Counter(
            r[PROVENANCE_KEY]["cell_id"] for r in run.read_jsonl(ACCEPTED_STREAM)
        )
        prior_drops = Counter(d["cell_id"] for d in run.read_jsonl(DROPS_STREAM))
        scheduler.restore_accepted(dict(prior_accepted))
        tried = {c.id: prior_accepted[c.id] + prior_drops[c.id] for c in cells}

        accepted: list[dict[str, Any]] = []
        with run.jsonl(ACCEPTED_STREAM) as out, run.jsonl(DROPS_STREAM) as drops_out:
            drops = DropLog(drops_out)
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
                scheduler.accept(cell.id)

        snapshot = scheduler.snapshot()
        snapshot["drops"] = {"by_layer": drops.by_layer(), "by_code": drops.by_code()}
        snapshot["skipped_layers"] = list(self.skipped_layers)
        run.write_stage("summary", snapshot)
        return RunResult(
            run=run,
            accepted=accepted,
            drops=drops,
            counts=scheduler.counts(),
            stop_reason=scheduler.stop_reason,
            layers=self.layers,
            skipped_layers=self.skipped_layers,
            snapshot=snapshot,
        )

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
