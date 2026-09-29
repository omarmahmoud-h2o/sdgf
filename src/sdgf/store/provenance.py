"""Per-record provenance (FRAMEWORK_DESIGN.md §7.7).

Every accepted record carries: spec_version, the generator and judge model ids with
their hosting (D12), the prompt hash, the random seed, the coverage cell, the tool
trace, the result of each validation layer, the repair count, and any human decisions.

A ProvenanceBuilder collects these while a record moves through generation, the
cascade and repair; build() freezes them into a Provenance. Provenance round-trips
through plain JSON (to_dict / from_dict) so it can be written next to the record in
the accepted stream and in the release. attach() puts it on the record under
PROVENANCE_KEY; the record's own fields are never touched.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping

from sdgf.spec.schema import Hosting, LayerName

PROVENANCE_KEY = "_provenance"
PROVENANCE_FORMAT = 1

LayerOutcome = Literal["pass", "fail_repairable", "fail_hard"]
HumanAction = Literal["accept", "reject", "relabel"]

_LAYER_OUTCOMES = ("pass", "fail_repairable", "fail_hard")
_HUMAN_ACTIONS = ("accept", "reject", "relabel")
_HOSTINGS = ("local", "provider_api")


class ProvenanceError(ValueError):
    """Provenance is incomplete, malformed, or would overwrite a record field."""


def prompt_hash(prompt: str) -> str:
    return "sha256:" + hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ModelRef:
    stage: str
    backend: str
    model: str
    hosting: Hosting
    version: str | None = None  # a pinned model revision, when the backend reports one

    def __post_init__(self) -> None:
        if self.hosting not in _HOSTINGS:
            raise ProvenanceError(f"model {self.model!r}: unknown hosting {self.hosting!r}")

    @classmethod
    def from_endpoint(cls, endpoint: Mapping[str, Any]) -> ModelRef:
        """From StageModel.endpoint(): {stage, backend, model, hosting}."""
        return cls(
            stage=endpoint["stage"],
            backend=endpoint["backend"],
            model=endpoint["model"],
            hosting=endpoint["hosting"],
            version=endpoint.get("version"),
        )


@dataclass(frozen=True)
class ToolTraceEntry:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    sensitivity: str | None = None
    cached: bool = False


@dataclass(frozen=True)
class LayerResult:
    layer: LayerName
    outcome: LayerOutcome
    attempt: int = 0  # 0 is the first generation, n is the n-th repair
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.outcome not in _LAYER_OUTCOMES:
            raise ProvenanceError(f"layer {self.layer}: unknown outcome {self.outcome!r}")
        if self.attempt < 0:
            raise ProvenanceError(f"layer {self.layer}: attempt must be >= 0")


@dataclass(frozen=True)
class HumanDecision:
    action: HumanAction
    reviewer: str
    reason: str | None = None
    new_label: Any = None  # set iff action is relabel
    decided_at: str | None = None  # ISO-8601; passed in so provenance stays deterministic

    def __post_init__(self) -> None:
        if self.action not in _HUMAN_ACTIONS:
            raise ProvenanceError(f"unknown human action {self.action!r}")
        if (self.action == "relabel") != (self.new_label is not None):
            raise ProvenanceError("new_label is required for relabel and only for relabel")


@dataclass(frozen=True)
class Provenance:
    spec_version: str
    cell_id: str
    seed: int
    prompt_hash: str
    models: tuple[ModelRef, ...]
    layer_results: tuple[LayerResult, ...] = ()
    tool_trace: tuple[ToolTraceEntry, ...] = ()
    repair_count: int = 0
    human_decisions: tuple[HumanDecision, ...] = ()
    run_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("spec_version", "cell_id", "prompt_hash"):
            if not getattr(self, name):
                raise ProvenanceError(f"provenance.{name} is required")
        if self.repair_count < 0:
            raise ProvenanceError("provenance.repair_count must be >= 0")
        stages = [m.stage for m in self.models]
        if "generator" not in stages:
            raise ProvenanceError("provenance.models must include the generator")
        if len(set(stages)) != len(stages):
            raise ProvenanceError(f"provenance.models repeats a stage: {stages}")

    def model(self, stage: str) -> ModelRef | None:
        return next((m for m in self.models if m.stage == stage), None)

    def check_accepted(self, layers: list[str] | tuple[str, ...]) -> None:
        """Raise unless every configured layer passed on the final attempt."""
        final_attempt = max((r.attempt for r in self.layer_results), default=0)
        last = {r.layer: r.outcome for r in self.layer_results if r.attempt == final_attempt}
        missing = [layer for layer in layers if layer not in last]
        failed = [layer for layer in layers if last.get(layer, "pass") != "pass"]
        if missing or failed:
            raise ProvenanceError(
                f"record is not accepted on attempt {final_attempt}: "
                f"missing layers {missing}, failed layers {failed}"
            )
        if final_attempt != self.repair_count:
            raise ProvenanceError(
                f"repair_count {self.repair_count} disagrees with final attempt {final_attempt}"
            )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["format"] = PROVENANCE_FORMAT
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Provenance:
        if d.get("format") != PROVENANCE_FORMAT:
            raise ProvenanceError(f"unsupported provenance format {d.get('format')!r}")
        try:
            return cls(
                spec_version=d["spec_version"],
                cell_id=d["cell_id"],
                seed=d["seed"],
                prompt_hash=d["prompt_hash"],
                models=tuple(ModelRef(**m) for m in d["models"]),
                layer_results=tuple(
                    LayerResult(**{**r, "errors": tuple(r.get("errors", ()))})
                    for r in d.get("layer_results", ())
                ),
                tool_trace=tuple(ToolTraceEntry(**t) for t in d.get("tool_trace", ())),
                repair_count=d.get("repair_count", 0),
                human_decisions=tuple(HumanDecision(**h) for h in d.get("human_decisions", ())),
                run_id=d.get("run_id"),
            )
        except (KeyError, TypeError) as e:
            raise ProvenanceError(f"malformed provenance: {e}") from None


class ProvenanceBuilder:
    """Mutable collector for one record's provenance, frozen by build()."""

    def __init__(
        self,
        spec_version: str,
        cell_id: str,
        seed: int,
        models: list[ModelRef] | list[Mapping[str, Any]],
        run_id: str | None = None,
    ):
        self.spec_version = spec_version
        self.cell_id = cell_id
        self.seed = seed
        self.run_id = run_id
        self.models = tuple(
            m if isinstance(m, ModelRef) else ModelRef.from_endpoint(m) for m in models
        )
        self.prompt_hash: str | None = None
        self.layer_results: list[LayerResult] = []
        self.tool_trace: list[ToolTraceEntry] = []
        self.human_decisions: list[HumanDecision] = []
        self.repair_count = 0

    def set_prompt(self, prompt: str) -> None:
        self.prompt_hash = prompt_hash(prompt)

    def add_tool_call(self, entry: ToolTraceEntry) -> None:
        self.tool_trace.append(entry)

    def add_layer_result(
        self, layer: LayerName, outcome: LayerOutcome, errors: list[str] | tuple[str, ...] = ()
    ) -> None:
        self.layer_results.append(
            LayerResult(
                layer=layer, outcome=outcome, attempt=self.repair_count, errors=tuple(errors)
            )
        )

    def start_repair(self) -> None:
        self.repair_count += 1

    def add_human_decision(self, decision: HumanDecision) -> None:
        self.human_decisions.append(decision)

    def build(self) -> Provenance:
        if self.prompt_hash is None:
            raise ProvenanceError("provenance.prompt_hash is required; call set_prompt()")
        return Provenance(
            spec_version=self.spec_version,
            cell_id=self.cell_id,
            seed=self.seed,
            prompt_hash=self.prompt_hash,
            models=self.models,
            layer_results=tuple(self.layer_results),
            tool_trace=tuple(self.tool_trace),
            repair_count=self.repair_count,
            human_decisions=tuple(self.human_decisions),
            run_id=self.run_id,
        )


def attach(record: Mapping[str, Any], provenance: Provenance) -> dict[str, Any]:
    """Return a copy of record with provenance under PROVENANCE_KEY."""
    if PROVENANCE_KEY in record:
        raise ProvenanceError(f"record already has a {PROVENANCE_KEY!r} field")
    return {**record, PROVENANCE_KEY: provenance.to_dict()}


def split(record: Mapping[str, Any]) -> tuple[dict[str, Any], Provenance]:
    """Inverse of attach(): the bare record and its provenance."""
    if PROVENANCE_KEY not in record:
        raise ProvenanceError(f"record has no {PROVENANCE_KEY!r} field")
    bare = {k: v for k, v in record.items() if k != PROVENANCE_KEY}
    return bare, Provenance.from_dict(record[PROVENANCE_KEY])
