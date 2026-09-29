"""Human-in-the-loop checkpoints (FRAMEWORK_DESIGN.md §7.10, §10, D3).

Review queue. Records L5 routes to people (low confidence, unusable judge output; judge
disagreement once L6 escalates) are items in a file-based queue:

    <dir>/<items>.jsonl      one ReviewItem.to_dict() per line, append-only
    <dir>/<decisions>.jsonl  one resolution per line, append-only

An item's id is a hash of its content, so the queue needs no counter and the same item
submitted twice is one item. pending() is every item with no decision. resolve() takes
one of three actions:

    accept    the intended label is right     gold entry with the intended label
    relabel   the record shows another label  gold entry with new_label
    reject    the record is unusable          no gold entry (a person gave no label)

Accepted and relabelled items are appended to the task's gold set, a JSONL of records
carrying the human label in their label field plus a "_gold" note (item id, action,
reviewer), so judge/calibration.gold_from_records() reads it as is and judges never see
the note. The gold entry is written before the decision and skipped if already there,
so a resolve cut short by a crash can simply be repeated. The gold set is per task, not
per spec_version: a changed rubric needs a fresh calibration, not fresh labels.

Plan approval. With hitl.approve_coverage_plan, generation waits until a person has
approved the coverage plan: require_plan_approval() raises ApprovalPending until an
approval artefact exists next to the plan. The approval pins the plan's sha256, so a
plan edited or rebuilt after approval needs approving again, while edits made before
approving are what gets approved.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from sdgf.store.artefacts import ArtefactStore, JsonlWriter, RunDir, _check_name, iter_jsonl
from sdgf.store.provenance import HumanAction, HumanDecision
from sdgf.validate.l5_judge import LABEL_FIELD, ReviewItem

ITEMS_STREAM = "review"
DECISIONS_STREAM = "review_decisions"
GOLD_KEY = "_gold"
APPROVAL_SUFFIX = "-approval"

_ACTIONS: tuple[HumanAction, ...] = ("accept", "reject", "relabel")


class ReviewError(ValueError):
    """An unknown or already resolved item, or a malformed decision."""


class ApprovalPending(RuntimeError):
    """The coverage plan needs a person's approval before generation can start."""

    def __init__(self, message: str, path: Path):
        super().__init__(message)
        self.path = path


def _canonical(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def item_id(item: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(dict(item)).encode("utf-8")).hexdigest()[:16]


def _now(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).isoformat()


# ── gold set ─────────────────────────────────────────────────────


class GoldSet:
    """The task's human-labelled records, appended to by review resolutions."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @classmethod
    def for_task(cls, store: ArtefactStore, task_name: str) -> GoldSet:
        """<store>/gold/<task>.jsonl, shared by every spec_version of the task."""
        return cls(store.root / "gold" / f"{_check_name('task name', task_name)}.jsonl")

    def records(self) -> list[dict[str, Any]]:
        return list(iter_jsonl(self.path))

    def item_ids(self) -> set[str]:
        return {r[GOLD_KEY]["item_id"] for r in self.records() if GOLD_KEY in r}

    def add(self, record: Mapping[str, Any]) -> None:
        with JsonlWriter(self.path) as w:
            w.write(dict(record))


# ── review queue ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Resolution:
    item_id: str
    decision: HumanDecision
    record: dict[str, Any] | None  # the record as resolved; None when rejected
    label: Any  # the human label; None when rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "decision": {
                "action": self.decision.action,
                "reviewer": self.decision.reviewer,
                "reason": self.decision.reason,
                "new_label": self.decision.new_label,
                "decided_at": self.decision.decided_at,
            },
            "record": self.record,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Resolution:
        return cls(d["item_id"], HumanDecision(**d["decision"]), d.get("record"), d.get("label"))


class ReviewQueue:
    """A file-based review queue; also a ReviewSink, so L5 can submit to it directly."""

    def __init__(
        self,
        directory: str | Path,
        gold: GoldSet | None = None,
        *,
        items: str = ITEMS_STREAM,
        decisions: str = DECISIONS_STREAM,
    ):
        self.directory = Path(directory)
        self.items_path = self.directory / f"{items}.jsonl"
        self.decisions_path = self.directory / f"{decisions}.jsonl"
        self.gold = gold

    @classmethod
    def for_run(cls, run: RunDir, gold: GoldSet | None = None) -> ReviewQueue:
        """The queue over a run's review stream, which the pipeline writes during the run."""
        return cls(run.path, gold)

    def submit(self, item: ReviewItem | Mapping[str, Any]) -> str:
        data = item.to_dict() if isinstance(item, ReviewItem) else dict(item)
        with JsonlWriter(self.items_path) as w:
            w.write(data)
        return item_id(data)

    def items(self) -> dict[str, dict[str, Any]]:
        """Every submitted item by id, in submission order, duplicates collapsed."""
        out: dict[str, dict[str, Any]] = {}
        for data in iter_jsonl(self.items_path):
            out.setdefault(item_id(data), data)
        return out

    def resolutions(self) -> dict[str, Resolution]:
        return {d["item_id"]: Resolution.from_dict(d) for d in iter_jsonl(self.decisions_path)}

    def pending(self) -> dict[str, dict[str, Any]]:
        done = self.resolutions()
        return {i: data for i, data in self.items().items() if i not in done}

    def get(self, id_: str) -> dict[str, Any]:
        items = self.items()
        if id_ not in items:
            raise ReviewError(f"no review item {id_!r} in {self.items_path}")
        return items[id_]

    def resolve(
        self,
        id_: str,
        action: str,
        *,
        reviewer: str,
        reason: str | None = None,
        new_label: Any = None,
        now: datetime | None = None,
    ) -> Resolution:
        if action not in _ACTIONS:
            raise ReviewError(f"unknown action {action!r}; use one of {list(_ACTIONS)}")
        if not reviewer:
            raise ReviewError("a reviewer is required")
        if action == "relabel" and new_label is None:
            raise ReviewError("relabel needs new_label")
        if action != "relabel" and new_label is not None:
            raise ReviewError(f"new_label is only for relabel, not {action}")
        item = self.get(id_)
        if id_ in self.resolutions():
            raise ReviewError(f"review item {id_!r} is already resolved")
        if action == "relabel" and new_label == item["intended_label"]:
            raise ReviewError("new_label equals the intended label; use accept")

        decision = HumanDecision(
            action=action,  # type: ignore[arg-type]
            reviewer=reviewer,
            reason=reason,
            new_label=new_label,
            decided_at=_now(now),
        )
        record: dict[str, Any] | None = None
        label: Any = None
        if action != "reject":
            label = item["intended_label"] if action == "accept" else new_label
            record = {**item["record"], LABEL_FIELD: label}
        resolution = Resolution(id_, decision, record, label)

        if record is not None and self.gold is not None and id_ not in self.gold.item_ids():
            self.gold.add(
                {
                    **record,
                    GOLD_KEY: {
                        "item_id": id_,
                        "action": action,
                        "reviewer": reviewer,
                        "intended_label": item["intended_label"],
                        "cell_id": item.get("cell_id"),
                        "code": item.get("code"),
                        "decided_at": decision.decided_at,
                    },
                }
            )
        with JsonlWriter(self.decisions_path) as w:
            w.write(resolution.to_dict())
        return resolution

    def resolved_records(self) -> list[dict[str, Any]]:
        """Records people accepted or relabelled, in decision order."""
        return [r.record for r in self.resolutions().values() if r.record is not None]


# ── coverage-plan approval ───────────────────────────────────────


def approval_stage(plan_stage: str) -> str:
    return plan_stage + APPROVAL_SUFFIX


def plan_sha256(store: ArtefactStore, spec_version: str, plan_stage: str) -> str:
    plan = store.read_shared(spec_version, plan_stage)
    return "sha256:" + hashlib.sha256(_canonical(plan).encode("utf-8")).hexdigest()


def approve_plan(
    store: ArtefactStore,
    spec_version: str,
    plan_stage: str,
    *,
    reviewer: str,
    note: str | None = None,
    now: datetime | None = None,
) -> Path:
    """Record a person's approval of the plan as it is now (after any edits)."""
    if not reviewer:
        raise ReviewError("a reviewer is required")
    return store.write_shared(
        spec_version,
        approval_stage(plan_stage),
        {
            "plan_stage": plan_stage,
            "plan_sha256": plan_sha256(store, spec_version, plan_stage),
            "reviewer": reviewer,
            "note": note,
            "approved_at": _now(now),
        },
    )


def plan_approved(store: ArtefactStore, spec_version: str, plan_stage: str) -> bool:
    stage = approval_stage(plan_stage)
    if not store.has_shared(spec_version, stage):
        return False
    approval = store.read_shared(spec_version, stage)
    return approval.get("plan_sha256") == plan_sha256(store, spec_version, plan_stage)


def require_plan_approval(store: ArtefactStore, spec_version: str, plan_stage: str) -> None:
    """Raise ApprovalPending unless the current plan has been approved."""
    if plan_approved(store, spec_version, plan_stage):
        return
    stage = approval_stage(plan_stage)
    changed = store.has_shared(spec_version, stage)
    path = store.shared_path(spec_version, stage)
    raise ApprovalPending(
        f"coverage plan {plan_stage} "
        + ("changed since it was approved" if changed else "is awaiting approval")
        + f"; review it and approve it (writes {path}), then run again",
        path,
    )
