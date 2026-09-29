"""HITL review queue and coverage-plan approval (hitl/queue.py)."""

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sdgf.hitl.queue import (
    GOLD_KEY,
    ApprovalPending,
    GoldSet,
    ReviewError,
    ReviewQueue,
    approval_stage,
    approve_plan,
    item_id,
    plan_approved,
    require_plan_approval,
)
from sdgf.judge.calibration import gold_from_records
from sdgf.pipeline import Pipeline
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore
from sdgf.validate.l5_judge import ReviewItem
from test_m4_checkpoint import TARGET, World
from test_pipeline import NO_JUDGE, valid_backend

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def with_hitl(fag, **flags):
    spec = fag.spec.model_copy(update={"hitl": fag.spec.hitl.model_copy(update=flags)})
    return dataclasses.replace(fag, spec=spec)


def review_item(n: int = 0, label: bool = False, code: str = "low_confidence") -> ReviewItem:
    record = {
        "messages": [
            {"turn": 1, "role": "customer", "content": f"Question {n} from Acme Test Pty Ltd."},
            {"turn": 2, "role": "assistant", "content": "The monthly fee is $10."},
        ],
        "label": label,
        "spans": [],
    }
    return ReviewItem(
        layer="L5",
        code=code,
        cell_id="non_corps_act|false|single_turn",
        attempt=0,
        record=record,
        intended_label=label,
        judge={"verdict": "breach", "confidence": {"verdict": 0.4}},
    )


@pytest.fixture
def queue(tmp_path):
    return ReviewQueue(tmp_path / "q", GoldSet(tmp_path / "gold.jsonl"))


# ── queue ────────────────────────────────────────────────────────


def test_submitted_items_are_pending_with_stable_ids(queue):
    ids = [queue.submit(review_item(n)) for n in range(3)]
    assert len(set(ids)) == 3
    assert list(queue.pending()) == ids
    assert ids[0] == item_id(review_item(0).to_dict())


def test_the_same_item_submitted_twice_is_one_item(queue):
    queue.submit(review_item(0))
    queue.submit(review_item(0))
    assert len(queue.pending()) == 1


def test_accept_keeps_the_intended_label_and_adds_a_gold_entry(queue):
    i = queue.submit(review_item(0, label=False))
    res = queue.resolve(i, "accept", reviewer="reviewer-a", now=NOW)
    assert res.label is False and res.record["label"] is False
    assert res.decision.action == "accept" and res.decision.decided_at == NOW.isoformat()
    assert queue.pending() == {}
    (gold,) = queue.gold.records()
    assert gold["label"] is False
    assert gold[GOLD_KEY]["item_id"] == i and gold[GOLD_KEY]["reviewer"] == "reviewer-a"


def test_relabel_sets_the_human_label_on_the_record_and_the_gold_entry(queue):
    i = queue.submit(review_item(0, label=False))
    res = queue.resolve(i, "relabel", reviewer="r", new_label=True, reason="it advises")
    assert res.record["label"] is True and res.decision.new_label is True
    (gold,) = queue.gold.records()
    assert gold["label"] is True
    assert gold[GOLD_KEY]["action"] == "relabel" and gold[GOLD_KEY]["intended_label"] is False


def test_reject_drops_the_record_and_adds_no_gold_entry(queue):
    i = queue.submit(review_item(0))
    res = queue.resolve(i, "reject", reviewer="r", reason="incoherent")
    assert res.record is None and res.label is None
    assert queue.gold.records() == []
    assert queue.pending() == {} and queue.resolved_records() == []


def test_resolved_records_are_the_accepted_and_relabelled_ones(queue):
    a, b, c = (queue.submit(review_item(n)) for n in range(3))
    queue.resolve(a, "accept", reviewer="r")
    queue.resolve(b, "reject", reviewer="r")
    queue.resolve(c, "relabel", reviewer="r", new_label=True)
    assert [r["label"] for r in queue.resolved_records()] == [False, True]


def test_gold_entries_feed_calibration_directly(queue):
    for n in range(2):
        queue.resolve(queue.submit(review_item(n)), "accept", reviewer="r")
    items = gold_from_records(queue.gold.records())
    assert [g.label for g in items] == [False, False]


def test_decisions_survive_a_new_queue_object(queue, tmp_path):
    i = queue.submit(review_item(0))
    queue.resolve(i, "accept", reviewer="r")
    again = ReviewQueue(tmp_path / "q", GoldSet(tmp_path / "gold.jsonl"))
    assert again.pending() == {} and i in again.resolutions()


def test_a_resolve_repeated_after_a_crash_does_not_duplicate_gold(queue):
    i = queue.submit(review_item(0))
    queue.resolve(i, "accept", reviewer="r")
    queue.decisions_path.unlink()  # as if the process died after writing gold
    queue.resolve(i, "accept", reviewer="r")
    assert len(queue.gold.records()) == 1


@pytest.mark.parametrize(
    "action, kw, message",
    [
        ("approve", {}, "unknown action"),
        ("accept", {"reviewer": ""}, "reviewer"),
        ("relabel", {}, "needs new_label"),
        ("accept", {"new_label": True}, "only for relabel"),
        ("relabel", {"new_label": False}, "use accept"),
    ],
)
def test_malformed_resolutions_are_refused(queue, action, kw, message):
    i = queue.submit(review_item(0, label=False))
    kw = {"reviewer": "r", **kw}
    with pytest.raises(ReviewError, match=message):
        queue.resolve(i, action, **kw)
    assert list(queue.pending()) == [i]


def test_unknown_and_already_resolved_items_are_refused(queue):
    with pytest.raises(ReviewError, match="no review item"):
        queue.resolve("0" * 16, "accept", reviewer="r")
    i = queue.submit(review_item(0))
    queue.resolve(i, "reject", reviewer="r")
    with pytest.raises(ReviewError, match="already resolved"):
        queue.resolve(i, "accept", reviewer="r")


def test_a_queue_without_a_gold_set_still_resolves(tmp_path):
    q = ReviewQueue(tmp_path / "q")
    res = q.resolve(q.submit(review_item(0)), "accept", reviewer="r")
    assert res.record is not None


def test_gold_set_for_task_is_shared_across_spec_versions(tmp_path):
    store = ArtefactStore(tmp_path / "store")
    assert GoldSet.for_task(store, "fag").path == tmp_path / "store" / "gold" / "fag.jsonl"


def test_pipeline_review_stream_is_resolvable_into_the_gold_set(fag, tmp_path):
    reviewed = with_hitl(fag, review_flagged=True)
    world = World(advise_every=0, confidence=0.5)
    store = ArtefactStore(tmp_path / "store")
    pipe = Pipeline(
        reviewed,
        store,
        model_overrides=world.backends(),
        target_size=TARGET,
        max_attempts_per_cell=1,
    )
    result = pipe.run("h1")
    gold = GoldSet.for_task(store, fag.spec.task.name)
    q = ReviewQueue.for_run(result.run, gold)
    pending = q.pending()
    assert pending and all(item["code"] == "low_confidence" for item in pending.values())
    first, *rest = pending
    q.resolve(first, "accept", reviewer="r")
    assert len(q.pending()) == len(rest)
    (entry,) = gold.records()
    assert entry["label"] == pending[first]["intended_label"]
    assert "_provenance" not in entry


# ── plan approval ────────────────────────────────────────────────


def test_a_run_waits_for_plan_approval_then_proceeds(fag, tmp_path):
    gated = with_hitl(fag, approve_coverage_plan=True)
    store = ArtefactStore(tmp_path / "store")
    pipe = Pipeline(
        gated,
        store,
        model_overrides={"generator": valid_backend()},
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    with pytest.raises(ApprovalPending, match="awaiting approval") as e:
        pipe.run("p1")
    assert not e.value.path.exists()
    assert pipe.models.backend("generator").calls == []  # no generation spend before approval
    assert store.has_shared(fag.spec_version, pipe.plan_stage)  # the plan is there to review

    approve_plan(store, fag.spec_version, pipe.plan_stage, reviewer="r", now=NOW)
    assert e.value.path.exists()
    result = pipe.run("p1")
    assert result.complete and len(result.accepted) == TARGET


def test_editing_the_plan_after_approval_needs_approval_again(fag, tmp_path):
    store = ArtefactStore(tmp_path / "store")
    pipe = Pipeline(
        fag,
        store,
        model_overrides={"generator": valid_backend()},
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    pipe.plan()
    sv, stage = fag.spec_version, pipe.plan_stage
    approve_plan(store, sv, stage, reviewer="r")
    assert plan_approved(store, sv, stage)
    require_plan_approval(store, sv, stage)

    plan = store.read_shared(sv, stage)
    plan["cells"][0]["quota"] += 1
    store.write_shared(sv, stage, plan)
    assert not plan_approved(store, sv, stage)
    with pytest.raises(ApprovalPending, match="changed since it was approved"):
        require_plan_approval(store, sv, stage)
    approval = json.loads(store.shared_path(sv, approval_stage(stage)).read_text())
    assert approval["data"]["reviewer"] == "r"


def test_without_the_checkpoint_no_approval_is_needed(fag, tmp_path):
    pipe = Pipeline(
        fag,
        tmp_path / "store",
        model_overrides={"generator": valid_backend()},
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    assert pipe.run("p1").complete


def test_approval_needs_a_reviewer(fag, tmp_path):
    store = ArtefactStore(tmp_path / "store")
    pipe = Pipeline(
        fag,
        store,
        model_overrides={"generator": valid_backend()},
        target_size=TARGET,
        layers=NO_JUDGE,
    )
    pipe.plan()
    with pytest.raises(ReviewError, match="reviewer"):
        approve_plan(store, fag.spec_version, pipe.plan_stage, reviewer="")
