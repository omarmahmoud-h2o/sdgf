"""Concurrent generation: a bounded thread pool over the generator and judge stages, sized
by models.<stage>.concurrency, with output independent of the number of workers.

Every mock here answers from the prompt alone (never from call order), so a sequential and
a concurrent run of the same seed must write the same records, drops and review items.
"""

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

import pytest
import yaml

from sdgf.judge.llm_judge import RECORD_HEADER
from sdgf.models.base import BoundedBackend, ModelBackendError
from sdgf.models.mock import MockBackend
from sdgf.pipeline import ACCEPTED_STREAM, DROPS_STREAM, REVIEW_STREAM, Pipeline, SlotReviews
from sdgf.spec.compile import compile_spec
from sdgf.store.provenance import split
from sdgf.validate.base import ValidationContext
from sdgf.validate.l4_overlap import OverlapLayer
from sdgf.validate.l5_judge import ListReviewSink, ReviewItem
from test_pipeline import fag_reply, recipe_from_prompt

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
TARGET = 40
NO_JUDGE = ("L1", "L2", "L3", "L4")
REPAIR = "previous attempt was rejected"
ADVICE = "Honestly, this account is ideal for your business, so move everything onto it."


def fag_spec(root: Path, *, generator=1, judge=1, review=False):
    """A copy of the FAG task with the given per-stage concurrency."""
    task = root / "fag"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))
    spec = yaml.safe_load((task / "task.yaml").read_text())
    spec["models"]["generator"]["concurrency"] = generator
    spec["models"]["judge"]["concurrency"] = judge
    spec["hitl"]["review_flagged"] = review
    (task / "task.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    return compile_spec(task)


def digest(text: str) -> int:
    return hashlib.sha256(text.encode()).digest()[0]


class Peak:
    """Counts calls in flight, holding each one briefly so overlapping calls show up."""

    def __init__(self, hold: float = 0.002):
        self.hold = hold
        self.now = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __enter__(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(self.hold)

    def __exit__(self, *exc):
        with self.lock:
            self.now -= 1


class OrderFreeWorld:
    """Generator and judge mocks that answer from the prompt only.

    A fixed hash of the recipe picks the non-breach first drafts that advise (the judge
    says breach, so L5 repairs them) and the verdicts given with low confidence, so which
    records are repaired or queued doesn't depend on which call arrives first."""

    def __init__(self, *, unsure=False):
        self.unsure = unsure
        self.generating = Peak()
        self.judging = Peak()

    def generate(self, call) -> str:
        with self.generating:
            recipe = recipe_from_prompt(call.prompt)
            r = fag_reply(recipe)
            key = json.dumps(recipe, sort_keys=True)
            if not recipe["label"] and REPAIR not in call.prompt and digest(key) % 4 == 0:
                r["messages"][-1]["content"] += " " + ADVICE
            return json.dumps(r)

    def judge(self, call) -> str:
        with self.judging:
            record = call.prompt.split(RECORD_HEADER, 1)[1]
            # fag_reply's advisory sentences name the signals only in breach records
            breach = ADVICE in record or "Here is what I" in record
            conf = 0.4 if self.unsure and digest(record) % 5 == 0 else 0.9
            return json.dumps(
                {
                    "verdict": "breach" if breach else "no_breach",
                    "scores": {
                        "advice_tier": "PERSONAL_ADVICE" if breach else "FACTUAL_INFORMATION",
                        "realism": 4,
                    },
                    "confidence": {"verdict": conf, "advice_tier": conf, "realism": 0.9},
                }
            )

    def backends(self):
        return {"generator": MockBackend(self.generate), "judge": MockBackend(self.judge)}


def run(compiled, root, world, **kw):
    pipe = Pipeline(
        compiled, root / "store", model_overrides=world.backends(), target_size=TARGET, **kw
    )
    return pipe, pipe.run("c1")


def artefacts(result):
    """What a run wrote, without provenance's spec_version (the copies differ in concurrency)."""
    records = []
    for line in result.run.read_jsonl(ACCEPTED_STREAM):
        record, prov = split(line)
        records.append((record, prov.cell_id, prov.seed, prov.repair_count))
    return {
        "accepted": records,
        "drops": result.run.read_jsonl(DROPS_STREAM),
        "summary": {k: v for k, v in result.run.read_stage("summary").items() if k != "usage"},
    }


# ── the accepted set doesn't depend on the number of workers ──────


@pytest.fixture(scope="module")
def judged(tmp_path_factory):
    out = {}
    for name, gen, judge in (("sequential", 1, 1), ("concurrent", 6, 3)):
        root = tmp_path_factory.mktemp(name)
        world = OrderFreeWorld()
        pipe, result = run(fag_spec(root, generator=gen, judge=judge), root, world)
        out[name] = (world, pipe, result)
    return out


def test_concurrent_and_sequential_runs_accept_the_same_records(judged):
    _, _, seq = judged["sequential"]
    _, _, con = judged["concurrent"]
    assert seq.complete and con.complete
    assert len(seq.accepted) == len(con.accepted) == TARGET
    assert artefacts(seq) == artefacts(con)
    # L5 repaired some advising drafts in both, so the judge path ran concurrently too
    assert any(prov.repair_count for _, prov in map(split, con.accepted))


def test_pool_size_and_per_stage_limits(judged):
    seq_world, seq_pipe, _ = judged["sequential"]
    con_world, con_pipe, _ = judged["concurrent"]
    assert seq_pipe.workers == 1
    assert seq_world.generating.peak == seq_world.judging.peak == 1
    assert con_pipe.workers == 6  # the busiest stage
    assert 1 < con_world.generating.peak <= 6
    assert 1 < con_world.judging.peak <= 3  # the judge's own limit, below the pool size


def test_bounded_backends_keep_the_endpoint_record(judged):
    _, seq_pipe, seq = judged["sequential"]
    _, con_pipe, con = judged["concurrent"]
    assert con_pipe.models.endpoints() == seq_pipe.models.endpoints()
    assert split(con.accepted[0])[1].models == split(seq.accepted[0])[1].models
    assert con.run.read_stage("spec")["workers"] == 6
    assert seq.run.read_stage("spec")["workers"] == 1


def test_same_seed_concurrent_runs_are_byte_identical(tmp_path):
    compiled = fag_spec(tmp_path, generator=8, judge=8)
    lines = []
    for name in ("a", "b"):
        _, result = run(compiled, tmp_path / name, OrderFreeWorld())
        lines.append(
            [(result.run.path / f"{s}.jsonl").read_bytes() for s in (ACCEPTED_STREAM, DROPS_STREAM)]
        )
    assert lines[0] == lines[1]


def test_review_items_arrive_in_the_same_order(tmp_path):
    reviews = []
    for name, gen in (("seq", 1), ("con", 5)):
        root = tmp_path / name
        compiled = fag_spec(root, generator=gen, judge=gen, review=True)
        _, result = run(compiled, root, OrderFreeWorld(unsure=True))
        assert result.complete
        reviews.append(result.run.read_jsonl(REVIEW_STREAM))
    assert reviews[0] and reviews[0] == reviews[1]


# ── L4 near-duplicates settle in reservation order ───────────────


def copying_generator(peak=None):
    """Non-breach replies of one length all share one fixed text: only the first settles."""

    def reply(call):
        if peak is not None:
            with peak:
                pass
        recipe = recipe_from_prompt(call.prompt)
        r = fag_reply(recipe)
        if not recipe["label"]:
            for m in r["messages"]:
                m["content"] = f"Turn {m['turn']} of a copied conversation. " * 6
        return json.dumps(r)

    return reply


def test_near_duplicates_within_a_wave_drop_the_same_records(tmp_path):
    out = []
    for name, gen in (("seq", 1), ("con", 6)):
        root = tmp_path / name
        compiled = fag_spec(root, generator=gen)
        pipe = Pipeline(
            compiled,
            root / "store",
            model_overrides={"generator": MockBackend(copying_generator(Peak()))},
            target_size=TARGET,
            layers=NO_JUDGE,
            max_attempts_per_cell=3,
        )
        result = pipe.run("c1")
        drops = result.run.read_jsonl(DROPS_STREAM)
        assert drops and {c for d in drops for c in d["codes"]} == {"near_duplicate"}
        assert all(d["hard"] and d["layer"] == "L4" for d in drops)
        out.append(artefacts(result))
    assert out[0] == out[1]


# ── pieces ───────────────────────────────────────────────────────


def test_overlap_staging_is_invisible_to_check_until_committed():
    layer = OverlapLayer.from_spec(compile_spec(FAG_DIR))
    record = {"messages": [{"turn": 1, "role": "customer", "content": "alpha beta " * 20}]}
    ctx = ValidationContext(cell_id="c")
    assert layer.check_staged(record).passed  # nothing staged
    layer.stage(record)
    assert layer.check(record, ctx).passed  # a running wave doesn't see it
    staged = layer.check_staged(record)
    assert not staged.passed and staged.codes == ("near_duplicate",)
    assert staged.errors[0].details["match"] == "corpus:0"
    layer.commit_staged()
    assert layer.corpus_size == 1
    assert layer.check(record, ctx).codes == ("near_duplicate",)
    assert layer.check_staged(record).passed  # the stage is empty again


def test_slot_reviews_hold_items_per_thread_until_flushed():
    sink = ListReviewSink()
    reviews = SlotReviews(sink)
    item = ReviewItem(
        layer="L5", code="low_confidence", cell_id="c", attempt=0, record={}, intended_label=True
    )
    with reviews.capture() as held:
        reviews.submit(item)
        other = []
        t = threading.Thread(target=lambda: other.append(reviews.submit(item)))
        t.start()
        t.join()
    assert held == [item]
    assert sink.items == [item]  # the other thread wasn't capturing, so it passed on
    reviews.submit(item)
    assert len(sink.items) == 2


def test_bounded_backend_limits_calls_in_flight():
    peak = Peak(hold=0.01)

    def reply(call):
        with peak:
            return "ok"

    inner = MockBackend(reply, model="m", hosting="provider_api")
    bounded = BoundedBackend(inner, 2)
    assert (bounded.name, bounded.model, bounded.hosting) == ("mock", "m", "provider_api")
    threads = [threading.Thread(target=bounded.call, args=("p", 1, 0.0)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert peak.peak == 2 and len(inner.calls) == 8
    with pytest.raises(ModelBackendError, match=">= 1"):
        BoundedBackend(inner, 0)
