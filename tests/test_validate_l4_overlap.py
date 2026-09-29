"""L4 overlap layer: seed copying, near-duplicates within the run and an opt-in held-out
check, all hard drops. Every text here is a synthetic string written for the test; the
held-out files are temporary files built in tmp_path."""

import copy
import json
from pathlib import Path

import pytest

from sdgf.spec.compile import compile_spec
from sdgf.validate.base import ValidationContext
from sdgf.validate.cascade import Cascade
from sdgf.validate.l1_schema import SchemaLayer
from sdgf.validate.l2_rules import RulesLayer
from sdgf.validate.l3_governance import GovernanceLayer
from sdgf.validate.l4_overlap import (
    EmbeddingIndex,
    OverlapEngine,
    OverlapError,
    OverlapLayer,
    ShingleIndex,
    jaccard,
    load_held_out,
    normalise,
    record_text,
    sentence_transformers_embedder,
    shingles,
)

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"
CTX = ValidationContext()

SEED_A = "The monthly fee on the Acme Test everyday account is ten dollars with no setup cost."
SEED_B = "Merchant terminals are delivered within five business days after the application."
OTHER = "Our branch in the fictional town of Testville opens at nine on weekdays for enquiries."


def rec(*contents):
    roles = ("customer", "assistant")
    return {
        "messages": [
            {"turn": i + 1, "role": roles[i % 2], "content": c} for i, c in enumerate(contents)
        ],
        "label": False,
        "spans": [],
    }


def layer(threshold=0.8, *, seeds=(SEED_A, SEED_B), held_out=None, corpus_max=None, k=5):
    return OverlapLayer(
        [
            OverlapEngine(
                build=lambda: ShingleIndex(k),
                thresholds={
                    "seed": threshold,
                    "held_out": threshold,
                    "corpus": threshold if corpus_max is None else corpus_max,
                },
            )
        ],
        seeds=[(f"seed:{i}", s) for i, s in enumerate(seeds)],
        held_out=held_out,
    )


# ── shingles and text ────────────────────────────────────────────


def test_normalise_casefolds_and_strips_punctuation():
    assert normalise("  Hello,   WORLD!\n") == "hello world"


def test_shingles_and_jaccard():
    assert shingles("abcdef", 3) == {"abc", "bcd", "cde", "def"}
    assert shingles("ab", 5) == {"ab"}
    assert shingles("  !! ", 5) == frozenset()
    assert jaccard(shingles(SEED_A), shingles(SEED_A.upper())) == 1.0
    assert jaccard(frozenset(), shingles(SEED_A)) == 0.0
    assert jaccard(shingles(SEED_A), shingles(OTHER)) < 0.3


def test_shingle_size_must_be_positive():
    with pytest.raises(OverlapError):
        shingles("abc", 0)
    with pytest.raises(OverlapError):
        ShingleIndex(0)


def test_record_text_prefers_messages_and_skips_code_owned_fields():
    r = {**rec("hi there", "hello"), "primary_topic": "Business Cards"}
    assert record_text(r) == "hi there\nhello"
    assert record_text({"question": "Q?", "answer": "A.", "_provenance": {"x": "secret"}}) == (
        "Q?\nA."
    )


def test_shingle_index_best_match():
    index = ShingleIndex()
    assert index.best(SEED_A) is None
    index.add("a", SEED_A)
    index.add("b", SEED_B)
    match = index.best(SEED_B + " Thanks.")
    assert match.key == "b" and 0.8 < match.score < 1.0
    assert len(index) == 2


# ── seeds ────────────────────────────────────────────────────────


def test_original_text_passes():
    assert layer().check(rec("Hi", OTHER), CTX).passed


def test_copied_seed_is_hard_dropped():
    v = layer().check(rec(SEED_A), CTX)
    assert v.hard and v.codes == ("seed_overlap",)
    d = v.errors[0].details
    assert d["source"] == "seed" and d["match"] == "seed:0" and d["score"] == 1.0
    assert d["engine"] == "shingle" and d["threshold"] == 0.8


def test_lightly_edited_seed_is_caught():
    edited = SEED_A.replace("ten dollars", "ten dollars a month")
    assert layer().check(rec(edited), CTX).codes == ("seed_overlap",)


def test_threshold_is_exclusive_and_configurable():
    score = jaccard(shingles(SEED_A), shingles(SEED_A + " Cheers."))
    assert layer(threshold=score).check(rec(SEED_A + " Cheers."), CTX).passed
    assert layer(threshold=score - 0.01).check(rec(SEED_A + " Cheers."), CTX).hard


def test_issue_never_repeats_matched_text():
    v = layer().check(rec(SEED_A), CTX)
    dumped = json.dumps([e.to_dict() for e in v.errors])
    assert "monthly fee" not in dumped and "Acme" not in dumped


def test_empty_record_text_passes():
    assert layer().check(rec(""), CTX).passed


# ── corpus near-duplicates ───────────────────────────────────────


def test_near_duplicate_only_after_remember():
    l4 = layer()
    first = rec("Hi", OTHER)
    assert l4.check(first, CTX).passed
    # Passing L4 doesn't add a record: a later layer may still reject it.
    assert l4.check(copy.deepcopy(first), CTX).passed
    l4.remember(first, key="SYN-1")
    assert l4.corpus_size == 1
    v = l4.check(rec("Hi", OTHER + " Thanks!"), CTX)
    assert v.hard and v.codes == ("near_duplicate",)
    assert v.errors[0].details["match"] == "SYN-1"


def test_remember_default_keys_count_up():
    l4 = layer()
    l4.remember(rec(OTHER))
    l4.remember(rec(SEED_B + " different"))
    assert l4.check(rec(OTHER), CTX).errors[0].details["match"] == "corpus:0"


def test_separate_near_duplicate_threshold():
    l4 = layer(threshold=0.99, corpus_max=0.5)
    l4.remember(rec(OTHER))
    assert l4.check(rec(OTHER + " We also open on Saturday mornings."), CTX).codes == (
        "near_duplicate",
    )


def test_distinct_records_accumulate_without_false_positives():
    l4 = layer()
    texts = [
        f"Customer {n} asked about the fictional fee schedule for plan {chr(65 + n)} "
        f"and the assistant quoted {n * 3 + 7} dollars for service tier {n}."
        for n in range(10)
    ]
    for i, t in enumerate(texts):
        assert l4.check(rec(OTHER[: 20 + i], t), CTX).passed
        l4.remember(rec(OTHER[: 20 + i], t))
    assert l4.corpus_size == 10


# ── held-out ─────────────────────────────────────────────────────


def test_held_out_off_by_default():
    l4 = layer()
    assert not l4.held_out_enabled
    assert l4.check(rec(OTHER), CTX).passed


def test_held_out_match_is_reported(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps(rec("Hello", OTHER)) + "\n\n" + json.dumps(rec(SEED_B)) + "\n")
    held = load_held_out([path])
    assert [k for k, _ in held] == ["eval.jsonl:1", "eval.jsonl:3"]
    l4 = layer(seeds=(), held_out=held)
    assert l4.held_out_enabled
    v = l4.check(rec("Hello", OTHER), CTX)
    assert v.hard and v.codes == ("held_out_overlap",)
    assert v.errors[0].details["match"] == "eval.jsonl:1"
    assert "Testville" not in json.dumps([e.to_dict() for e in v.errors])


def test_held_out_csv_with_columns_and_literal_lists(tmp_path):
    path = tmp_path / "eval.csv"
    turns = [{"role": "user", "Content": "Hello"}, {"role": "assistant", "Content": OTHER}]
    path.write_text(
        "id,conversation\n" + f'1,"{turns!r}"\n' + "2,\n" + f'3,"{SEED_B}"\n',
        encoding="utf-8",
    )
    held = load_held_out([path], columns=["conversation"])
    assert held == [("eval.csv:1", f"Hello {OTHER}"), ("eval.csv:3", SEED_B)]
    assert layer(seeds=(), held_out=held).check(rec("Hello", OTHER), CTX).codes == (
        "held_out_overlap",
    )


def test_held_out_load_errors(tmp_path):
    with pytest.raises(OverlapError, match="not found"):
        load_held_out([tmp_path / "missing.jsonl"])
    txt = tmp_path / "eval.txt"
    txt.write_text("x\n")
    with pytest.raises(OverlapError, match=".jsonl or .csv"):
        load_held_out([txt])
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    with pytest.raises(OverlapError, match="bad.jsonl:1"):
        load_held_out([bad])
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("a,b\n1,2\n")
    with pytest.raises(OverlapError, match="missing columns"):
        load_held_out([csv_path], columns=["text"])


def test_layer_keeps_no_held_out_text(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps(rec(OTHER)) + "\n")
    l4 = layer(seeds=(), held_out=load_held_out([path]))
    state = repr(vars(l4)) + repr([vars(i) for idx in l4._indexes for i in idx.values()])
    assert OTHER not in state


# ── engines ──────────────────────────────────────────────────────


def fake_embed(texts):
    # Bag of a few words: deterministic, no model needed.
    vocab = ("fee", "merchant", "branch", "account", "terminal")
    return [[normalise(t).split().count(w) for w in vocab] for t in texts]


def test_embedding_index_cosine():
    index = EmbeddingIndex(fake_embed)
    index.add("a", "fee fee account")
    index.add("empty", "   ")
    index.add("zero", "nothing matching here")
    assert len(index) == 1
    assert index.best("account fee fee").score == pytest.approx(1.0)
    assert index.best("zzz") is None


def test_embedding_engine_runs_with_its_own_thresholds():
    l4 = OverlapLayer(
        [
            OverlapEngine(build=ShingleIndex, thresholds={"seed": 0.9}),
            OverlapEngine(build=lambda: EmbeddingIndex(fake_embed), thresholds={"seed": 0.95}),
        ],
        seeds=[("seed:0", "What is the account fee?")],
    )
    v = l4.check(rec("Tell me the fee for the account please."), CTX)
    assert v.hard and [e.details["engine"] for e in v.errors] == ["embedding"]
    assert set(l4.scores(rec("account fee"))) == {("shingle", "seed"), ("embedding", "seed")}


def test_engine_config_errors():
    with pytest.raises(OverlapError):
        OverlapEngine(build=ShingleIndex, thresholds={"test": 0.5})
    with pytest.raises(OverlapError):
        OverlapEngine(build=ShingleIndex, thresholds={"seed": 1.5})
    with pytest.raises(OverlapError):
        OverlapLayer([])


def test_source_without_threshold_is_not_checked():
    l4 = OverlapLayer(
        [OverlapEngine(build=ShingleIndex, thresholds={"corpus": 0.8})],
        seeds=[("seed:0", SEED_A)],
    )
    assert l4.check(rec(SEED_A), CTX).passed


def test_sentence_transformers_adapter_is_lazy(monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(Exception, match="sdgf\\[embeddings\\]"):
        sentence_transformers_embedder()

    class FakeModel:
        def __init__(self, name):
            self.name = name

        def encode(self, texts):
            return [[1.0, float(len(t))] for t in texts]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeModel)
    )
    assert sentence_transformers_embedder("fake")(["ab"]) == [[1.0, 2.0]]


# ── FAG spec ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fag():
    return compile_spec(FAG_DIR)


def test_from_spec_uses_seeds_and_overlap_max(fag):
    l4 = OverlapLayer.from_spec(fag)
    assert not l4.held_out_enabled
    for seed in fag.seeds:
        v = l4.check(seed, CTX)
        assert v.codes == ("seed_overlap",)
        assert v.errors[0].details["threshold"] == fag.spec.thresholds.overlap_max


def test_fag_seeds_are_not_near_duplicates_of_each_other(fag):
    l4 = OverlapLayer.from_spec(fag)
    assert l4.scores(fag.seeds[0])[("shingle", "seed")].score == 1.0
    for i, seed in enumerate(fag.seeds):
        rest = OverlapLayer(
            l4.engines,
            seeds=[(f"seed:{j}", record_text(s)) for j, s in enumerate(fag.seeds) if j != i],
        )
        assert rest.check(seed, CTX).passed


def test_from_spec_held_out_and_embedding(fag, tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps(rec("Hello", OTHER)) + "\n")
    l4 = OverlapLayer.from_spec(
        fag,
        held_out_paths=[path],
        near_duplicate_max=0.5,
        embedder=fake_embed,
        embedding_thresholds={"seed": 0.999},
    )
    assert l4.held_out_enabled and len(l4.engines) == 2
    assert l4.engines[0].thresholds["corpus"] == 0.5
    assert "held_out_overlap" in l4.check(rec("Hello", OTHER), CTX).codes
    with pytest.raises(OverlapError, match="own thresholds"):
        OverlapLayer.from_spec(fag, embedder=fake_embed)


def test_from_spec_needs_overlap_max(fag):
    spec = fag.spec.model_copy(
        update={"thresholds": fag.spec.thresholds.model_copy(update={"overlap_max": None})}
    )
    unset = type(fag)(**{**vars(fag), "spec": spec})
    with pytest.raises(OverlapError, match="overlap_max"):
        OverlapLayer.from_spec(unset)


def test_copied_fag_seed_passes_l1_to_l3_and_drops_at_l4(fag):
    cascade = Cascade(
        [
            SchemaLayer.from_spec(fag),
            RulesLayer.from_spec(fag),
            GovernanceLayer.from_spec(fag),
            OverlapLayer.from_spec(fag),
        ]
    )
    result = cascade.run(copy.deepcopy(fag.seeds[5]), CTX)
    assert result.failed_layer == "L4" and result.hard and not result.repairable
    assert result.layers_run == ("L1", "L2", "L3", "L4")
