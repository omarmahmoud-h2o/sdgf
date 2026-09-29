"""Diversity metrics on small hand-worked inputs: distinct-n, self-BLEU, the optional
embedding cluster entropy, and the overall/per-cell report. All texts are synthetic."""

import math
import random

import pytest

from sdgf.evaluation.diversity import (
    UNKNOWN_CELL,
    DiversityError,
    cell_of,
    cluster_entropy,
    distinct_n,
    diversity_report,
    ngrams,
    self_bleu,
    tokenize,
)
from sdgf.store.provenance import PROVENANCE_KEY


def rec(cell, *contents):
    roles = ("customer", "assistant")
    r = {
        "messages": [
            {"turn": i + 1, "role": roles[i % 2], "content": c} for i, c in enumerate(contents)
        ],
        "label": False,
        "spans": [],
    }
    if cell is not None:
        r[PROVENANCE_KEY] = {"cell_id": cell}
    return r


# ── tokens and n-grams ───────────────────────────────────────────


def test_tokenize_casefolds_and_drops_punctuation():
    assert tokenize("The Fee, the FEE!") == ["the", "fee", "the", "fee"]


def test_ngrams():
    assert ngrams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert ngrams(["a"], 2) == []
    with pytest.raises(DiversityError):
        ngrams(["a"], 0)


# ── distinct-n ───────────────────────────────────────────────────


def test_distinct_n_known_values():
    texts = ["a b a b", "a c"]
    # unigrams: a b a b a c -> 3 unique / 6
    assert distinct_n(texts, 1) == pytest.approx(3 / 6)
    # bigrams: ab ba ab | ac -> {ab, ba, ac} 3 / 4
    assert distinct_n(texts, 2) == pytest.approx(3 / 4)


def test_distinct_n_all_unique_and_all_repeated():
    assert distinct_n(["one two three", "four five six"], 1) == 1.0
    assert distinct_n(["same same", "same"], 1) == pytest.approx(1 / 3)


def test_distinct_n_none_without_ngrams():
    assert distinct_n([], 1) is None
    assert distinct_n(["single"], 2) is None


# ── self-BLEU ────────────────────────────────────────────────────


def test_self_bleu_identical_texts_is_one():
    text = "the monthly fee on the test account is ten dollars"
    assert self_bleu([text, text, text]) == pytest.approx(1.0)


def test_self_bleu_disjoint_texts_is_zero_without_smoothing():
    assert self_bleu(["alpha beta gamma delta", "one two three four"], smooth=False) == 0.0


def test_self_bleu_disjoint_texts_is_small_with_smoothing():
    score = self_bleu(["alpha beta gamma delta", "one two three four"])
    assert 0.0 < score < 0.2


def test_self_bleu_hand_computed_bigram():
    # hyp "a b c" vs ref "a b d", max_n=2: p1 = 2/3, p2 = 1/2, equal lengths -> BP 1
    expected = math.sqrt(2 / 3 * 1 / 2)
    assert self_bleu(["a b c", "a b d"], max_n=2) == pytest.approx(expected)


def test_self_bleu_brevity_penalty():
    # hyp "a b" vs ref "a b c d": precisions 1, BP exp(1 - 4/2); hyp "a b c d" vs "a b":
    # p1 = 2/4, p2 = 1/3, longer than ref so BP 1
    short = math.exp(1 - 4 / 2)
    long = math.sqrt(2 / 4 * 1 / 3)
    assert self_bleu(["a b", "a b c d"], max_n=2) == pytest.approx((short + long) / 2)


def test_self_bleu_orders_ranked():
    similar = ["the fee is ten dollars a month", "the fee is ten dollars per month"]
    varied = ["the fee is ten dollars a month", "terminals arrive within five business days"]
    assert self_bleu(similar) > self_bleu(varied)


def test_self_bleu_needs_two_texts():
    assert self_bleu([]) is None
    assert self_bleu(["only one"]) is None
    assert self_bleu(["only one", "", "!!!"]) is None


def test_self_bleu_sampling_is_seeded():
    texts = [f"text number {i} about topic {i % 3}" for i in range(12)]
    a = self_bleu(texts, max_hypotheses=4, rng=random.Random(7))
    b = self_bleu(texts, max_hypotheses=4, rng=random.Random(7))
    assert a == b
    assert self_bleu(texts, max_hypotheses=100) == self_bleu(texts)
    with pytest.raises(DiversityError, match="random.Random"):
        self_bleu(texts, max_hypotheses=4)
    with pytest.raises(DiversityError):
        self_bleu(texts, max_n=0)


# ── cluster entropy ──────────────────────────────────────────────


def test_cluster_entropy_identical_vectors_is_zero():
    assert cluster_entropy([[1.0, 0.0]] * 5, k=3) == 0.0


def test_cluster_entropy_two_even_groups_is_ln2():
    vectors = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]
    assert cluster_entropy(vectors, k=2) == pytest.approx(math.log(2))


def test_cluster_entropy_uneven_groups():
    vectors = [[1.0, 0.0]] * 3 + [[0.0, 1.0]]
    expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
    assert cluster_entropy(vectors, k=2) == pytest.approx(expected)


def test_cluster_entropy_is_scale_invariant_and_seeded():
    vectors = [[1.0, 0.0], [5.0, 0.1], [0.0, 2.0], [0.1, 9.0], [1.0, 1.0]]
    assert cluster_entropy(vectors, k=3, seed=1) == cluster_entropy(vectors, k=3, seed=1)
    scaled = [[3 * x for x in v] for v in vectors]
    assert cluster_entropy(vectors, k=3, seed=1) == pytest.approx(
        cluster_entropy(scaled, k=3, seed=1)
    )


def test_cluster_entropy_edge_cases():
    assert cluster_entropy([]) is None
    with pytest.raises(DiversityError):
        cluster_entropy([[1.0]], k=0)


# ── report ───────────────────────────────────────────────────────


def fake_embed(texts):
    # one axis per leading word: texts starting with the same word share a direction
    words = sorted({t.split()[0] for t in texts})
    return [[1.0 if t.split()[0] == w else 0.0 for w in words] for t in texts]


def test_cell_of():
    assert cell_of(rec("c1", "hi")) == "c1"
    assert cell_of(rec(None, "hi")) == UNKNOWN_CELL


def test_report_overall_and_per_cell():
    records = [
        rec("a", "fees question", "fees are ten dollars"),
        rec("a", "fees question", "fees are ten dollars"),
        rec("b", "terminal question", "terminals arrive in five days"),
    ]
    report = diversity_report(records, embed=fake_embed, k=4)
    assert set(report.per_cell) == {"a", "b"}
    assert report.overall.size == 3
    # cell a holds two identical conversations
    assert report.per_cell["a"].self_bleu == pytest.approx(1.0)
    assert report.per_cell["a"].cluster_entropy == 0.0
    assert report.per_cell["b"].self_bleu is None
    assert report.per_cell["b"].size == 1
    assert report.overall.cluster_entropy == pytest.approx(
        -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3))
    )
    assert report.overall.distinct[1] < 1.0
    assert set(report.overall.distinct) == {1, 2}


def test_report_ignores_code_owned_fields():
    a = rec("a", "same words here", "and here")
    b = rec("a", "same words here", "and here")
    b["label"] = True
    b["spans"] = [{"turn": 2, "text": "and here", "category": "X"}]
    assert diversity_report([a, b]).overall.self_bleu == pytest.approx(1.0)


def test_report_without_embedder_has_no_entropy():
    report = diversity_report([rec("a", "one two"), rec("a", "three four")])
    assert report.overall.cluster_entropy is None


def test_report_to_dict_and_determinism():
    records = [rec(f"c{i % 2}", f"text {i} about fees and terminals {i * 3}") for i in range(10)]
    a = diversity_report(records, max_hypotheses=3, seed=5).to_dict()
    b = diversity_report(records, max_hypotheses=3, seed=5).to_dict()
    assert a == b
    assert set(a) == {"overall", "per_cell"}
    assert set(a["overall"]) == {"size", "distinct", "self_bleu", "cluster_entropy"}
    assert set(a["overall"]["distinct"]) == {"1", "2"}


def test_report_empty():
    report = diversity_report([], embed=fake_embed)
    assert report.overall.size == 0
    assert report.overall.self_bleu is None
    assert report.overall.cluster_entropy is None
    assert report.per_cell == {}
