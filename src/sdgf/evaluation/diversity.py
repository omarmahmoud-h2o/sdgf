"""Diversity metrics (FRAMEWORK_DESIGN.md §8): lexical and semantic, overall and per cell.

    distinct-n        unique n-grams / total n-grams over the whole set (higher is more
                      diverse; Li et al. 2016)
    self-BLEU         mean sentence BLEU of each text against all the others as references
                      (lower is more diverse; Zhu et al. 2018)
    cluster entropy   Shannon entropy, in nats, of k-means cluster sizes over normalised
                      embeddings (higher is more diverse; at most ln k)

distinct-n and self-BLEU are pure Python. Cluster entropy needs an embed callable
(texts -> vectors), e.g. validate.l4_overlap.sentence_transformers_embedder, and is only
computed when one is passed; k-means is numpy with an explicit seed.

Texts are compared by validate.l4_overlap.record_text, so code-owned labels and facts
don't make records of one cell look alike. §16 Q2 leaves the final metric pair open.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sdgf.store.provenance import PROVENANCE_KEY
from sdgf.validate.l4_overlap import record_text

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]

DEFAULT_NS: tuple[int, ...] = (1, 2)
BLEU_MAX_N = 4
SMOOTHING_EPSILON = 0.1
UNKNOWN_CELL = "_unknown"

_TOKEN = re.compile(r"\w+")


class DiversityError(ValueError):
    """A diversity metric was asked for with invalid parameters."""


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    if n < 1:
        raise DiversityError("n-gram order must be at least 1")
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


# ── distinct-n ───────────────────────────────────────────────────


def distinct_n(texts: Iterable[str], n: int) -> float | None:
    """Unique n-grams over total n-grams across all texts; None when there are none."""
    seen: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        grams = ngrams(tokenize(text), n)
        total += len(grams)
        seen.update(grams)
    return len(seen) / total if total else None


# ── self-BLEU ────────────────────────────────────────────────────


def _counts(tokens: Sequence[str], max_n: int) -> list[Counter[tuple[str, ...]]]:
    return [Counter(ngrams(tokens, n)) for n in range(1, max_n + 1)]


def _sentence_bleu(
    hyp: list[Counter[tuple[str, ...]]],
    hyp_len: int,
    refs: Sequence[list[Counter[tuple[str, ...]]]],
    ref_lens: Sequence[int],
    smooth: bool,
) -> float:
    log_precisions: list[float] = []
    for order, hyp_counts in enumerate(hyp):
        total = sum(hyp_counts.values())
        if not total:  # hypothesis shorter than this order: use the orders it has
            continue
        max_ref: Counter[tuple[str, ...]] = Counter()
        for ref in refs:
            max_ref |= ref[order]
        clipped = sum(min(c, max_ref[g]) for g, c in hyp_counts.items())
        if not clipped:
            if not smooth:
                return 0.0
            clipped = SMOOTHING_EPSILON  # Chen & Cherry 2014, method 1
        log_precisions.append(math.log(clipped / total))
    if not log_precisions:
        return 0.0
    closest = min(ref_lens, key=lambda r: (abs(r - hyp_len), r))
    bp = 1.0 if hyp_len > closest else math.exp(1 - closest / hyp_len)
    return bp * math.exp(sum(log_precisions) / len(log_precisions))


def self_bleu(
    texts: Sequence[str],
    *,
    max_n: int = BLEU_MAX_N,
    smooth: bool = True,
    max_hypotheses: int | None = None,
    rng: random.Random | None = None,
) -> float | None:
    """Mean BLEU of each text against every other text; None with fewer than two non-empty
    texts. It is quadratic in the set size, so max_hypotheses caps how many texts are
    scored (every text stays a reference), drawn with the given rng."""
    if max_n < 1:
        raise DiversityError("max_n must be at least 1")
    tokenized = [t for t in (tokenize(text) for text in texts) if t]
    if len(tokenized) < 2:
        return None
    counts = [_counts(t, max_n) for t in tokenized]
    lens = [len(t) for t in tokenized]
    indices = list(range(len(tokenized)))
    if max_hypotheses is not None and max_hypotheses < len(indices):
        if max_hypotheses < 1:
            raise DiversityError("max_hypotheses must be at least 1")
        if rng is None:
            raise DiversityError("sampling hypotheses needs an explicit random.Random")
        indices = sorted(rng.sample(indices, max_hypotheses))
    scores = [
        _sentence_bleu(
            counts[i],
            lens[i],
            [c for j, c in enumerate(counts) if j != i],
            [n for j, n in enumerate(lens) if j != i],
            smooth,
        )
        for i in indices
    ]
    return sum(scores) / len(scores)


# ── cluster entropy (optional embeddings) ────────────────────────


def _kmeans(x: np.ndarray, k: int, seed: int, iterations: int) -> np.ndarray:
    """k-means++ init then Lloyd iterations; returns a cluster index per row."""
    gen = np.random.default_rng(seed)
    centres = [x[gen.integers(len(x))]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centres], axis=0)
        total = d2.sum()
        if total <= 0:
            break
        centres.append(x[gen.choice(len(x), p=d2 / total)])
    c = np.array(centres)
    assign: np.ndarray | None = None
    for _ in range(iterations):
        new = ((x[:, None, :] - c[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
        if assign is not None and np.array_equal(new, assign):
            break
        assign = new
        c = np.array(
            [x[assign == j].mean(axis=0) if (assign == j).any() else c[j] for j in range(len(c))]
        )
    return assign if assign is not None else np.zeros(len(x), dtype=int)


def cluster_entropy(
    vectors: Sequence[Sequence[float]], *, k: int = 8, seed: int = 0, iterations: int = 50
) -> float | None:
    """Entropy (nats) of k-means cluster sizes over L2-normalised vectors; None when empty.
    k is capped at the number of distinct vectors, so identical texts score 0."""
    if k < 1:
        raise DiversityError("k must be at least 1")
    if not len(vectors):
        return None
    x = np.asarray(vectors, dtype=float)
    if x.ndim != 2:
        raise DiversityError("vectors must be a 2-D array of equal-length rows")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)
    k = min(k, len(np.unique(x.round(12), axis=0)))
    assign = _kmeans(x, k, seed, iterations)
    p = np.bincount(assign) / len(assign)
    p = p[p > 0]
    return max(0.0, float(-(p * np.log(p)).sum()))


# ── report ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiversityScores:
    size: int
    distinct: dict[int, float | None]
    self_bleu: float | None
    cluster_entropy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "distinct": {str(n): v for n, v in self.distinct.items()},
            "self_bleu": self.self_bleu,
            "cluster_entropy": self.cluster_entropy,
        }


@dataclass(frozen=True)
class DiversityReport:
    overall: DiversityScores
    per_cell: dict[str, DiversityScores] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "per_cell": {cell: s.to_dict() for cell, s in self.per_cell.items()},
        }


def cell_of(record: Mapping[str, Any]) -> str:
    provenance = record.get(PROVENANCE_KEY)
    if isinstance(provenance, Mapping) and provenance.get("cell_id"):
        return str(provenance["cell_id"])
    return UNKNOWN_CELL


def score_texts(
    texts: Sequence[str],
    *,
    ns: Sequence[int] = DEFAULT_NS,
    bleu_max_n: int = BLEU_MAX_N,
    max_hypotheses: int | None = None,
    rng: random.Random | None = None,
    vectors: Sequence[Sequence[float]] | None = None,
    k: int = 8,
    seed: int = 0,
) -> DiversityScores:
    return DiversityScores(
        size=len(texts),
        distinct={n: distinct_n(texts, n) for n in ns},
        self_bleu=self_bleu(texts, max_n=bleu_max_n, max_hypotheses=max_hypotheses, rng=rng),
        cluster_entropy=None if vectors is None else cluster_entropy(vectors, k=k, seed=seed),
    )


def diversity_report(
    records: Sequence[Mapping[str, Any]],
    *,
    ns: Sequence[int] = DEFAULT_NS,
    bleu_max_n: int = BLEU_MAX_N,
    max_hypotheses: int | None = None,
    seed: int = 0,
    embed: Embedder | None = None,
    k: int = 8,
    cell: Callable[[Mapping[str, Any]], str] = cell_of,
) -> DiversityReport:
    """Diversity of accepted records overall and per cell (from provenance cell_id).
    Texts are embedded once when embed is given; seed drives every random draw."""
    texts = [record_text(r) for r in records]
    vectors = [list(v) for v in embed(texts)] if embed is not None and texts else None
    groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        groups[cell(r)].append(i)

    def scores(idx: Sequence[int], label: str) -> DiversityScores:
        return score_texts(
            [texts[i] for i in idx],
            ns=ns,
            bleu_max_n=bleu_max_n,
            max_hypotheses=max_hypotheses,
            rng=random.Random(f"{seed}:{label}"),
            vectors=None if vectors is None else [vectors[i] for i in idx],
            k=k,
            seed=seed,
        )

    return DiversityReport(
        overall=scores(range(len(texts)), "overall"),
        per_cell={c: scores(idx, c) for c, idx in sorted(groups.items())},
    )
