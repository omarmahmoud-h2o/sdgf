"""L4: overlap (FRAMEWORK_DESIGN.md §6.4, §9.2).

Compares a candidate's text with three sources and drops it (fail_hard, never repaired)
when it is too similar to any of them:

    seed       copying a few-shot seed                          issue code seed_overlap
    held_out   contamination of an evaluation set               issue code held_out_overlap
    corpus     a near-duplicate of a record accepted this run   issue code near_duplicate

Similarity is Jaccard over character shingles of normalised text (casefolded, punctuation
and whitespace collapsed), pure Python. An optional embedding engine (cosine over
sentence-transformers vectors, or any embed callable) can run alongside it with its own
thresholds, since cosine and Jaccard don't share a scale.

The held-out check runs only when an explicit held-out path is passed at run time
(load_held_out / from_spec(held_out_paths=...)); the spec has no field for it, so nothing
compiled or cached ever names it. Its documents are reduced to shingle sets or vectors on
load and their text isn't kept, so the layer can't hand held-out content to a prompt.
Issues name the source, the matched item's key, the score and the threshold, never the
matched text.

Records join the corpus only through remember(), which the pipeline calls once a record is
accepted: a candidate that passes L4 and then fails L5 must not block later candidates.

Concurrent generation validates a batch of candidates against the corpus as it stood when
the batch started, so check() must not see records accepted mid-batch. The pipeline
settles the batch in a fixed order instead: check_staged() compares each candidate with
the ones already staged from the same batch, stage() adds it, and commit_staged() moves
the batch into the corpus. The outcome is the same as checking each record against every
earlier one, whatever order the batch's checks ran in.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from sdgf.governance._util import iter_strings, lazy_import
from sdgf.validate.base import Layer, LayerVerdict, Record, ValidationContext, ValidationIssue

if TYPE_CHECKING:
    from sdgf.spec.compile import CompiledSpec

Source = Literal["seed", "held_out", "corpus"]
SOURCES: tuple[Source, ...] = ("seed", "held_out", "corpus")
ISSUE_CODES: dict[Source, str] = {
    "seed": "seed_overlap",
    "held_out": "held_out_overlap",
    "corpus": "near_duplicate",
}

SHINGLE_SIZE = 5
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


class OverlapError(ValueError):
    """The overlap layer or a held-out file is misconfigured."""


def normalise(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", text.casefold())).strip()


def shingles(text: str, k: int = SHINGLE_SIZE) -> frozenset[str]:
    """Character k-shingles of the normalised text; a text shorter than k is one shingle."""
    if k < 1:
        raise OverlapError("shingle size must be at least 1")
    norm = normalise(text)
    if not norm:
        return frozenset()
    if len(norm) <= k:
        return frozenset({norm})
    return frozenset(norm[i : i + k] for i in range(len(norm) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def record_text(record: Mapping[str, Any]) -> str:
    """The text L4 compares: message contents when the record has messages, else every
    public string field. Code-owned labels and facts would otherwise inflate similarity
    between records of the same cell, so messages win when present."""
    messages = record.get("messages")
    if isinstance(messages, list) and messages:
        return "\n".join(
            str(m.get("content", "")) for m in messages if isinstance(m, Mapping)
        ).strip()
    return "\n".join(text for _, text in iter_strings(record)).strip()


# ── similarity indexes ───────────────────────────────────────────


@dataclass(frozen=True)
class Match:
    key: str
    score: float


class SimilarityIndex(ABC):
    """Items are reduced to a comparable form on add(); best() returns the closest one."""

    engine: str

    @abstractmethod
    def add(self, key: str, text: str) -> None: ...

    @abstractmethod
    def best(self, text: str) -> Match | None: ...

    @abstractmethod
    def __len__(self) -> int: ...


class ShingleIndex(SimilarityIndex):
    engine = "shingle"

    def __init__(self, k: int = SHINGLE_SIZE) -> None:
        if k < 1:
            raise OverlapError("shingle size must be at least 1")
        self.k = k
        self._items: list[tuple[str, frozenset[str]]] = []

    def add(self, key: str, text: str) -> None:
        self._items.append((key, shingles(text, self.k)))

    def best(self, text: str) -> Match | None:
        query = shingles(text, self.k)
        if not query:
            return None
        top: Match | None = None
        for key, items in self._items:
            if not items:
                continue
            # Jaccard can't exceed the size ratio, so most items are skipped uncompared.
            bound = min(len(query), len(items)) / max(len(query), len(items))
            if top is not None and bound <= top.score:
                continue
            score = jaccard(query, items)
            if top is None or score > top.score:
                top = Match(key, score)
        return top

    def __len__(self) -> int:
        return len(self._items)


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class EmbeddingIndex(SimilarityIndex):
    """Cosine similarity over vectors from any embed callable (texts -> vectors)."""

    engine = "embedding"

    def __init__(self, embed: Embedder) -> None:
        self.embed = embed
        self._items: list[tuple[str, list[float]]] = []

    def _vector(self, text: str) -> list[float] | None:
        (vector,) = self.embed([text])
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(x * x for x in values))
        return [x / norm for x in values] if norm else None

    def add(self, key: str, text: str) -> None:
        vector = self._vector(text) if normalise(text) else None
        if vector is not None:
            self._items.append((key, vector))

    def best(self, text: str) -> Match | None:
        if not self._items or not normalise(text):
            return None
        query = self._vector(text)
        if query is None:
            return None
        top: Match | None = None
        for key, vector in self._items:
            score = sum(a * b for a, b in zip(query, vector, strict=True))
            if top is None or score > top.score:
                top = Match(key, score)
        return top

    def __len__(self) -> int:
        return len(self._items)


def sentence_transformers_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Optional adapter; sentence-transformers is imported only when this is called."""
    module = lazy_import("sentence_transformers", "sdgf[embeddings]")
    model = module.SentenceTransformer(model_name)

    def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [list(v) for v in model.encode(list(texts))]

    return embed


@dataclass(frozen=True)
class OverlapEngine:
    """One similarity measure and the maximum score it allows per source. A source with no
    threshold isn't checked by this engine."""

    build: Callable[[], SimilarityIndex]
    thresholds: Mapping[Source, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for source, value in self.thresholds.items():
            if source not in SOURCES:
                raise OverlapError(f"unknown overlap source {source!r}; expected {SOURCES}")
            if not 0.0 <= value <= 1.0:
                raise OverlapError(f"{source} threshold must be within [0, 1], got {value}")


# ── held-out loading ─────────────────────────────────────────────


def _flatten_cell(value: str) -> str:
    """A CSV cell may hold a python-literal list of {role, content} dicts (as in
    scripts/check_test_overlap.py); otherwise it's plain text."""
    if value.lstrip().startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, Mapping):
                    for key in ("content", "Content", "text"):
                        if key in item:
                            out.append(str(item[key]))
                            break
                else:
                    out.append(str(item))
            return " ".join(out)
    return value


def load_held_out(
    paths: Iterable[str | Path],
    *,
    columns: Sequence[str] | None = None,
    text_of: Callable[[Mapping[str, Any]], str] = record_text,
) -> list[tuple[str, str]]:
    """Read held-out documents as (key, text), key being "<file name>:<line or row>".

    .jsonl files hold one record per line (text via text_of); .csv files one document per
    row, joining `columns` (all columns if None). Only call this with paths given
    explicitly at run time."""
    docs: list[tuple[str, str]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise OverlapError(f"held-out file not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open(encoding="utf-8") as f:
                for n, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise OverlapError(f"{path.name}:{n}: invalid JSON: {e.msg}") from e
                    if not isinstance(record, Mapping):
                        raise OverlapError(f"{path.name}:{n}: expected a JSON object")
                    docs.append((f"{path.name}:{n}", text_of(record)))
        elif suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                wanted = list(columns) if columns is not None else list(reader.fieldnames or [])
                missing = [c for c in wanted if c not in (reader.fieldnames or [])]
                if missing:
                    raise OverlapError(f"{path.name}: missing columns {missing}")
                for n, row in enumerate(reader, start=1):
                    parts = [_flatten_cell(row[c]) for c in wanted if row.get(c)]
                    if parts:
                        docs.append((f"{path.name}:{n}", " ".join(parts)))
        else:
            raise OverlapError(f"held-out file must be .jsonl or .csv, got {path.name}")
    return docs


# ── the layer ────────────────────────────────────────────────────


class OverlapLayer(Layer):
    name = "L4"

    def __init__(
        self,
        engines: Sequence[OverlapEngine],
        *,
        seeds: Iterable[tuple[str, str]] = (),
        held_out: Iterable[tuple[str, str]] | None = None,
        text_of: Callable[[Mapping[str, Any]], str] = record_text,
    ) -> None:
        if not engines:
            raise OverlapError("L4 needs at least one overlap engine")
        self.engines = tuple(engines)
        self.text_of = text_of
        self.held_out_enabled = held_out is not None
        self._indexes: list[dict[Source, SimilarityIndex]] = [
            {source: engine.build() for source in SOURCES} for engine in self.engines
        ]
        self._corpus_size = 0
        self._staged: list[tuple[str, SimilarityIndex]] = []  # (engine, index) per engine
        self._staged_records: list[Record] = []
        seeds = list(seeds)
        held = list(held_out) if held_out is not None else []
        for indexes in self._indexes:
            for key, text in seeds:
                indexes["seed"].add(key, text)
            for key, text in held:
                indexes["held_out"].add(key, text)

    @classmethod
    def from_spec(
        cls,
        compiled: CompiledSpec,
        *,
        held_out_paths: Iterable[str | Path] | None = None,
        held_out_columns: Sequence[str] | None = None,
        shingle_size: int = SHINGLE_SIZE,
        near_duplicate_max: float | None = None,
        embedder: Embedder | None = None,
        embedding_thresholds: Mapping[Source, float] | None = None,
        text_of: Callable[[Mapping[str, Any]], str] = record_text,
    ) -> OverlapLayer:
        """Seeds from the spec; every source's maximum defaults to thresholds.overlap_max.

        held_out_paths is a run-time argument only; without it the held-out check is off."""
        overlap_max = compiled.spec.thresholds.overlap_max
        if overlap_max is None:
            raise OverlapError("L4 needs thresholds.overlap_max to be set")
        engines = [
            OverlapEngine(
                build=lambda: ShingleIndex(shingle_size),
                thresholds={
                    "seed": overlap_max,
                    "held_out": overlap_max,
                    "corpus": overlap_max if near_duplicate_max is None else near_duplicate_max,
                },
            )
        ]
        if embedder is not None:
            if not embedding_thresholds:
                raise OverlapError("an embedding engine needs its own thresholds")
            engines.append(
                OverlapEngine(
                    build=lambda: EmbeddingIndex(embedder), thresholds=embedding_thresholds
                )
            )
        seeds = [(f"seed:{i}", text_of(s)) for i, s in enumerate(compiled.seeds)]
        held_out = (
            load_held_out(held_out_paths, columns=held_out_columns, text_of=text_of)
            if held_out_paths is not None
            else None
        )
        return cls(engines, seeds=seeds, held_out=held_out, text_of=text_of)

    @property
    def corpus_size(self) -> int:
        return self._corpus_size

    def remember(self, record: Record, key: str | None = None) -> None:
        """Add an accepted record to the near-duplicate corpus."""
        key = key if key is not None else f"corpus:{self._corpus_size}"
        text = self.text_of(record)
        for indexes in self._indexes:
            indexes["corpus"].add(key, text)
        self._corpus_size += 1

    def scores(self, record: Record) -> dict[tuple[str, Source], Match]:
        """The closest item per (engine, source), for the §8 overlap metric."""
        text = self.text_of(record)
        out: dict[tuple[str, Source], Match] = {}
        for indexes in self._indexes:
            for source, index in indexes.items():
                if len(index) and (match := index.best(text)) is not None:
                    out[(index.engine, source)] = match
        return out

    # ── staging (a batch settled in order) ───────────────────────

    def stage(self, record: Record) -> None:
        """Hold an accepted record for commit_staged(); only check_staged() sees it."""
        if not self._staged:
            self._staged = [(e, e.build()) for e in self.engines]
        key = f"corpus:{self._corpus_size + len(self._staged_records)}"
        text = self.text_of(record)
        for _, index in self._staged:
            index.add(key, text)
        self._staged_records.append(record)

    def check_staged(self, record: Record) -> LayerVerdict:
        """Near-duplicate check against the staged records only."""
        text = self.text_of(record)
        issues: list[ValidationIssue] = []
        for engine, index in self._staged:
            issue = self._issue(engine, "corpus", index, text)
            if issue is not None:
                issues.append(issue)
        return self.verdict(issues, repairable=False)

    def commit_staged(self) -> None:
        for record in self._staged_records:
            self.remember(record)
        self._staged, self._staged_records = [], []

    # ── check ────────────────────────────────────────────────────

    def _issue(
        self, engine: OverlapEngine, source: Source, index: SimilarityIndex, text: str
    ) -> ValidationIssue | None:
        limit = engine.thresholds.get(source)
        if limit is None or not len(index):
            return None
        match = index.best(text)
        if match is None or match.score <= limit:
            return None
        return ValidationIssue(
            code=ISSUE_CODES[source],
            message=(
                f"{index.engine} similarity {match.score:.3f} to {source} item "
                f"{match.key!r} exceeds {limit:.3f}; overlap failures are dropped, "
                "not repaired"
            ),
            details={
                "source": source,
                "engine": index.engine,
                "match": match.key,
                "score": round(match.score, 4),
                "threshold": limit,
            },
        )

    def check(self, record: Record, context: ValidationContext) -> LayerVerdict:
        text = self.text_of(record)
        issues: list[ValidationIssue] = []
        for engine, indexes in zip(self.engines, self._indexes, strict=True):
            for source in SOURCES:
                issue = self._issue(engine, source, indexes[source], text)
                if issue is not None:
                    issues.append(issue)
        return self.verdict(issues, repairable=False)
