"""Retrieval-augmented keyword extraction (FRAMEWORK_DESIGN.md §6.2 step 1).

Ported from DS²-Instruct's bm25_retriever.py and keyword_expansion_from_corpus.py:
each iteration samples the current keywords, builds a query from the task description
plus the sample, retrieves the top passages with BM25 and asks the expansion model to
extract domain terms missing from the current list.

Fixes from §12.1:
- the BM25 index is built once over the corpus and persisted to the artefact store
  (keyed by a hash of the corpus and BM25 parameters), instead of being rebuilt over
  the whole corpus on every retrieval call;
- corpus paths are explicit and resolved against the task directory, not the working
  directory;
- the extraction prompt carries the real current keyword list (keywords.extraction_prompt),
  and an unparseable reply raises KeywordParseError.

BM25 is pure Python, scoring as rank_bm25.BM25Okapi (k1 1.5, b 0.75, epsilon 0.25 floor
for negative idf), so the optional `retrieval` extra isn't needed.

Settings come from coverage.params["retrieval"]; the backend is models.expansion.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sdgf.coverage.keywords import (
    ExpansionSettings,
    KeywordError,
    KeywordParseError,
    extraction_prompt,
    parse_keywords,
)
from sdgf.models.base import ModelBackend
from sdgf.spec.compile import CompiledSpec
from sdgf.store.artefacts import ArtefactStore

PARAMS_KEY = "retrieval"
INDEX_FORMAT = 1
TEXT_FIELDS = ("text", "content", "webpage")


class RetrievalError(KeywordError):
    """The retrieval corpus, index or settings are unusable."""


def tokenize(text: str) -> list[str]:
    """Port of bm25_retriever.tokenize: lower-case, punctuation to spaces, split."""
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


# ── corpus ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class Document:
    id: str
    text: str


def _doc_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Mapping):
        for name in TEXT_FIELDS:
            value = obj.get(name)
            if isinstance(value, str) and value:
                return value
    return ""


def load_documents(paths: Iterable[str | Path], min_chars: int = 50) -> list[Document]:
    """Read .jsonl, .json (a list) and .txt files; documents shorter than min_chars skipped.

    A document's id is <file name>:<line or item number>, so hits can be traced to source.
    """
    docs: list[Document] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise RetrievalError(f"retrieval corpus file not found: {path}")
        items: list[tuple[int, Any]]
        if path.suffix == ".jsonl":
            items = []
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    items.append((lineno, json.loads(line)))
                except json.JSONDecodeError as e:
                    raise RetrievalError(f"{path}:{lineno}: corrupt JSON: {e}") from None
        elif path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                raise RetrievalError(f"{path}: corrupt JSON: {e}") from None
            if not isinstance(data, list):
                raise RetrievalError(f"{path}: a .json corpus must be a list of documents")
            items = list(enumerate(data, 1))
        elif path.suffix == ".txt":
            items = [(1, path.read_text(encoding="utf-8"))]
        else:
            raise RetrievalError(f"{path}: unsupported corpus file type (use .jsonl, .json, .txt)")
        for n, item in items:
            text = _doc_text(item).strip()
            if len(text) >= min_chars:
                docs.append(Document(f"{path.name}:{n}", text))
    return docs


def compute_corpus_hash(documents: Sequence[Document], k1: float, b: float, epsilon: float) -> str:
    h = hashlib.sha256()
    h.update(json.dumps({"format": INDEX_FORMAT, "k1": k1, "b": b, "epsilon": epsilon}).encode())
    for doc in documents:
        for part in (doc.id, doc.text):
            data = part.encode("utf-8")
            h.update(str(len(data)).encode("ascii") + b":" + data)
    return h.hexdigest()


# ── BM25 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Hit:
    doc_id: str
    text: str
    score: float


class BM25Index:
    """An inverted BM25 index. Build with from_documents; persist with to_dict/from_dict."""

    def __init__(
        self,
        documents: Sequence[Document],
        postings: Mapping[str, Sequence[tuple[int, int]]],
        doc_len: Sequence[int],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        corpus_hash: str | None = None,
    ) -> None:
        if not documents:
            raise RetrievalError("cannot build a BM25 index over an empty corpus")
        if len(doc_len) != len(documents):
            raise RetrievalError("doc_len does not match the documents")
        self.documents = tuple(documents)
        self.postings = {t: tuple((int(d), int(f)) for d, f in p) for t, p in postings.items()}
        self.doc_len = tuple(int(n) for n in doc_len)
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_hash = corpus_hash or compute_corpus_hash(self.documents, k1, b, epsilon)
        n = len(self.documents)
        self.avgdl = sum(self.doc_len) / n
        # rank_bm25.BM25Okapi idf: negative values are floored at epsilon * mean idf.
        idf = {
            t: math.log(n - len(p) + 0.5) - math.log(len(p) + 0.5) for t, p in self.postings.items()
        }
        floor = epsilon * (sum(idf.values()) / len(idf)) if idf else 0.0
        self.idf = {t: (v if v >= 0 else floor) for t, v in idf.items()}

    @classmethod
    def from_documents(
        cls,
        documents: Sequence[Document],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> BM25Index:
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_len: list[int] = []
        for i, doc in enumerate(documents):
            tokens = tokenize(doc.text)
            doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                postings.setdefault(term, []).append((i, tf))
        return cls(documents, postings, doc_len, k1=k1, b=b, epsilon=epsilon)

    def __len__(self) -> int:
        return len(self.documents)

    def scores(self, query: str) -> list[float]:
        """A score per document; repeated query terms count again, as in rank_bm25."""
        out = [0.0] * len(self.documents)
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for term in tokenize(query):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for d, tf in self.postings[term]:
                norm = k1 * (1 - b + b * self.doc_len[d] / avgdl)
                out[d] += idf * tf * (k1 + 1) / (tf + norm)
        return out

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        """Top documents by score, ties in corpus order; documents sharing no term excluded."""
        if top_k < 1:
            raise RetrievalError("top_k must be at least 1")
        terms = set(tokenize(query))
        matched = {d for t in terms for d, _ in self.postings.get(t, ())}
        scores = self.scores(query)
        ranked = sorted(matched, key=lambda d: (-scores[d], d))[:top_k]
        return [Hit(self.documents[d].id, self.documents[d].text, scores[d]) for d in ranked]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": INDEX_FORMAT,
            "corpus_hash": self.corpus_hash,
            "params": {"k1": self.k1, "b": self.b, "epsilon": self.epsilon},
            "documents": [{"id": d.id, "text": d.text} for d in self.documents],
            "doc_len": list(self.doc_len),
            "postings": {t: [list(p) for p in ps] for t, ps in sorted(self.postings.items())},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BM25Index:
        if data.get("format") != INDEX_FORMAT:
            raise RetrievalError(f"unsupported BM25 index format {data.get('format')!r}")
        try:
            docs = [Document(d["id"], d["text"]) for d in data["documents"]]
            params = data["params"]
            return cls(
                docs,
                {t: [tuple(p) for p in ps] for t, ps in data["postings"].items()},
                data["doc_len"],
                k1=params["k1"],
                b=params["b"],
                epsilon=params["epsilon"],
                corpus_hash=data["corpus_hash"],
            )
        except (KeyError, TypeError, ValueError) as e:
            raise RetrievalError(f"corrupt BM25 index: {e!r}") from None


def index_stage_name(hash_: str) -> str:
    return f"bm25-{hash_[:16]}"


def load_or_build_index(
    store: ArtefactStore,
    spec_version: str,
    documents: Sequence[Document],
    *,
    k1: float = 1.5,
    b: float = 0.75,
    epsilon: float = 0.25,
) -> tuple[BM25Index, bool]:
    """The persisted index for this corpus, building and saving it only if absent.

    Returns (index, built). Stored in the spec_version's shared area, so every run and
    every retrieval call reuses one build; a changed corpus or parameter gets a new file.
    """
    hash_ = compute_corpus_hash(documents, k1, b, epsilon)
    stage = index_stage_name(hash_)
    if store.has_shared(spec_version, stage):
        index = BM25Index.from_dict(store.read_shared(spec_version, stage))
        if index.corpus_hash != hash_:
            raise RetrievalError(f"stored index {stage} does not match the corpus")
        return index, False
    index = BM25Index.from_documents(documents, k1=k1, b=b, epsilon=epsilon)
    store.write_shared(spec_version, stage, index.to_dict())
    return index, True


# ── settings ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievalSettings:
    corpus: tuple[str, ...] = ()  # paths relative to the task directory
    min_chars: int = 50
    iterations: int = 1
    sample_size: int = 10  # keywords added to the query
    top_k: int = 5  # passages retrieved per iteration
    passage_chars: int = 3000
    max_keywords: int | None = None
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        for name in ("sample_size", "top_k", "passage_chars"):
            if getattr(self, name) < 1:
                raise RetrievalError(f"{PARAMS_KEY}.{name} must be at least 1")
        for name in ("iterations", "min_chars"):
            if getattr(self, name) < 0:
                raise RetrievalError(f"{PARAMS_KEY}.{name} must not be negative")
        if self.max_keywords is not None and self.max_keywords < 1:
            raise RetrievalError(f"{PARAMS_KEY}.max_keywords must be at least 1")
        if self.k1 < 0 or not 0 <= self.b <= 1:
            raise RetrievalError(f"{PARAMS_KEY}: need k1 >= 0 and 0 <= b <= 1")

    @classmethod
    def from_params(cls, params: Mapping[str, Any] | None) -> RetrievalSettings:
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(params) - known)
        if unknown:
            raise RetrievalError(f"unknown {PARAMS_KEY} settings: {unknown}")
        if "corpus" in params:
            corpus = params["corpus"]
            params["corpus"] = (corpus,) if isinstance(corpus, str) else tuple(corpus)
        return cls(**params)

    def corpus_paths(self, base: Path) -> list[Path]:
        return [p if p.is_absolute() else base / p for p in map(Path, self.corpus)]


def build_query(task_description: str, sample: Sequence[str]) -> str:
    """Port of bm25_retriever.build_query; underscores become spaces so terms tokenise."""
    return " ".join([task_description.strip(), *(kw.replace("_", " ") for kw in sample)])


# ── extraction ───────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    keywords: list[str]
    added: list[str]
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "keywords": list(self.keywords),
            "added": list(self.added),
            "history": [dict(h) for h in self.history],
        }


class RetrievalExtractor:
    def __init__(
        self,
        backend: ModelBackend,
        index: BM25Index,
        task_description: str,
        settings: RetrievalSettings | None = None,
        *,
        prompt_settings: ExpansionSettings | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> None:
        if not task_description.strip():
            raise RetrievalError("retrieval needs a task description")
        self.backend = backend
        self.index = index
        self.task_description = task_description
        self.settings = settings or RetrievalSettings()
        self.prompt_settings = prompt_settings or ExpansionSettings()
        self.max_tokens = max_tokens
        self.temperature = temperature

    @classmethod
    def from_spec(
        cls,
        compiled: CompiledSpec,
        backend: ModelBackend,
        store: ArtefactStore,
        **kwargs: Any,
    ) -> RetrievalExtractor:
        """Load the corpus named in coverage.params.retrieval and its persisted index."""
        params = compiled.spec.coverage.params
        settings = RetrievalSettings.from_params(params.get(PARAMS_KEY))
        if not settings.corpus:
            raise RetrievalError(f"coverage.params.{PARAMS_KEY}.corpus names no files")
        docs = load_documents(settings.corpus_paths(compiled.task_dir), settings.min_chars)
        index, _ = load_or_build_index(
            store,
            compiled.spec_version,
            docs,
            k1=settings.k1,
            b=settings.b,
            epsilon=settings.epsilon,
        )
        config = compiled.spec.models.expansion
        if config is not None:
            kwargs.setdefault("max_tokens", config.max_tokens)
            kwargs.setdefault("temperature", config.temperature)
        kwargs.setdefault(
            "prompt_settings", ExpansionSettings.from_params(params.get("keyword_expansion"))
        )
        return cls(backend, index, compiled.spec.task.description, settings, **kwargs)

    def _room(self, current: Sequence[str]) -> int | None:
        cap = self.settings.max_keywords
        return None if cap is None else max(cap - len(current), 0)

    def extract(self, passage: str, current: Sequence[str]) -> list[str]:
        prompt = extraction_prompt(
            self.task_description,
            passage[: self.settings.passage_chars],
            list(current),
            self.prompt_settings,
        )
        response = self.backend.call(prompt, self.max_tokens, self.temperature)
        keywords = parse_keywords(response.text)
        if not keywords:
            raise KeywordParseError("extraction", response.text)
        return keywords

    def run_iteration(
        self, iteration: int, current: list[str], rng: random.Random
    ) -> dict[str, Any]:
        """Retrieve for a sample of `current` and extend it in place with new terms."""
        s = self.settings
        known = set(current)
        sample = rng.sample(current, min(s.sample_size, len(current)))
        hits = self.index.search(build_query(self.task_description, sample), s.top_k)
        added: list[str] = []
        passages: list[dict[str, Any]] = []
        for hit in hits:
            if self._room(current) == 0:
                break
            new = []
            for kw in self.extract(hit.text, current):
                if self._room(current) == 0:
                    break
                if kw not in known:
                    known.add(kw)
                    current.append(kw)
                    new.append(kw)
            added.extend(new)
            passages.append({"doc_id": hit.doc_id, "score": hit.score, "added": new})
        return {
            "iteration": iteration,
            "sample": sample,
            "passages": passages,
            "added": added,
            "ending_count": len(current),
        }

    def run(self, keywords: Sequence[str], rng: random.Random) -> RetrievalResult:
        current = list(dict.fromkeys(keywords))
        start = set(current)
        result = RetrievalResult(keywords=current, added=[])
        for i in range(1, self.settings.iterations + 1):
            if self._room(current) == 0:
                break
            result.history.append(self.run_iteration(i, current, rng))
        result.added = [kw for kw in current if kw not in start]
        return result
