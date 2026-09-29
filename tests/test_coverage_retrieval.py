import json
import math
import random

import pytest

from sdgf.coverage import retrieval
from sdgf.coverage.keywords import KeywordParseError
from sdgf.coverage.retrieval import (
    BM25Index,
    Document,
    RetrievalError,
    RetrievalExtractor,
    RetrievalSettings,
    build_query,
    compute_corpus_hash,
    load_documents,
    load_or_build_index,
    tokenize,
)
from sdgf.models.mock import MockBackend
from sdgf.spec.compile import compile_spec
from sdgf.store.artefacts import ArtefactStore

SV = "a" * 64
DESCRIPTION = "Write exam questions about fictional small-business finance."

CORPUS = [
    Document("c:1", "Cash flow forecasting helps a small business plan its payroll."),
    Document("c:2", "An overdraft is a short-term credit facility linked to an account."),
    Document("c:3", "Invoice factoring converts unpaid invoices into cash flow."),
    Document("c:4", "Depreciation spreads the cost of equipment over its useful life."),
    Document("c:5", "Working capital is current assets minus current liabilities."),
]


def naive_bm25(docs, query, k1=1.5, b=0.75, epsilon=0.25):
    """rank_bm25.BM25Okapi.get_scores, written out directly."""
    toks = [tokenize(d.text) for d in docs]
    n = len(toks)
    avgdl = sum(map(len, toks)) / n
    df = {}
    for t in toks:
        for w in set(t):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log(n - f + 0.5) - math.log(f + 0.5) for w, f in df.items()}
    eps = epsilon * sum(idf.values()) / len(idf)
    idf = {w: (v if v >= 0 else eps) for w, v in idf.items()}
    out = []
    for t in toks:
        s = 0.0
        for q in tokenize(query):
            tf = t.count(q)
            s += idf.get(q, 0.0) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(t) / avgdl))
        out.append(s)
    return out


# ── tokenising and corpus loading ────────────────────────────────


def test_tokenize_matches_original():
    assert tokenize("Cash-flow, GST & BAS!") == ["cash", "flow", "gst", "bas"]


def test_load_documents_formats_and_min_chars(tmp_path):
    long = "x" * 60
    (tmp_path / "a.jsonl").write_text(
        json.dumps({"text": long})
        + "\n\n"
        + json.dumps({"content": "short"})
        + "\n"
        + json.dumps({"webpage": long + "w"})
        + "\n"
    )
    (tmp_path / "b.json").write_text(json.dumps([{"content": long + "c"}, long + "s"]))
    (tmp_path / "c.txt").write_text(long + "t")
    docs = load_documents([tmp_path / "a.jsonl", tmp_path / "b.json", tmp_path / "c.txt"])
    assert [d.id for d in docs] == ["a.jsonl:1", "a.jsonl:4", "b.json:1", "b.json:2", "c.txt:1"]
    assert docs[1].text.endswith("w")


@pytest.mark.parametrize(
    "name, content, match",
    [
        ("bad.jsonl", "{not json\n", "bad.jsonl:1"),
        ("bad.json", '{"text": "x"}', "must be a list"),
        ("bad.csv", "a,b\n", "unsupported"),
    ],
)
def test_load_documents_errors(tmp_path, name, content, match):
    (tmp_path / name).write_text(content)
    with pytest.raises(RetrievalError, match=match):
        load_documents([tmp_path / name])


def test_load_documents_missing_file(tmp_path):
    with pytest.raises(RetrievalError, match="not found"):
        load_documents([tmp_path / "nope.jsonl"])


# ── BM25 ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query", ["cash flow", "overdraft credit account", "invoice invoice cash", "unknownterm"]
)
def test_scores_match_bm25okapi(query):
    index = BM25Index.from_documents(CORPUS)
    assert index.scores(query) == pytest.approx(naive_bm25(CORPUS, query))


def test_scores_match_rank_bm25_when_installed():
    rank_bm25 = pytest.importorskip("rank_bm25")
    ref = rank_bm25.BM25Okapi([tokenize(d.text) for d in CORPUS])
    index = BM25Index.from_documents(CORPUS)
    for q in ("cash flow", "current assets"):
        assert index.scores(q) == pytest.approx(list(ref.get_scores(tokenize(q))))


def test_search_ranks_relevant_documents():
    index = BM25Index.from_documents(CORPUS)
    hits = index.search("cash flow invoices", top_k=2)
    assert [h.doc_id for h in hits] == ["c:3", "c:1"]
    assert hits[0].score > hits[1].score > 0
    assert hits[0].text == CORPUS[2].text


def test_search_excludes_unmatched_and_breaks_ties_in_corpus_order():
    docs = [Document("x", "alpha beta"), Document("y", "gamma delta"), Document("z", "alpha beta")]
    index = BM25Index.from_documents(docs)
    assert [h.doc_id for h in index.search("alpha", top_k=10)] == ["x", "z"]
    assert index.search("nothing here") == []


def test_search_top_k_must_be_positive():
    with pytest.raises(RetrievalError, match="top_k"):
        BM25Index.from_documents(CORPUS).search("cash", top_k=0)


def test_empty_corpus_rejected():
    with pytest.raises(RetrievalError, match="empty corpus"):
        BM25Index.from_documents([])


def test_round_trip_preserves_scores():
    index = BM25Index.from_documents(CORPUS)
    data = json.loads(json.dumps(index.to_dict()))
    loaded = BM25Index.from_dict(data)
    assert loaded.corpus_hash == index.corpus_hash
    assert loaded.scores("cash flow payroll") == index.scores("cash flow payroll")
    assert loaded.search("overdraft") == index.search("overdraft")


def test_from_dict_rejects_bad_data():
    data = BM25Index.from_documents(CORPUS).to_dict()
    with pytest.raises(RetrievalError, match="format"):
        BM25Index.from_dict({**data, "format": 99})
    del data["postings"]
    with pytest.raises(RetrievalError, match="corrupt"):
        BM25Index.from_dict(data)


def test_corpus_hash_depends_on_text_ids_and_params():
    base = compute_corpus_hash(CORPUS, 1.5, 0.75, 0.25)
    edited = [Document(CORPUS[0].id, CORPUS[0].text + "!"), *CORPUS[1:]]
    renamed = [Document("other", CORPUS[0].text), *CORPUS[1:]]
    assert compute_corpus_hash(edited, 1.5, 0.75, 0.25) != base
    assert compute_corpus_hash(renamed, 1.5, 0.75, 0.25) != base
    assert compute_corpus_hash(CORPUS, 1.2, 0.75, 0.25) != base
    assert compute_corpus_hash(list(CORPUS), 1.5, 0.75, 0.25) == base


# ── persistence (§12.1: build once) ──────────────────────────────


def test_index_built_once_and_persisted(tmp_path, monkeypatch):
    store = ArtefactStore(tmp_path)
    index, built = load_or_build_index(store, SV, CORPUS)
    assert built
    stage = retrieval.index_stage_name(index.corpus_hash)
    assert store.has_shared(SV, stage)

    calls = []
    monkeypatch.setattr(
        BM25Index, "from_documents", classmethod(lambda cls, *a, **k: calls.append(1))
    )
    again, built_again = load_or_build_index(store, SV, CORPUS)
    assert not built_again and calls == []
    assert again.search("cash flow") == index.search("cash flow")


def test_changed_corpus_builds_a_new_index(tmp_path):
    store = ArtefactStore(tmp_path)
    first, _ = load_or_build_index(store, SV, CORPUS)
    second, built = load_or_build_index(store, SV, CORPUS[:3])
    assert built and second.corpus_hash != first.corpus_hash
    assert len(second) == 3


def test_stored_index_mismatch_raises(tmp_path):
    store = ArtefactStore(tmp_path)
    index, _ = load_or_build_index(store, SV, CORPUS)
    stage = retrieval.index_stage_name(index.corpus_hash)
    store.write_shared(SV, stage, {**index.to_dict(), "corpus_hash": "b" * 64})
    with pytest.raises(RetrievalError, match="does not match"):
        load_or_build_index(store, SV, CORPUS)


def test_many_searches_do_not_rebuild(monkeypatch):
    index = BM25Index.from_documents(CORPUS)
    calls = []
    real = retrieval.tokenize
    monkeypatch.setattr(retrieval, "tokenize", lambda t: calls.append(t) or real(t))
    for _ in range(5):
        index.search("cash flow")
    assert calls == ["cash flow"] * 10  # the query only (search + scores), never documents


# ── settings and query ───────────────────────────────────────────


def test_settings_from_params():
    s = RetrievalSettings.from_params({"corpus": "docs.jsonl", "top_k": 3})
    assert s.corpus == ("docs.jsonl",) and s.top_k == 3
    with pytest.raises(RetrievalError, match="unknown"):
        RetrievalSettings.from_params({"topk": 3})
    with pytest.raises(RetrievalError, match="top_k"):
        RetrievalSettings.from_params({"top_k": 0})
    with pytest.raises(RetrievalError, match="b <= 1"):
        RetrievalSettings.from_params({"b": 2})


def test_corpus_paths_resolve_against_task_dir(tmp_path):
    s = RetrievalSettings(corpus=("docs.jsonl", str(tmp_path / "abs.jsonl")))
    assert s.corpus_paths(tmp_path / "task") == [
        tmp_path / "task" / "docs.jsonl",
        tmp_path / "abs.jsonl",
    ]


def test_build_query():
    assert build_query(" Finance. ", ["cash_flow", "gst"]) == "Finance. cash flow gst"


# ── extraction ───────────────────────────────────────────────────


def test_extraction_passes_current_keywords_and_adds_new_ones():
    backend = MockBackend(["invoice_factoring, cash_flow", "payroll, forecasting"])
    index = BM25Index.from_documents(CORPUS)
    settings = RetrievalSettings(top_k=2, sample_size=1)
    extractor = RetrievalExtractor(backend, index, DESCRIPTION, settings)
    result = extractor.run(["cash_flow"], random.Random(0))

    assert result.keywords == ["cash_flow", "invoice_factoring", "payroll", "forecasting"]
    assert result.added == ["invoice_factoring", "payroll", "forecasting"]
    first, second = (c.prompt for c in backend.calls)
    assert "Current Keywords: cash_flow\n" in first
    assert "Current Keywords: cash_flow, invoice_factoring\n" in second  # §12.1: real list
    # the task description is part of the query, as in the original, so "small business"
    # ranks c:1 first
    assert CORPUS[0].text in first and CORPUS[2].text in second
    [it] = result.history
    assert it["sample"] == ["cash_flow"]
    assert [p["doc_id"] for p in it["passages"]] == ["c:1", "c:3"]
    assert it["passages"][0]["added"] == ["invoice_factoring"]


def test_passage_truncated_to_passage_chars():
    backend = MockBackend(["cash"])
    index = BM25Index.from_documents(CORPUS)
    settings = RetrievalSettings(top_k=1, passage_chars=10)
    RetrievalExtractor(backend, index, DESCRIPTION, settings).run(["overdraft"], random.Random(0))
    assert "\nCash flow \n" in backend.calls[0].prompt
    assert CORPUS[0].text not in backend.calls[0].prompt


def test_unparseable_reply_raises_instead_of_mock_keywords():
    backend = MockBackend(["  ,  "])
    extractor = RetrievalExtractor(backend, BM25Index.from_documents(CORPUS), DESCRIPTION)
    with pytest.raises(KeywordParseError, match="extraction"):
        extractor.run(["cash_flow"], random.Random(0))


def test_max_keywords_stops_extraction():
    backend = MockBackend(["a1, a2, a3", "b1"], cycle=True)
    settings = RetrievalSettings(top_k=5, iterations=3, max_keywords=3)
    extractor = RetrievalExtractor(backend, BM25Index.from_documents(CORPUS), DESCRIPTION, settings)
    result = extractor.run(["cash_flow"], random.Random(0))
    assert result.keywords == ["cash_flow", "a1", "a2"]
    assert len(backend.calls) == 1 and len(result.history) == 1


def test_seeded_runs_are_deterministic():
    def run(seed):
        backend = MockBackend(["x"], cycle=True)
        settings = RetrievalSettings(sample_size=2, iterations=2)
        extractor = RetrievalExtractor(
            backend, BM25Index.from_documents(CORPUS), DESCRIPTION, settings
        )
        extractor.run(["cash_flow", "overdraft", "depreciation", "gst"], random.Random(seed))
        return [c.prompt for c in backend.calls]

    assert run(3) == run(3)


# ── from_spec ────────────────────────────────────────────────────

TASK_YAML = """\
task:
  name: toyret
  version: "0.1"
  type: classification_spans
  generation_mode: label_first
  description: Write exam questions about fictional small-business finance.
output_schema: {}
rubric:
  verdict:
    values: [pass, fail]
seeds:
  path: seeds.jsonl
coverage:
  target_size: 10
  axes:
    - name: label
      values: [true, false]
  params:
    keyword_expansion:
      example_single: ledger
    retrieval:
      corpus: [corpus/docs.jsonl]
      top_k: 1
models:
  generator:
    backend: mock
    model: mock-1
  expansion:
    backend: mock
    model: mock-exp
    temperature: 0.3
    max_tokens: 111
validation:
  layers: [L1, L2]
thresholds:
  fidelity_min: 0.95
  kappa_min: 0.8
  coverage_min_cell_fill: 0.9
  balance_tolerance: 0.05
  distinct_n_min: 0.3
  self_bleu_max: 0.6
  semantic_diversity_min: 1.0
  residual_error_max: 0.05
  overlap_max: 0.8
  cost_per_record_max: 0.05
"""


def _toy_task(tmp_path, corpus=True):
    task = tmp_path / "task"
    (task / "corpus").mkdir(parents=True)
    (task / "task.yaml").write_text(TASK_YAML)
    (task / "seeds.jsonl").write_text('{"id": "s1", "label": true}\n')
    if corpus:
        (task / "corpus" / "docs.jsonl").write_text(
            "".join(json.dumps({"text": d.text}) + "\n" for d in CORPUS)
        )
    return compile_spec(task / "task.yaml")


def test_from_spec_loads_corpus_relative_to_task_and_persists(tmp_path, monkeypatch):
    compiled = _toy_task(tmp_path)
    monkeypatch.chdir(tmp_path)  # corpus path must not depend on the working directory
    store = ArtefactStore(tmp_path / "store")
    backend = MockBackend(["receivables"])
    extractor = RetrievalExtractor.from_spec(compiled, backend, store)
    assert len(extractor.index) == 5
    assert (extractor.max_tokens, extractor.temperature) == (111, 0.3)
    assert extractor.prompt_settings.example_single == "ledger"
    stage = retrieval.index_stage_name(extractor.index.corpus_hash)
    assert store.has_shared(compiled.spec_version, stage)

    result = extractor.run(["overdraft"], random.Random(0))
    assert result.added == ["receivables"]
    assert backend.calls[0].max_tokens == 111

    again = RetrievalExtractor.from_spec(compiled, MockBackend(["x"]), store)
    assert again.index.corpus_hash == extractor.index.corpus_hash


def test_from_spec_errors(tmp_path):
    compiled = _toy_task(tmp_path, corpus=False)
    with pytest.raises(RetrievalError, match="not found"):
        RetrievalExtractor.from_spec(compiled, MockBackend(["x"]), ArtefactStore(tmp_path / "s"))
