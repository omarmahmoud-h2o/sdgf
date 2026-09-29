import json

import pytest

from sdgf.store.artefacts import ArtefactError, ArtefactStore, iter_jsonl

V1 = "a" * 64
V2 = "b" * 64


def test_open_run_creates_manifest_keyed_by_spec_version_and_run_id(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "run-1")
    assert run.path == tmp_path / V1 / "runs" / "run-1"
    assert not run.resumed
    meta = json.loads((run.path / "run.json").read_text())
    assert meta["spec_version"] == V1 and meta["run_id"] == "run-1"


def test_open_run_generates_unique_run_ids(tmp_path):
    store = ArtefactStore(tmp_path)
    a, b = store.open_run(V1), store.open_run(V1)
    assert a.run_id != b.run_id
    assert store.runs(V1) == sorted([a.run_id, b.run_id])


def test_reopening_a_run_resumes_it(tmp_path):
    store = ArtefactStore(tmp_path)
    store.open_run(V1, "r")
    assert store.open_run(V1, "r").resumed


def test_same_run_id_under_other_spec_version_is_a_separate_run(tmp_path):
    store = ArtefactStore(tmp_path)
    store.open_run(V1, "r").write_stage("plan", {"x": 1})
    other = store.open_run(V2, "r")
    assert not other.resumed
    assert not other.has_stage("plan")


def test_tampered_manifest_is_rejected(tmp_path):
    store = ArtefactStore(tmp_path)
    run = store.open_run(V1, "r")
    (run.path / "run.json").write_text(json.dumps({"spec_version": V2, "run_id": "r"}))
    with pytest.raises(ArtefactError, match="manifest"):
        store.open_run(V1, "r")


def test_run_dir_without_manifest_is_rejected(tmp_path):
    stray = tmp_path / V1 / "runs" / "r"
    stray.mkdir(parents=True)
    (stray / "accepted.jsonl").write_text("{}\n")
    with pytest.raises(ArtefactError, match="without a manifest"):
        ArtefactStore(tmp_path).open_run(V1, "r")


@pytest.mark.parametrize("bad", ["../x", "a/b", "", ".hidden", "a..b"])
def test_unsafe_names_are_rejected(tmp_path, bad):
    store = ArtefactStore(tmp_path)
    with pytest.raises(ArtefactError, match="invalid"):
        store.open_run(V1, bad)
    with pytest.raises(ArtefactError, match="invalid"):
        store.open_run(bad, "r")
    with pytest.raises(ArtefactError, match="invalid"):
        store.open_run(V1, "ok").write_stage(bad, {})


def test_stage_write_and_read_roundtrip(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")
    assert not run.has_stage("coverage_plan")
    run.write_stage("coverage_plan", {"cells": [{"id": "c1", "quota": 3}]})
    assert run.has_stage("coverage_plan")
    assert run.read_stage("coverage_plan") == {"cells": [{"id": "c1", "quota": 3}]}
    assert not list(run.path.glob("*.tmp"))


def test_read_missing_stage_raises(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")
    with pytest.raises(ArtefactError, match="not found"):
        run.read_stage("coverage_plan")


def test_stage_from_other_spec_version_is_rejected(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")
    path = run.write_stage("plan", {"x": 1})
    payload = json.loads(path.read_text())
    payload["spec_version"] = V2
    path.write_text(json.dumps(payload))
    with pytest.raises(ArtefactError, match="different spec_version"):
        run.read_stage("plan")


def test_stage_resume_skips_compute_when_artefact_exists(tmp_path):
    store = ArtefactStore(tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return {"n": len(calls)}

    assert store.open_run(V1, "r").stage("plan", compute) == {"n": 1}
    assert store.open_run(V1, "r").stage("plan", compute) == {"n": 1}
    assert calls == [1]


def test_stage_is_recomputed_for_a_new_spec_version(tmp_path):
    store = ArtefactStore(tmp_path)
    calls = []
    store.open_run(V1, "r").stage("plan", lambda: calls.append(1) or "v1")
    assert store.open_run(V2, "r").stage("plan", lambda: calls.append(2) or "v2") == "v2"
    assert calls == [1, 2]


def test_shared_stage_is_reused_across_runs_of_same_spec_version(tmp_path):
    store = ArtefactStore(tmp_path)
    store.open_run(V1, "r1").stage("coverage_plan", lambda: {"p": 1}, shared=True)
    run2 = store.open_run(V1, "r2")
    assert run2.has_stage("coverage_plan", shared=True)
    assert not run2.has_stage("coverage_plan")
    assert run2.stage("coverage_plan", lambda: pytest.fail("recomputed"), shared=True) == {"p": 1}
    assert not store.open_run(V2, "r1").has_stage("coverage_plan", shared=True)


def test_failed_compute_writes_nothing(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")

    def boom():
        raise RuntimeError("model down")

    with pytest.raises(RuntimeError):
        run.stage("plan", boom)
    assert not run.has_stage("plan")


def test_jsonl_writer_flushes_each_record(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")
    with run.jsonl("accepted") as w:
        w.write({"id": 1})
        # visible to a reader before the writer closes
        assert run.read_jsonl("accepted") == [{"id": 1}]
        w.write_many([{"id": 2}, {"id": 3}])
    assert [r["id"] for r in run.read_jsonl("accepted")] == [1, 2, 3]


def test_jsonl_appends_across_resumed_runs(tmp_path):
    store = ArtefactStore(tmp_path)
    with store.open_run(V1, "r").jsonl("drops") as w:
        w.write({"id": 1})
    with store.open_run(V1, "r").jsonl("drops") as w:
        w.write({"id": 2})
    assert [r["id"] for r in store.open_run(V1, "r").read_jsonl("drops")] == [1, 2]


def test_partial_last_line_from_crash_is_skipped_and_trimmed(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")
    path = run.jsonl_path("accepted")
    path.write_text('{"id": 1}\n{"id": 2, "trunc')
    assert run.read_jsonl("accepted") == [{"id": 1}]
    with run.jsonl("accepted") as w:
        w.write({"id": 3})
    assert [r["id"] for r in run.read_jsonl("accepted")] == [1, 3]


def test_corrupt_middle_line_raises(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"id": 1}\nnot json\n{"id": 3}\n')
    with pytest.raises(ArtefactError, match=r"x.jsonl:2"):
        list(iter_jsonl(path))


def test_missing_stream_reads_empty(tmp_path):
    run = ArtefactStore(tmp_path).open_run(V1, "r")
    assert run.read_jsonl("accepted") == []


def test_closed_writer_refuses_writes(tmp_path):
    w = ArtefactStore(tmp_path).open_run(V1, "r").jsonl("accepted")
    w.close()
    with pytest.raises(ArtefactError, match="closed"):
        w.write({"id": 1})


def test_latest_run(tmp_path):
    store = ArtefactStore(tmp_path)
    assert store.latest_run(V1) is None
    for run_id, created in (("b", "2026-01-01T00:00:00+00:00"), ("a", "2026-01-02T00:00:00+00:00")):
        manifest = store.open_run(V1, run_id).path / "run.json"
        meta = json.loads(manifest.read_text())
        manifest.write_text(json.dumps({**meta, "created_at": created}))
    assert store.latest_run(V1) == "a"
