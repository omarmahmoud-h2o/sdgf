"""M6 checkpoint: agentic tools end to end on FAG, through the gateway, the shared cache
and provenance."""

import shutil
from pathlib import Path

import pytest
from test_pipeline import NO_JUDGE
from test_tools_builtin import CALC, LOOKUP, agent_reply

from sdgf.generate.generator import TOOL_RESULTS_HEADER
from sdgf.models.base import ToolCall
from sdgf.models.mock import MockBackend
from sdgf.pipeline import DROPS_STREAM, Pipeline
from sdgf.store.provenance import split
from sdgf.tools.builtin import CALCULATOR, CATALOGUE_LOOKUP

FAG_DIR = Path(__file__).resolve().parents[1] / "tasks" / "fag"


def fag_with_tools(tmp_path, tools):
    task = tmp_path / "fag_tools"
    shutil.copytree(FAG_DIR, task, ignore=shutil.ignore_patterns("__pycache__"))
    with (task / "task.yaml").open("a") as f:
        f.write("\ntools:\n" + "".join(f"  - name: {t}\n" for t in tools))
    return task


def run(task, store, reply, run_id, target_size=4):
    pipe = Pipeline(
        task,
        store,
        model_overrides={"generator": MockBackend(reply)},
        target_size=target_size,
        layers=NO_JUDGE,
    )
    return pipe.run(run_id)


def traces(result):
    return [split(rec)[1].tool_trace for rec in result.accepted]


def test_fag_with_builtin_tools_runs_and_every_record_carries_its_trace(tmp_path):
    task = fag_with_tools(tmp_path, [CATALOGUE_LOOKUP, CALCULATOR])
    result = run(task, tmp_path / "store", agent_reply, "r1")
    assert result.complete and len(result.accepted) == 4
    for trace in traces(result):
        assert [(t.tool, t.arguments) for t in trace] == [
            (LOOKUP.name, LOOKUP.arguments),
            (CALC.name, CALC.arguments),
        ]
        assert all(t.sensitivity == "public" and t.error is None for t in trace)
    # Same arguments every record, so only the first record's calls reach the handlers.
    assert [t.cached for t in traces(result)[0]] == [False, False]
    assert all(t.cached for trace in traces(result)[1:] for t in trace)


def test_second_run_of_the_same_spec_is_served_from_the_shared_cache(tmp_path):
    task = fag_with_tools(tmp_path, [CATALOGUE_LOOKUP, CALCULATOR])
    store = tmp_path / "store"
    first = run(task, store, agent_reply, "r1")
    second = run(task, store, agent_reply, "r2")
    assert second.complete
    assert all(t.cached for trace in traces(second) for t in trace)
    assert [t.result for t in traces(second)[0]] == [t.result for t in traces(first)[0]]


def test_unlisted_tool_is_denied_traced_and_the_record_still_completes(tmp_path):
    task = fag_with_tools(tmp_path, [CALCULATOR])  # the catalogue is not allowed

    def reply(call):
        if TOOL_RESULTS_HEADER not in call.prompt:
            return [LOOKUP, CALC]
        return agent_reply(call)

    result = run(task, tmp_path / "store", reply, "r1", target_size=2)
    assert result.complete
    for trace in traces(result):
        assert trace[0].tool == CATALOGUE_LOOKUP and trace[0].error.startswith("not_allowed")
        assert trace[0].result is None
        assert trace[1].tool == CALCULATOR and trace[1].error is None


def test_spec_listing_an_unknown_tool_is_refused_at_stage_0(tmp_path):
    task = fag_with_tools(tmp_path, ["no_such_tool"])
    with pytest.raises(Exception, match="not in the tool registry"):
        run(task, tmp_path / "store", agent_reply, "r1")


def test_tool_calls_without_tools_in_the_spec_fail_the_record(tmp_path):
    call = ToolCall(CALCULATOR, {"expression": "1 + 1"})
    result = run(FAG_DIR, tmp_path / "store", lambda c: [call], "r1", target_size=1)
    assert not result.complete and not result.accepted  # stopped by the run budget
    drop = result.run.read_jsonl(DROPS_STREAM)[0]
    assert drop["layer"] == "generate" and drop["codes"] == ["tool_calls"]
