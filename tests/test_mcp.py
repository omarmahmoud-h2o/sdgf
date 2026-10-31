"""sdgf.mcp: the MCP tools an agent uses, driven in-process by the MCP SDK's client and
once over stdio as `python -m sdgf.mcp`. Jobs run with MockBackends from
tests/cli_backends.py; no model or API calls."""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio
from mcp import Client, StdioServerParameters

from sdgf.mcp.server import ServerConfig, build_server, main

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
TARGET = 20
ENV = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}
WAIVE = ["semantic_diversity_min", "kappa_min", "residual_error_max"]
TOOLS = {
    "list_tasks",
    "describe_task",
    "validate_spec",
    "plan",
    "start_run",
    "start_release",
    "start_evaluate",
    "job_status",
    "wait_job",
    "list_jobs",
    "cancel_job",
    "restart_job",
    "get_records",
    "list_reviews",
}


def server(tmp_path, roots=(TASKS,), **kwargs):
    config = ServerConfig(
        store=tmp_path / "store",
        task_roots=list(roots),
        releases=tmp_path / "releases",
        backends="cli_backends:fag_world",
        env=ENV,
        **kwargs,
    )
    return build_server(config)


class ToolFailed(Exception):
    def __init__(self, error):
        super().__init__(error)
        self.error = error


def session(srv, calls):
    """Run calls(call) against srv, where call(name, **args) returns the tool's result."""

    async def go():
        async with Client(srv) as client:

            async def call(name, **args):
                result = await client.call_tool(name, args)
                if result.is_error:
                    text = result.content[0].text
                    raise ToolFailed(json.loads(text[text.index("{") :]))
                return result.structured_content

            return await calls(call)

    return anyio.run(go)


def finished(call):
    async def wait(job_id, **kwargs):
        status = await call("wait_job", job_id=job_id, timeout_s=120, **kwargs)
        assert status["state"] == "finished", status
        return status

    return wait


# ── what an agent can and can't do ───────────────────────────────


def test_the_tools_leave_peoples_decisions_out(tmp_path):
    srv = server(tmp_path)

    async def tools():
        async with Client(srv) as client:
            return (await client.list_tools()).tools

    listed = anyio.run(tools)
    assert {t.name for t in listed} == TOOLS
    params = {p for t in listed for p in t.input_schema.get("properties", {})}
    # no approving, reviewing, paths, model overrides or release directories
    assert not params & {"reviewer", "approve", "store", "releases", "backends", "path", "held_out"}
    assert all(t.description for t in listed)


def test_tasks_by_name_and_errors_with_codes(tmp_path):
    async def calls(call):
        listed = await call("list_tasks")
        assert {"fag", "cfa"} <= {t["name"] for t in listed["tasks"]}
        described = await call("describe_task", task="fag")
        assert described["status"] == "ok" and described["task_type"] == "classification_spans"
        with pytest.raises(ToolFailed) as e:
            await call("describe_task", task="../tasks/fag")
        assert e.value.error["code"] == "unknown_task" and e.value.error["hint"]
        with pytest.raises(ToolFailed) as e:
            await call("job_status", job_id="job-nope")
        assert e.value.error["code"] == "unknown_job"

    session(server(tmp_path), calls)


# ── running ──────────────────────────────────────────────────────


def test_run_then_read_records_a_page_at_a_time(tmp_path):
    async def calls(call):
        started = await call("start_run", task="fag", target_size=TARGET)
        assert started["state"] in ("running", "finished")
        brief = await finished(call)(started["job_id"])
        assert brief["status"] == "ok" and brief["data"]["accepted"] == TARGET
        assert "counts" not in brief["data"] and "counts" in brief["omitted"]
        full = await call("job_status", job_id=started["job_id"], detail=True)
        assert sum(full["data"]["counts"].values()) == TARGET

        page = await call("get_records", task="fag", limit=8)
        assert page["total"] == TARGET and len(page["records"]) == 8
        assert page["next_offset"] == 8
        assert all("_provenance" not in r for r in page["records"])
        last = await call("get_records", task="fag", offset=16, limit=8)
        assert len(last["records"]) == 4 and last["next_offset"] is None
        with pytest.raises(ToolFailed) as e:
            await call("get_records", task="fag", limit=500)
        assert e.value.error["code"] == "bad_page"

        jobs = await call("list_jobs")
        assert [j["job_id"] for j in jobs["jobs"]] == [started["job_id"]]

    session(server(tmp_path), calls)


def test_only_the_operators_waivers_are_allowed(tmp_path):
    async def refused(call):
        with pytest.raises(ToolFailed) as e:
            await call("start_release", task="fag", waive=["kappa_min"])
        assert e.value.error["code"] == "waiver_not_allowed"

    session(server(tmp_path), refused)

    async def released(call):
        started = await call("start_release", task="fag", target_size=TARGET, waive=WAIVE)
        done = await finished(call)(started["job_id"])
        assert done["status"] == "ok" and done["data"]["released"]
        evaluated = await finished(call)((await call("start_evaluate", task="fag"))["job_id"])
        assert evaluated["status"] == "incomplete"  # nothing waived this time
        assert {f["metric"] for f in evaluated["data"]["gate_failures"]} == set(WAIVE)

    session(server(tmp_path, waivable=frozenset(WAIVE)), released)


def test_a_plan_awaiting_approval_is_reported_not_approved(tmp_path):
    shutil.copytree(TASKS / "fag", tmp_path / "tasks" / "fag")
    task = tmp_path / "tasks" / "fag" / "task.yaml"
    text = task.read_text()
    task.write_text(text.replace("approve_coverage_plan: false", "approve_coverage_plan: true"))

    async def calls(call):
        planned = await call("plan", task="fag", target_size=TARGET)
        assert planned["approval_required"] and not planned["approved"]
        assert "--approve --reviewer" in planned["hint"]
        started = await call("start_run", task="fag", target_size=TARGET)
        done = await finished(call)(started["job_id"])
        assert done["status"] == "approval_pending"
        assert "awaiting approval" in done["data"]["message"]

    session(server(tmp_path, roots=[tmp_path / "tasks"]), calls)


# ── command line ─────────────────────────────────────────────────


def test_http_binds_only_to_loopback(tmp_path, capsys):
    code = main(["--store", str(tmp_path), "--tasks", str(TASKS), "--http", "--host", "0.0.0.0"])
    assert code == 2 and "no authentication" in capsys.readouterr().err


def test_the_stdio_server_runs_as_a_module(tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "sdgf.mcp", "--store", str(tmp_path / "store"), "--tasks", str(TASKS)],
        env=ENV,
    )

    async def go():
        async with Client(params) as client:
            result = await client.call_tool("validate_spec", {"task": "fag"})
            return result.structured_content

    assert anyio.run(go)["task"] == "fag"
