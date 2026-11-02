"""A shared sdgf server: bearer-token clients over real streamable HTTP (uvicorn in a
thread), each seeing only its own jobs and drafts under its own limits, with every call
audited, through sdgf.client. MockBackends only; no model calls."""

import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

pytest.importorskip("mcp")

import anyio
import uvicorn

from sdgf.client import SdgfClient, SdgfToolError, call
from sdgf.clients import ClientPolicy, ClientRegistry, token_digest
from sdgf.clients import main as clients_main
from sdgf.jobs import RUN_JOBS_STAGE
from sdgf.mcp.server import AUDIT_FILE, ServerConfig, build_server, main
from sdgf.service import ServiceError
from sdgf.store.artefacts import ArtefactStore

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
ENV = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}
TOKENS = {"alpha": "tok-alpha", "beta": "tok-beta", "gamma": "tok-gamma"}
REGISTRY = ClientRegistry(
    [
        ClientPolicy("alpha", token_digest(TOKENS["alpha"]), max_running=1),
        ClientPolicy("beta", token_digest(TOKENS["beta"])),
        ClientPolicy("gamma", token_digest(TOKENS["gamma"]), max_tokens=1),
    ]
)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def serving(tmp_path, backends="cli_backends:fag_world"):
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    srv = build_server(
        ServerConfig(
            store=tmp_path / "store",
            task_roots=[TASKS],
            backends=backends,
            env=ENV,
            drafts=tmp_path / "drafts",
            dry_run_backends="cli_backends:fag_world",
            clients=REGISTRY,
            public_url=base,
        )
    )
    app = srv.streamable_http_app(host="127.0.0.1")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.02)
    try:
        yield f"{base}/mcp"
    finally:
        server.should_exit = True
        thread.join(10)


def as_client(url, name, body):
    async def go():
        async with SdgfClient(url, token=TOKENS[name]) as sdgf:
            return await body(sdgf)

    return anyio.run(go)


async def code_of(make_call):
    """Await a call expected to fail; return its error code."""
    with pytest.raises(SdgfToolError) as e:
        await make_call()
    return e.value.code


# ── who may call ─────────────────────────────────────────────────


def test_a_call_needs_a_known_token(tmp_path):
    with serving(tmp_path) as url:
        assert call(url, "my_usage", token=TOKENS["beta"])["client"] == "beta"
        for token in (None, "not-a-token"):
            with pytest.raises(Exception):  # noqa: B017 - the HTTP 401 surfaces as a group
                call(url, "list_tasks", token=token)


# ── each client's own jobs, drafts and limits ────────────────────


def test_clients_see_only_their_own_jobs_and_drafts(tmp_path):
    with serving(tmp_path, backends="cli_backends:fag_slow") as url:

        async def alpha(sdgf):
            job = await sdgf.call("start_run", task="fag", target_size=20)
            draft = await sdgf.call("create_draft", name="d1", from_task="fag")
            assert draft["state"] == "editing"
            assert await code_of(lambda: sdgf.call("start_run", task="fag")) == "too_many_jobs"
            return job["job_id"]

        job_id = as_client(url, "alpha", alpha)

        async def beta(sdgf):
            assert (await sdgf.call("list_jobs"))["jobs"] == []
            assert (await sdgf.call("list_drafts"))["drafts"] == []
            for tool in ("job_status", "cancel_job", "restart_job"):
                assert await code_of(lambda t=tool: sdgf.call(t, job_id=job_id)) == "unknown_job"
            assert await code_of(lambda: sdgf.call("read_draft", name="d1")) == "unknown_draft"

        as_client(url, "beta", beta)

        async def alpha_again(sdgf):
            assert [j["job_id"] for j in (await sdgf.call("list_jobs"))["jobs"]] == [job_id]
            return await sdgf.call("cancel_job", job_id=job_id)

        assert as_client(url, "alpha", alpha_again)["state"] == "cancelled"
    assert (tmp_path / "drafts" / "alpha" / "d1" / "spec" / "task.yaml").is_file()


def test_a_spent_budget_stops_new_jobs(tmp_path):
    with serving(tmp_path) as url:

        async def gamma(sdgf):
            job = await sdgf.call("start_run", task="fag", target_size=20)
            done = await sdgf.wait(job["job_id"], poll_s=30)
            assert done["state"] == "finished"
            used = await sdgf.call("my_usage")
            assert used["tokens"] > 1 and used["runs"] == 1 and used["limits"]["max_tokens"] == 1
            assert await code_of(lambda: sdgf.call("start_run", task="fag")) == "budget_exhausted"
            restart = await code_of(lambda: sdgf.call("restart_job", job_id=job["job_id"]))
            assert restart == "budget_exhausted"
            return done

        done = as_client(url, "gamma", gamma)
    run = ArtefactStore(tmp_path / "store").open_run(done["spec_version"], done["run_id"])
    recorded = run.read_stage(RUN_JOBS_STAGE)
    assert [(e["job_id"], e["client"]) for e in recorded] == [(done["job_id"], "gamma")]


def test_every_call_is_audited_without_its_text(tmp_path):
    with serving(tmp_path) as url:

        async def alpha(sdgf):
            await sdgf.call("create_draft", name="d1", from_task="fag")
            await sdgf.call("write_draft_file", name="d1", file="task.yaml", content="x" * 5000)
            await code_of(lambda: sdgf.call("describe_task", task="../tasks/fag"))

        as_client(url, "alpha", alpha)
    audit = (tmp_path / "store" / AUDIT_FILE).read_text().splitlines()
    lines = [json.loads(line) for line in audit]
    assert [e["tool"] for e in lines] == ["create_draft", "write_draft_file", "describe_task"]
    assert all(e["client"] == "alpha" for e in lines)
    assert lines[1]["args"]["content"] == {"chars": 5000} and lines[1]["ok"]
    assert lines[2]["ok"] is False and lines[2]["error"] == "unknown_task"


# ── configuration ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "entries, message",
    [
        ([{"id": "a"}], "token_sha256"),
        ([{"id": "local", "token_sha256": "x"}], "local"),
        ([{"id": "a", "token_sha256": "x", "admin": True}], "unknown keys"),
        ([{"id": "a", "token_sha256": "x"}, {"id": "a", "token_sha256": "y"}], "duplicate"),
    ],
)
def test_clients_files_are_checked(tmp_path, entries, message):
    path = tmp_path / "clients.json"
    path.write_text(json.dumps({"clients": entries}))
    with pytest.raises(ServiceError, match=message):
        ClientRegistry.load(path)


def test_a_new_client_entry_verifies_its_token(tmp_path, capsys):
    assert clients_main(["new", "team-a"]) == 0
    printed = json.loads(capsys.readouterr().out)
    (tmp_path / "clients.json").write_text(json.dumps({"clients": [printed["entry"]]}))
    registry = ClientRegistry.load(tmp_path / "clients.json")
    assert registry.verify(printed["token"]).id == "team-a"
    assert registry.verify(printed["token"] + "x") is None
    assert printed["token"] not in (tmp_path / "clients.json").read_text()


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--clients", "c.json"], "needs --http"),
        (["--http", "--host", "0.0.0.0"], "no authentication"),
        (["--http", "--clients", "missing.json"], "not a clients file"),
    ],
)
def test_the_command_line_refuses_unsafe_setups(tmp_path, capsys, extra, message):
    assert main(["--store", str(tmp_path), "--tasks", str(TASKS), *extra]) == 2
    assert message in capsys.readouterr().err
