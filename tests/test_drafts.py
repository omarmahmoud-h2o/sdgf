"""sdgf.drafts: agent-written task specs, validated in a subprocess without the server's
environment, and approved by a person before any real run; the people's side through the
CLI and the agents' side through the MCP tools. MockBackends only; no model calls."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sdgf.drafts import APPROVED, EDITING, REJECTED, SUBMITTED, DraftError, DraftStore

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
ENV = {"PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT / "tests")])}


@pytest.fixture
def drafts(tmp_path):
    return DraftStore(tmp_path / "drafts", approved_root=tmp_path / "tasks", env=ENV)


def fag_copy(drafts, name="fag2"):
    drafts.create(name, from_dir=TASKS / "fag")
    return name


def first_seed():
    return json.loads((TASKS / "fag" / "seeds.jsonl").read_text().splitlines()[0])


# ── the draft's life ─────────────────────────────────────────────


def test_submit_approve_and_any_edit_withdraws(drafts, tmp_path):
    name = fag_copy(drafts)
    assert drafts.state(name) == EDITING
    with pytest.raises(DraftError) as e:
        drafts.approve(name, reviewer="R")
    assert e.value.code == "not_submitted"

    submitted = drafts.submit(name)
    assert submitted["state"] == SUBMITTED and submitted["validation"]["ok"]
    approved = drafts.approve(name, reviewer="R", note="looks right")
    assert approved["state"] == APPROVED and approved["approved"]["reviewer"] == "R"
    promoted = tmp_path / "tasks" / name
    assert {p.name for p in promoted.iterdir()} == {"task.yaml", "seeds.jsonl", "hooks.py"}

    drafts.append_seeds(name, [first_seed()])
    assert drafts.state(name) == EDITING  # the approved bytes are no longer the draft's
    drafts.submit(name)
    with pytest.raises(DraftError) as e:
        drafts.approve(name, reviewer="R")  # task fag2 now exists
    assert e.value.code == "task_exists"
    assert drafts.approve(name, reviewer="R", replace=True)["state"] == APPROVED


def test_an_edit_after_submitting_needs_a_new_submission(drafts):
    name = fag_copy(drafts)
    drafts.submit(name)
    task = drafts.spec_dir(name) / "task.yaml"
    drafts.write_file(name, "task.yaml", task.read_text() + "\n# edited\n")
    with pytest.raises(DraftError) as e:
        drafts.approve(name, reviewer="R")
    assert e.value.code == "not_submitted" and "changed since" in str(e.value)


def test_a_rejection_reaches_the_agent_until_it_edits(drafts):
    name = fag_copy(drafts)
    drafts.submit(name)
    drafts.reject(name, reviewer="R", reason="the breach rule ignores tier 2")
    read = drafts.read(name)
    assert read["state"] == REJECTED and "tier 2" in read["rejected"]["reason"]
    drafts.delete_file(name, "hooks.py")
    assert drafts.state(name) == EDITING


def test_validation_reports_a_broken_spec_without_raising(drafts):
    name = fag_copy(drafts)
    task = drafts.spec_dir(name) / "task.yaml"
    drafts.write_file(name, "task.yaml", task.read_text().replace("thresholds:", "thresholdz:", 1))
    report = drafts.validate(name)
    assert not report["ok"] and "thresholds" in report["error"]["message"]
    with pytest.raises(DraftError) as e:
        drafts.submit(name)
    assert e.value.code == "draft_invalid"


# ── a draft's hooks.py is untrusted code ─────────────────────────


def test_validation_runs_hooks_without_the_servers_environment(drafts, tmp_path, monkeypatch):
    monkeypatch.setenv("SDGF_TEST_API_KEY", "s3cret")
    name = fag_copy(drafts)
    hooks = (drafts.spec_dir(name) / "hooks.py").read_text()
    seen = tmp_path / "seen.txt"
    probe = f"\nimport os\nopen({str(seen)!r}, 'w').write(os.environ.get('SDGF_TEST_API_KEY', 'absent'))\n"
    drafts.write_file(name, "hooks.py", hooks + probe)
    assert drafts.validate(name)["ok"]
    assert seen.read_text() == "absent"


def test_slow_hooks_time_out(tmp_path):
    drafts = DraftStore(tmp_path / "drafts", env=ENV, validate_timeout_s=2)
    name = fag_copy(drafts)
    hooks = (drafts.spec_dir(name) / "hooks.py").read_text()
    drafts.write_file(name, "hooks.py", hooks + "\nimport time\ntime.sleep(60)\n")
    report = drafts.validate(name)
    assert not report["ok"] and report["error"]["code"] == "validation_timeout"


def test_hooks_can_be_closed_to_agents(tmp_path):
    drafts = DraftStore(tmp_path / "drafts", allow_hooks=False, env=ENV)
    name = fag_copy(drafts)  # a copied, already approved hooks.py stays
    assert (drafts.spec_dir(name) / "hooks.py").is_file()
    with pytest.raises(DraftError) as e:
        drafts.write_file(name, "hooks.py", "print('hi')\n")
    assert e.value.code == "hooks_not_allowed"
    assert drafts.read(name)["hooks_editable"] is False


@pytest.mark.parametrize(
    "call, code",
    [
        (lambda d: d.create("../escape"), "bad_name"),
        (lambda d: d.create("_store"), "bad_name"),
        (lambda d: d.read("nope"), "unknown_draft"),
        (lambda d: d.write_file("fag2", "evil.py", "x"), "bad_file"),
        (lambda d: d.write_file("fag2", "task.yaml", "x" * (512 * 1024 + 1)), "too_large"),
        (lambda d: d.append_seeds("fag2", ["not an object"]), "bad_seeds"),
        (lambda d: d.append_seeds("fag2", []), "bad_seeds"),
        (lambda d: d.create("fag2"), "draft_exists"),
    ],
)
def test_bad_requests_are_named(drafts, call, code):
    fag_copy(drafts)
    with pytest.raises(DraftError) as e:
        call(drafts)
    assert e.value.code == code


# ── a person's side: the CLI ─────────────────────────────────────


def sdgf(*args):
    proc = subprocess.run(
        [sys.executable, "-m", "sdgf.cli", *map(str, args)],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, **ENV},
        cwd=ROOT,
        timeout=120,
    )
    out = json.loads(proc.stdout) if proc.stdout.strip() else None
    return proc.returncode, out, proc.stderr


def test_people_review_and_approve_with_the_cli(drafts, tmp_path):
    name = fag_copy(drafts)
    drafts.submit(name)
    shutil.copytree(TASKS / "fag", tmp_path / "tasks" / name)  # an older version exists
    (tmp_path / "tasks" / name / "hooks.py").write_text("# the old rules\n")
    root = ["drafts", "--drafts", tmp_path / "drafts"]

    code, listed, _ = sdgf(*root, "list")
    assert code == 0 and listed["drafts"][0]["state"] == SUBMITTED
    code, shown, _ = sdgf(*root, "show", name, "--into", tmp_path / "tasks")
    assert code == 0 and shown["validation"]["ok"]
    assert "-# the old rules" in shown["diff_against_approved"]["hooks.py"]

    approve = (*root, "approve", name, "--reviewer", "R", "--into", tmp_path / "tasks")
    code, _, err = sdgf(*approve)
    assert code == 2 and "already exists" in err
    code, approved, _ = sdgf(*approve, "--replace")
    assert code == 0 and approved["state"] == APPROVED
    promoted = (tmp_path / "tasks" / name / "hooks.py").read_text()
    assert promoted == (TASKS / "fag" / "hooks.py").read_text()


# ── an agent's side: the MCP tools ───────────────────────────────


def test_an_agent_drafts_dry_runs_and_submits_and_a_person_approves(tmp_path):
    pytest.importorskip("mcp")
    from sdgf.mcp.server import ServerConfig, build_server
    from test_mcp import ToolFailed, finished, session

    shutil.copytree(TASKS / "fag", tmp_path / "tasks" / "fag")
    srv = build_server(
        ServerConfig(
            store=tmp_path / "store",
            task_roots=[tmp_path / "tasks"],
            backends="cli_backends:fag_world",
            env=ENV,
            drafts=tmp_path / "drafts",
            allow_hooks=False,
            dry_run_backends="cli_backends:fag_world",
            dry_run_max_target=10,
        )
    )

    async def agent(call):
        created = await call("create_draft", name="fag2", from_task="fag")
        assert created["state"] == EDITING
        with pytest.raises(ToolFailed) as e:
            await call("write_draft_file", name="fag2", file="hooks.py", content="x = 1\n")
        assert e.value.error["code"] == "hooks_not_allowed"
        assert (await call("validate_draft", name="fag2"))["ok"]

        started = await call("dry_run_draft", name="fag2")
        assert started["task"] == "draft:fag2"  # no server path
        dry = await finished(call)(started["job_id"])
        assert dry["status"] == "ok" and dry["data"]["accepted"] == 10
        assert [j["dry_run"] for j in (await call("list_jobs"))["jobs"]] == [True]

        assert (await call("submit_draft", name="fag2"))["state"] == SUBMITTED
        with pytest.raises(ToolFailed) as e:
            await call("start_run", task="fag2")  # not approved, so not a task
        assert e.value.error["code"] == "unknown_task"

    session(srv, agent)
    DraftStore(tmp_path / "drafts", approved_root=tmp_path / "tasks").approve("fag2", reviewer="R")

    async def after(call):
        assert "fag2" in {t["name"] for t in (await call("list_tasks"))["tasks"]}
        assert (await call("read_draft", name="fag2"))["state"] == APPROVED

    session(srv, after)
