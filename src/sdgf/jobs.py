"""Jobs: the long sdgf steps (run, resume, evaluate, release) as background processes.

An agent's tool call can't block for an hour, so JobManager.start returns a job id at
once and the step runs in its own process (python -m sdgf.jobs JOB_DIR). The process
boundary also keeps a task's hooks.py, and whatever it does, out of the caller's process.

A job is a directory in the artefact store; each file has one writer:

    jobs/<id>/job.json     the request (the manager)
    jobs/<id>/lock         flock'd by the worker for its whole life: held means running
    jobs/<id>/worker.json  pid, spec_version and run_id, once the worker has compiled the spec
    jobs/<id>/result.json  the end state, created once: finished, failed, timed_out, cancelled
    jobs/<id>/log.txt      the worker's stdout and stderr

A job's state is result.json's if there is one, else running while the lock is held,
else lost: the worker died without a result (killed, crashed, the machine restarted).
Because the state lives in the store, a new manager sees every job, and a worker
outlives the manager that started it.

A killed run keeps everything it accepted (Pipeline resume), so cancel is a kill and
restart(job_id) starts a new job on the same run id, which continues the run. The
worker enforces its own time limit, so a job times out even if no manager is running.

POSIX only: the lock is flock and cancel signals the worker's process group.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdgf.models.usage import USAGE_STAGE
from sdgf.pipeline import DEFAULT_MAX_ROUNDS
from sdgf.service import Outcome, RunOptions, Service, ServiceError, TraceSettings
from sdgf.service.core import DEFAULT_OPTIONS
from sdgf.store.artefacts import ArtefactStore, iter_jsonl, new_run_id

KINDS = ("run", "resume", "evaluate", "release")

RUNNING = "running"
FINISHED = "finished"  # the step returned an Outcome; its status says ok / incomplete / ...
FAILED = "failed"
TIMED_OUT = "timed_out"
CANCELLED = "cancelled"
LOST = "lost"

DEFAULT_TIMEOUT_S = 3600.0
DEFAULT_MAX_RUNNING = 2
LOG_TAIL_LINES = 20

JOBS_DIR = "jobs"
JOB_FILE = "job.json"
LOCK_FILE = "lock"
WORKER_FILE = "worker.json"
RESULT_FILE = "result.json"
LOG_FILE = "log.txt"

_SRC_ROOT = Path(__file__).resolve().parents[1]  # the directory holding the sdgf package

# What a worker keeps of the caller's environment when it must not inherit the rest
# (e.g. API keys) because it runs code an agent wrote.
ENV_KEEP = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TZ", "SYSTEMROOT")


def child_env(extra: Mapping[str, str] = {}, *, inherit: bool = True) -> dict[str, str]:
    """The environment for an sdgf subprocess: all of this one's (inherit) or only
    ENV_KEEP, then extra, with the sdgf package importable."""
    base = dict(os.environ) if inherit else {k: os.environ[k] for k in ENV_KEEP if k in os.environ}
    env = {**base, **extra}
    paths = [str(_SRC_ROOT), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    return env


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _write_result(job_dir: Path, result: Mapping[str, Any]) -> bool:
    """Create result.json unless it exists; True if this call wrote it. The worker, its
    timer and a cancelling manager may race to it, and the first one wins."""
    tmp = job_dir / f"{RESULT_FILE}.{os.getpid()}.{threading.get_ident()}.tmp"
    payload = {**result, "finished_at": _now()}
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    try:
        os.link(tmp, job_dir / RESULT_FILE)  # atomic, and fails if it already exists
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def _lock_held(path: Path) -> bool:
    try:
        fd = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _log_tail(path: Path, lines: int = LOG_TAIL_LINES) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except FileNotFoundError:
        return []


class JobManager:
    """Starts, watches, cancels and restarts jobs in one artefact store.

    The caller fixes everything an agent must not choose: where releases go (releases),
    which model overrides a job may use (backends, a MODULE:ATTR reference), how long a
    job may run and how many may run at once, and the largest target size. service
    resolves task names, so an agent-facing manager passes one built with allow_paths=False.
    inherit_env=False gives workers only ENV_KEEP plus env, for jobs running code an agent
    wrote (a draft's hooks.py), so it can't read the server's API keys.
    """

    def __init__(
        self,
        store: ArtefactStore | str | Path,
        *,
        service: Service | None = None,
        releases: str | Path | None = None,
        backends: str | None = None,
        max_running: int = DEFAULT_MAX_RUNNING,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_target_size: int | None = None,
        env: Mapping[str, str] | None = None,
        inherit_env: bool = True,
    ):
        self.store = store if isinstance(store, ArtefactStore) else ArtefactStore(store)
        self.service = service if service is not None else Service(self.store)
        self.releases = Path(releases).resolve() if releases is not None else None
        self.backends = backends
        self.max_running = max_running
        self.timeout_s = timeout_s
        self.max_target_size = max_target_size
        self.env = dict(env or {})
        self.inherit_env = inherit_env
        self.root = self.store.root.resolve() / JOBS_DIR
        self._procs: dict[str, subprocess.Popen[bytes]] = {}

    # ── starting ─────────────────────────────────────────────────

    def start(
        self,
        kind: str,
        task: str | Path,
        *,
        run_id: str | None = None,
        options: RunOptions = DEFAULT_OPTIONS,
        waive: Iterable[str] = (),
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        version: str | None = None,
        trace: TraceSettings | None = None,
        label: str | None = None,
    ) -> dict[str, Any]:
        """Start a job and return its status at once. run and release make a run id when
        none is given; resume and evaluate default to the spec_version's latest run.
        label is the task name statuses show, when task is a path a caller must not see."""
        if kind not in KINDS:
            raise ServiceError(f"unknown job kind {kind!r}; have {list(KINDS)}", code="bad_kind")
        if kind == "release" and self.releases is None:
            raise ServiceError("no release directory is configured", code="no_releases")
        task_dir = self.service.task_dir(task)
        if not task_dir.exists():
            raise ServiceError(f"no task at {task_dir}", code="unknown_task")
        running = [j["job_id"] for j in self.list() if j["state"] == RUNNING]
        if len(running) >= self.max_running:
            raise ServiceError(
                f"{len(running)} jobs are running, the limit is {self.max_running}",
                code="too_many_jobs",
                hint=f"wait for or cancel one of {running}",
            )
        if run_id is None and kind in ("run", "release"):
            run_id = new_run_id()
        return self._spawn(
            {
                "kind": kind,
                "task": label or str(task),
                "task_ref": str(task),
                "task_dir": str(task_dir.resolve()),
                "run_id": run_id,
                "options": dataclasses.asdict(options),
                "waive": list(waive),
                "max_rounds": max_rounds,
                "version": version,
                "trace": dataclasses.asdict(trace) if trace is not None else None,
                "releases": str(self.releases) if self.releases is not None else None,
                "backends": self.backends,
                "timeout_s": self.timeout_s,
                "max_target_size": self.max_target_size,
            }
        )

    def restart(self, job_id: str) -> dict[str, Any]:
        """A new job continuing a finished, failed, timed-out, cancelled or lost one's run.

        It keeps the old job's request and takes this manager's current limits."""
        status = self.status(job_id)
        if status["state"] == RUNNING:
            raise ServiceError(f"job {job_id} is still running", code="job_running")
        job = self._job(job_id)
        options = RunOptions(**job["options"])
        trace = TraceSettings(**job["trace"]) if job["trace"] else None
        restarted = self.start(
            job["kind"],
            job.get("task_ref", job["task"]),
            run_id=status["run_id"],
            options=options,
            waive=job["waive"],
            max_rounds=job["max_rounds"],
            version=job["version"],
            trace=trace,
            label=job["task"],
        )
        return {**restarted, "restarted_from": job_id}

    def _spawn(self, request: dict[str, Any]) -> dict[str, Any]:
        job_id = "job-" + new_run_id()
        job_dir = self.root / job_id
        job_dir.mkdir(parents=True)
        job = {"id": job_id, "created_at": _now(), **request}
        _write_json(job_dir / JOB_FILE, job)
        # The manager takes the lock and hands it to the worker, so the job counts as
        # running from the moment it exists; the lock goes when the worker exits.
        fd = os.open(job_dir / LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (job_dir / LOG_FILE).open("ab") as log:
                proc = subprocess.Popen(
                    [sys.executable, "-m", "sdgf.jobs", str(job_dir), "--lock-fd", str(fd)],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(fd,),
                    start_new_session=True,  # its own process group, so cancel reaches it all
                    env=child_env(self.env, inherit=self.inherit_env),
                )
        except OSError as e:
            _write_result(
                job_dir,
                {
                    "state": FAILED,
                    "error": {"code": "spawn_failed", "message": str(e), "hint": None},
                },
            )
            raise ServiceError(f"could not start job {job_id}: {e}", code="spawn_failed") from e
        finally:
            os.close(fd)
        self._procs[job_id] = proc
        _write_json(job_dir / JOB_FILE, {**job, "pid": proc.pid})
        return self.status(job_id)

    # ── watching ─────────────────────────────────────────────────

    def status(self, job_id: str) -> dict[str, Any]:
        job = self._job(job_id)
        job_dir = self.root / job_id
        self._reap(job_id)
        result = _read_json(job_dir / RESULT_FILE)
        if result is not None:
            state = result["state"]
        else:
            state = RUNNING if _lock_held(job_dir / LOCK_FILE) else LOST
        worker = _read_json(job_dir / WORKER_FILE) or {}
        run_id = worker.get("run_id", job["run_id"])
        spec_version = worker.get("spec_version")
        status: dict[str, Any] = {
            "job_id": job_id,
            "kind": job["kind"],
            "task": job["task"],
            "state": state,
            "run_id": run_id,
            "spec_version": spec_version,
            "created_at": job["created_at"],
            "started_at": worker.get("started_at"),
            "finished_at": (result or {}).get("finished_at"),
            "progress": self._progress(spec_version, run_id),
        }
        if result is not None:
            status.update({k: result[k] for k in ("status", "data", "error") if k in result})
        if state in (FAILED, TIMED_OUT, LOST):
            status["log_tail"] = _log_tail(job_dir / LOG_FILE)
        if state == LOST:
            status["error"] = {
                "code": "worker_lost",
                "message": "the job's process ended without a result",
                "hint": "restart the job to continue its run",
            }
        return status

    def list(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        jobs = [self.status(d.name) for d in self.root.iterdir() if (d / JOB_FILE).is_file()]
        return sorted(jobs, key=lambda j: (j["created_at"], j["job_id"]))

    def wait(self, job_id: str, timeout: float | None = None, poll: float = 0.2) -> dict[str, Any]:
        """Block until the job leaves running, or timeout seconds pass; return its status."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            status = self.status(job_id)
            if status["state"] != RUNNING:
                return status
            if deadline is not None and time.monotonic() >= deadline:
                return status
            time.sleep(poll)

    def _progress(self, spec_version: str | None, run_id: str | None) -> dict[str, Any] | None:
        """Read from the run directory, which the pipeline rewrites after every wave."""
        if spec_version is None or run_id is None or run_id not in self.store.runs(spec_version):
            return None
        run = self.store.open_run(spec_version, run_id)
        cells = run.read_stage("cells") if run.has_stage("cells") else None
        usage = run.read_stage(USAGE_STAGE) if run.has_stage(USAGE_STAGE) else {}
        return {
            "accepted": sum(1 for _ in iter_jsonl(run.jsonl_path("accepted"))),
            "target": sum(c["quota"] for c in cells) if cells is not None else None,
            "usage": usage.get("total") if isinstance(usage, dict) else None,
        }

    # ── stopping ─────────────────────────────────────────────────

    def cancel(self, job_id: str, grace: float = 5.0) -> dict[str, Any]:
        """Stop a running job: SIGTERM to its process group, SIGKILL after grace seconds.

        A job that already ended keeps its own end state."""
        job = self._job(job_id)
        job_dir = self.root / job_id
        lock = job_dir / LOCK_FILE
        if self.status(job_id)["state"] == RUNNING:
            # The lock is held, so job["pid"] is still our worker, not a reused pid.
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(job["pid"], sig)
                except ProcessLookupError:
                    break
                deadline = time.monotonic() + grace
                while _lock_held(lock) and time.monotonic() < deadline:
                    time.sleep(0.05)
                if not _lock_held(lock):
                    break
            _write_result(
                job_dir,
                {
                    "state": CANCELLED,
                    "error": {
                        "code": "cancelled",
                        "message": "the job was cancelled",
                        "hint": "restart the job to continue its run",
                    },
                },
            )
        return self.status(job_id)

    def _job(self, job_id: str) -> dict[str, Any]:
        job = None
        if job_id and "/" not in job_id and job_id not in (".", ".."):
            job = _read_json(self.root / job_id / JOB_FILE)
        if job is None:
            raise ServiceError(f"unknown job {job_id!r}", code="unknown_job")
        return job

    def _reap(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is not None:
            del self._procs[job_id]


# ── worker ───────────────────────────────────────────────────────


def _execute(job_dir: Path, job: Mapping[str, Any]) -> Outcome:
    from sdgf.cli import load_backends  # the CLI's MODULE:ATTR loader; set by the manager

    store = ArtefactStore(job_dir.parent.parent)
    service = Service(store, notify=lambda m: print(f"sdgf: {m}", file=sys.stderr, flush=True))
    kind = job["kind"]
    task = job["task_dir"]
    compiled = service.compile(task)
    options = RunOptions(**job["options"])
    limit = job["max_target_size"]
    target = options.target_size or compiled.spec.coverage.target_size
    if limit is not None and target > limit:
        raise ServiceError(
            f"target size {target} is over this server's limit of {limit}",
            code="limit_exceeded",
            hint=f"pass a target_size of at most {limit}",
        )
    run_id = job["run_id"]
    if kind in ("resume", "evaluate"):
        run_id = service.existing_run_id(compiled, run_id)
    _write_json(
        job_dir / WORKER_FILE,
        {
            "pid": os.getpid(),
            "spec_version": compiled.spec_version,
            "run_id": run_id,
            "started_at": _now(),
        },
    )
    backends = load_backends(job["backends"]) if job["backends"] else None
    trace = TraceSettings(**job["trace"]) if job["trace"] else None
    if kind == "run":
        return service.run(task, run_id, options=options, backends=backends, trace=trace)
    if kind == "resume":
        return service.resume(task, run_id, options=options, backends=backends, trace=trace)
    if kind == "evaluate":
        return service.evaluate(
            task, run_id, options=options, backends=backends, waive=job["waive"]
        )
    return service.release(
        task,
        job["releases"],
        run_id,
        options=options,
        backends=backends,
        trace=trace,
        max_rounds=job["max_rounds"],
        waive=job["waive"],
        version=job["version"],
    )


def _expire(job_dir: Path, timeout_s: float) -> None:
    _write_result(
        job_dir,
        {
            "state": TIMED_OUT,
            "error": {
                "code": "timed_out",
                "message": f"the job ran for over {timeout_s:g}s",
                "hint": "restart the job to continue its run",
            },
        },
    )
    os._exit(1)  # the pipeline is kill-safe; a restart resumes the run


def work(job_dir: Path) -> int:
    job = _read_json(job_dir / JOB_FILE)
    if job is None:
        print(f"sdgf: no job at {job_dir}", file=sys.stderr)
        return 2
    timer = threading.Timer(job["timeout_s"], _expire, args=(job_dir, job["timeout_s"]))
    timer.daemon = True
    timer.start()
    try:
        outcome = _execute(job_dir, job)
    except ServiceError as e:
        _write_result(job_dir, {"state": FAILED, "error": e.to_dict()})
        return 1
    except Exception as e:  # noqa: BLE001 - any failure of the step ends the job as failed
        traceback.print_exc()
        error = {
            "code": "step_failed",
            "message": f"{type(e).__name__}: {e}",
            "hint": "see log_tail; fix the cause, then restart the job",
        }
        _write_result(job_dir, {"state": FAILED, "error": error})
        return 1
    finally:
        timer.cancel()
    _write_result(job_dir, {"state": FINISHED, "status": outcome.status, "data": outcome.data})
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sdgf.jobs", description="run one sdgf job")
    parser.add_argument("job_dir", type=Path)
    parser.add_argument("--lock-fd", type=int, required=True)
    args = parser.parse_args(argv)
    # Keep the lock for this process's life, but don't pass it on to any child process,
    # which would otherwise hold it, and the job would look running, after we exit.
    os.set_inheritable(args.lock_fd, False)
    return work(args.job_dir)


if __name__ == "__main__":
    sys.exit(main())
