"""
Background jobs: anything slower than a few seconds runs detached, with a log.

Training takes hours to days, so nothing the workbench starts may depend on the
workbench staying open.  Each job is started in its own session (`setsid`), which puts it
out of reach of the SIGHUP a closing terminal or a dropped SSH connection sends, and its
output goes to a log file rather than the screen.  Quitting the workbench, closing VS Code
or losing the connection leaves every job running; reopening the workbench finds them
again from their records.

    output/workbench/jobs/<id>/
        job.json     what was run: command, environment overrides, GPU, links
        log.txt      everything the command printed (stdout and stderr)
        exit_code    written by the wrapper when the command ends
        cancelled    written when the job is cancelled from the workbench

A job's status is derived, never stored: exit_code present → succeeded/failed;
`cancelled` present → cancelled; otherwise running if its process still exists (the
process start time is checked too, so a recycled PID is not mistaken for the job), and
otherwise "stopped" — it died without reporting, e.g. the machine rebooted.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

from workbench import paths
from workbench.util import read_json, slug, timestamp, write_json

# The command runs as "$@" inside a tiny bash wrapper whose only job is to record the
# exit code: the command itself is exactly what a person would type.
_WRAPPER = (
    '"$@"; rc=$?; '
    'printf "%s" "$rc" > "$WB_JOB_DIR/exit_code"; '
    'printf "\\n[workbench] %s  finished, exit code %s\\n" "$(date "+%F %T")" "$rc"; '
    'exit $rc'
)

_children: dict[str, subprocess.Popen] = {}   # this session's launches, reaped on poll


@dataclass
class Job:
    id: str
    title: str
    kind: str                      # train / eval / detect / prepare / baseline / analysis ...
    cmd: list[str]
    env: dict = field(default_factory=dict)
    gpu: str | None = None
    created: float = 0.0
    pid: int | None = None
    start_ticks: int | None = None
    links: dict = field(default_factory=dict)   # model, run, survey, detection ...

    # ── Where things are ──
    @property
    def dir(self) -> str:
        return os.path.join(paths.JOBS_DIR, self.id)

    @property
    def log_path(self) -> str:
        return os.path.join(self.dir, "log.txt")

    def command_line(self) -> str:
        """The command as a person would type it, environment included."""
        env = [f"{k}={shlex.quote(v)}" for k, v in self.env.items()]
        if self.gpu is not None:
            env.insert(0, f"CUDA_VISIBLE_DEVICES={self.gpu}")
        return " ".join(env + [shlex.join(self.cmd)])

    # ── Status ──
    def exit_code(self) -> int | None:
        try:
            with open(os.path.join(self.dir, "exit_code")) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    def cancelled(self) -> bool:
        return os.path.exists(os.path.join(self.dir, "cancelled"))

    def alive(self) -> bool:
        if self.id in _children and _children[self.id].poll() is not None:
            return False
        if not self.pid:
            return False
        try:
            with open(f"/proc/{self.pid}/stat") as f:
                stat = f.read().rsplit(")", 1)[1].split()
        except OSError:
            return False
        if stat[0] == "Z":
            return False
        return self.start_ticks is None or int(stat[19]) == self.start_ticks

    def status(self) -> str:
        if self.cancelled():
            return "cancelled"
        code = self.exit_code()
        if code is not None:
            return "succeeded" if code == 0 else "failed"
        return "running" if self.alive() else "stopped"

    def finished_at(self) -> float | None:
        for name in ("exit_code", "cancelled"):
            p = os.path.join(self.dir, name)
            if os.path.exists(p):
                return os.path.getmtime(p)
        if self.status() == "stopped" and os.path.exists(self.log_path):
            return os.path.getmtime(self.log_path)
        return None

    def duration(self) -> float:
        return (self.finished_at() or time.time()) - self.created

    def tail(self, n: int = 40) -> list[str]:
        """Last n lines of the log, with progress-bar carriage returns resolved."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 256 * 1024))
                data = f.read().decode(errors="replace")
        except OSError:
            return []
        lines = [line.rsplit("\r", 1)[-1] for line in data.replace("\r\n", "\n").split("\n")]
        while lines and not lines[-1].strip():
            lines.pop()
        return lines[-n:]

    def last_line(self) -> str:
        for line in reversed(self.tail(15)):
            if line.strip() and not line.startswith("[workbench]"):
                return line.strip()
        return ""

    def save(self) -> None:
        write_json(os.path.join(self.dir, "job.json"), asdict(self))


def _proc_start_ticks(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return int(f.read().rsplit(")", 1)[1].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def launch(title: str, kind: str, cmd: list[str], env: dict | None = None,
           gpu: str | None = None, links: dict | None = None) -> Job:
    """Start `cmd` from the repository root, detached, logging to the job's log.txt.

    `gpu` is a CUDA device index (as a string) or "cpu"; it becomes CUDA_VISIBLE_DEVICES,
    so every framework sees exactly one card and calls it cuda:0.
    """
    paths.ensure_state_dir()
    job_id = f"{timestamp()}_{slug(kind)}"
    n = 1
    while os.path.exists(os.path.join(paths.JOBS_DIR, job_id)):
        n += 1
        job_id = f"{timestamp()}_{slug(kind)}-{n}"

    job = Job(id=job_id, title=title, kind=kind, cmd=list(cmd), env=dict(env or {}),
              gpu=None if gpu in (None, "auto") else str(gpu), created=time.time(),
              links=dict(links or {}))
    os.makedirs(job.dir, exist_ok=True)

    run_env = os.environ.copy()
    run_env.update(job.env)
    run_env.update({"PYTHONUNBUFFERED": "1", "WB_JOB_ID": job.id, "WB_JOB_DIR": job.dir})
    if job.gpu is not None:
        run_env["CUDA_VISIBLE_DEVICES"] = "" if job.gpu == "cpu" else job.gpu

    with open(job.log_path, "ab") as log:
        header = (f"[workbench] {time.strftime('%F %T')}  {title}\n"
                  f"[workbench] {job.command_line()}\n\n")
        log.write(header.encode())
        log.flush()
        proc = subprocess.Popen(
            ["bash", "-c", _WRAPPER, "wb-job", *job.cmd],
            cwd=paths.ROOT, env=run_env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    _children[job.id] = proc
    job.pid = proc.pid
    job.start_ticks = _proc_start_ticks(proc.pid)
    job.save()
    return job


def python_cmd(*args: str) -> list[str]:
    """A command run with the workbench's own interpreter (the `waterfowl` env)."""
    return [sys.executable, *args]


def load(job_id: str) -> Job | None:
    payload = read_json(os.path.join(paths.JOBS_DIR, job_id, "job.json"))
    if not payload:
        return None
    known = {f for f in Job.__dataclass_fields__}
    return Job(**{k: v for k, v in payload.items() if k in known})


def all_jobs() -> list[Job]:
    if not os.path.isdir(paths.JOBS_DIR):
        return []
    found = [load(e.name) for e in os.scandir(paths.JOBS_DIR) if e.is_dir()]
    return sorted((j for j in found if j), key=lambda j: j.created, reverse=True)


def running() -> list[Job]:
    return [j for j in all_jobs() if j.status() == "running"]


def cancel(job: Job, grace: float = 15.0) -> bool:
    """Stop a job and everything it started (its whole process group)."""
    if job.status() != "running" or not job.pid:
        return False
    with open(os.path.join(job.dir, "cancelled"), "w") as f:
        f.write(timestamp() + "\n")
    try:
        os.killpg(job.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.time() + grace
    while time.time() < deadline and job.alive():
        time.sleep(0.25)
    if job.alive():
        try:
            os.killpg(job.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    with open(job.log_path, "a") as f:
        f.write(f"\n[workbench] {time.strftime('%F %T')}  cancelled from the workbench\n")
    return True


def forget(job: Job) -> None:
    """Delete a finished job's record and log."""
    if job.status() == "running":
        raise RuntimeError("cannot remove a running job — cancel it first")
    import shutil
    shutil.rmtree(job.dir, ignore_errors=True)
