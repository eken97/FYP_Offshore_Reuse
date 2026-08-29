"""
pool.py — runs at most N processes at once, refilling a slot as each finishes.

Generic: knows nothing about OpenFAST, cases, or machines. Takes a list of jobs
(each just an argv + working directory + log file) and a worker cap, and never
lets more than `max_workers` run at the same time.
"""

import ctypes
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Job:
    job_id: str
    argv: list
    cwd: Path
    log_path: Path


_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


# Stops Windows from sleeping the machine while jobs are running.
def keep_awake_on() -> None:
    # No-op off Windows: SetThreadExecutionState is a Win32 call, and the
    # pipeline itself is platform-independent.
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
    )


# Releases the keep-awake request.
def keep_awake_off() -> None:
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)


# Runs `jobs` with at most `max_workers` running at once, refilling a slot as
# each finishes. `can_start(job)` is asked before every NEW launch (not for jobs
# already running) — return False from it to stop launching new jobs while
# letting already-running ones finish; this is the overnight-deadline hook. It
# receives the specific job under consideration (not just called generically),
# so e.g. a job not yet built by a concurrent builder thread can be skipped
# without blocking every job behind it — the launch loop scans `pending` for
# the first job that's currently launchable rather than only checking the head.
#
# `wait_for_more()` is only consulted when nothing is running AND nothing in
# `pending` is currently launchable — return True to idle and re-check rather
# than exit, e.g. while a producer thread is still building more jobs. Return
# False (or pass None, the default) to exit immediately, reporting whatever is
# left in `pending` as unstarted — this is the original single-shot behavior,
# unchanged when `wait_for_more` isn't passed.
#
# Returns {job_id: {"rc": int, "wall_s": float}} for every job that was
# actually started — a job never launched because can_start() said no is
# simply absent from the result.
def run_pool(jobs, max_workers: int, poll_s: float = 10, can_start=None, wait_for_more=None) -> dict:
    pending = list(jobs)
    running = {}
    results = {}

    keep_awake_on()
    try:
        while True:
            while len(running) < max_workers:
                idx = next(
                    (i for i, j in enumerate(pending) if can_start is None or can_start(j)),
                    None,
                )
                if idx is None:
                    break
                job = pending.pop(idx)
                log_file = open(job.log_path, "w", encoding="utf-8", errors="replace")
                proc = subprocess.Popen(
                    job.argv, cwd=str(job.cwd), stdout=log_file, stderr=subprocess.STDOUT
                )
                running[job.job_id] = {"proc": proc, "log_file": log_file, "start": time.time()}
                print(f"[launch] {job.job_id}  (pid {proc.pid})")

            if not running:
                if pending and wait_for_more is not None and wait_for_more():
                    time.sleep(poll_s)
                    continue
                if pending:
                    print(f"[stop] {len(pending)} case(s) left unstarted")
                break

            time.sleep(poll_s)

            finished_ids = [jid for jid, r in running.items() if r["proc"].poll() is not None]
            for jid in finished_ids:
                r = running.pop(jid)
                r["log_file"].close()
                wall_s = time.time() - r["start"]
                rc = r["proc"].returncode
                results[jid] = {"rc": rc, "wall_s": wall_s}
                print(f"[done]   {jid}  rc={rc}  wall={wall_s:.1f}s")

            if running or pending:
                print(f"  running={list(running.keys())}  queued={len(pending)}  done={len(results)}/{len(jobs)}")
    except KeyboardInterrupt:
        print("\n[interrupt] terminating running jobs...")
        for r in running.values():
            r["proc"].terminate()
        for r in running.values():
            try:
                r["proc"].wait(timeout=30)
            finally:
                r["log_file"].close()
        raise
    finally:
        keep_awake_off()

    return results
