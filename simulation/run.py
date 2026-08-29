"""
run.py — runs OpenFAST on this machine's assigned cases.

Skips cases that already finished (so an interrupted overnight batch just
picks up where it left off next time), refuses to touch a case built by the
other machine, respects an optional overnight deadline, and logs one JSON line
per finished case to this machine's own manifest file.

Run it directly:
  python run.py --dry-run
  python run.py --stop-at 07:00
  python run.py --only TS07 --workers 1
"""

import argparse
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import config
import pool


# A case is finished once its .outb exists and is newer than its .fst — this
# is the whole resume mechanism: re-running this script later just skips these.
def is_finished(case_dir: Path) -> bool:
    fst = case_dir / config.FST_FILE
    outb = case_dir / config.FST_FILE.replace(".fst", ".outb")
    return fst.exists() and outb.exists() and outb.stat().st_mtime > fst.stat().st_mtime


# A case counts as done even if its local folder was deleted after merging to
# the drive (e.g. to free space when the drive has moved to the other
# machine) — checked against the persistent local staging manifest, which is
# never itself deleted, unlike the case folder it was originally written
# alongside. This is what actually prevents a wasted rebuild+rerun once the
# drive isn't attached and run_root() falls back to _staging/.
def already_completed(case_id: str) -> bool:
    manifest_path = config.PROJECT_DIR / "_staging" / config.CAMPAIGN_NAME / f"manifest_{config.WORKER}.jsonl"
    if not manifest_path.exists():
        return False
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("case_id") == case_id and rec.get("rc") == 0:
            return True
    return False


# Refuses to run a case that owner.json says belongs to the other machine.
def check_owner(case_dir: Path) -> None:
    owner_path = case_dir / "owner.json"
    if not owner_path.exists():
        raise RuntimeError(f"{case_dir} has no owner.json — build it first with build.py")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner["machine"] != config.WORKER:
        raise RuntimeError(
            f"{case_dir} is owned by {owner['machine']}, not {config.WORKER} — refusing to run it"
        )


# Builds a can_start() callback from --stop-at or --max-hours. Returns False
# once starting one more case (at this machine's flat EST_HOURS_PER_CASE
# estimate) wouldn't finish before the deadline. Returns None (= no limit) if
# neither argument was given. Never causes a running case to be killed.
def make_deadline(stop_at: str = None, max_hours: float = None):
    if stop_at is not None:
        hh, mm = (int(x) for x in stop_at.split(":"))
        now = datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline <= now:
            deadline += timedelta(days=1)
    elif max_hours is not None:
        deadline = datetime.now() + timedelta(hours=max_hours)
    else:
        return None

    def can_start(job=None) -> bool:
        return datetime.now() + timedelta(hours=config.est_hours_per_case()) <= deadline

    return can_start


# Appends one JSON line to this machine's own manifest file. Each machine writes
# only its own file, so nothing ever needs merging across machines.
def append_manifest(rec: dict) -> None:
    root, _ = config.run_root()
    manifest_path = root / f"manifest_{config.WORKER}.jsonl"
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# Short git commit hash of the project, recorded per run for thesis provenance.
def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(config.PROJECT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                         help=f"override the worker count (default {config.CORES} "
                              f"on {config.WORKER}); can go above or below it")
    parser.add_argument("--only", help="run only this case id (must belong to this machine)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-at", help="stop launching new cases past this time, e.g. 07:00")
    parser.add_argument("--max-hours", type=float, help="stop launching new cases after this many hours")
    args = parser.parse_args()

    cases = config.my_cases()
    if args.only:
        cases = [c for c in cases if c.case_id == args.only]
        if not cases:
            raise SystemExit(f"{args.only} is not assigned to {config.WORKER}")

    todo = []
    for case in cases:
        case_dir = config.case_dir(case)
        if is_finished(case_dir):
            print(f"[skip] {case.case_id} already finished")
            continue
        if already_completed(case.case_id):
            print(f"[skip] {case.case_id} already completed (merged, local copy pruned)")
            continue
        check_owner(case_dir)
        todo.append(case)

    print(f"\n{len(todo)} case(s) to run on {config.WORKER}")
    for case in todo:
        print(f"  {case.case_id}  ({config.condition_folder(case)}/{config.seed_folder(case)})")

    deadline = make_deadline(args.stop_at, args.max_hours)
    if deadline is not None:
        allowed_now = deadline()
        print(f"Deadline check: starting a case right now is "
              f"{'ALLOWED' if allowed_now else 'BLOCKED'} (estimate {config.est_hours_per_case()}h/case)")

    if not todo:
        return
    if args.dry_run:
        return

    max_workers = min(len(todo), args.workers or config.CORES)
    print(f"max_workers={max_workers}"
          f"{' (override, default is ' + str(config.CORES) + ')' if args.workers else ''}")

    jobs = [
        pool.Job(
            job_id=case.case_id,
            argv=[str(config.OPENFAST_EXE), config.FST_FILE],
            cwd=config.case_dir(case),
            log_path=config.case_dir(case) / "openfast.log",
        )
        for case in todo
    ]

    code_sha = _git_sha()
    results = pool.run_pool(jobs, max_workers, can_start=deadline)

    for case in todo:
        if case.case_id not in results:
            continue
        r = results[case.case_id]
        case_dir = config.case_dir(case)
        outb = case_dir / config.FST_FILE.replace(".fst", ".outb")
        append_manifest({
            "case_id": case.case_id,
            "machine": config.WORKER,
            "rc": r["rc"],
            "wall_s": r["wall_s"],
            "outb_bytes": outb.stat().st_size if outb.exists() else None,
            "code_sha": code_sha,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        })

    print("\nSummary:")
    for case in todo:
        status = results.get(case.case_id, {"rc": "NOT STARTED (deadline)"})
        print(f"  {case.case_id}: {status}")


if __name__ == "__main__":
    main()
