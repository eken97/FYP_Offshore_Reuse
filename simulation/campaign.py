"""
campaign.py — interleaved build+run: builds this machine's next cases in a
background thread while already-built cases run through pool.run_pool.

Exists because build-all-then-run-all (build.py then run.py) pays TurbSim's
serial cost once per BATCH, not once per campaign — fine for a single batch
(the 8-case test), but ~52 batches deep into the real 414-run campaign that
adds up to ~13 hours of dead time in front of compute, per machine. Since a
build (~2-3 min) is trivial next to a run (~2.7-4.4h), a single builder
thread run ahead of the execution pool hides all but the very first batch's
build time.

Resume works exactly like run.py/build.py: re-running this script skips
already-finished cases (run.is_finished) and already-built ones
(build.is_built) — nothing is tracked separately.

Run it directly:
  python campaign.py --dry-run
  python campaign.py --stop-at 07:00
  python campaign.py --smoke --workers 1
"""

import argparse
import threading
from datetime import datetime, timezone

import build
import config
import pool
import run as run_mod


# Deletes every --force'd case's directory, synchronously, before the builder
# thread or the pool starts. This matters specifically because of a race that
# --force otherwise opens up: is_built() can read True from a STALE case.json
# left over from an earlier session (the exact situation --force exists to
# override), which the runner trusts immediately at t=0 — before the builder
# thread has done anything. If the builder then tore that same directory down
# mid-rebuild while the runner had already launched OpenFAST against the old
# files, OpenFAST loses the race and fails ("input file ... was not found").
# Wiping everything up front, before either thread starts, guarantees
# is_built() reads False for every --force target from the very first check —
# the runner can then never observe a stale "built" case that's about to be
# deleted out from under it. (Found by hitting exactly this failure during
# the first --force smoke test, 30.07.2026 — TS08's .fst vanished mid-launch.)
def clean_for_force(cases) -> None:
    for case in cases:
        case_dir = config.case_dir(case)
        if case_dir.exists():
            config.rmtree_retry(case_dir)
            print(f"[force] wiped {case.case_id}")


# Builds this machine's cases, in assignment order, one at a time — TurbSim is
# fast enough (~2-3 min/case) that a single thread easily stays ahead of the
# execution pool; running it concurrently would just steal cores from
# whatever's already executing. Skips a case that's already built; force-
# rebuilds one that's only partially built (folder exists, no case.json —
# e.g. a previous run was killed mid-TurbSim, or clean_for_force() above just
# ran). A build failure is logged and skipped rather than raised, so one bad
# case doesn't stall every case behind it — it simply never becomes "ready"
# and run_pool reports it as unstarted.
def build_worker(cases, tmax: float, run_turbsim_flag: bool, stop_event: threading.Event) -> None:
    pool.keep_awake_on()
    try:
        for case in cases:
            if stop_event.is_set():
                return
            case_dir = config.case_dir(case)
            if build.is_built(case_dir):
                continue
            force = case_dir.exists()
            try:
                build.build_case(case, tmax, force=force, run_turbsim_flag=run_turbsim_flag)
                print(f"[built]  {case.case_id}")
            except Exception as exc:
                print(f"[build FAILED] {case.case_id}: {exc}")
    finally:
        pool.keep_awake_off()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None,
                         help=f"override the worker count (default {config.CORES} "
                              f"on {config.WORKER}); can go above or below it")
    parser.add_argument("--only", help="run only this case id (must belong to this machine)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true",
                         help=f"use TMax={config.SMOKE_TMAX}s instead of {config.TMAX}s")
    parser.add_argument("--no-turbsim", action="store_true",
                         help="skip running TurbSim.exe (still writes wind.inp)")
    parser.add_argument("--force", action="store_true",
                         help="rebuild+rerun even cases already finished or already built")
    parser.add_argument("--stop-at", help="stop launching new cases past this time, e.g. 07:00")
    parser.add_argument("--max-hours", type=float, help="stop launching new cases after this many hours")
    args = parser.parse_args()

    tmax = config.SMOKE_TMAX if args.smoke else config.TMAX
    cases = config.my_cases()
    if args.only:
        cases = [c for c in cases if c.case_id == args.only]
        if not cases:
            raise SystemExit(f"{args.only} is not assigned to {config.WORKER}")

    todo = []
    for case in cases:
        case_dir = config.case_dir(case)
        if not args.force and run_mod.is_finished(case_dir):
            print(f"[skip] {case.case_id} already finished")
        elif not args.force and run_mod.already_completed(case.case_id):
            print(f"[skip] {case.case_id} already completed (merged, local copy pruned)")
        else:
            todo.append(case)

    print(f"\n{len(todo)} case(s) to build+run on {config.WORKER}")
    for case in todo:
        print(f"  {case.case_id}  ({config.condition_folder(case)}/{config.seed_folder(case)})")

    deadline = run_mod.make_deadline(args.stop_at, args.max_hours)
    if deadline is not None:
        print(f"Deadline check: starting a case right now is "
              f"{'ALLOWED' if deadline() else 'BLOCKED'} (estimate {config.est_hours_per_case()}h/case)")

    if not todo:
        return
    if args.dry_run:
        return

    if args.force:
        clean_for_force(todo)

    max_workers = min(len(todo), args.workers or config.CORES)
    print(f"max_workers={max_workers}"
          f"{' (override, default is ' + str(config.CORES) + ')' if args.workers else ''}")

    stop_event = threading.Event()
    builder = threading.Thread(
        target=build_worker, args=(todo, tmax, not args.no_turbsim, stop_event), daemon=True
    )
    builder.start()

    case_by_id = {c.case_id: c for c in todo}
    # Cases that failed an owner check are remembered so they aren't re-checked
    # (and re-logged) on every ~10s poll — should never actually trigger here,
    # since build_worker only ever builds config.my_cases(), but kept as the
    # same defensive check run.py does before handing a case to the pool.
    _owner_bad: set[str] = set()

    def can_start_job(job: pool.Job) -> bool:
        if deadline is not None and not deadline():
            return False
        case_dir = config.case_dir(case_by_id[job.job_id])
        if not build.is_built(case_dir):
            return False
        if job.job_id in _owner_bad:
            return False
        try:
            run_mod.check_owner(case_dir)
        except RuntimeError as exc:
            print(f"[owner FAILED] {job.job_id}: {exc}")
            _owner_bad.add(job.job_id)
            return False
        return True

    def wait_for_more() -> bool:
        if deadline is not None and not deadline():
            return False
        return builder.is_alive()

    jobs = [
        pool.Job(
            job_id=case.case_id,
            argv=[str(config.OPENFAST_EXE), config.FST_FILE],
            cwd=config.case_dir(case),
            log_path=config.case_dir(case) / "openfast.log",
        )
        for case in todo
    ]

    code_sha = run_mod._git_sha()
    try:
        results = pool.run_pool(jobs, max_workers, can_start=can_start_job, wait_for_more=wait_for_more)
    finally:
        stop_event.set()

    for case in todo:
        if case.case_id not in results:
            continue
        r = results[case.case_id]
        case_dir = config.case_dir(case)
        outb = case_dir / config.FST_FILE.replace(".fst", ".outb")
        run_mod.append_manifest({
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
        status = results.get(case.case_id, {"rc": "NOT STARTED (deadline/build/owner)"})
        print(f"  {case.case_id}: {status}")


if __name__ == "__main__":
    main()
