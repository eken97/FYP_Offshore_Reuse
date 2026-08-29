"""
Step 10 -- Batch driver.

Walks a root folder, finds every <COND>/<SEED>/ run folder beneath it
(identified by owner.json presence, same convention Simulation/merge.py and
stage0_qa.py already use), and drives Stage 0/2/3 across all of them,
skipping runs whose Stage 2 output is already up to date
(stage2_histograms.process_run's own stamp check). This is the automation
wrapper around Steps 0-9, which each only know how to process ONE run or
condition folder at a time.

SELECT does NOT enumerate the 69 load cases by name -- it only holds a
handful of on/off switches (which stages to run, force) plus ONE root
folder path. discover_runs(root) finds however many run folders exist
underneath that path automatically, whether it's 6 (now, LC_V20_H3p5_T8)
or 414 (the full campaign) -- no code or config change needed as the
campaign grows. A hard-coded dict rather than argparse, to match every
other script in this pipeline's plain-constants style, and because "here's
the exact dict I ran" is a more auditable record for the thesis than a
remembered command-line invocation.

Do not run this while the real 414-run campaign is executing on any
machine -- Stage 2/3 write to disk and would contend for it (see
docs/decisions.md). Stage 0 (read-only) is fine at any time.
"""
import ctypes
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sd_geometry as sdg
import stage0_qa as s0
import stage2_corrosion as s2c
import stage2_histograms as s2
import stage2_joints as s2j
import stage2_joints_thickness as s2jt
import stage2_joints_thickness_corrosion as s2jtc
import stage3_damage as s3
import stage3_damage_corrosion as s3c
import stage3_joint_damage as s3j
import stage3_joint_damage_corrosion as s3jc
import stage3_joint_damage_thickness as s3jt
import stage3_joint_damage_thickness_corrosion as s3jtc
import joint_thickness_override as jto
import stress

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# Real, full-campaign deliverable CSVs (the ones this driver's own default
# out_path/results_dir args point to) live in their own subfolder -- keeps
# them separate from RESULTS_DIR's self-check scratch (_..._selfcheck/,
# already .gitignore'd) and from ad-hoc dev-fixture test output, added
# 15.08.2026 once results/ started accumulating both kinds side by side.
REAL_RESULTS_DIR = RESULTS_DIR / "real_campaign"
# Dev/test fixture data (LC_V20_H3p5_T8, _staging/) deliberately stays at
# its original location, not moved alongside the code -- see
# See docs/decisions.md, 10.08.2026 folder-reorg session.
DEV_FIXTURE_DIR = PROJECT / "data" / "example"
# endregion


# region --- keep-awake (Windows) ---
# Same technique as Simulation/pool.py's keep_awake_on()/off() and
# Validation/run_wave_period_study.py's own copy -- duplicated here rather
# than imported (matches this codebase's existing convention: each
# long-running driver carries its own copy rather than sharing one import
# across Simulation/Validation/Postprocessing). A full real-campaign run is
# ~4.0-4.6h single-core (measured 10.08.2026, ~35-40s/seed x 414 seeds) --
# without this, Windows sleeping mid-run would pause Stage 2 for however
# long the machine was asleep, not lose data (skip-if-exists resumes
# cleanly) but silently stall an unattended overnight run.
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


def keep_awake_on() -> None:
    # No-op off Windows: SetThreadExecutionState is a Win32 call, and the
    # pipeline itself is platform-independent.
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(
        _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED | _ES_DISPLAY_REQUIRED
    )


def keep_awake_off() -> None:
    if os.name != "nt":
        return
    ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
# endregion


# region --- discovery ---
def discover_runs(root):
    """
    Every <COND>/<SEED>/ run folder under root, identified by owner.json
    presence -- the same convention Simulation/merge.py's
    discover_staged_cases() and stage0_qa.py's _discover_cases() already
    use. Sorted for deterministic, reproducible ordering across reruns.
    """
    return sorted(p.parent for p in Path(root).rglob("owner.json"))
# endregion


# region --- transient-write retry ---
def _with_retry(fn, *args, attempts=20, delay_s=3.0, **kwargs):
    """
    Retries a transient PermissionError on a write. Simulation/config.py's
    rmtree_retry() documents the same failure hit against BOTH the
    cloud-synced tree and the external drive (30.07.2026) -- not
    sync-provider-specific, more likely sync/indexing/AV momentarily locking a
    just-written file. Same 20 attempts / 3s (60s total) budget as that
    precedent. stage2_histograms.process_run's own os.replace() atomic
    rename is what makes a retry here safe: a retried call either sees the
    previous attempt's complete .tmp file still mid-flight (rare) or starts
    clean, never a half-written .npz.
    """
    for attempt in range(attempts):
        try:
            return fn(*args, **kwargs)
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)
# endregion


# region --- stage runners ---
def run_stage0(roots, out_path=REAL_RESULTS_DIR / "stage0_qa.csv"):
    """Read-only QA sweep. Safe to run at any time, even during the freeze."""
    rows = s0.qa_sweep(roots)
    s0.write_qa_csv(rows, out_path)
    print(f"  stage0: scanned {len(rows)} run(s) -> {out_path}")
    return rows


def _stage2_worker(case_dir, force):
    """
    Module-level (picklable) worker for ProcessPoolExecutor -- each worker
    process re-imports stage2_histograms fresh, so it independently re-runs
    find_drive() and re-reads its own .outb; no shared state between
    workers, matching Stage 2's existing atomic-rename write discipline
    (stage2_histograms.process_run's own os.replace()), which is what makes
    concurrent workers safe to write into the same stage2/ tree at all --
    two workers never touch the same seed's .npz.
    """
    rel = f"{case_dir.parent.name}/{case_dir.name}"
    try:
        npz_path = _with_retry(s2.process_run, case_dir, force=force)
        return rel, npz_path, None
    except Exception as exc:
        return rel, None, exc


def _stage2_corrosion_worker(case_dir, years, force):
    """Same contract as _stage2_worker, calling stage2_corrosion.process_run_corroded
    instead -- each worker independently re-runs stage2_corrosion.find_drive()."""
    rel = f"{case_dir.parent.name}/{case_dir.name}"
    try:
        npz_path = _with_retry(s2c.process_run_corroded, case_dir, years, force=force)
        return rel, npz_path, None
    except Exception as exc:
        return rel, None, exc


def _stage2_joints_worker(case_dir, force):
    """Same contract as _stage2_worker, calling stage2_joints.process_run instead --
    each worker independently re-runs stage2_joints.find_drive()."""
    rel = f"{case_dir.parent.name}/{case_dir.name}"
    try:
        npz_path = _with_retry(s2j.process_run, case_dir, force=force)
        return rel, npz_path, None
    except Exception as exc:
        return rel, None, exc


def _stage2_joints_thickness_worker(case_dir, scenario, force):
    """Same contract as _stage2_worker, calling
    stage2_joints_thickness.process_run_retrofit instead (can-thickness
    retrofit, no corrosion, ALL 120 connections) -- each worker
    independently re-runs stage2_joints_thickness.find_drive()."""
    rel = f"{case_dir.parent.name}/{case_dir.name}"
    try:
        npz_path = _with_retry(s2jt.process_run_retrofit, case_dir, scenario, force=force)
        return rel, npz_path, None
    except Exception as exc:
        return rel, None, exc


def _stage2_joints_thickness_corrosion_worker(case_dir, scenario, years, force):
    """Same contract as _stage2_worker, calling
    stage2_joints_thickness_corrosion.process_run_retrofit_corroded instead
    (retrofit THEN corrosion, splash-zone connections only) -- each worker
    independently re-runs stage2_joints_thickness_corrosion.find_drive()."""
    rel = f"{case_dir.parent.name}/{case_dir.name}"
    try:
        npz_path = _with_retry(s2jtc.process_run_retrofit_corroded, case_dir, scenario, years, force=force)
        return rel, npz_path, None
    except Exception as exc:
        return rel, None, exc


def _fmt_duration(seconds):
    """H:MM:SS for progress lines -- seconds may be None (unknown yet, e.g.
    ETA before the first result lands) or inf (n_done==0)."""
    if seconds is None or not (seconds == seconds) or seconds == float("inf"):
        return "?"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _progress_prefix(n_done, total, start_time):
    """[done/total, pct%, elapsed H:MM:SS, ETA H:MM:SS] -- ETA is a plain
    linear extrapolation (elapsed / n_done * remaining), the simplest honest
    estimate given uneven per-run cost and, under workers>1, uneven
    completion order. Recomputed fresh each call, not smoothed/windowed --
    good enough for "how far along are we," not a precision instrument."""
    elapsed = time.time() - start_time
    pct = 100.0 * n_done / total if total else 0.0
    eta = (elapsed / n_done * (total - n_done)) if n_done else None
    return f"[{n_done}/{total}, {pct:5.1f}%, elapsed {_fmt_duration(elapsed)}, ETA {_fmt_duration(eta)}]"


def _run_stage2_generic(worker_fn, case_dirs, workers, *worker_args):
    """
    Shared sequential/parallel driver behind run_stage2/run_stage2_corrosion/
    run_stage2_joints -- worker_fn is one of the three module-level (picklable)
    workers above, worker_args are its extra positional args after case_dir
    (e.g. (force,) for the member/joint tracks, (years, force) for corrosion).
    One bad run (e.g. a corrupt .outb) must not stop the rest -- same posture
    as merge.py's per-case status collection: catch, report, keep going, and
    let the caller decide whether the failure count matters.

    Every completed run (OK or FAILED) prints a progress prefix -- see
    _progress_prefix() -- so an unattended multi-hour run (the real 414-run
    campaign, ~hours per track) has a visible "how far along" signal instead
    of just a scrolling list of filenames. Skip-if-cached runs (stamp
    already matches, force=False) still count as "done" here -- they return
    from worker_fn just as fast as a real compute, so the counter/ETA stay
    accurate either way.

    workers=1 keeps the original sequential path unchanged, byte-for-byte
    (aside from the added progress prefix). workers>1 fans the SAME per-run
    call out across a ProcessPoolExecutor -- each run's Stage 2 computation
    reads its own .outb and writes its own .npz, with no shared state
    between runs, so this is embarrassingly parallel in principle. In
    practice the external drive is the shared bottleneck (every worker
    reads/writes it at once), so don't expect linear speedup with worker
    count -- a 4.2x speedup measured on 4 cores against a local disk is
    a starting point, not a guarantee on 8 cores against real
    drive-hosted I/O.
    """
    total = len(case_dirs)
    start_time = time.time()
    n_ok, n_failed = 0, 0

    if workers <= 1:
        for case_dir in case_dirs:
            # Root-agnostic label (condition/seed folder names) -- case_dirs
            # can come from the dev fixture, the real campaign on D:, or
            # anywhere else discover_runs() was pointed at, so a single
            # fixed "relative to X" base would break (ValueError) the
            # moment root isn't under X.
            rel, npz_path, exc = worker_fn(case_dir, *worker_args)
            if exc is None:
                n_ok += 1
            else:
                n_failed += 1
            prefix = _progress_prefix(n_ok + n_failed, total, start_time)
            if exc is None:
                print(f"  {prefix} stage2 OK: {rel} -> {npz_path.name}")
            else:
                print(f"  {prefix} stage2 FAILED: {rel}: {exc}")
        return n_ok, n_failed

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker_fn, case_dir, *worker_args): case_dir for case_dir in case_dirs}
        for future in as_completed(futures):
            rel, npz_path, exc = future.result()
            if exc is None:
                n_ok += 1
            else:
                n_failed += 1
            prefix = _progress_prefix(n_ok + n_failed, total, start_time)
            if exc is None:
                print(f"  {prefix} stage2 OK: {rel} -> {npz_path.name}")
            else:
                print(f"  {prefix} stage2 FAILED: {rel}: {exc}")
    return n_ok, n_failed


def run_stage2(case_dirs, force=False, workers=1):
    """Member-track (uncorroded) Stage 2 over every discovered run folder.
    See _run_stage2_generic's docstring for the sequential/parallel contract."""
    return _run_stage2_generic(_stage2_worker, case_dirs, workers, force)


def run_stage2_corrosion(case_dirs, years, force=False, workers=1):
    """Corrosion-aware member-track Stage 2 (stage2_corrosion.py -- 32
    splash-zone members x 2 ends only, NOT all 112) over every discovered
    run folder. See _run_stage2_generic's docstring for the
    sequential/parallel contract."""
    return _run_stage2_generic(_stage2_corrosion_worker, case_dirs, workers, years, force)


def run_stage2_joints(case_dirs, force=False, workers=1):
    """Joint-track Stage 2 (stage2_joints.py -- 120 brace-to-chord
    connections) over every discovered run folder. Does NOT itself check
    stress.HOTSPOT_JOINT_VERIFIED -- that gate lives in main(), which prints
    a warning banner but does not block (soft review gate, see
    stage3_joint_damage.py's own module docstring). See
    _run_stage2_generic's docstring for the sequential/parallel contract."""
    return _run_stage2_generic(_stage2_joints_worker, case_dirs, workers, force)


def run_stage2_joints_thickness(case_dirs, scenario, force=False, workers=1):
    """Can-thickness-retrofit joint-track Stage 2 (stage2_joints_thickness.py
    -- ALL 120 connections, no corrosion) over every discovered run folder,
    under `scenario` ("A"/"B"). See _run_stage2_generic's docstring for the
    sequential/parallel contract."""
    return _run_stage2_generic(_stage2_joints_thickness_worker, case_dirs, workers, scenario, force)


def run_stage2_joints_thickness_corrosion(case_dirs, scenario, years, force=False, workers=1):
    """Retrofit+corrosion composed joint-track Stage 2
    (stage2_joints_thickness_corrosion.py -- splash-zone connections only,
    same scope as the corrosion-only track) over every discovered run
    folder, under `scenario`. See _run_stage2_generic's docstring for the
    sequential/parallel contract."""
    return _run_stage2_generic(_stage2_joints_thickness_corrosion_worker, case_dirs, workers,
                                scenario, years, force)


def run_stage3(category="B1", out_path=REAL_RESULTS_DIR / "member_track" / "stage3_damage.csv",
               matrix_raw_path=REAL_RESULTS_DIR / "member_track" / "member_damage_matrix_raw.csv",
               matrix_weighted_path=REAL_RESULTS_DIR / "member_track" / "member_damage_matrix_weighted.csv"):
    """
    Stage 3 aggregates over whatever condition folders currently exist
    under stage2/ (compute_stage3's own discovery, not this module's
    discover_runs) -- it is not scoped to a single call's case_dirs, since
    a full campaign run is expected to accumulate stage2/ across many
    separate invocations over time.

    Builds the per-bin aggregation ONCE (s3.build_per_bin) and feeds it to
    both compute_stage3's per-point/per-end 25yr life output AND the
    member x load-case overview matrix (added 10.08.2026) -- avoids reading
    the whole Stage 2 cache twice for what would otherwise be two separate
    compute_stage3()/compute_member_matrix() calls.
    """
    p_bin, raw_total = s3.load_bin_probabilities()
    bin_names = s3.load_bin_names()
    per_bin = s3.build_per_bin(s3.STAGE2_DIR, category)

    rows, meta = s3.stage3_from_per_bin(per_bin, p_bin, raw_total, category)
    df = s3.write_stage3_damage(rows, out_path)
    print(f"  stage3: {len(df)} point(s), bins available {meta['bins_available']} "
          f"of {meta['n_bins_total_campaign']} total campaign bins -> {out_path}")

    raw_rows, weighted_rows, bin_columns = s3.matrix_from_per_bin(per_bin, p_bin, bin_names)
    s3.write_member_matrix_csv(raw_rows, bin_columns, "worst_bin_damage", matrix_raw_path)
    s3.write_member_matrix_csv(weighted_rows, bin_columns, "worst_bin_contribution", matrix_weighted_path)
    print(f"  matrix: {len(raw_rows)} member(s) x {len(bin_columns)} bin(s) -> "
          f"{matrix_raw_path}, {matrix_weighted_path}")

    return df, meta


def run_stage3_corrosion(sd_path, years=(5, 10, 15, 20, 25), step_years=5.0,
                          out_path=REAL_RESULTS_DIR / "member_track" / "corrosion" / "stage3_damage_corrosion.csv"):
    """
    Corrosion Stage 3: aggregates every condition folder currently under
    stage2_corrosion.STAGE2_CORROSION_DIR (drive-routed), writes the
    summed-over-years deliverable CSV plus the confirmed 10 per-year-step
    member x load-case matrices (5 years x {raw, weighted} -- see
    stage3_damage_corrosion.write_year_step_matrices). sd_path is the
    campaign's SubDyn.dat (any one run's copy -- geometry is
    campaign-constant, same reasoning as stage3_joint_damage.py's own
    build_scf_lookup()).
    """
    p_bin, raw_total = s3.load_bin_probabilities()
    bin_names = s3.load_bin_names()
    model = sdg.read_subdyn_model(sd_path)
    per_bin = s3c.build_per_bin_corrosion(s2c.STAGE2_CORROSION_DIR, model)

    rows, meta = s3c.stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, step_years=step_years)
    df = s3c.write_stage3_damage_corrosion(rows, out_path)
    print(f"  stage3_corrosion: {len(df)} point(s), bins available {meta['bins_available']} "
          f"of {meta['n_bins_total_campaign']} total campaign bins -> {out_path}")

    s3c.write_year_step_matrices(per_bin, p_bin, bin_names, years, step_years=step_years,
                                  results_dir=REAL_RESULTS_DIR / "member_track" / "corrosion")
    print(f"  matrix_corrosion: {len(years)} year-step(s) x 2 metric(s) -> "
          f"{REAL_RESULTS_DIR / 'member_track' / 'corrosion'}")

    return df, meta


def run_stage3_joints(sd_path, sd_sum_path,
                       out_path=REAL_RESULTS_DIR / "joint_track" / "stage3_joint_damage.csv"):
    """
    Joint Stage 3: aggregates every condition folder currently under
    stage2_joints.STAGE2_JOINTS_DIR (drive-routed), writes the deliverable
    CSV plus the 184-row-per-metric connection x load-case matrices (raw +
    weighted -- see stage3_joint_damage.write_joint_matrices). sd_path/
    sd_sum_path are any one run's copies (geometry/SCF are
    campaign-constant, built once here rather than per run).
    """
    p_bin, raw_total = s3.load_bin_probabilities()
    bin_names = s3.load_bin_names()
    mudline_z = s3j.mudline_z_from_sd(sd_path)
    scf_lookup = s3j.build_scf_lookup(sd_path, sd_sum_path)
    per_bin = s3j.build_per_bin(s2j.STAGE2_JOINTS_DIR, scf_lookup, mudline_z)

    rows, meta = s3j.stage3_from_per_bin(per_bin, p_bin, raw_total)
    df = s3j.write_stage3_joint_damage(rows, out_path)
    print(f"  stage3_joints: {len(df)} connection-group(s), bins available {meta['bins_available']} "
          f"of {meta['n_bins_total_campaign']} total campaign bins -> {out_path}")

    s3j.write_joint_matrices(per_bin, p_bin, bin_names, results_dir=REAL_RESULTS_DIR / "joint_track")
    print(f"  matrix_joints: {len(rows)} connection-group(s) x 2 metric(s) -> "
          f"{REAL_RESULTS_DIR / 'joint_track'}")

    return df, meta


def run_stage3_joints_thickness(sd_path, sd_sum_path, scenario,
                                 out_path=None):
    """Can-thickness-retrofit joint Stage 3 (uncorroded, ALL 120
    connections): aggregates every condition folder currently under
    stage2_joints_thickness.STAGE2_JOINTS_THICKNESS_DIR/<scenario>, writes
    the deliverable CSV plus the 184-row-per-metric connection x load-case
    matrices (raw + weighted -- see stage3_joint_damage.matrix_from_per_bin_
    joints/write_joint_matrix_csv, reused unmodified since this scenario's
    per_bin has the identical group-key shape). sd_path/sd_sum_path are any
    one run's copies (geometry/SCF are campaign-constant)."""
    if out_path is None:
        out_path = REAL_RESULTS_DIR / "joint_track" / f"stage3_joint_damage_thickness_{scenario}.csv"
    stage2_root = s2jt.STAGE2_JOINTS_THICKNESS_DIR / scenario
    p_bin, raw_total = s3.load_bin_probabilities()
    bin_names = s3.load_bin_names()
    mudline_z = s3j.mudline_z_from_sd(sd_path)
    scf_lookup = s3jt.build_scf_lookup_retrofit(sd_path, sd_sum_path, scenario)
    per_bin = s3jt.build_per_bin_retrofit(stage2_root, scf_lookup, mudline_z)

    rows, meta = s3j.stage3_from_per_bin(per_bin, p_bin, raw_total)
    df = s3jt.write_stage3_joint_damage_thickness(rows, out_path)
    print(f"  stage3_joints_thickness[{scenario}]: {len(df)} connection-group(s), "
          f"bins available {meta['bins_available']} of {meta['n_bins_total_campaign']} "
          f"total campaign bins -> {out_path}")

    raw_rows, weighted_rows, bin_columns = s3j.matrix_from_per_bin_joints(per_bin, p_bin, bin_names)
    raw_path = REAL_RESULTS_DIR / "joint_track" / f"joint_damage_matrix_thickness_{scenario}_raw.csv"
    weighted_path = REAL_RESULTS_DIR / "joint_track" / f"joint_damage_matrix_thickness_{scenario}_weighted.csv"
    s3j.write_joint_matrix_csv(raw_rows, bin_columns, "worst_bin_damage", raw_path)
    s3j.write_joint_matrix_csv(weighted_rows, bin_columns, "worst_bin_contribution", weighted_path)
    print(f"  matrix_joints_thickness[{scenario}]: {len(rows)} connection-group(s) x 2 metric(s) "
          f"-> {REAL_RESULTS_DIR / 'joint_track'}")

    return df, meta


def run_stage3_joints_thickness_corrosion(sd_path, sd_sum_path, scenario,
                                           years=(5, 10, 15, 20, 25), step_years=5.0,
                                           out_path=None):
    """Retrofit+corrosion composed joint Stage 3: aggregates every condition
    folder currently under
    stage2_joints_thickness_corrosion.STAGE2_JOINTS_THICKNESS_CORROSION_DIR
    /<scenario>, writes the summed-over-years deliverable CSV plus, per year
    in `years`, the splash-only connection x load-case matrices (raw +
    weighted -- see stage3_joint_damage_corrosion.write_year_step_matrices_
    joints, reused unmodified via its `prefix` argument since this
    scenario's per_bin has the identical (connection-group, year) key
    shape). Each year's matrix is THAT 5-year block's own damage
    contribution only, not cumulative through that year -- same convention
    as the member-track corrosion matrices."""
    if out_path is None:
        out_path = REAL_RESULTS_DIR / "joint_track" / "corrosion" / f"stage3_joint_damage_thickness_corrosion_{scenario}.csv"
    stage2_root = s2jtc.STAGE2_JOINTS_THICKNESS_CORROSION_DIR / scenario
    p_bin, raw_total = s3.load_bin_probabilities()
    bin_names = s3.load_bin_names()
    splash, model, mudline_z, k_groups = s3jtc.build_geometry(sd_path, sd_sum_path)
    per_bin = s3jtc.build_per_bin_retrofit_corrosion(
        stage2_root, splash, k_groups, model, mudline_z, scenario)

    rows, meta = s3jc.stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, step_years=step_years)
    df = s3jtc.write_stage3_joint_damage_thickness_corrosion(rows, out_path)
    print(f"  stage3_joints_thickness_corrosion[{scenario}]: {len(df)} connection-group(s), "
          f"bins available {meta['bins_available']} of {meta['n_bins_total_campaign']} "
          f"total campaign bins -> {out_path}")

    out_paths = s3jc.write_year_step_matrices_joints(
        per_bin, p_bin, bin_names, years, step_years=step_years,
        results_dir=REAL_RESULTS_DIR / "joint_track" / "corrosion",
        prefix=f"joint_damage_matrix_thickness_corrosion_{scenario}")
    print(f"  matrix_joints_thickness_corrosion[{scenario}]: {len(years)} year-step(s) x 2 metric(s) "
          f"-> {REAL_RESULTS_DIR / 'joint_track' / 'corrosion'}")

    return df, meta
# endregion


# region --- driver ---
# The only thing you edit to change what a run of this script does. Not a
# by-load-case switchboard -- see module docstring.
SELECT = dict(
    # Campaign run-data root: the folder holding the per-condition run
    # folders and their .outb files. Point OC4_RUN_ROOT at yours.
    root=Path(os.environ.get("OC4_RUN_ROOT", "runs")),
    run_stage0=False,
    # Member-uncorroded, member-corrosion, and joint-uncorroded all already
    # ran to completion in the real campaign runs (see docs/decisions.md) -- OFF by
    # default now so this
    # invocation doesn't redundantly redo ~4-10h of already-complete work.
    # Flip back to True for a genuine full-campaign rerun.
    run_stage2=False,
    run_stage3=False,
    run_stage2_corrosion=False,
    run_stage3_corrosion=False,
    run_stage2_joints=False,
    run_stage3_joints=False,
    # Can-thickness retrofit tracks, both scenarios (see
    # joint_thickness_override.py's module docstring): scenario A is the
    # the author's own read of UpWind D4.2.5 Fig 3-3 (X top/bottom +10mm, X mid
    # +5mm, K/Y +5mm both sides, bottom-K thick-scenario only); scenario B
    # is a flat +12mm both sides everywhere, same bottom-K restriction.
    # Each runs twice: uncorroded (all 120 connections) + corrosion
    # composed on top (splash-zone connections only, 25yr horizon).
    thickness_scenarios=["A", "B"],
    run_stage2_joints_thickness=True,
    run_stage3_joints_thickness=True,
    run_stage2_joints_thickness_corrosion=True,
    run_stage3_joints_thickness_corrosion=True,
    force=False,
    category="B1",
    # 1 = original sequential behavior. >1 fans Stage 2 out across a
    # ProcessPoolExecutor -- see _run_stage2_generic's docstring for why
    # this won't scale linearly (shared external-drive I/O). Hardcoded per
    # this project's own "auditable exact dict" convention, not
    # auto-detected: the real campaign ran on an 8-physical-core machine
    # with 7 workers, leaving one core free for the OS during unattended
    # overnight runs. Set this to suit your own machine. A 4.2x speedup
    # measured on 4 cores against a local disk does NOT necessarily
    # transfer 1:1 to 8 cores against real drive-hosted I/O, so re-time a
    # small real-drive batch before trusting it for a full run. Applies to
    # all three Stage 2 tracks.
    workers=7,
    # 25yr horizon, confirmed by the author 15.08.2026 -- extending to 50yr
    # is a one-line change here (see docs/decisions.md), not a
    # rebuild.
    corrosion_years=[5, 10, 15, 20, 25],
    corrosion_step_years=5.0,
)


def main(select=SELECT):
    root = Path(select["root"])
    case_dirs = discover_runs(root)
    print(f"discovered {len(case_dirs)} run folder(s) under {root}")
    assert case_dirs, f"no run folders (owner.json found) under {root}"

    # Geometry is campaign-constant (same SubDyn.dat/SD.sum.yaml every run)
    # -- any one run folder's copy is representative, same assumption
    # stage3_joint_damage.py's own build_scf_lookup() already makes.
    sd_path = case_dirs[0] / s2.SD_NAME
    sd_sum_candidates = list(case_dirs[0].glob("*.SD.sum.yaml"))

    workers = select.get("workers", 1)
    force = select.get("force", False)

    keep_awake_on()
    try:
        if select.get("run_stage0"):
            run_stage0([root])

        if select.get("run_stage2"):
            n_ok, n_failed = run_stage2(case_dirs, force=force, workers=workers)
            print(f"stage2 done: {n_ok} ok, {n_failed} failed")
            if n_failed:
                print("WARNING: some runs failed stage2 -- stage3 (if enabled) will proceed "
                      "on whatever stage2/ data currently exists, which may be incomplete.")

        if select.get("run_stage3"):
            run_stage3(category=select.get("category", "B1"))

        if select.get("run_stage2_corrosion"):
            years = select.get("corrosion_years", [5, 10, 15, 20, 25])
            n_ok, n_failed = run_stage2_corrosion(case_dirs, years, force=force, workers=workers)
            print(f"stage2_corrosion done: {n_ok} ok, {n_failed} failed")
            if n_failed:
                print("WARNING: some runs failed stage2_corrosion -- stage3_corrosion (if enabled) "
                      "will proceed on whatever data currently exists, which may be incomplete.")

        if select.get("run_stage3_corrosion"):
            run_stage3_corrosion(sd_path, years=select.get("corrosion_years", [5, 10, 15, 20, 25]),
                                  step_years=select.get("corrosion_step_years", 5.0))

        if select.get("run_stage2_joints") or select.get("run_stage3_joints"):
            if not stress.HOTSPOT_JOINT_VERIFIED:
                print("WARNING: stress.HOTSPOT_JOINT_VERIFIED is False -- joint track results "
                      "are build/plumbing-only, not independently reviewed. Proceeding anyway "
                      "(soft review gate, see stage3_joint_damage.py's own module docstring).")

        if select.get("run_stage2_joints"):
            n_ok, n_failed = run_stage2_joints(case_dirs, force=force, workers=workers)
            print(f"stage2_joints done: {n_ok} ok, {n_failed} failed")
            if n_failed:
                print("WARNING: some runs failed stage2_joints -- stage3_joints (if enabled) "
                      "will proceed on whatever data currently exists, which may be incomplete.")

        if select.get("run_stage3_joints"):
            assert sd_sum_candidates, f"no *.SD.sum.yaml found in {case_dirs[0]}"
            run_stage3_joints(sd_path, sd_sum_candidates[0])

        thickness_stages_selected = (
            select.get("run_stage2_joints_thickness") or select.get("run_stage3_joints_thickness")
            or select.get("run_stage2_joints_thickness_corrosion")
            or select.get("run_stage3_joints_thickness_corrosion")
        )
        if thickness_stages_selected and not stress.HOTSPOT_JOINT_VERIFIED:
            print("WARNING: stress.HOTSPOT_JOINT_VERIFIED is False -- joint-thickness-retrofit "
                  "results are build/plumbing-only, not independently reviewed. Proceeding "
                  "anyway (soft review gate, same as the uncorroded joint track).")

        scenarios = select.get("thickness_scenarios", list(jto.SCENARIOS))
        years = select.get("corrosion_years", [5, 10, 15, 20, 25])
        step_years = select.get("corrosion_step_years", 5.0)

        for scenario in scenarios:
            if select.get("run_stage2_joints_thickness"):
                n_ok, n_failed = run_stage2_joints_thickness(case_dirs, scenario, force=force, workers=workers)
                print(f"stage2_joints_thickness[{scenario}] done: {n_ok} ok, {n_failed} failed")
                if n_failed:
                    print(f"WARNING: some runs failed stage2_joints_thickness[{scenario}] -- "
                          f"stage3 (if enabled) will proceed on whatever data currently exists.")

            if select.get("run_stage3_joints_thickness"):
                assert sd_sum_candidates, f"no *.SD.sum.yaml found in {case_dirs[0]}"
                run_stage3_joints_thickness(sd_path, sd_sum_candidates[0], scenario)

            if select.get("run_stage2_joints_thickness_corrosion"):
                n_ok, n_failed = run_stage2_joints_thickness_corrosion(
                    case_dirs, scenario, years, force=force, workers=workers)
                print(f"stage2_joints_thickness_corrosion[{scenario}] done: {n_ok} ok, {n_failed} failed")
                if n_failed:
                    print(f"WARNING: some runs failed stage2_joints_thickness_corrosion[{scenario}] -- "
                          f"stage3 (if enabled) will proceed on whatever data currently exists.")

            if select.get("run_stage3_joints_thickness_corrosion"):
                assert sd_sum_candidates, f"no *.SD.sum.yaml found in {case_dirs[0]}"
                run_stage3_joints_thickness_corrosion(
                    sd_path, sd_sum_candidates[0], scenario, years=years, step_years=step_years)
    finally:
        keep_awake_off()


if __name__ == "__main__":
    main()
# endregion
