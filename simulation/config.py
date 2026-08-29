"""
config.py — settings and decisions for the real 414-run OC4 K13 metocean campaign.

This file makes no OpenFAST run happen and edits no OpenFAST input file. It only
answers questions that every other script in this folder needs answered:
  - Which machine is this?
  - Where do project files and run data live?
  - What are the campaign's cases, and which machine runs which?

Run it directly (`python config.py`) to print a readiness report.
"""

import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import NamedTuple

import pandas as pd

# ---------------------------------------------------------------------------
# Where things are.
#
# Everything is resolved from the repository root or from environment
# variables, so a clone works on any machine without editing this file.
# See docs/setup.md for the variables and how to set them.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

# Root the campaign writes under. Defaults to the repo itself.
PROJECT = Path(os.environ.get("OC4_PROJECT_ROOT", REPO_ROOT))
PROJECT_DIR = PROJECT / "simulation"

# A name for this machine, used only to look up its share of the work in
# assignment.json when the campaign is split across several machines.
WORKER = os.environ.get("OC4_WORKER", socket.gethostname())

# Concurrent OpenFAST processes. One core left for the OS by default.
CORES = int(os.environ.get("OC4_CORES", max(1, (os.cpu_count() or 2) - 1)))

# ---------------------------------------------------------------------------
# Executables and template files.
#
# These are NOT resolved strictly at import time -- importing this module must
# succeed on a machine that has no OpenFAST installed, so that the
# post-processing side can be used on its own. print_report() is what tells
# you whether they actually exist.
#
# The OpenFAST input decks are NREL's, not this project's, and are not
# redistributed here. Run scripts/fetch_openfast_inputs.py to stage them into
# inputs/ from the OpenFAST r-test. See docs/setup.md.
# ---------------------------------------------------------------------------

OPENFAST_EXE = Path(os.environ.get("OC4_OPENFAST_EXE", "openfast"))
TURBSIM_EXE = Path(os.environ.get("OC4_TURBSIM_EXE", "turbsim"))
BASELINE_5MW = Path(os.environ.get(
    "OC4_RTEST_5MW_BASELINE", REPO_ROOT / "inputs" / "base_files" / "5MW_Baseline"))

SOURCE_DIR = PROJECT / "inputs" / "r-test-baseline"
TURBSIM_BASE = PROJECT / "inputs" / "base_files" / "TurbSim_base.inp"

# The 9 files copied into every case folder (source names, as they exist in
# SOURCE_DIR — the InflowWind one gets renamed after copying, see of_inputs.py).
SOURCE_FILES = [
    "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.fst",
    "NRELOffshrBsline5MW_InflowWind_12mps.dat",
    "NRELOffshrBsline5MW_OC4Jacket_AeroDyn.dat",
    "NRELOffshrBsline5MW_OC4Jacket_ElastoDyn.dat",
    "NRELOffshrBsline5MW_OC4Jacket_ElastoDyn_Tower.dat",
    "NRELOffshrBsline5MW_OC4Jacket_HydroDyn.dat",
    "NRELOffshrBsline5MW_OC4Jacket_ServoDyn.dat",
    "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat",
    "SeaState.dat",
]

# File-name constants used by of_inputs.py once a case folder exists (its own
# working copies, not the source names above).
FST_FILE = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.fst"
INFLOW_SOURCE_NAME = "NRELOffshrBsline5MW_InflowWind_12mps.dat"  # name as copied from SOURCE_DIR
INFLOW_FILE = "NRELOffshrBsline5MW_InflowWind.dat"                # name after point_inflow_file() renames it
AERODYN_FILE = "NRELOffshrBsline5MW_OC4Jacket_AeroDyn.dat"
ELASTODYN_FILE = "NRELOffshrBsline5MW_OC4Jacket_ElastoDyn.dat"
ELASTODYN_TOWER_FILE = "NRELOffshrBsline5MW_OC4Jacket_ElastoDyn_Tower.dat"
HYDRODYN_FILE = "NRELOffshrBsline5MW_OC4Jacket_HydroDyn.dat"
SERVODYN_FILE = "NRELOffshrBsline5MW_OC4Jacket_ServoDyn.dat"
SUBDYN_FILE = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
SEASTATE_FILE = "SeaState.dat"

# ---------------------------------------------------------------------------
# Campaign-wide simulation settings (decided in prior sessions).
# ---------------------------------------------------------------------------

TMAX = 700.0          # 600 s usable + 100 s discarded transient
DT = 0.01             # DT=0.05 is known to hang
DT_OUT = 0.05
TRANSIENT = 100.0     # discarded later at post-processing, not via TStart
WAVE_TMAX = 750.0     # must exceed TMAX or the irregular sea repeats
WAVE_PKSHP = 1.0      # Pierson-Moskowitz
PLEXP = 0.14          # OC4 wind shear exponent (TurbSim default of 0.20 is wrong for this site)
SMOKE_TMAX = 30.0     # used for the cheap end-to-end check before the real 700 s runs

# Flat mean runtime estimate per case, used only for the deadline arithmetic
# in run.py/campaign.py (is a case worth starting given the remaining window?).
# Deliberately NOT scaled by wind speed, even though measured runtime varies
# ~20% across V=4-15 m/s -- that precision is not needed for how the deadline
# is used in practice.
#
# The default below is the mean measured over the real 700 s campaign runs on
# two 4-to-8-core consumer machines (3.20 h and 3.23 h respectively). Override
# it for your own hardware with OC4_EST_HOURS_PER_CASE.
EST_HOURS_PER_CASE = float(os.environ.get("OC4_EST_HOURS_PER_CASE", "3.2"))


# This machine's flat mean estimated hours per case -- see EST_HOURS_PER_CASE.
def est_hours_per_case() -> float:
    return EST_HOURS_PER_CASE


# ---------------------------------------------------------------------------
# The real campaign's cases — 69 wind/wave bins from the author's own OC4 K13
# binning tool output, each run at 6 seeds (IEC 61400-1 cl.7.5 fatigue rule),
# for 414 total runs.
# ---------------------------------------------------------------------------

class Case(NamedTuple):
    case_id: str
    v_hub: float   # hub-height mean wind speed [m/s]
    hs: float      # significant wave height [m]
    tp: float      # peak spectral period [s]
    seed: int
    mode: str      # "operating" or "idle"


BINS_CSV = REPO_ROOT / "data" / "oc4_k13_bins.csv"

# 6 seeds, reused across every bin (same pattern already used for the test
# cases — a seed only needs to be distinct WITHIN a condition, not globally).
# Chosen as clearly-distinct round numbers, not overlapping any seed used by
# the earlier TestScenario test cases (TS01-24), to avoid any confusion
# between real-campaign data and leftover test/validation data on disk.
CAMPAIGN_SEEDS = [100001, 200002, 300003, 400004, 500005, 600006]


# Loads the 69 bins from data/oc4_k13_bins.csv and expands each into
# 6 Case entries (one per CAMPAIGN_SEEDS value) -> 414 total. Cross-checks
# each row's own wind/wave parameters against the Excel's own NAME column
# using condition_folder()'s exact naming formula (duplicated here rather
# than calling condition_folder() itself, since that also appends "_PARKED"
# for idle mode, which the NAME column never does) — 414 cases is too
# many to eyeball by hand, so this makes a parsing/formatting mistake fail
# loudly at import time instead of silently producing a wrong folder name.
def _load_real_cases() -> list[Case]:
    df = pd.read_csv(BINS_CSV)
    cases = []
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        v_hub = float(row.Vw_representative_ms)
        hs = float(row.Hs_representative_m)
        tp = float(row.Tp_representative_s)
        mode = "idle" if "parked" in str(row.Vw_bin).lower() else "operating"

        hs_str = f"{hs:.1f}".replace(".", "p")
        raw_name = f"LC_V{v_hub:g}_H{hs_str}_T{tp:g}"
        if raw_name != row.NAME:
            raise AssertionError(
                f"Bin row {idx}: computed name {raw_name!r} != NAME column {row.NAME!r} "
                f"— check BINS_CSV parsing before trusting CASES."
            )

        for seed_idx, seed in enumerate(CAMPAIGN_SEEDS, start=1):
            cases.append(Case(f"LC{idx:02d}_S{seed_idx}", v_hub, hs, tp, seed, mode))
    return cases


CASES = _load_real_cases()


# Builds the shared per-condition folder name (e.g. "LC_V4_H1p0_T6") — TS01 and TS02
# both resolve to the same string here, which is what puts them in one folder together.
def condition_folder(case: Case) -> str:
    hs_str = f"{case.hs:.1f}".replace(".", "p")
    name = f"LC_V{case.v_hub:g}_H{hs_str}_T{case.tp:g}"
    if case.mode == "idle":
        name += "_PARKED"
    return name


# Builds the seed subfolder name inside a condition folder (e.g. "S123456").
def seed_folder(case: Case) -> str:
    return f"S{case.seed}"


# The TurbSim wind seed for a case — just the case's own seed value.
def wind_seed(case: Case) -> int:
    return case.seed


# The wave seed for a case. Deliberately different from the wind seed (co-varying,
# not fixed) so each of the 6 real-campaign realizations is a genuinely independent draw.
def wave_seed(case: Case) -> int:
    return case.seed + 1


# Turbulence intensity (%) at a given wind speed, from the OC4/UpWind K13 NTM curve.
# This IS already the 90% characteristic value — do not apply a further quantile factor.
def ntm_ti_percent(v_hub: float, i15: float = 0.14, a: float = 5.0) -> float:
    sigma_1 = i15 * (15 + a * v_hub) / (a + 1)
    return 100.0 * sigma_1 / v_hub


# NREL 5MW rotor-speed schedule (wind speed [m/s], rotor speed [rpm]), rounded to
# one decimal — we only need to start near the right operating point, since the
# controller takes over immediately and the first 100 s is discarded anyway.
_ROTOR_SPEED_CURVE = [
    (3.0, 6.9),
    (5.0, 7.5),
    (7.0, 8.5),
    (7.8, 8.9),
    (8.9, 10.1),
    (10.2, 11.6),
    (11.4, 12.0),
    (17.0, 12.0),
    (25.0, 12.0),
]

# NREL 5MW collective blade-pitch schedule (wind speed [m/s], pitch [deg]), read
# from the author's own copy of the reference turbine's steady-state operating
# figure (BlPitch1 vs wind speed, "Region 3" above-rated curve — same source as
# _ROTOR_SPEED_CURVE above, confirmed by the matching flat-12rpm region). Added
# 30.07.2026 after finding set_operating() left every case at the baseline
# template's BlPitch=0 regardless of wind speed — at V=23 m/s that's ~21 deg
# short of the real steady-state pitch, so the controller has to swing pitch
# that far in well under a second against a live turbulent inflow. Confirmed
# as the root cause of a "Tower strike" fatal error hit by 4/6 seeds tested at
# V=23/Hs=4.0/Tp=10 (all failing at ~t=0.77-0.78s, essentially independent of
# which seed) — see docs/decisions.md.
_PITCH_CURVE = [
    (11.4, 0.0),
    (15.0, 10.45),
    (20.0, 17.47),
    (25.0, 23.47),
]


# Linearly interpolates a (x, y) curve; clamps flat below/above its first/last
# point. Shared by initial_rot_speed() and initial_pitch_deg() below — both are
# the same "look up an approximate steady-state value before the controller
# takes over" pattern, just different NREL 5MW reference curves.
def _interp_curve(x: float, curve: list[tuple[float, float]]) -> float:
    xs = [point[0] for point in curve]
    ys = [point[1] for point in curve]

    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            frac = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + frac * (ys[i + 1] - ys[i])


# Initial rotor speed (rpm) for a case, before OpenFAST's own controller takes over.
def initial_rot_speed(v_hub: float) -> float:
    if v_hub <= 0:
        return 0.0
    return _interp_curve(v_hub, _ROTOR_SPEED_CURVE)


# Initial collective blade pitch (deg) for an operating case, before the
# controller takes over. Below rated (v_hub <= 11.4) this is 0, matching the
# baseline template's own default — only above-rated cases need a nonzero
# starting guess.
def initial_pitch_deg(v_hub: float) -> float:
    if v_hub <= 0:
        return 0.0
    return _interp_curve(v_hub, _PITCH_CURVE)


# ---------------------------------------------------------------------------
# Static assignment — which machine runs which case. This is the whole
# anti-double-running mechanism: it is checked once, at import time, so a
# mistake here (a duplicate, a missing case) crashes every script rather than
# letting two machines run the same case.
# ---------------------------------------------------------------------------

# File-backed, not hand-written — the real 414-case split has to be COMPUTED
# (by plan_assignment.py, from measured throughput) rather than typed by
# hand like the old test-case ASSIGNMENT dict was. Git-tracked (small,
# deterministic, meaningful campaign state — like owner.json per case, not
# disposable run data like the actual result folders).
ASSIGNMENT_FILE = PROJECT_DIR / "assignment.json"


def _load_assignment() -> dict:
    if not ASSIGNMENT_FILE.exists():
        return {}
    return json.loads(ASSIGNMENT_FILE.read_text(encoding="utf-8"))


ASSIGNMENT = _load_assignment()


# Verifies an assignment dict covers every case exactly once and only names
# known workers. Defaults to the module-level ASSIGNMENT, but takes an
# explicit `assignment` param so plan_assignment.py/rebalance.py can validate
# a CANDIDATE dict before writing it to disk, reusing this exact logic rather
# than duplicating it. A no-op on an empty dict — this is what lets config.py
# still import cleanly before plan_assignment.py has ever run (it needs to
# `import config` for CASES before any assignment.json exists). Called once,
# below, at import time on the module's own ASSIGNMENT — so a bad file
# crashes every script immediately, same as the old hand-written dict did.
def check_assignment(assignment: dict | None = None) -> None:
    if assignment is None:
        assignment = ASSIGNMENT
    if not assignment:
        return

    all_ids = [c.case_id for c in CASES]
    assigned_ids = [cid for ids in assignment.values() for cid in ids]

    seen = set()
    duplicates = {cid for cid in assigned_ids if cid in seen or seen.add(cid)}
    if duplicates:
        raise AssertionError(f"Case(s) assigned to more than one machine: {duplicates}")

    missing = set(all_ids) - set(assigned_ids)
    if missing:
        raise AssertionError(f"Case(s) not assigned to any machine: {missing}")

    unknown = set(assigned_ids) - set(all_ids)
    if unknown:
        raise AssertionError(f"ASSIGNMENT references unknown case id(s): {unknown}")

    # NB: deliberately does NOT require this machine to appear in the
    # assignment -- plan_assignment.py validates a candidate split that may
    # name machines other than the one planning it. my_cases() is where a
    # missing entry for THIS machine is reported.


check_assignment()


# Returns this machine's assigned Case objects, in ASSIGNMENT's order. Raises
# a clear error (rather than a confusing KeyError) if plan_assignment.py
# hasn't been run yet.
def my_cases() -> list[Case]:
    if not ASSIGNMENT:
        raise RuntimeError(
            "No assignment.json found — run plan_assignment.py first to compute "
            "the split across machines."
        )
    if WORKER not in ASSIGNMENT:
        raise RuntimeError(
            f"This machine ({WORKER!r}) is not named in assignment.json "
            f"(it names {sorted(ASSIGNMENT)}). Set OC4_WORKER to one of those "
            f"names, or re-run plan_assignment.py."
        )
    my_ids = ASSIGNMENT[WORKER]
    by_id = {c.case_id: c for c in CASES}
    return [by_id[cid] for cid in my_ids]


# ---------------------------------------------------------------------------
# Run-data location: the external drive if attached, otherwise a local
# staging folder. The drive is scanned for by a marker file rather than a
# hardcoded letter, because the same physical drive will very likely enumerate
# as a different letter once it is moved between machines.
#
# OC4_RUN_ROOT overrides the whole mechanism if you would rather just
# name a path outright.
# ---------------------------------------------------------------------------

DRIVE_MARKER = os.environ.get("OC4_DRIVE_MARKER", ".oc4_campaign_drive")
DRIVE_SUBFOLDER = os.environ.get("OC4_DRIVE_SUBFOLDER", "OC4_CAMPAIGN")
CAMPAIGN_NAME = "Simulation"


# Scans drive letters D:-Z: for the marker file; returns the campaign folder
# on whichever drive has it, or None if the external drive isn't attached.
def find_drive() -> Path | None:
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = Path(f"{letter}:/{DRIVE_SUBFOLDER}")
        if (candidate / DRIVE_MARKER).exists():
            return candidate
    return None


# Where all run data should go: the external drive if attached, else local staging.
# Returns (path, is_staging) so callers can print/log which one is in effect.
def run_root() -> tuple[Path, bool]:
    override = os.environ.get("OC4_RUN_ROOT")
    if override:
        return Path(override), False
    drive = find_drive()
    if drive is not None:
        return drive / CAMPAIGN_NAME, False
    return PROJECT_DIR / "_staging" / CAMPAIGN_NAME, True


# The actual folder a specific case's OpenFAST files live in (condition/seed nested).
def case_dir(case: Case) -> Path:
    root, _ = run_root()
    return root / condition_folder(case) / seed_folder(case)


# shutil.rmtree(path) with retries — a directory that just had heavy file
# churn (a fresh multi-MB wind.bts, or in one case a 112MB SubDyn .sum.yaml)
# can transiently PermissionError/WinError 5 on rmtree. Originally attributed
# to a cloud-sync engine locking the cloud-synced _staging/ fallback
# path, but hit the IDENTICAL error against the external drive itself
# (30.07.2026, 3 occurrences across two sessions — up to 20s of retry budget
# wasn't consistently enough) — so cloud sync isn't the only cause; Explorer
# thumbnailing/indexing or antivirus scanning a just-written file are just as
# plausible, and this can happen on EITHER run_root() target, not just
# _staging/. 20 attempts / 3s (60s total) covers what's been seen so far.
def rmtree_retry(path: Path, attempts: int = 20, delay_s: float = 3.0) -> None:
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay_s)


# Raises if the drive/folder holding `path` doesn't have at least need_gb free.
def require_free_gb(path: Path, need_gb: float) -> None:
    existing = path
    while not existing.exists():
        existing = existing.parent
    free_gb = shutil.disk_usage(existing).free / 1e9
    if free_gb < need_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free at {existing} — need at least {need_gb:.1f} GB."
        )


# ---------------------------------------------------------------------------
# Readiness report.
# ---------------------------------------------------------------------------

# Prints the full readiness check: paths, exes, base files, run root, cases, packages.
def print_report() -> None:
    print(f"Machine:       {WORKER}  ({CORES} concurrent runs)")
    print(f"Project:       {PROJECT}  (exists: {PROJECT.exists()})")
    print(f"OpenFAST exe:  {OPENFAST_EXE}  (exists: {OPENFAST_EXE.exists()})")
    print(f"TurbSim exe:   {TURBSIM_EXE}  (exists: {TURBSIM_EXE.exists()})")

    discon = BASELINE_5MW / "ServoData" / "DISCON.dll"
    print(f"DISCON.dll:    {discon}  (exists: {discon.exists()})")

    missing_sources = [f for f in SOURCE_FILES if not (SOURCE_DIR / f).exists()]
    print(f"Base files:    {len(SOURCE_FILES) - len(missing_sources)}/{len(SOURCE_FILES)} present"
          + (f"  MISSING: {missing_sources}" if missing_sources else ""))

    root, is_staging = run_root()
    print(f"Run root:      {root}  ({'STAGING (drive not attached)' if is_staging else 'external drive'})")
    require_free_gb_note = ""
    existing = root
    while not existing.exists():
        existing = existing.parent
    free_gb = shutil.disk_usage(existing).free / 1e9
    print(f"Free space:    {free_gb:.1f} GB at {existing}")

    print(f"\n{len(CASES)} cases loaded ({len(CASES) // len(CAMPAIGN_SEEDS)} bins x "
          f"{len(CAMPAIGN_SEEDS)} seeds), NAME cross-check passed.")

    # One row per BIN (not per seed — rot/pitch/TI only depend on v_hub, and
    # 414 lines is too many to scan; the per-bin sanity check is what matters).
    seen_conditions = set()
    print("Per-bin sanity check (rotor speed / pitch):")
    for c in CASES:
        key = (c.v_hub, c.hs, c.tp, c.mode)
        if key in seen_conditions:
            continue
        seen_conditions.add(key)
        ti = ntm_ti_percent(c.v_hub) if c.v_hub > 0 else 0.0
        rot = initial_rot_speed(c.v_hub)
        pitch = 90.0 if c.mode == "idle" else initial_pitch_deg(c.v_hub)
        print(f"  V={c.v_hub:>4g} Hs={c.hs:>4.1f} Tp={c.tp:>4g} mode={c.mode:<9} "
              f"TI={ti:5.1f}% RotSpeed={rot:5.2f}rpm Pitch={pitch:5.2f}deg  -> {condition_folder(c)}")

    try:
        mine = my_cases()
        print(f"\nMy cases ({WORKER}): {len(mine)} of {len(CASES)} "
              f"({len(mine) / len(CAMPAIGN_SEEDS):.0f} bins)")
    except RuntimeError as exc:
        print(f"\nMy cases ({WORKER}): {exc}")

    cpu_ok = CORES <= (os.cpu_count() or 0)
    print(f"\nCPU check:     CORES={CORES} <= os.cpu_count()={os.cpu_count()}: {cpu_ok}")

    for pkg in ("numpy", "pandas", "openfast_toolbox"):
        try:
            __import__(pkg)
            print(f"Package OK:    {pkg}")
        except ImportError:
            print(f"Package MISSING: {pkg}")


if __name__ == "__main__":
    print_report()
