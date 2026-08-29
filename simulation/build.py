"""
build.py — creates and configures the case folders (no OpenFAST run yet, except
TurbSim to generate each case's wind file).

For each of this machine's assigned cases, this:
  1. creates <run_root>/<condition_folder>/<seed_folder>/
  2. copies the 9 base input files + wind.inp template into it
  3. calls of_inputs.py functions, in order, to configure that case
  4. runs TurbSim to produce wind.bts
  5. writes owner.json (who built this) and case.json (what values were used)

Run it directly: `python build.py [--smoke] [--only TS07] [--force] [--no-turbsim]`
"""

import argparse
import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path

import config
import of_inputs


# Creates the (condition, seed) folder and copies in the base files + wind.inp template.
def make_case_dir(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for fname in config.SOURCE_FILES:
        shutil.copy2(config.SOURCE_DIR / fname, case_dir / fname)
    shutil.copy2(config.TURBSIM_BASE, case_dir / "wind.inp")


# Writes owner.json — which machine built this case, and when. run.py and merge.py
# both check this before touching a case folder, to prevent double-running.
def stamp_owner(case_dir: Path, case: config.Case) -> None:
    owner = {
        "case_id": case.case_id,
        "condition_folder": config.condition_folder(case),
        "seed_folder": config.seed_folder(case),
        "machine": config.WORKER,
        "hostname": socket.gethostname(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_dir / "owner.json").write_text(json.dumps(owner, indent=2), encoding="utf-8")


# Calls the of_inputs.py setters in the order that matters, then writes case.json —
# the resolved parameter record, so it's clear afterwards exactly what this case ran with.
def configure(case: config.Case, case_dir: Path, tmax: float, run_turbsim_flag: bool) -> None:
    of_inputs.fix_baseline_paths(case_dir)
    of_inputs.point_inflow_file(case_dir)
    of_inputs.set_tmax(case_dir, tmax)
    of_inputs.set_waves(case_dir, hs=case.hs, tp=case.tp, seed=config.wave_seed(case))
    of_inputs.set_subdyn_outall(case_dir, True)

    ti_percent = config.ntm_ti_percent(case.v_hub)
    of_inputs.setup_turbsim(
        case_dir, v_hub=case.v_hub, seed=config.wind_seed(case), tmax=tmax, ti_percent=ti_percent
    )

    turbsim_elapsed = None
    if run_turbsim_flag:
        turbsim_elapsed = of_inputs.run_turbsim(case_dir)
    of_inputs.set_wind_turbulent(case_dir)

    if case.mode == "idle":
        of_inputs.set_idle_parked(case_dir)
        rot_speed = 0.0
        pitch_deg = 90.0
    else:
        rot_speed = config.initial_rot_speed(case.v_hub)
        pitch_deg = config.initial_pitch_deg(case.v_hub)
        of_inputs.set_operating(case_dir, rot_speed, pitch_deg)

    case_record = {
        "case_id": case.case_id,
        "v_hub": case.v_hub,
        "hs": case.hs,
        "tp": case.tp,
        "wind_seed": config.wind_seed(case),
        "wave_seed": config.wave_seed(case),
        "mode": case.mode,
        "tmax": tmax,
        "ti_percent": ti_percent,
        "rot_speed": rot_speed,
        "pitch_deg": pitch_deg,
        "turbsim_elapsed_s": turbsim_elapsed,
    }
    (case_dir / "case.json").write_text(json.dumps(case_record, indent=2), encoding="utf-8")


# True once a case's build finished cleanly — case.json is written last in
# configure(), after every of_inputs setter and TurbSim, so its presence is a
# reliable completion marker. Used by campaign.py's builder thread to skip
# already-built cases and to tell a partial build (folder exists, case.json
# doesn't — e.g. killed mid-TurbSim) apart from a finished one.
def is_built(case_dir: Path) -> bool:
    return (case_dir / "case.json").exists()


# Builds one case end-to-end: resolves its folder, refuses to clobber an existing
# one unless --force, then creates + stamps + configures it. Returns the folder.
def build_case(case: config.Case, tmax: float, force: bool, run_turbsim_flag: bool) -> Path:
    case_dir = config.case_dir(case)
    if case_dir.exists():
        if not force:
            raise FileExistsError(f"{case_dir} already exists — pass --force to rebuild it")
        config.rmtree_retry(case_dir)

    make_case_dir(case_dir)
    stamp_owner(case_dir, case)
    configure(case, case_dir, tmax, run_turbsim_flag)
    return case_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                         help=f"use TMax={config.SMOKE_TMAX}s instead of {config.TMAX}s")
    parser.add_argument("--only", help="build only this case id (must belong to this machine)")
    parser.add_argument("--force", action="store_true",
                         help="rebuild a case folder that already exists")
    parser.add_argument("--no-turbsim", action="store_true",
                         help="skip running TurbSim.exe (still writes wind.inp)")
    args = parser.parse_args()

    tmax = config.SMOKE_TMAX if args.smoke else config.TMAX
    cases = config.my_cases()

    if args.only:
        my_ids = {c.case_id for c in cases}
        if args.only not in my_ids:
            owner = next(
                (machine for machine, ids in config.ASSIGNMENT.items() if args.only in ids),
                "an unknown machine",
            )
            raise SystemExit(f"{args.only} is not assigned to {config.WORKER} — it belongs to {owner}")
        cases = [c for c in cases if c.case_id == args.only]

    print(f"Building {len(cases)} case(s) on {config.WORKER} at TMax={tmax}s")
    for case in cases:
        print(f"\n--- {case.case_id}  ({config.condition_folder(case)}/{config.seed_folder(case)}) ---")
        case_dir = build_case(case, tmax, args.force, run_turbsim_flag=not args.no_turbsim)
        print(f"  -> {case_dir}")


if __name__ == "__main__":
    main()
