"""
of_inputs.py — functions that edit OpenFAST input files for one case folder.

Each function takes a case folder and already-decided values (computed by
config.py) and writes them into the right OpenFAST input file. These functions
know nothing about the campaign's 8 cases, the run root, or how a value was
chosen — that is build.py's job, which calls these in the right order.
"""

import re
import subprocess
import time
from pathlib import Path

from openfast_toolbox.io import FASTInputFile

import config


# Regex-substitutes one parameter's value in a raw text file. Used only for
# file types FASTInputFile can't handle reliably (SubDyn.dat has a known
# parsing bug) or where raw-text editing matches the existing convention
# (TurbSim's wind.inp). Raises KeyError on zero matches — a silent no-op here
# is exactly how a name typo (e.g. "WakeMod" instead of "Wake_Mod") would
# otherwise pass unnoticed.
def set_param_text(text: str, name: str, value) -> str:
    pattern = r'^(\s*)(\S+|".+?")(\s+' + re.escape(name) + r'\s)'
    new_text, n = re.subn(
        pattern,
        lambda m: m.group(1) + str(value) + m.group(3),
        text, flags=re.MULTILINE,
    )
    if n == 0:
        raise KeyError(f"Parameter '{name}' not found in text — check the exact name.")
    return new_text


# Replaces the baseline template's relative "../5MW_Baseline/" path references
# with an absolute path, in every file that has them. Without this, OpenFAST
# can't find the shared blade/airfoil/DISCON files when the .fst is run from
# this case folder instead of the original r-test location.
def fix_baseline_paths(case_dir: Path) -> None:
    old = "../5MW_Baseline/"
    new = str(config.BASELINE_5MW).replace("\\", "/") + "/"
    files = [
        case_dir / config.FST_FILE,
        case_dir / config.ELASTODYN_FILE,
        case_dir / config.AERODYN_FILE,
        case_dir / config.SERVODYN_FILE,
    ]
    for f in files:
        text = f.read_text(encoding="utf-8")
        text = text.replace(old, new)
        f.write_text(text, encoding="utf-8")


# Renames the copied InflowWind file to its working name and repoints the .fst
# at it. Without this, the .fst still points at a file OUTSIDE the case folder
# (the baseline's shared, fixed-12mps template) — any per-case wind settings
# written later would silently have no effect. This exact trap has bitten the
# validation campaign 3 times already.
def point_inflow_file(case_dir: Path) -> None:
    (case_dir / config.INFLOW_SOURCE_NAME).rename(case_dir / config.INFLOW_FILE)

    fst_path = case_dir / config.FST_FILE
    fst = FASTInputFile(str(fst_path))
    fst["InflowFile"] = f'"{config.INFLOW_FILE}"'
    fst.write(str(fst_path))


# Sets the simulation duration and time steps in the .fst file.
def set_tmax(case_dir: Path, tmax: float, dt: float = config.DT, dt_out: float = config.DT_OUT) -> None:
    fst_path = case_dir / config.FST_FILE
    fst = FASTInputFile(str(fst_path))
    fst["TMax"] = tmax
    fst["DT"] = dt
    fst["DT_Out"] = dt_out
    fst.write(str(fst_path))


# Sets the irregular sea state (height, period, seed) AND WaveTMax, so the sea
# does not repeat within the run — the baseline template's default (60 s)
# makes the sea loop every 60 s if this is left untouched.
def set_waves(case_dir: Path, hs: float, tp: float, seed: int,
              wave_tmax: float = config.WAVE_TMAX, pkshp: float = config.WAVE_PKSHP,
              wave_dir: float = 0.0) -> None:
    ss_path = case_dir / config.SEASTATE_FILE
    ss = FASTInputFile(str(ss_path))
    ss["WaveMod"] = 2          # irregular (JONSWAP / Pierson-Moskowitz)
    ss["WaveHs"] = hs
    ss["WaveTp"] = tp
    ss["WaveSeed(1)"] = seed
    ss["WavePkShp"] = pkshp    # 1.0 = Pierson-Moskowitz
    ss["WaveTMax"] = wave_tmax
    ss["WaveDir"] = wave_dir
    ss.write(str(ss_path))


# Configures the case's own wind.inp (already copied there by build.py) with
# wind speed, seed, duration, turbulence intensity, and wind shear.
def setup_turbsim(case_dir: Path, v_hub: float, seed: int, tmax: float,
                   ti_percent: float, plexp: float = config.PLEXP) -> None:
    inp_path = case_dir / "wind.inp"
    text = inp_path.read_text(encoding="utf-8")
    text = set_param_text(text, "RandSeed1", seed)
    text = set_param_text(text, "URef", v_hub)
    text = set_param_text(text, "AnalysisTime", tmax)
    text = set_param_text(text, "UsableTime", tmax)
    text = set_param_text(text, "IECturbc", ti_percent)
    text = set_param_text(text, "PLExp", plexp)
    text = set_param_text(text, "ScaleIEC", 1)
    inp_path.write_text(text, encoding="utf-8")


# Runs TurbSim.exe on the case's wind.inp, producing wind.bts. Writes stdout to
# a log FILE rather than capturing it in memory — capturing output on a
# subprocess risks a PIPE-buffer deadlock once the run takes a while.
def run_turbsim(case_dir: Path, timeout: float = 1800) -> float:
    log_path = case_dir / "turbsim.log"
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log_file:
        result = subprocess.run(
            [str(config.TURBSIM_EXE), "wind.inp"],
            cwd=case_dir, stdout=log_file, stderr=subprocess.STDOUT, timeout=timeout,
        )
    elapsed = time.time() - t0
    if result.returncode != 0:
        raise RuntimeError(f"TurbSim failed in {case_dir} (rc={result.returncode}) — see {log_path}")
    if not (case_dir / "wind.bts").exists():
        raise RuntimeError(f"TurbSim reported success but wind.bts is missing in {case_dir}")
    return elapsed


# Points InflowWind at the case's own wind.bts file (turbulent wind, WindType=3).
def set_wind_turbulent(case_dir: Path) -> None:
    ifw_path = case_dir / config.INFLOW_FILE
    ifw = FASTInputFile(str(ifw_path))
    bts_path = str(case_dir / "wind.bts").replace("\\", "/")
    ifw["WindType"] = 3
    ifw["FileName_BTS"] = f'"{bts_path}"'
    ifw.write(str(ifw_path))


# Configures a normal operating case: full aero + controller, all structural
# DOFs enabled, initial rotor speed as computed by config.initial_rot_speed(),
# initial blade pitch as computed by config.initial_pitch_deg(). The pitch
# default matters above rated wind speed — the baseline template's own default
# is 0 deg, correct only up to rated (11.4 m/s); leaving it at 0 for an
# above-rated case forces the controller to swing pitch tens of degrees in the
# first fraction of a second against a live turbulent inflow, which is exactly
# what caused a "Tower strike" fatal error in 4/6 seeds tested at V=23 m/s
# before this was added (30.07.2026) — see config._PITCH_CURVE.
def set_operating(case_dir: Path, rot_speed: float, pitch_deg: float = 0.0) -> None:
    fst_path = case_dir / config.FST_FILE
    fst = FASTInputFile(str(fst_path))
    fst["CompAero"] = 2
    fst["CompServo"] = 1
    fst.write(str(fst_path))

    ed_path = case_dir / config.ELASTODYN_FILE
    ed = FASTInputFile(str(ed_path))
    ed["RotSpeed"] = rot_speed
    ed["BlPitch(1)"] = pitch_deg
    ed["BlPitch(2)"] = pitch_deg
    ed["BlPitch(3)"] = pitch_deg
    for dof in ["FlapDOF1", "FlapDOF2", "EdgeDOF", "DrTrDOF", "GenDOF", "YawDOF",
                "TwFADOF1", "TwFADOF2", "TwSSDOF1", "TwSSDOF2",
                "PtfmSgDOF", "PtfmSwDOF", "PtfmHvDOF", "PtfmRDOF", "PtfmPDOF", "PtfmYDOF"]:
        ed[dof] = True
    ed.write(str(ed_path))


# Configures the idle/parked case: rotor feathered and free-wheeling under aero
# torque alone. comp_servo=0 (the recommended default) never loads DISCON.dll;
# comp_servo=1 is a real init-abort risk with this baseline's ServoDyn file
# (VSContrl=0 validates Thevenin inputs that are all placeholders here).
def set_idle_parked(case_dir: Path, blpitch_deg: float = 90.0, comp_servo: int = 0) -> None:
    fst_path = case_dir / config.FST_FILE
    fst = FASTInputFile(str(fst_path))
    fst["CompAero"] = 2
    fst["CompServo"] = comp_servo
    fst.write(str(fst_path))

    ed_path = case_dir / config.ELASTODYN_FILE
    ed = FASTInputFile(str(ed_path))
    ed["BlPitch(1)"] = blpitch_deg
    ed["BlPitch(2)"] = blpitch_deg
    ed["BlPitch(3)"] = blpitch_deg
    ed["RotSpeed"] = 0.0
    ed["GenDOF"] = True
    ed["DrTrDOF"] = False   # forum gotcha: left True, causes azimuth 360->0 jumps + RotSpeed noise
    ed["YawDOF"] = False
    for dof in ["FlapDOF1", "FlapDOF2", "EdgeDOF",
                "TwFADOF1", "TwFADOF2", "TwSSDOF1", "TwSSDOF2",
                "PtfmSgDOF", "PtfmSwDOF", "PtfmHvDOF", "PtfmRDOF", "PtfmPDOF", "PtfmYDOF"]:
        ed[dof] = True
    ed.write(str(ed_path))

    ad_path = case_dir / config.AERODYN_FILE
    ad = FASTInputFile(str(ad_path))
    ad["Wake_Mod"] = 0   # no induction — correct for a parked/idling rotor
    ad["UA_Mod"] = 0     # avoids Beddoes-Leishman, a known NaN source at +-180 deg AoA
    ad.write(str(ad_path))

    if comp_servo == 1:
        sd_path = case_dir / config.SERVODYN_FILE
        sd = FASTInputFile(str(sd_path))
        sd["PCMode"] = 0
        sd["VSContrl"] = 0
        sd["GenTiStr"] = True
        sd["TimGenOn"] = 9999.9
        sd["HSSBrMode"] = 0
        sd["YCMode"] = 0
        sd.write(str(sd_path))


# Toggles SubDyn's OutAll flag (all 112 members' end-node forces/moments).
# SubDyn.dat can't be parsed by FASTInputFile (a known library bug), so this
# uses the raw-text set_param_text helper instead.
def set_subdyn_outall(case_dir: Path, enabled: bool = True) -> None:
    sd_path = case_dir / config.SUBDYN_FILE
    text = sd_path.read_text(encoding="utf-8")
    text = set_param_text(text, "OutAll", "True" if enabled else "False")
    sd_path.write_text(text, encoding="utf-8")
