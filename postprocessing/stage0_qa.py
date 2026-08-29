"""
Step 3 -- Stage 0 QA sweep.

Reads every case folder found under one or more roots and produces one
summary row per run: does the duration match what was requested, did the
run terminate normally, does the SubDyn geometry match what's expected,
is the inertial (FM/MM) channel family still negligible on THIS run (not
just assumed from LC_999 in an earlier session), and do a handful of
environment/mode channels land near the run's own target -- all without
reading the ~2700-channel bulk fatigue data for runs that failed anyway.

Scope, confirmed 05.08.2026 (see the build plan): alongside OutAll,
Stage 0 reads exactly four context channels -- RotSpeed/BldPitch1 (mode
check against case.json) and Wind1VelX/Wave1Elev (environment check
against the bin's v_hub/hs target). Deliberately NOT read: tower-base
moments, platform motion, base reactions -- considered and left out.

Steps:
    1. Discover every case folder under a root (identified by owner.json
       presence, same convention Simulation/merge.py already uses).
    2. Read the .outb header WITHOUT the integrity assert (see
       outb_reader._read_outb_header_raw) -- a crashed run must show up as
       a row in the output, not kill the whole sweep.
    3. Compare header duration against case.json's tmax and check the
       openfast.log tail -- this is the ONLY reliable way to catch a
       crashed run: a truncated file can still be internally
       size-consistent (see the LC_V23_H4p0_T10/S654321 fixture).
    4. For runs that pass (3), do the more expensive checks: SubDyn md5,
       NaN/Inf and the FM/MM-vs-FK/MK inertial ratio on a member sample,
       and the four context channels.
"""
import json
import hashlib
from pathlib import Path

import numpy as np

import outb_reader as obr
import sd_geometry as sdg

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# Dev/test fixture data stays at its original location -- see
# See docs/decisions.md, 10.08.2026 folder-reorg session.
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

REAL_BIN_ROOT = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8"
OLD_TEST_ROOT = DEV_FIXTURE_DIR / "_staging" / "TestScenario"

OUTB_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb"
SD_NAME = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
LOG_NAME = "openfast.log"
# endregion

# region --- QA parameters ---
# Members sampled for the inertial (FM/MM vs FK/MK) ratio check -- the same
# members spot-checked by hand earlier this session (legs, braces, TP
# stubs, grouted, piles -- one from every propset group), not exhaustive
# but not cherry-picked either.
INERTIAL_SAMPLE_MEMBERS = [1, 5, 17, 33, 41, 60, 80, 101, 109]

CONTEXT_CHANNELS = ["RotSpeed", "BldPitch1", "Wind1VelX", "Wave1Elev"]

# Allow up to 2 output samples of slack between header duration and
# case.json's tmax (covers ordinary off-by-one at the record boundary,
# still catches an aborted run by a wide margin -- e.g. 60s requested vs
# 0.75s actual).
DURATION_TOL_MULT = 2

# Mode-check thresholds -- deliberately loose (this is a sanity check for
# a mixed-up input file, not a physics validation): idle should show near-
# zero rotor speed and a near-feathered blade; operating should show
# neither.
IDLE_ROTSPEED_MAX_RPM = 3.0
IDLE_BLDPITCH_MIN_DEG = 60.0

# Environment-check tolerances -- also loose on purpose. Wind1VelX is a
# single-point sample of a turbulent field, and Hs estimated from
# 4*std(elevation) is itself an approximation (valid for a narrow-band
# Gaussian sea surface), so this is a "wrong input file" trip-wire, not a
# precision check.
WIND_REL_TOL = 0.25
WAVE_HS_REL_TOL = 0.35
# endregion


def _discover_cases(root):
    """Every case folder under root, identified by owner.json presence."""
    return sorted(p.parent for p in Path(root).rglob("owner.json"))


def _log_tail_ok(log_path):
    """
    True if the log shows normal termination. Search the last ~2000 chars
    rather than just the last non-empty line -- OpenFAST's fatal-error
    message is followed by blank lines and "Aborting OpenFAST.", so the
    literal last line alone is not a reliable signal either way.
    """
    if not log_path.exists():
        return False, "(missing openfast.log)"
    text = log_path.read_text(encoding="latin-1", errors="replace")
    tail_chunk = text[-2000:]
    ok = "OpenFAST terminated normally" in tail_chunk
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last_line = lines[-1] if lines else ""
    return ok, last_line


def qa_one_run(case_dir):
    """Return a dict -- one QA row for a single case folder."""
    case_dir = Path(case_dir)
    # Root-agnostic label (condition/seed folder names) -- case_dir can come
    # from the dev fixture, the real campaign on D:, or anywhere else the
    # caller pointed at, so a single fixed "relative to X" base would break
    # (ValueError) the moment root isn't under X. This was the one real bug
    # found while dry-running the pipeline against real-campaign-shaped
    # paths (10.08.2026) -- qa_sweep() is exactly the function
    # run_pipeline.py's run_stage0() calls with an arbitrary root.
    row = dict(case_dir=f"{case_dir.parent.name}/{case_dir.name}")

    case_json = json.loads((case_dir / "case.json").read_text())
    owner_json = json.loads((case_dir / "owner.json").read_text())
    row.update(
        case_id=case_json["case_id"], mode=case_json["mode"],
        machine=owner_json.get("machine"),
        v_hub=case_json["v_hub"], hs=case_json["hs"], tp=case_json["tp"],
        tmax_case_json=case_json["tmax"],
    )

    outb_path = case_dir / OUTB_NAME
    header = obr._read_outb_header_raw(outb_path)  # never raises on a bad file
    dt = header["t_incr"]
    duration = (header["n_t"] - 1) * dt
    size_consistent = (
        header["data_offset"] + header["n_t"] * header["n_chan"] * 8 == header["filesize"]
    )
    duration_ok = abs(duration - case_json["tmax"]) <= DURATION_TOL_MULT * dt
    log_ok, log_tail = _log_tail_ok(case_dir / LOG_NAME)

    row.update(
        n_chan=header["n_chan"], n_t=header["n_t"], dt=dt, duration_s=duration,
        size_consistent=size_consistent, duration_ok=duration_ok,
        log_ok=log_ok, log_tail=log_tail[:80],
    )

    status = "OK" if (duration_ok and log_ok) else "CRASHED"
    row["status"] = status

    empty_extra = dict(
        subdyn_md5="", subdyn_md5_matches_known="",
        any_nan="", any_inf="",
        force_elastic_scale_max="", moment_elastic_scale_max="",
        force_inertial_max="", moment_inertial_max="",
        inertial_force_ratio_max="", inertial_moment_ratio_max="",
        rotspeed_mean="", bldpitch_mean="", mode_check_ok="",
        wind1velx_mean="", wind_rel_diff="",
        wave_hs_estimate="", wave_hs_rel_diff="", env_check_ok="",
    )
    if status != "OK":
        # Do not trust further reads on a run that didn't finish normally
        # -- report what's already known and stop.
        row.update(empty_extra)
        return row

    # SubDyn md5, against sd_geometry's independently-confirmed constant.
    sd_path = case_dir / SD_NAME
    sd_md5 = hashlib.md5(sd_path.read_bytes()).hexdigest()
    row["subdyn_md5"] = sd_md5
    row["subdyn_md5_matches_known"] = (sd_md5 == sdg.KNOWN_CAMPAIGN_MD5)

    # Context channels: mode check + environment sanity check.
    _, ctx = obr.read_channels(outb_path, header, CONTEXT_CHANNELS)
    rotspeed_mean = float(np.mean(ctx[:, 0]))
    bldpitch_mean = float(np.mean(ctx[:, 1]))
    wind1velx_mean = float(np.mean(ctx[:, 2]))
    wave_hs_estimate = float(4.0 * np.std(ctx[:, 3]))  # Hs ~= 4*sigma, narrow-band Gaussian sea

    if case_json["mode"] == "idle":
        mode_check_ok = (abs(rotspeed_mean) < IDLE_ROTSPEED_MAX_RPM
                          and bldpitch_mean > IDLE_BLDPITCH_MIN_DEG)
    else:
        mode_check_ok = (abs(rotspeed_mean) >= IDLE_ROTSPEED_MAX_RPM
                          and bldpitch_mean <= IDLE_BLDPITCH_MIN_DEG)

    wind_rel_diff = abs(wind1velx_mean - case_json["v_hub"]) / case_json["v_hub"]
    if case_json["hs"] > 0:
        wave_hs_rel_diff = abs(wave_hs_estimate - case_json["hs"]) / case_json["hs"]
        wave_ok = wave_hs_rel_diff <= WAVE_HS_REL_TOL
    else:
        wave_hs_rel_diff = float("nan")
        wave_ok = True
    env_check_ok = (wind_rel_diff <= WIND_REL_TOL) and wave_ok

    row.update(
        rotspeed_mean=rotspeed_mean, bldpitch_mean=bldpitch_mean,
        mode_check_ok=mode_check_ok,
        wind1velx_mean=wind1velx_mean, wind_rel_diff=wind_rel_diff,
        wave_hs_estimate=wave_hs_estimate, wave_hs_rel_diff=wave_hs_rel_diff,
        env_check_ok=env_check_ok,
    )

    # NaN/Inf + inertial (FM/MM vs FK/MK) ratio on the member sample.
    k_names = obr.member_end_channels(INERTIAL_SAMPLE_MEMBERS, obr.ALL_K_COMPONENTS)
    m_names = obr.member_end_channels(INERTIAL_SAMPLE_MEMBERS, obr.ALL_M_COMPONENTS)
    _, k_arr = obr.read_channels(outb_path, header, k_names)
    _, m_arr = obr.read_channels(outb_path, header, m_names)

    row["any_nan"] = bool(np.isnan(k_arr).any() or np.isnan(m_arr).any())
    row["any_inf"] = bool(np.isinf(k_arr).any() or np.isinf(m_arr).any())

    # ALL_K_COMPONENTS / ALL_M_COMPONENTS are the same length (6) and in
    # matching order (FKxe<->FMxe, ..., MKze<->MMze) -- member_end_channels
    # repeats that order per member-end, so column j of k_arr and m_arr
    # are always the same physical component. See outb_reader.py.
    #
    # Deliberately NOT a per-member self-ratio (m_std/k_std column by
    # column): member 101 (a TP interface stub) has elastic force std of
    # ~1e-8 N -- effectively exact zero -- so dividing its own tiny
    # inertial noise by its own near-zero elastic denominator explodes
    # into a meaningless multi-billion "ratio" that swamps the real
    # signal. That degeneracy was already found earlier this session and
    # is exactly why members 101-104 are flagged not_assessable downstream
    # (Stage 3), not evidence the inertial family matters. The physically
    # meaningful question is whether the inertial contribution is small
    # relative to the STRUCTURE'S overall elastic force/moment scale, so
    # compare against the largest elastic std across the whole sample
    # (matches how this was checked by hand earlier: worst inertial ~42
    # N*m against ~3.3e4-3.2e5 N*m of elastic bending).
    k_std = k_arr.std(axis=0)
    m_std = m_arr.std(axis=0)
    n_comp = len(obr.ALL_K_COMPONENTS)
    is_force = np.array([c.startswith("F") for c in obr.ALL_K_COMPONENTS])
    force_mask = np.tile(is_force, len(k_names) // n_comp)

    force_elastic_scale = float(k_std[force_mask].max())
    moment_elastic_scale = float(k_std[~force_mask].max())
    force_inertial_max = float(m_std[force_mask].max())
    moment_inertial_max = float(m_std[~force_mask].max())

    row["force_elastic_scale_max"] = force_elastic_scale
    row["moment_elastic_scale_max"] = moment_elastic_scale
    row["force_inertial_max"] = force_inertial_max
    row["moment_inertial_max"] = moment_inertial_max
    row["inertial_force_ratio_max"] = force_inertial_max / force_elastic_scale
    row["inertial_moment_ratio_max"] = moment_inertial_max / moment_elastic_scale

    return row


def qa_sweep(roots):
    """Run qa_one_run over every case found under one or more roots."""
    rows = []
    for root in roots:
        for case_dir in _discover_cases(root):
            rows.append(qa_one_run(case_dir))
    return rows


def write_qa_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def _self_check():
    rows = qa_sweep([REAL_BIN_ROOT, OLD_TEST_ROOT])
    out_path = RESULTS_DIR / "stage0_qa.csv"
    write_qa_csv(rows, out_path)
    print(f"scanned {len(rows)} runs, wrote {out_path}")

    ok_rows = [r for r in rows if r["status"] == "OK"]
    crashed_rows = [r for r in rows if r["status"] == "CRASHED"]
    print(f"  OK: {len(ok_rows)}   CRASHED: {len(crashed_rows)}")
    for r in crashed_rows:
        print(f"    CRASHED: {r['case_dir']}  duration={r['duration_s']:.2f}s "
              f"vs tmax={r['tmax_case_json']}  size_consistent={r['size_consistent']}  "
              f"log_tail={r['log_tail']!r}")

    real_bin_rows = [r for r in rows if r["case_dir"].startswith("LC_V20_H3p5_T8")]
    print(f"\n  real bin LC_V20_H3p5_T8: {len(real_bin_rows)} runs, "
          f"all OK at 700.0s: "
          f"{all(r['status'] == 'OK' and abs(r['duration_s'] - 700.0) < 1e-6 for r in real_bin_rows)}")

    n_chan_counts = {}
    for r in ok_rows:
        n_chan_counts[r["n_chan"]] = n_chan_counts.get(r["n_chan"], 0) + 1
    print(f"  n_chan distribution across OK runs (informational, expected to vary): "
          f"{n_chan_counts}")

    md5_mismatches = [r for r in ok_rows if not r["subdyn_md5_matches_known"]]
    print(f"  SubDyn md5 mismatches: {len(md5_mismatches)}")

    nan_inf = [r for r in ok_rows if r["any_nan"] or r["any_inf"]]
    print(f"  runs with NaN/Inf in sampled channels: {len(nan_inf)}")

    mode_fail = [r for r in ok_rows if not r["mode_check_ok"]]
    env_fail = [r for r in ok_rows if not r["env_check_ok"]]
    print(f"  mode-check failures: {len(mode_fail)}   env-check failures: {len(env_fail)}")

    max_force_ratio = max(r["inertial_force_ratio_max"] for r in ok_rows)
    max_moment_ratio = max(r["inertial_moment_ratio_max"] for r in ok_rows)
    max_force_abs = max(r["force_inertial_max"] for r in ok_rows)
    max_moment_abs = max(r["moment_inertial_max"] for r in ok_rows)
    print(f"  worst inertial-vs-elastic-scale ratio across all OK runs: "
          f"force={max_force_ratio:.4f}  moment={max_moment_ratio:.4f}")
    print(f"  worst inertial ABSOLUTE std across all OK runs: "
          f"force={max_force_abs:.4g} N   moment={max_moment_abs:.4g} N*m  "
          f"(negligibility finding from LC_999 holds if both ratios and both "
          f"absolutes are small)")


if __name__ == "__main__":
    _self_check()
