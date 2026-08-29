"""
Joint-track corrosion, Step J3 -- Stage 2 per-run joint histogram file,
splash-zone scope, corrosion years.

NEW file. Reuses stage2_joints.py's own already-tested machinery UNEDITED --
build_full_connections() (topology, unaffected by corrosion), _connection_key(),
_connection_signals(), iter_assessment_rows(), build_point_table() -- rather
than reimplementing any of the position/segment/hotspot logic. The only
genuinely new code here is the year loop and the corroded-input construction:

  stage2_joints._connection_signals(c, outb_path, header, t_cutoff) reads
  EVERY geometry input off the connection dict `c` itself (brace_D/brace_t/
  phi_deg/chord_a_D/T/chord_b_D/T) -- nothing hardcoded from module state.
  So calling it, UNEDITED, on a COPY of `c` with those fields overwritten by
  their corroded-at-year values reproduces the exact same signal-building
  code path (read .outb -> nominal_components -> rotate_to_joint_axes) at
  any year. stress.hotspot_joint() already takes its SCF values as plain
  arguments (supplied here by scf_corrosion.compute_scf_corroded, Step J2).
  iter_assessment_rows() only needs a `connections` list (year-independent
  topology) and a `scf_index` (year-specific, from Step J2) -- reused as-is.

DESIGN: ONE .outb READ PER (CONNECTION, YEAR) -- NOT one bulk read per run
like the member track. _connection_signals() re-reads its own few channels
per call; for the splash connections x N_years, that's a small number of
cheap memmap reads -- stage2_joints.py's own uncorroded design already
treats per-connection reads as fine, unbatched (rainflow dominates the cost
budget, not I/O); the same reasoning applies with a year axis added.

SPLASH-ZONE SCOPE: filtered via sd_geometry.environment_zone() on each
connection's node z, same convention stage3_joint_damage.py already uses to
derive zone from z. 24 of 120 connections measured splash-zone in an earlier
session (16 K + 8 X, zero TY -- TY family is top/mudbrace level, entirely
outside the splash band) -- reconfirmed by this module's own self-check
below, not assumed.

DEV-FIXTURE-ONLY STATUS, same as stage2_joints.py: stress.HOTSPOT_JOINT_VERIFIED
is the author's own sign-off (not independently reviewed, see
docs/decisions.md) -- this module builds/self-checks against the same
dev fixture and is NOT wired into run_pipeline.py or the real campaign drive.
"""
import json
import os
from pathlib import Path

import numpy as np
import rainflow

import fatigue_config as cfg
import outb_reader as obr
import rainflow_hist as rhist
import sd_geometry as sdg
import stress
import scf
import stage2_joints as s2j
import joint_geometry_corrosion as jgc
import scf_corrosion as scfc

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

# Same D:-Z: scan as stage2_joints.find_drive() -- duplicated rather than
# imported (this module stays outside Simulation/'s import graph, see
# module docstring / docs/decisions.md). A SEPARATE subfolder
# from BOTH stage2_joints/ (uncorroded) and stage2_corrosion/ (member track)
# -- a different point table shape from either, must never collide.
_DRIVE_MARKER = ".oc4_campaign_drive"
_DRIVE_SUBFOLDER = "OC4_CAMPAIGN"


def find_drive():
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = Path(f"{letter}:/{_DRIVE_SUBFOLDER}")
        if (candidate / _DRIVE_MARKER).exists():
            return candidate
    return None


_drive = find_drive()
if _drive is not None:
    STAGE2_JOINTS_CORROSION_DIR = _drive / "Postprocessing" / "stage2_joints_corrosion"
else:
    STAGE2_JOINTS_CORROSION_DIR = RESULTS_DIR / "_stage2_joints_corrosion_local"
    print(f"WARNING: external drive not found (checked D:-Z: for a '{_DRIVE_SUBFOLDER}' "
          f"folder containing '{_DRIVE_MARKER}'). Joint corrosion Stage 2 cache falling "
          f"back to local {STAGE2_JOINTS_CORROSION_DIR}")

OUTB_NAME = s2j.OUTB_NAME
SD_NAME = s2j.SD_NAME
SD_SUM_NAME = s2j.SD_SUM_NAME
# endregion


# region --- splash-zone connection scope ---
def splash_connections(connections, model):
    """Every connection whose node sits in the splash band (sd_geometry.
    environment_zone on the node's z) -- the only connections this module
    ever touches."""
    mudline_z = min(z for _, _, z in model["joints"].values())
    out = []
    for c in connections:
        z = model["joints"][c["node"]][2]
        if sdg.environment_zone(z, mudline_z) == "splash":
            out.append(c)
    return out
# endregion


# region --- corroded connection copy ---
def _corroded_connection(c, model, other_row, year):
    """A COPY of connection `c` with brace_D/brace_t/chord_a_D/chord_a_T/
    chord_b_D/chord_b_T overwritten by their corroded-at-year values --
    every OTHER field (phi_deg, member/end ids, family, ...) is untouched,
    since stage2_joints._connection_signals() reads exactly those
    geometrically-unaffected fields straight through. See module docstring."""
    g = jgc.corroded_connection_geometry(c, model, other_row, year)
    corroded = dict(c, brace_D=g["brace_D"], brace_t=g["brace_t"])
    if c["family"] != "X":
        Da, Ta, _Wa = jgc.corroded_chord_segment(model, c["chord_a_member"], year)
        Db, Tb, _Wb = jgc.corroded_chord_segment(model, c["chord_b_member"], year)
        corroded.update(chord_a_D=Da, chord_a_T=Ta, chord_b_D=Db, chord_b_T=Tb)
    return corroded
# endregion


def _scf_index(rows):
    index = {}
    for r in rows:
        key = (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
               r["chord_t_scenario"], r["direction"])
        index.setdefault(key, []).append(r)
    return index


def _k_pairing(connections):
    """{(sub_joint_id, chord_t_scenario): [row_a, row_b]}, restricted to the
    connections passed in -- reuses scf._group_k_planes() (a pure topology
    grouping, unaffected by corrosion)."""
    return scf._group_k_planes(connections)


# region --- provenance stamp ---
def build_stamp(outb_path, header, model, case_json, owner_json, n_points, years):
    return dict(
        pipeline_version=cfg.PIPELINE_VERSION,
        stage="joint_corrosion",
        hotspot_joint_verified=stress.HOTSPOT_JOINT_VERIFIED,
        scf_equations_verified=scf.SCF_EQUATIONS_VERIFIED,
        n_points=n_points,
        corrosion_years=list(years),
        corrosion_rate_mm_per_year_per_surface=sdg.CORROSION_RATE_MM_PER_YEAR_PER_SURFACE,
        leg_diameter_min_m=sdg.LEG_DIAMETER_MIN_M,
        theta_8_deg=stress.THETA_8_DEG.tolist(),
        bin_edges_mpa=cfg.BIN_EDGES_MPA.tolist(),
        wohler_exponents=list(cfg.WOHLER_EXPONENTS),
        units=dict(stress="MPa", force="N", moment="N*m"),
        force_family_brace=list(obr.FATIGUE_COMPONENTS),
        force_family_chord=["MKxe", "MKye"],
        transient_cutoff_s=cfg.TRANSIENT_CUTOFF_S,
        subdyn_md5=model["md5"],
        outb_path=str(outb_path),
        outb_size_bytes=header["filesize"],
        outb_mtime=Path(outb_path).stat().st_mtime,
        dt_s=header["t_incr"],
        n_t=header["n_t"],
        n_chan=header["n_chan"],
        library_versions=dict(numpy=np.__version__, rainflow=rainflow.__version__),
        case_json=case_json,
        owner_json=owner_json,
    )


def _stamp_recompute_key(stamp):
    return (
        stamp["pipeline_version"], stamp["n_points"], tuple(stamp["corrosion_years"]),
        tuple(sorted((k, tuple(sorted(v.items())))
                     for k, v in stamp["corrosion_rate_mm_per_year_per_surface"].items())),
        stamp["leg_diameter_min_m"],
        tuple(stamp["bin_edges_mpa"]), tuple(stamp["wohler_exponents"]),
        stamp["transient_cutoff_s"], stamp["subdyn_md5"],
        stamp["outb_size_bytes"], stamp["outb_mtime"],
    )
# endregion


# region --- write side ---
def _npz_json_paths(case_dir, out_root=STAGE2_JOINTS_CORROSION_DIR):
    case_dir = Path(case_dir)
    cond, seed = case_dir.parent.name, case_dir.name
    npz_path = out_root / cond / f"{seed}.npz"
    return npz_path, npz_path.with_suffix(".json")


def process_run_corroded(case_dir, years, out_root=STAGE2_JOINTS_CORROSION_DIR, force=False):
    """Compute (or skip, if a stamp-matching .npz/.json pair already
    exists) the corrosion-aware joint Stage 2 file for one run, at every
    year in `years`, splash-zone connections only. Returns the .npz path."""
    case_dir = Path(case_dir)
    outb_path = case_dir / OUTB_NAME
    sd_path = case_dir / SD_NAME
    sd_sum_path = case_dir / SD_SUM_NAME
    case_json = json.loads((case_dir / "case.json").read_text())
    owner_path = case_dir / "owner.json"
    owner_json = json.loads(owner_path.read_text()) if owner_path.exists() else {}

    header = obr.read_outb_header(outb_path)
    connections, model = s2j.build_full_connections(sd_path, sd_sum_path)
    splash = splash_connections(connections, model)
    k_groups = _k_pairing(splash)

    # scf_index at year=0 fixes the point ordering (the SET of (connection,
    # treatment, side, position, segment) rows is identical at every year --
    # only their numeric SCF/damage values change, see module docstring).
    scf_index_y0 = _scf_index(scfc.compute_scf_corroded(splash, model, year=0))
    point_table = s2j.build_point_table(splash, scf_index_y0, model)
    n_points = len(point_table)

    stamp = build_stamp(outb_path, header, model, case_json, owner_json, n_points, years)
    npz_path, json_path = _npz_json_paths(case_dir, out_root)

    if not force and npz_path.exists() and json_path.exists():
        try:
            existing_stamp = json.loads(json_path.read_text())
            if _stamp_recompute_key(existing_stamp) == _stamp_recompute_key(stamp):
                return npz_path
        except (json.JSONDecodeError, KeyError):
            pass

    n_bins = cfg.N_BINS
    n_years = len(years)
    exponents = cfg.WOHLER_EXPONENTS
    counts = np.zeros((n_points, n_years, n_bins), dtype=np.float64)
    sum_r = {m: np.zeros((n_points, n_years, n_bins), dtype=np.float64) for m in exponents}
    n_under = np.zeros((n_points, n_years), dtype=np.float64)
    n_over = np.zeros((n_points, n_years), dtype=np.float64)

    for y_idx, year in enumerate(years):
        scf_index_year = _scf_index(scfc.compute_scf_corroded(splash, model, year=year))

        signal_cache = {}

        def _signals_for(c):
            key = s2j._connection_key(c)
            if key not in signal_cache:
                if c["family"] == "K":
                    pair = k_groups[(c["sub_joint_id"], c["chord_t_scenario"])]
                    other = pair[0] if pair[1] is c else pair[1]
                else:
                    other = None
                corroded_c = _corroded_connection(c, model, other, year)
                signal_cache[key] = s2j._connection_signals(
                    corroded_c, outb_path, header, cfg.TRANSIENT_CUTOFF_S)
            return signal_cache[key]

        point_id = 0
        for c, sr, variant_positions in s2j.iter_assessment_rows(splash, scf_index_year):
            cs = _signals_for(c)
            AC_base, AC_att = sr["SCF_AC_base"], sr["SCF_AC_att"]
            AS, MIP, MOP = sr["SCF_AS"], sr["SCF_MIP"], sr["SCF_MOP"]

            if c["family"] == "X":
                result_a = stress.hotspot_joint(cs["sig_ax"], cs["sig_mip"], cs["sig_mop"],
                                                 cs["sig_cb_a"], AC_base, AC_att, AS, MIP, MOP)
                result_b = result_a
            else:
                result_a = stress.hotspot_joint(cs["sig_ax"], cs["sig_mip"], cs["sig_mop"],
                                                 cs["sig_cb_a"], AC_base, AC_att, AS, MIP, MOP)
                result_b = stress.hotspot_joint(cs["sig_ax"], cs["sig_mip"], cs["sig_mop"],
                                                 cs["sig_cb_b"], AC_base, AC_att, AS, MIP, MOP)

            for pos, seg in variant_positions:
                signal = result_b[pos] if seg == "b" else result_a[pos]
                cycles = list(rainflow.extract_cycles(signal))
                c_hist, sr_hist, nu, no = rhist.cycles_to_histogram(cycles)
                counts[point_id, y_idx, :] = c_hist
                for m in exponents:
                    sum_r[m][point_id, y_idx, :] = sr_hist[m]
                n_under[point_id, y_idx] = nu
                n_over[point_id, y_idx] = no
                point_id += 1

        assert point_id == n_points, (
            f"year {year}: iter_assessment_rows produced {point_id} signals but "
            f"build_point_table counted {n_points} -- point ordering diverged across years"
        )

    save_kwargs = dict(
        counts=counts, n_under=n_under, n_over=n_over,
        bin_edges_mpa=cfg.BIN_EDGES_MPA, years=np.array(years, dtype=np.int64),
        point_id=np.array([r["point_id"] for r in point_table], dtype=np.int64),
        node=np.array([r["node"] for r in point_table], dtype=np.int64),
        sub_joint_id=np.array([r["sub_joint_id"] for r in point_table], dtype="<U16"),
        plane_id=np.array([r["plane_id"] for r in point_table], dtype=np.int64),
        family=np.array([r["family"] for r in point_table], dtype="<U4"),
        type_label=np.array([r["type_label"] for r in point_table], dtype="<U4"),
        treatment=np.array([r["treatment"] for r in point_table], dtype="<U4"),
        side=np.array([r["side"] for r in point_table], dtype="<U8"),
        brace_member=np.array([r["brace_member"] for r in point_table], dtype=np.int64),
        brace_end=np.array([r["brace_end"] for r in point_table], dtype=np.int64),
        chord_t_scenario=np.array([r["chord_t_scenario"] for r in point_table], dtype="<U8"),
        direction=np.array([r["direction"] for r in point_table], dtype="<U16"),
        position=np.array([r["position"] for r in point_table], dtype="<U2"),
        segment=np.array([r["segment"] for r in point_table], dtype="<U2"),
        z=np.array([r["z"] for r in point_table], dtype=np.float64),
        brace_D0=np.array([r["brace_D"] for r in point_table], dtype=np.float64),
        brace_t0=np.array([r["brace_t"] for r in point_table], dtype=np.float64),
        chord_D0=np.array([r["chord_D"] for r in point_table], dtype=np.float64),
        chord_T0=np.array([r["chord_T"] for r in point_table], dtype=np.float64),
    )
    for m in exponents:
        save_kwargs[f"sum_r{m}"] = sum_r[m]

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = npz_path.with_name(npz_path.name + ".tmp")
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(f, **save_kwargs)
    os.replace(tmp_npz, npz_path)

    tmp_json = json_path.with_name(json_path.name + ".tmp")
    tmp_json.write_text(json.dumps(stamp, indent=2))
    os.replace(tmp_json, json_path)

    return npz_path
# endregion


# region --- read side ---
def load_stage2_joints_corrosion(npz_path):
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)
    exponents = sorted(int(k[len("sum_r"):]) for k in data.files if k.startswith("sum_r"))
    sum_r = {m: data[f"sum_r{m}"] for m in exponents}
    n_points = data["counts"].shape[0]
    point_table = [
        dict(
            point_id=int(data["point_id"][i]), node=int(data["node"][i]),
            sub_joint_id=str(data["sub_joint_id"][i]), plane_id=int(data["plane_id"][i]),
            family=str(data["family"][i]), type_label=str(data["type_label"][i]),
            treatment=str(data["treatment"][i]), side=str(data["side"][i]),
            brace_member=int(data["brace_member"][i]), brace_end=int(data["brace_end"][i]),
            chord_t_scenario=str(data["chord_t_scenario"][i]), direction=str(data["direction"][i]),
            position=str(data["position"][i]), segment=str(data["segment"][i]),
            z=float(data["z"][i]),
            brace_D0=float(data["brace_D0"][i]), brace_t0=float(data["brace_t0"][i]),
            chord_D0=float(data["chord_D0"][i]), chord_T0=float(data["chord_T0"][i]),
        )
        for i in range(n_points)
    ]
    json_path = npz_path.with_suffix(".json")
    stamp = json.loads(json_path.read_text()) if json_path.exists() else None
    return dict(
        counts=data["counts"], sum_r=sum_r, n_under=data["n_under"], n_over=data["n_over"],
        bin_edges_mpa=data["bin_edges_mpa"], years=data["years"].tolist(),
        point_table=point_table, stamp=stamp,
    )
# endregion


def _self_check():
    import time

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    assert case_dir.exists(), f"missing fixture: {case_dir}"

    print(f"HOTSPOT_JOINT_VERIFIED = {stress.HOTSPOT_JOINT_VERIFIED}")
    print(f"SCF_EQUATIONS_VERIFIED = {scf.SCF_EQUATIONS_VERIFIED}")

    years = [5, 10, 15, 20, 25]
    out_root = RESULTS_DIR / "_stage2_joints_corrosion_selfcheck"
    npz_path, json_path = _npz_json_paths(case_dir, out_root)
    for p in (npz_path, json_path):
        if p.exists():
            p.unlink()

    print(f"\nprocessing {case_dir.parent.name}/{case_dir.name} -> {npz_path}")
    print(f"years = {years}")
    t0 = time.time()
    p1 = process_run_corroded(case_dir, years, out_root=out_root)
    dt_first = time.time() - t0
    print(f"  first run: {dt_first:.1f} s")
    assert p1 == npz_path and npz_path.exists() and json_path.exists()

    npz_bytes_1 = npz_path.read_bytes()
    t0 = time.time()
    p2 = process_run_corroded(case_dir, years, out_root=out_root)
    dt_second = time.time() - t0
    print(f"  second run (skip-if-exists): {dt_second:.2f} s")
    assert p2 == npz_path
    assert dt_second < 1.0, "skip-if-exists did not skip"
    assert npz_path.read_bytes() == npz_bytes_1

    stage2c = load_stage2_joints_corrosion(npz_path)
    n_points = len(stage2c["point_table"])
    print(f"\n  loaded: {n_points} points, counts shape {stage2c['counts'].shape}, "
          f"years {stage2c['years']}")
    assert stage2c["counts"].shape == (n_points, len(years), cfg.N_BINS)

    families = {}
    for r in stage2c["point_table"]:
        families[r["family"]] = families.get(r["family"], 0) + 1
    print(f"  signals by family: {families}")
    # 16 K connections x 2 treatments x 2 sides x 14 signals = 896
    # 8 X connections x 1 treatment x 2 sides x 8 signals = 128
    assert families == {"K": 896, "X": 128}, (
        "unexpected splash-zone family/signal composition -- see module docstring's "
        "'16 K + 8 X, zero TY' measured scope"
    )

    print(f"\n  n_over max: {stage2c['n_over'].max()} (expect 0.0)")
    assert stage2c["n_over"].max() == 0.0

    # --- independent recheck: one K connection, chord side, K-treatment,
    # crown-toe segment a, year 25 -- recompute directly, never touching
    # this module's own arrays.
    connections, model = s2j.build_full_connections(case_dir / SD_NAME, case_dir / SD_SUM_NAME)
    splash = splash_connections(connections, model)
    k_groups = _k_pairing(splash)
    target_row = next(
        r for r in stage2c["point_table"]
        if r["family"] == "K" and r["treatment"] == "K" and r["side"] == "chord"
        and r["position"] == "1" and r["segment"] == "a"
    )
    c = next(c for c in splash if c["node"] == target_row["node"]
             and c["brace_member"] == target_row["brace_member"]
             and c["brace_end"] == target_row["brace_end"]
             and c["chord_t_scenario"] == target_row["chord_t_scenario"])
    pair = k_groups[(c["sub_joint_id"], c["chord_t_scenario"])]
    other = pair[0] if pair[1] is c else pair[1]

    year_idx = 4  # year=25
    year = years[year_idx]
    sr = next(r for r in scfc.compute_scf_corroded(splash, model, year=year)
              if r["node"] == c["node"] and r["sub_joint_id"] == c["sub_joint_id"]
              and r["brace_member"] == c["brace_member"] and r["brace_end"] == c["brace_end"]
              and r["chord_t_scenario"] == c["chord_t_scenario"] and r["direction"] == c["direction"]
              and r["treatment"] == "K" and r["side"] == "chord")

    corroded_c = _corroded_connection(c, model, other, year)
    header = obr.read_outb_header(case_dir / OUTB_NAME)
    cs = s2j._connection_signals(corroded_c, case_dir / OUTB_NAME, header, cfg.TRANSIENT_CUTOFF_S)
    result_direct = stress.hotspot_joint(
        cs["sig_ax"], cs["sig_mip"], cs["sig_mop"], cs["sig_cb_a"],
        sr["SCF_AC_base"], sr["SCF_AC_att"], sr["SCF_AS"], sr["SCF_MIP"], sr["SCF_MOP"])
    cycles_direct = list(rainflow.extract_cycles(result_direct["1"]))

    print(f"\n  independent recheck: node {c['node']} brace M{c['brace_member']}J{c['brace_end']} "
          f"K-treatment, chord-side, crown toe, segment a, year {year} "
          f"(point_id {target_row['point_id']}):")
    for m in cfg.WOHLER_EXPONENTS:
        d_direct = rhist.damage_from_cycles(cycles_direct, m)
        d_stored = float(np.sum(stage2c["sum_r"][m][target_row["point_id"], year_idx, :]))
        rel_diff = abs(d_direct - d_stored) / d_direct if d_direct > 0 else abs(d_direct - d_stored)
        print(f"    m={m}: direct={d_direct:.6e}  stored={d_stored:.6e}  rel.diff={rel_diff:.3e}")
        assert rel_diff < 1e-9, f"m={m}: stored histogram doesn't match direct recompute"

    # --- damage should increase with year at this point/exponent (thinner
    # section + rising SCF -> more damage), same monotonic check as the
    # member-track corrosion self-check.
    pid = target_row["point_id"]
    d_by_year = [float(np.sum(stage2c["sum_r"][3][pid, y_idx2, :])) for y_idx2 in range(len(years))]
    print(f"\n  damage vs year (point {pid}, m=3): {[f'{d:.3e}' for d in d_by_year]}")
    assert all(d_by_year[i] <= d_by_year[i + 1] for i in range(len(d_by_year) - 1)), \
        "damage should be monotonically non-decreasing with corrosion year"

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
