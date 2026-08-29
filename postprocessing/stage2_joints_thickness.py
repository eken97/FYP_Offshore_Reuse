"""
Joint-track can-thickness retrofit, Stage 2 -- full-population, no
corrosion. Mirrors stage2_joints.py's own process_run exactly (same
counts/sum_r shape, (n_points, n_bins), no year axis), reusing its already-
tested machinery UNEDITED (build_full_connections, _connection_key,
_connection_signals, iter_assessment_rows, build_point_table) -- same
pattern stage2_joints_corrosion.py already established for corrosion.

SCOPE: ALL 120 connections, not the 24 splash-zone ones the corrosion
tracks use -- the can retrofit is a structure-wide intervention (see
joint_thickness_override.py's module docstring), not a corrosion-only
local effect.

One entry point per scenario ("A"/"B"), selected by the `scenario`
argument threaded through process_run_retrofit -- NOT baked into separate
files, since the two scenarios are the same computation with a different
override spec (see joint_thickness_override.SCENARIOS).
"""
import json
import os
from pathlib import Path

import numpy as np
import rainflow

import fatigue_config as cfg
import outb_reader as obr
import rainflow_hist as rhist
import stress
import scf
import stage2_joints as s2j
import joint_thickness_override as jto
import scf_thickness_override as sto

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

# Same D:-Z: scan convention as every other Stage 2 module -- duplicated,
# not imported, see stage2_joints_corrosion.py's own docstring for why.
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
    STAGE2_JOINTS_THICKNESS_DIR = _drive / "Postprocessing" / "stage2_joints_thickness"
else:
    STAGE2_JOINTS_THICKNESS_DIR = RESULTS_DIR / "_stage2_joints_thickness_local"
    print(f"WARNING: external drive not found (checked D:-Z: for a '{_DRIVE_SUBFOLDER}' "
          f"folder containing '{_DRIVE_MARKER}'). Joint retrofit Stage 2 cache falling "
          f"back to local {STAGE2_JOINTS_THICKNESS_DIR}")

OUTB_NAME = s2j.OUTB_NAME
SD_NAME = s2j.SD_NAME
SD_SUM_NAME = s2j.SD_SUM_NAME
# endregion


def _retrofitted_connection(c, model, scenario):
    """A COPY of connection `c` with brace_D/brace_t/chord_a_D/chord_a_T/
    chord_b_D/chord_b_T overwritten by their retrofitted-under-`scenario`
    values -- every OTHER field untouched, since
    stage2_joints._connection_signals() reads exactly those geometrically-
    affected fields straight through (same pattern as
    stage2_joints_corrosion._corroded_connection)."""
    g = jto.retrofit_connection_geometry(c, model, scenario)
    retrofitted = dict(c, brace_D=g["brace_D"], brace_t=g["brace_t"])
    if c["family"] != "X":
        retrofitted.update(chord_a_D=g["chord_a_D"], chord_a_T=g["chord_a_T"],
                            chord_b_D=g["chord_b_D"], chord_b_T=g["chord_b_T"])
    return retrofitted


def _scf_index(rows):
    index = {}
    for r in rows:
        key = (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
               r["chord_t_scenario"], r["direction"])
        index.setdefault(key, []).append(r)
    return index


# region --- provenance stamp ---
def build_stamp(outb_path, header, model, case_json, owner_json, n_points, scenario):
    return dict(
        pipeline_version=cfg.PIPELINE_VERSION,
        stage="joint_thickness",
        scenario=scenario,
        hotspot_joint_verified=stress.HOTSPOT_JOINT_VERIFIED,
        scf_equations_verified=scf.SCF_EQUATIONS_VERIFIED,
        n_points=n_points,
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
        stamp["pipeline_version"], stamp["scenario"], stamp["n_points"],
        tuple(stamp["bin_edges_mpa"]), tuple(stamp["wohler_exponents"]),
        stamp["transient_cutoff_s"], stamp["subdyn_md5"],
        stamp["outb_size_bytes"], stamp["outb_mtime"],
    )
# endregion


# region --- write side ---
def _npz_json_paths(case_dir, scenario, out_root=STAGE2_JOINTS_THICKNESS_DIR):
    case_dir = Path(case_dir)
    cond, seed = case_dir.parent.name, case_dir.name
    npz_path = out_root / scenario / cond / f"{seed}.npz"
    return npz_path, npz_path.with_suffix(".json")


def process_run_retrofit(case_dir, scenario, out_root=STAGE2_JOINTS_THICKNESS_DIR, force=False):
    """Compute (or skip, if a stamp-matching .npz/.json pair already
    exists) the can-thickness-retrofit joint Stage 2 file for one run,
    under `scenario` ("A"/"B"), ALL 120 connections. Returns the .npz
    path."""
    if scenario not in jto.SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {jto.SCENARIOS}")
    case_dir = Path(case_dir)
    outb_path = case_dir / OUTB_NAME
    sd_path = case_dir / SD_NAME
    sd_sum_path = case_dir / SD_SUM_NAME
    case_json = json.loads((case_dir / "case.json").read_text())
    owner_path = case_dir / "owner.json"
    owner_json = json.loads(owner_path.read_text()) if owner_path.exists() else {}

    header = obr.read_outb_header(outb_path)
    connections, model = s2j.build_full_connections(sd_path, sd_sum_path)

    scf_index = _scf_index(sto.compute_scf_retrofit(connections, model, scenario))
    point_table = s2j.build_point_table(connections, scf_index, model)
    n_points = len(point_table)

    stamp = build_stamp(outb_path, header, model, case_json, owner_json, n_points, scenario)
    npz_path, json_path = _npz_json_paths(case_dir, scenario, out_root)

    if not force and npz_path.exists() and json_path.exists():
        try:
            existing_stamp = json.loads(json_path.read_text())
            if _stamp_recompute_key(existing_stamp) == _stamp_recompute_key(stamp):
                return npz_path
        except (json.JSONDecodeError, KeyError):
            pass

    n_bins = cfg.N_BINS
    exponents = cfg.WOHLER_EXPONENTS
    counts = np.zeros((n_points, n_bins), dtype=np.float64)
    sum_r = {m: np.zeros((n_points, n_bins), dtype=np.float64) for m in exponents}
    n_under = np.zeros(n_points, dtype=np.float64)
    n_over = np.zeros(n_points, dtype=np.float64)

    signal_cache = {}

    def _signals_for(c):
        key = s2j._connection_key(c)
        if key not in signal_cache:
            retrofitted_c = _retrofitted_connection(c, model, scenario)
            signal_cache[key] = s2j._connection_signals(
                retrofitted_c, outb_path, header, cfg.TRANSIENT_CUTOFF_S)
        return signal_cache[key]

    point_id = 0
    for c, sr, variant_positions in s2j.iter_assessment_rows(connections, scf_index):
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
            counts[point_id, :] = c_hist
            for m in exponents:
                sum_r[m][point_id, :] = sr_hist[m]
            n_under[point_id] = nu
            n_over[point_id] = no
            point_id += 1

    assert point_id == n_points

    save_kwargs = dict(
        counts=counts, n_under=n_under, n_over=n_over,
        bin_edges_mpa=cfg.BIN_EDGES_MPA,
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
        brace_D=np.array([r["brace_D"] for r in point_table], dtype=np.float64),
        brace_t=np.array([r["brace_t"] for r in point_table], dtype=np.float64),
        chord_D=np.array([r["chord_D"] for r in point_table], dtype=np.float64),
        chord_T=np.array([r["chord_T"] for r in point_table], dtype=np.float64),
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
def load_stage2_joints_thickness(npz_path):
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
            brace_D=float(data["brace_D"][i]), brace_t=float(data["brace_t"][i]),
            chord_D=float(data["chord_D"][i]), chord_T=float(data["chord_T"][i]),
        )
        for i in range(n_points)
    ]
    json_path = npz_path.with_suffix(".json")
    stamp = json.loads(json_path.read_text()) if json_path.exists() else None
    return dict(
        counts=data["counts"], sum_r=sum_r, n_under=data["n_under"], n_over=data["n_over"],
        bin_edges_mpa=data["bin_edges_mpa"], point_table=point_table, stamp=stamp,
    )
# endregion


def _self_check():
    import time

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    assert case_dir.exists(), f"missing fixture: {case_dir}"

    print(f"HOTSPOT_JOINT_VERIFIED = {stress.HOTSPOT_JOINT_VERIFIED}")
    print(f"SCF_EQUATIONS_VERIFIED = {scf.SCF_EQUATIONS_VERIFIED}")

    for scenario in jto.SCENARIOS:
        out_root = RESULTS_DIR / "_stage2_joints_thickness_selfcheck"
        npz_path, json_path = _npz_json_paths(case_dir, scenario, out_root)
        for p in (npz_path, json_path):
            if p.exists():
                p.unlink()

        print(f"\n=== scenario {scenario} ===")
        print(f"processing {case_dir.parent.name}/{case_dir.name} -> {npz_path}")
        t0 = time.time()
        p1 = process_run_retrofit(case_dir, scenario, out_root=out_root)
        dt_first = time.time() - t0
        print(f"  first run: {dt_first:.1f} s")
        assert p1 == npz_path and npz_path.exists() and json_path.exists()

        npz_bytes_1 = npz_path.read_bytes()
        t0 = time.time()
        p2 = process_run_retrofit(case_dir, scenario, out_root=out_root)
        dt_second = time.time() - t0
        print(f"  second run (skip-if-exists): {dt_second:.2f} s")
        assert p2 == npz_path
        # Loose bound (vs. the ~80s first-run cost) -- the invariant being
        # tested is "skip avoids the .outb read + rainflow work", not a tight
        # wall-clock number. Building the 368-row SCF index + point table
        # (SD.dat/DCM parse, geometry, scf.py dispatch) still happens before
        # the stamp check either way -- same as stage2_joints.py's own
        # process_run -- so a few seconds of that overhead is expected.
        assert dt_second < 10.0, "skip-if-exists did not skip"
        assert npz_path.read_bytes() == npz_bytes_1

        s2t = load_stage2_joints_thickness(npz_path)
        n_points = len(s2t["point_table"])
        print(f"  loaded: {n_points} points, counts shape {s2t['counts'].shape}")
        # full population, matches stage2_joints.py's own uncorroded scope
        # exactly: 128 K + 24 TY... wait, TY carries 1 treatment x 2 sides
        # x 8 signals per connection, K carries 2 treatments x 2 sides x 14,
        # X carries 1 treatment x 2 sides x 8 -- see stage2_joints.py's own
        # module docstring for the full derivation; just confirm the total
        # matches the uncorroded reference build's own point count.
        connections, model = s2j.build_full_connections(case_dir / SD_NAME, case_dir / SD_SUM_NAME)
        ref_scf_index = _scf_index(scf.compute_all_scf(connections))
        ref_point_table = s2j.build_point_table(connections, ref_scf_index, model)
        assert n_points == len(ref_point_table), (
            f"scenario {scenario}: point count {n_points} != uncorroded reference "
            f"{len(ref_point_table)} -- retrofit must not change WHICH signals exist, "
            f"only their magnitudes"
        )

        print(f"  n_over max: {s2t['n_over'].max()} (expect 0.0)")
        assert s2t["n_over"].max() == 0.0

        # --- independent recheck: the real X-bottom worst point (node
        # 39/brace 47, chord side, position 3 saddle -- the same connection
        # docs/decisions.md ballpark work anchored to)
        # matches a direct recompute.
        target_row = next(
            r for r in s2t["point_table"]
            if r["node"] == 39 and r["brace_member"] == 47 and r["side"] == "chord"
            and r["position"] == "3"
        )
        c = next(c for c in connections if c["node"] == 39 and c["brace_member"] == 47
                 and c["direction"] == target_row["direction"])
        sr = next(r for r in sto.compute_scf_retrofit(connections, model, scenario)
                  if r["node"] == c["node"] and r["sub_joint_id"] == c["sub_joint_id"]
                  and r["brace_member"] == c["brace_member"] and r["brace_end"] == c["brace_end"]
                  and r["chord_t_scenario"] == c["chord_t_scenario"] and r["direction"] == c["direction"]
                  and r["treatment"] == "X" and r["side"] == "chord")
        retrofitted_c = _retrofitted_connection(c, model, scenario)
        header = obr.read_outb_header(case_dir / OUTB_NAME)
        cs = s2j._connection_signals(retrofitted_c, case_dir / OUTB_NAME, header, cfg.TRANSIENT_CUTOFF_S)
        result_direct = stress.hotspot_joint(
            cs["sig_ax"], cs["sig_mip"], cs["sig_mop"], cs["sig_cb_a"],
            sr["SCF_AC_base"], sr["SCF_AC_att"], sr["SCF_AS"], sr["SCF_MIP"], sr["SCF_MOP"])
        cycles_direct = list(rainflow.extract_cycles(result_direct["3"]))

        print(f"\n  independent recheck: node 39/brace 47, X-treatment, chord-side, "
              f"saddle (position 3), point_id {target_row['point_id']}:")
        for m in cfg.WOHLER_EXPONENTS:
            d_direct = rhist.damage_from_cycles(cycles_direct, m)
            d_stored = float(np.sum(s2t["sum_r"][m][target_row["point_id"], :]))
            rel_diff = abs(d_direct - d_stored) / d_direct if d_direct > 0 else abs(d_direct - d_stored)
            print(f"    m={m}: direct={d_direct:.6e}  stored={d_stored:.6e}  rel.diff={rel_diff:.3e}")
            assert rel_diff < 1e-9, f"m={m}: stored histogram doesn't match direct recompute"

        # --- process the other 5 seeds too (Stage 3's aggregate_condition
        # needs all 6 to compute a real seed mean/std) ---
        print(f"\n  processing remaining 5 seeds ...")
        for seed_dir in sorted((DEV_FIXTURE_DIR / "LC_V20_H3p5_T8").glob("S*")):
            if seed_dir.name == case_dir.name:
                continue
            t0 = time.time()
            process_run_retrofit(seed_dir, scenario, out_root=out_root)
            print(f"    {seed_dir.name}: {time.time() - t0:.1f} s")

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
