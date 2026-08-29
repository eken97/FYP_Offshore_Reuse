"""
Joint-track can-thickness retrofit + corrosion composed, Stage 2.

Direct analogue of stage2_joints_corrosion.py, substituting
joint_thickness_override.retrofit_and_corrode_connection_geometry (retrofit
FIRST, defining a new year-0 baseline, THEN corrosion loss from that
baseline) for joint_geometry_corrosion.corroded_connection_geometry, and
scf_thickness_override.compute_scf_retrofit_corroded for
scf_corrosion.compute_scf_corroded. Same splash-zone-only scope as the
corrosion track (corrosion is only physically defined there) -- the
retrofit itself already applies structure-wide via
stage2_joints_thickness.py; this module only adds corrosion ON TOP of the
already-thickened splash-zone connections.

Reuses stage2_joints.py's own machinery (build_full_connections,
_connection_key, _connection_signals, iter_assessment_rows,
build_point_table) and stage2_joints_corrosion.py's splash_connections()
UNEDITED -- same "thin driver" pattern as every other module in this
retrofit build.
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
import stage2_joints_corrosion as s2jc
import joint_thickness_override as jto
import scf_thickness_override as sto

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

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
    STAGE2_JOINTS_THICKNESS_CORROSION_DIR = _drive / "Postprocessing" / "stage2_joints_thickness_corrosion"
else:
    STAGE2_JOINTS_THICKNESS_CORROSION_DIR = RESULTS_DIR / "_stage2_joints_thickness_corrosion_local"
    print(f"WARNING: external drive not found (checked D:-Z: for a '{_DRIVE_SUBFOLDER}' "
          f"folder containing '{_DRIVE_MARKER}'). Joint retrofit+corrosion Stage 2 cache "
          f"falling back to local {STAGE2_JOINTS_THICKNESS_CORROSION_DIR}")

OUTB_NAME = s2j.OUTB_NAME
SD_NAME = s2j.SD_NAME
SD_SUM_NAME = s2j.SD_SUM_NAME

splash_connections = s2jc.splash_connections   # reused unedited
# endregion


def _retrofitted_corroded_connection(c, model, other_row, scenario, year):
    """A COPY of connection `c` with brace_D/brace_t/chord_a_D/chord_a_T/
    chord_b_D/chord_b_T overwritten by their retrofit-then-corroded values
    -- same pattern as stage2_joints_corrosion._corroded_connection and
    stage2_joints_thickness._retrofitted_connection, composed."""
    g = jto.retrofit_and_corrode_connection_geometry(c, model, other_row, scenario, year)
    out = dict(c, brace_D=g["brace_D"], brace_t=g["brace_t"])
    if c["family"] != "X":
        # chord_a/b_T need the SAME retrofit-then-corrosion treatment as
        # chord_T itself -- recomputed directly (not via chord_D/chord_T,
        # which are the post-max/min-pick values) since chord_a and chord_b
        # are the two individual physical leg segments.
        g_retrofit = jto.retrofit_connection_geometry(c, model, scenario)
        leg_a_mid, leg_b_mid = c["chord_members"][0][0], c["chord_members"][1][0]
        Da, Ta = jto.corrode_from_baseline(
            g_retrofit["chord_a_D"], g_retrofit["chord_a_T"], sdg.member_class(model, leg_a_mid), year)
        Db, Tb = jto.corrode_from_baseline(
            g_retrofit["chord_b_D"], g_retrofit["chord_b_T"], sdg.member_class(model, leg_b_mid), year)
        out.update(chord_a_D=Da, chord_a_T=Ta, chord_b_D=Db, chord_b_T=Tb)
    return out


def _scf_index(rows):
    index = {}
    for r in rows:
        key = (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
               r["chord_t_scenario"], r["direction"])
        index.setdefault(key, []).append(r)
    return index


def _k_pairing(connections):
    return scf._group_k_planes(connections)


# region --- provenance stamp ---
def build_stamp(outb_path, header, model, case_json, owner_json, n_points, scenario, years):
    return dict(
        pipeline_version=cfg.PIPELINE_VERSION,
        stage="joint_thickness_corrosion",
        scenario=scenario,
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
        stamp["pipeline_version"], stamp["scenario"], stamp["n_points"], tuple(stamp["corrosion_years"]),
        tuple(sorted((k, tuple(sorted(v.items())))
                     for k, v in stamp["corrosion_rate_mm_per_year_per_surface"].items())),
        stamp["leg_diameter_min_m"],
        tuple(stamp["bin_edges_mpa"]), tuple(stamp["wohler_exponents"]),
        stamp["transient_cutoff_s"], stamp["subdyn_md5"],
        stamp["outb_size_bytes"], stamp["outb_mtime"],
    )
# endregion


# region --- write side ---
def _npz_json_paths(case_dir, scenario, out_root=STAGE2_JOINTS_THICKNESS_CORROSION_DIR):
    case_dir = Path(case_dir)
    cond, seed = case_dir.parent.name, case_dir.name
    npz_path = out_root / scenario / cond / f"{seed}.npz"
    return npz_path, npz_path.with_suffix(".json")


def process_run_retrofit_corroded(case_dir, scenario, years,
                                   out_root=STAGE2_JOINTS_THICKNESS_CORROSION_DIR, force=False):
    """Compute (or skip) the retrofit+corrosion joint Stage 2 file for one
    run, under `scenario`, at every year in `years`, splash-zone
    connections only. Returns the .npz path."""
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
    splash = splash_connections(connections, model)
    k_groups = _k_pairing(splash)

    scf_index_y0 = _scf_index(sto.compute_scf_retrofit_corroded(splash, model, scenario, year=0))
    point_table = s2j.build_point_table(splash, scf_index_y0, model)
    n_points = len(point_table)

    stamp = build_stamp(outb_path, header, model, case_json, owner_json, n_points, scenario, years)
    npz_path, json_path = _npz_json_paths(case_dir, scenario, out_root)

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
        scf_index_year = _scf_index(sto.compute_scf_retrofit_corroded(splash, model, scenario, year=year))

        signal_cache = {}

        def _signals_for(c):
            key = s2j._connection_key(c)
            if key not in signal_cache:
                if c["family"] == "K":
                    pair = k_groups[(c["sub_joint_id"], c["chord_t_scenario"])]
                    other = pair[0] if pair[1] is c else pair[1]
                else:
                    other = None
                rc = _retrofitted_corroded_connection(c, model, other, scenario, year)
                signal_cache[key] = s2j._connection_signals(
                    rc, outb_path, header, cfg.TRANSIENT_CUTOFF_S)
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

        assert point_id == n_points

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
def load_stage2_joints_thickness_corrosion(npz_path):
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
    for scenario in jto.SCENARIOS:
        out_root = RESULTS_DIR / "_stage2_joints_thickness_corrosion_selfcheck"
        npz_path, json_path = _npz_json_paths(case_dir, scenario, out_root)
        for p in (npz_path, json_path):
            if p.exists():
                p.unlink()

        print(f"\n=== scenario {scenario} ===")
        print(f"processing {case_dir.parent.name}/{case_dir.name} -> {npz_path}")
        t0 = time.time()
        p1 = process_run_retrofit_corroded(case_dir, scenario, years, out_root=out_root)
        dt_first = time.time() - t0
        print(f"  first run: {dt_first:.1f} s")
        assert p1 == npz_path and npz_path.exists() and json_path.exists()

        npz_bytes_1 = npz_path.read_bytes()
        t0 = time.time()
        p2 = process_run_retrofit_corroded(case_dir, scenario, years, out_root=out_root)
        dt_second = time.time() - t0
        print(f"  second run (skip-if-exists): {dt_second:.2f} s")
        assert p2 == npz_path
        assert dt_second < 10.0, "skip-if-exists did not skip"
        assert npz_path.read_bytes() == npz_bytes_1

        s2tc = load_stage2_joints_thickness_corrosion(npz_path)
        n_points = len(s2tc["point_table"])
        print(f"  loaded: {n_points} points, counts shape {s2tc['counts'].shape}, years {s2tc['years']}")
        assert s2tc["counts"].shape == (n_points, len(years), cfg.N_BINS)

        families = {}
        for r in s2tc["point_table"]:
            families[r["family"]] = families.get(r["family"], 0) + 1
        print(f"  signals by family: {families}")
        assert families == {"K": 896, "X": 128}, "unexpected splash-zone composition"

        print(f"  n_over max: {s2tc['n_over'].max()} (expect 0.0)")
        assert s2tc["n_over"].max() == 0.0

        # --- year=0 damage must match the pure-retrofit stage2 file exactly
        # (retrofit-only IS retrofit+corrosion-at-year-0) -- decisive cross-
        # module check.
        # --- NOTE: `years` (e.g. [5,10,15,20,25]) never includes year=0 in
        # production -- that's deliberate, matching stage2_joints_corrosion.py's
        # own 5-year-STEP convention (each step's damage rate applies to the
        # PRECEDING 5-year block, see stage3_joint_damage_corrosion.py's
        # docstring). To test the decisive "retrofit+corrosion at year=0
        # equals pure retrofit" boundary property, run a SEPARATE, tiny,
        # explicit years=[0] pass here (self-check scratch output only,
        # never wired into run_pipeline.py) rather than assuming index 0 of
        # the production `years` list means year 0 -- that wrong assumption
        # is exactly the bug this comment replaces (self-check-only, caught
        # and fixed same session, no production code was ever wrong).
        import stage2_joints_thickness as s2jt
        year0_out_root = RESULTS_DIR / "_stage2_joints_thickness_corrosion_selfcheck_year0only"
        year0_npz = process_run_retrofit_corroded(case_dir, scenario, [0], out_root=year0_out_root, force=True)
        s2tc_year0 = load_stage2_joints_thickness_corrosion(year0_npz)

        retrofit_only_path = RESULTS_DIR / "_stage2_joints_thickness_selfcheck" / scenario / "LC_V20_H3p5_T8" / "S100001.npz"
        assert retrofit_only_path.exists(), f"run stage2_joints_thickness.py first: {retrofit_only_path}"
        s2t = s2jt.load_stage2_joints_thickness(retrofit_only_path)
        s2t_by_key = {(r["node"], r["brace_member"], r["brace_end"], r["chord_t_scenario"],
                       r["direction"], r["treatment"], r["side"], r["position"], r["segment"]): r["point_id"]
                      for r in s2t["point_table"]}
        max_rel_diff = 0.0
        n_checked = 0
        for r in s2tc_year0["point_table"]:
            key = (r["node"], r["brace_member"], r["brace_end"], r["chord_t_scenario"],
                   r["direction"], r["treatment"], r["side"], r["position"], r["segment"])
            pid_retrofit = s2t_by_key[key]
            pid_corr = r["point_id"]
            for m in cfg.WOHLER_EXPONENTS:
                d_retrofit = float(np.sum(s2t["sum_r"][m][pid_retrofit, :]))
                d_year0 = float(np.sum(s2tc_year0["sum_r"][m][pid_corr, 0, :]))
                rel_diff = abs(d_retrofit - d_year0) / d_retrofit if d_retrofit > 0 else abs(d_retrofit - d_year0)
                max_rel_diff = max(max_rel_diff, rel_diff)
            n_checked += 1
        print(f"  explicit year=0 pass vs retrofit-only: {n_checked} points, max rel.diff = {max_rel_diff:.3e}")
        assert max_rel_diff < 1e-9

        # --- damage rises with year (production `years` list, e.g. year=5
        # through year=25) at a representative splash K point ---
        pid0 = s2tc["point_table"][0]["point_id"]
        d_by_year = [float(np.sum(s2tc["sum_r"][3][pid0, y_idx, :])) for y_idx in range(len(years))]
        print(f"  damage vs year (point {pid0}, m=3): {[f'{d:.3e}' for d in d_by_year]}")
        assert all(d_by_year[i] <= d_by_year[i + 1] for i in range(len(d_by_year) - 1))

        # --- process the other 5 seeds too (Stage 3's aggregate_condition
        # needs all 6 to compute a real seed mean/std) ---
        print(f"\n  processing remaining 5 seeds ...")
        for seed_dir in sorted((DEV_FIXTURE_DIR / "LC_V20_H3p5_T8").glob("S*")):
            if seed_dir.name == case_dir.name:
                continue
            t0 = time.time()
            process_run_retrofit_corroded(seed_dir, scenario, years, out_root=out_root)
            print(f"    {seed_dir.name}: {time.time() - t0:.1f} s")

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
