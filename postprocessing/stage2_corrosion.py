"""
Step C3 -- Corrosion-aware Stage 2 (member track only).

Direct analogue of stage2_histograms.py, scoped to the 32 splash-zone
members only (see sd_geometry.member_zone) and looped over a caller-supplied
list of corrosion-exposure years. Everything else -- point-per-(member,end),
per-theta rainflow, counts + sum(range^m) histogram storage -- is identical
to the uncorroded Stage 2; only the section (D, t) fed into
stress.member_end_stress_history() changes per year, via
sd_geometry.corroded_section().

WHY A SEPARATE MODULE, NOT A PARAMETER ON stage2_histograms.process_run():
the uncorroded Stage 2 output is the campaign's primary, already-verified
deliverable (worst 7.4%/25yr result) -- this module must never risk
perturbing that code path. Splash-only + a year loop is also a genuinely
different point table (32 points x 2 ends x N_years rows, not 112 x 2), so
sharing one function would need year=None as a special case threaded through
every array shape below -- more confusing than a parallel module that reuses
the same underlying stress.py/rainflow_hist.py primitives.

DESIGN: ONE .outb READ PER RUN, corrosion steps looped IN-MEMORY. The
expensive part of Stage 2 is the bulk channel read (obr.read_channels), not
the stress+rainflow math (cheap numpy per point/theta). So this module reads
each splash member's (N, Mkx, Mky) channels ONCE, then loops corrosion years
computing stress.member_end_stress_history() with a different (D, t) each
time -- not re-reading the .outb per year. See fatigue_pipeline_build /
docs/decisions.md for the cost reasoning behind this choice
(the author's 14.08.2026 question about extending the horizon to 50yr).

DRIVE ROUTING (added when wiring into run_pipeline.py): same find_drive()
D:-Z: scan as stage2_histograms.py, writing to
<drive>/Postprocessing/stage2_corrosion -- a SEPARATE subfolder from the
member-uncorroded track's stage2/, since these are a different point table
shape (64 splash points x years, not 224 points) and must never be mistaken
for each other by anything that globs stage2/<COND>/*.npz.

Corrosion rule (confirmed against the UpWind Design Basis,
14.08.2026 -- see sd_geometry.corroded_section() and fatigue_postpro_design
memory): legs flooded (both surfaces, 0.15 mm/yr each = 0.30 mm/yr total),
braces external-only (0.15 mm/yr total) -- both at the Design Basis's
fatigue-design HALVED rate. Splash-zone members only; non-splash members
carry zero corrosion and are NOT part of this module's output at all (their
year-0 Stage 2 result from stage2_histograms.py is already correct and
unaffected by any of this).

Steps:
    1. build_point_table_corrosion(model) -- one row per (member, end) for
       splash-zone members only, plus member_class ('leg'/'brace').
    2. build_stamp_corrosion(...) -- extends stage2_histograms.build_stamp
       with the years list and the corrosion-rate table, so changing either
       invalidates the cache.
    3. process_run_corroded(case_dir, years, out_root, force) -- one .outb
       read, then for every splash point x every year x every theta:
       corroded (D,t) -> stress -> rainflow -> histogram. Stored shape gains
       a year axis: (n_points, n_years, n_theta, n_bins).
    4. load_stage2_corrosion(npz_path) -- the read side.
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

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

# Same D:-Z: scan as stage2_histograms.find_drive() -- duplicated rather
# than imported (this module stays outside Simulation/'s import graph, see
# module docstring / docs/decisions.md). A SEPARATE subfolder
# (stage2_corrosion, not stage2) from the member-uncorroded track's cache --
# different point table shape, must never collide.
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
    STAGE2_CORROSION_DIR = _drive / "Postprocessing" / "stage2_corrosion"
else:
    STAGE2_CORROSION_DIR = RESULTS_DIR / "_stage2_corrosion_local"
    print(f"WARNING: external drive not found (checked D:-Z: for a '{_DRIVE_SUBFOLDER}' "
          f"folder containing '{_DRIVE_MARKER}'). Corrosion Stage 2 cache falling back to "
          f"local {STAGE2_CORROSION_DIR}")

OUTB_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb"
SD_NAME = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
# endregion


# region --- point table (splash-zone members only) ---
def splash_member_ids(model, mudline_z):
    """Every member whose worst zone touched anywhere along its length is
    'splash' (sd_geometry.member_zone) -- the only members this module ever
    touches. Sorted for deterministic point ordering."""
    return sorted(
        mid for mid in model["members"]
        if sdg.member_zone(model, mid, mudline_z)[0] == "splash"
    )


def build_point_table_corrosion(model, mudline_z):
    """
    One row per (member, end) for splash-zone members only: point_id,
    member_id, end, member_class ('leg'/'brace'), D0, t0 (uncorroded
    section, year-0), z, propset. zone is always 'splash' by construction
    (see splash_member_ids) -- not stored as a column since it never varies.
    """
    member_ids = splash_member_ids(model, mudline_z)
    rows = []
    point_id = 0
    for mid in member_ids:
        D0, t0, pid = sdg.member_section(model, mid)
        cls = sdg.member_class(model, mid)
        for end in (1, 2):
            z = sdg.member_end_z(model, mid, end)
            rows.append(dict(
                point_id=point_id, member_id=mid, end=end,
                member_class=cls, D0=D0, t0=t0, z=z, propset=pid,
            ))
            point_id += 1
    return rows
# endregion


# region --- provenance stamp ---
def build_stamp_corrosion(outb_path, header, model, case_json, owner_json, years):
    """
    Same fields as stage2_histograms.build_stamp, plus `years` (the exact
    corrosion-exposure horizon this file was computed for) and the
    corrosion-rate table itself -- both go into the recompute key, so a
    change to either (e.g. extending 25yr -> 50yr, or a rate correction)
    invalidates every cached corrosion file, not just new years appended.
    """
    return dict(
        pipeline_version=cfg.PIPELINE_VERSION,
        n_theta=stress.N_THETA,
        theta_rad=stress.THETA_RAD.tolist(),
        theta_convention=(
            "theta=0 origin per stress.py; member track, SCF=1: "
            "sigma(theta) = sig_ax + sig_ipb*cos(theta) - sig_opb*sin(theta)"
        ),
        bin_edges_mpa=cfg.BIN_EDGES_MPA.tolist(),
        wohler_exponents=list(cfg.WOHLER_EXPONENTS),
        units=dict(stress="MPa", force="N", moment="N*m"),
        force_family=list(obr.FATIGUE_COMPONENTS),
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
        corrosion_years=list(years),
        corrosion_rate_mm_per_year_per_surface=sdg.CORROSION_RATE_MM_PER_YEAR_PER_SURFACE,
        leg_diameter_min_m=sdg.LEG_DIAMETER_MIN_M,
    )


def _stamp_recompute_key(stamp):
    """Same idea as stage2_histograms._stamp_recompute_key, plus the
    corrosion-specific fields (years, rates, leg/brace threshold)."""
    return (
        stamp["pipeline_version"], stamp["n_theta"],
        tuple(stamp["bin_edges_mpa"]), tuple(stamp["wohler_exponents"]),
        stamp["transient_cutoff_s"], stamp["subdyn_md5"],
        stamp["outb_size_bytes"], stamp["outb_mtime"],
        tuple(stamp["corrosion_years"]),
        tuple(sorted((k, tuple(sorted(v.items())))
                     for k, v in stamp["corrosion_rate_mm_per_year_per_surface"].items())),
        stamp["leg_diameter_min_m"],
    )
# endregion


# region --- write side ---
def _npz_json_paths(case_dir, out_root=STAGE2_CORROSION_DIR):
    case_dir = Path(case_dir)
    cond, seed = case_dir.parent.name, case_dir.name
    npz_path = out_root / cond / f"{seed}.npz"
    return npz_path, npz_path.with_suffix(".json")


def process_run_corroded(case_dir, years, out_root=STAGE2_CORROSION_DIR, force=False):
    """
    Compute (or skip, if a stamp-matching .npz/.json pair already exists)
    the corrosion-aware Stage 2 file for one run, at every year in `years`.
    Returns the .npz path.
    """
    case_dir = Path(case_dir)
    outb_path = case_dir / OUTB_NAME
    sd_path = case_dir / SD_NAME
    case_json = json.loads((case_dir / "case.json").read_text())
    owner_path = case_dir / "owner.json"
    owner_json = json.loads(owner_path.read_text()) if owner_path.exists() else {}

    header = obr.read_outb_header(outb_path)
    model = sdg.read_subdyn_model(sd_path)
    mudline_z = min(z for _, _, z in model["joints"].values())
    stamp = build_stamp_corrosion(outb_path, header, model, case_json, owner_json, years)

    npz_path, json_path = _npz_json_paths(case_dir, out_root)

    if not force and npz_path.exists() and json_path.exists():
        try:
            existing_stamp = json.loads(json_path.read_text())
            if _stamp_recompute_key(existing_stamp) == _stamp_recompute_key(stamp):
                return npz_path  # up to date, nothing to do
        except (json.JSONDecodeError, KeyError):
            pass  # sidecar unreadable/incomplete -- fall through and recompute

    point_table = build_point_table_corrosion(model, mudline_z)
    member_ids = splash_member_ids(model, mudline_z)
    n_points = len(point_table)
    n_years = len(years)
    n_theta = stress.N_THETA
    n_bins = cfg.N_BINS
    exponents = cfg.WOHLER_EXPONENTS

    counts = np.zeros((n_points, n_years, n_theta, n_bins), dtype=np.float64)
    sum_r = {m: np.zeros((n_points, n_years, n_theta, n_bins), dtype=np.float64) for m in exponents}
    n_under = np.zeros((n_points, n_years, n_theta), dtype=np.float64)
    n_over = np.zeros((n_points, n_years, n_theta), dtype=np.float64)

    # ONE bulk read, splash members only (32 of 112) -- see module docstring
    # on why this is cheaper than reading all 112 and discarding 80.
    names = obr.member_end_channels(member_ids, obr.FATIGUE_COMPONENTS)
    t_full, arr = obr.read_channels(outb_path, header, names)
    n_comp = len(obr.FATIGUE_COMPONENTS)

    for i, mid in enumerate(member_ids):
        for e_idx, end in enumerate((1, 2)):
            point_id = i * 2 + e_idx  # matches build_point_table_corrosion's row order
            base = point_id * n_comp
            N, Mkx, Mky = arr[:, base], arr[:, base + 1], arr[:, base + 2]

            # ONE channel read, then loop corrosion years IN-MEMORY -- the
            # module's central cost-control decision, see docstring.
            for y_idx, year in enumerate(years):
                D, wall_t = sdg.corroded_section(model, mid, year)
                _t_trim, sigma = stress.member_end_stress_history(t_full, N, Mkx, Mky, D, wall_t)
                for k_theta in range(n_theta):
                    cycles = list(rainflow.extract_cycles(sigma[:, k_theta]))
                    c, sr, nu, no = rhist.cycles_to_histogram(cycles)
                    counts[point_id, y_idx, k_theta, :] = c
                    for m in exponents:
                        sum_r[m][point_id, y_idx, k_theta, :] = sr[m]
                    n_under[point_id, y_idx, k_theta] = nu
                    n_over[point_id, y_idx, k_theta] = no

    save_kwargs = dict(
        counts=counts, n_under=n_under, n_over=n_over,
        bin_edges_mpa=cfg.BIN_EDGES_MPA, theta_rad=stress.THETA_RAD,
        years=np.array(years, dtype=np.int64),
        point_id=np.array([r["point_id"] for r in point_table], dtype=np.int64),
        member_id=np.array([r["member_id"] for r in point_table], dtype=np.int64),
        end=np.array([r["end"] for r in point_table], dtype=np.int64),
        member_class=np.array([r["member_class"] for r in point_table], dtype="<U8"),
        D0=np.array([r["D0"] for r in point_table], dtype=np.float64),
        t0=np.array([r["t0"] for r in point_table], dtype=np.float64),
        z=np.array([r["z"] for r in point_table], dtype=np.float64),
        propset=np.array([r["propset"] for r in point_table], dtype=np.int64),
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
def load_stage2_corrosion(npz_path):
    """Read a corrosion Stage 2 .npz (+ .json sidecar) back into a dict:
    counts, sum_r, n_under, n_over, bin_edges_mpa, theta_rad, years,
    point_table (list of dicts), stamp."""
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    exponents = sorted(int(k[len("sum_r"):]) for k in data.files if k.startswith("sum_r"))
    sum_r = {m: data[f"sum_r{m}"] for m in exponents}

    n_points = data["counts"].shape[0]
    point_table = [
        dict(
            point_id=int(data["point_id"][i]),
            member_id=int(data["member_id"][i]), end=int(data["end"][i]),
            member_class=str(data["member_class"][i]),
            D0=float(data["D0"][i]), t0=float(data["t0"][i]), z=float(data["z"][i]),
            propset=int(data["propset"][i]),
        )
        for i in range(n_points)
    ]

    json_path = npz_path.with_suffix(".json")
    stamp = json.loads(json_path.read_text()) if json_path.exists() else None

    return dict(
        counts=data["counts"], sum_r=sum_r, n_under=data["n_under"], n_over=data["n_over"],
        bin_edges_mpa=data["bin_edges_mpa"], theta_rad=data["theta_rad"],
        years=data["years"].tolist(), point_table=point_table, stamp=stamp,
    )
# endregion


def _self_check():
    import time

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    assert case_dir.exists(), f"missing fixture: {case_dir}"

    years = [5, 10, 15, 20, 25]
    out_root = RESULTS_DIR / "_stage2_corrosion_selfcheck"
    npz_path, json_path = _npz_json_paths(case_dir, out_root)
    for p in (npz_path, json_path):
        if p.exists():
            p.unlink()

    print(f"processing {case_dir.parent.name}/{case_dir.name} -> {npz_path}")
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
    assert dt_second < 1.0, "skip-if-exists did not skip -- stamp comparison is wrong"
    assert npz_path.read_bytes() == npz_bytes_1, "skip path must not touch the file at all"

    # Changing the year list must invalidate the cache (different stamp key).
    t0 = time.time()
    p3 = process_run_corroded(case_dir, years + [30], out_root=out_root)
    dt_extended = time.time() - t0
    print(f"  extended years (must recompute, not skip): {dt_extended:.1f} s")
    assert dt_extended > 1.0, "extending the year list should have forced a recompute"
    # restore the 5-step file for the rest of this check
    p1 = process_run_corroded(case_dir, years, out_root=out_root, force=True)

    stage2c = load_stage2_corrosion(npz_path)
    n_points = len(stage2c["point_table"])
    print(f"\n  loaded: {n_points} points, counts shape {stage2c['counts'].shape}, "
          f"years {stage2c['years']}")
    assert n_points == 32 * 2, f"expected 64 splash points (32 members x 2 ends), got {n_points}"
    assert stage2c["counts"].shape == (n_points, len(years), stress.N_THETA, cfg.N_BINS)
    assert stage2c["stamp"]["subdyn_md5"] == sdg.KNOWN_CAMPAIGN_MD5

    n_legs = sum(1 for r in stage2c["point_table"] if r["member_class"] == "leg")
    n_braces = sum(1 for r in stage2c["point_table"] if r["member_class"] == "brace")
    print(f"  member_class census: {n_legs} leg-points, {n_braces} brace-points "
          f"(expect 8*2=16 leg, 24*2=48 brace)")
    assert (n_legs, n_braces) == (16, 48)

    assert stage2c["n_over"].max() == 0.0, "n_over > 0 somewhere -- BIN_HI_MPA too low"

    # Independent verification: for one point at one year, recompute D,t via
    # sd_geometry directly and the stress signal via stress.py directly
    # (never touching this module's own arrays on the way there), then
    # compare the stored damage against a fresh rainflow of that signal.
    model = sdg.read_subdyn_model(case_dir / SD_NAME)
    row = stage2c["point_table"][0]
    mid, end = row["member_id"], row["end"]
    y_idx, year = 2, years[2]  # year 15
    print(f"\n  independent recheck: member {mid} end {end} year {year} (point_id {row['point_id']}):")

    D_direct, t_direct = sdg.corroded_section(model, mid, year)
    print(f"    corroded_section() direct: D={D_direct*1000:.2f}mm t={t_direct*1000:.2f}mm")

    names = [f"M{mid}J{end}{c}" for c in obr.FATIGUE_COMPONENTS]
    header = obr.read_outb_header(case_dir / OUTB_NAME)
    t_full, arr = obr.read_channels(case_dir / OUTB_NAME, header, names)
    N, Mkx, Mky = arr[:, 0], arr[:, 1], arr[:, 2]
    _t_trim, sigma_direct = stress.member_end_stress_history(t_full, N, Mkx, Mky, D_direct, t_direct)

    k_theta = 3
    cycles_direct = list(rainflow.extract_cycles(sigma_direct[:, k_theta]))
    pid = row["point_id"]
    for m in cfg.WOHLER_EXPONENTS:
        d_direct = rhist.damage_from_cycles(cycles_direct, m)
        d_stored = float(np.sum(stage2c["sum_r"][m][pid, y_idx, k_theta, :]))
        rel_diff = abs(d_direct - d_stored) / d_direct if d_direct > 0 else abs(d_direct - d_stored)
        print(f"    m={m}: direct={d_direct:.6e}  stored={d_stored:.6e}  rel.diff={rel_diff:.3e}")
        assert rel_diff < 1e-9, f"m={m}: stored histogram doesn't match direct recompute"

    # Sanity: damage at year 25 should exceed damage at year 5 for the same
    # point/theta/m (thinner section -> higher stress -> more damage),
    # monotonic in year for every point this jacket has.
    d_by_year = []
    for y_idx2, yr in enumerate(years):
        d = float(np.sum(stage2c["sum_r"][3][pid, y_idx2, k_theta, :]))
        d_by_year.append(d)
    print(f"\n  damage vs year (point {pid}, theta_idx {k_theta}, m=3): "
          f"{[f'{d:.3e}' for d in d_by_year]}")
    assert all(d_by_year[i] <= d_by_year[i + 1] for i in range(len(d_by_year) - 1)), \
        "damage should be monotonically non-decreasing with corrosion year"

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
