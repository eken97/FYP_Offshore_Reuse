"""
Step 7 -- Stage 2 per-run histogram file.

Drives Step 4 (stress.py) and Step 6 (rainflow_hist.py) across every
assessment point of one run and writes the compressed result to disk: one
`.npz` per run at `stage2/<COND>/<SEED>.npz`, plus a sibling `.json`
provenance sidecar. Nothing upstream of this step writes the full-length
stress signal to disk (see stress.py's docstring) -- this is the first
point in the pipeline where anything durable is produced.

THE JOINT SEAM (see stress.py, sd_geometry.py, and the build plan): this
module only ever computes the MEMBER track (SCF=1 everywhere). A future
joint track forks BEFORE rainflow, at nominal_components() -- it cannot
reuse this module's per-theta rainflow output, since
SCF*(A+B) != SCF_A*A + SCF_B*B and rainflow is not linear. That is why the
point table below is a table (point_id, kind, ...), not a hard-coded
(112, 2, 16, n_bins) array: a joint track appends rows with
kind="joint_brace" later, no format change.

Steps:
    1. build_point_table(model) -- one row per (member, end) assessment
       point: point_id, kind, member_id, end, D, t, z, zone, propset,
       not_assessable, not_assessable_reason. Zone is the MEMBER's worst
       zone (sd_geometry.member_zone), shared by both ends, since that is
       what decides the S-N curve later -- not the two ends' own endpoint
       zones. not_assessable flags the 12 known-degenerate members
       (101-104/105-108/109-112) with a reason, WITHOUT excluding them --
       see docs/decisions.md, "all 112 members get a damage
       number eventually". This is a different mechanism from
       fatigue_config.SCREENING_EXCLUDED_MEMBER_IDS (diagnostic-tool-only
       scope) -- that constant is deliberately never imported here.
    2. build_stamp(...) -- full provenance: pipeline version, theta
       convention, bin edges, units, force family, SubDyn md5, outb
       path/size/mtime, transient cutoff, dt, library versions, and
       case.json/owner.json copied in verbatim.
    3. process_run(case_dir) -- reads all fatigue channels for every member
       in ONE memmap read (see bin_range_check.py's performance note),
       computes stress + rainflow histogram for every (point, theta), and
       writes counts + sum(R^m) for every m in WOHLER_EXPONENTS. Skips
       recompute if a valid, stamp-matching .npz/.json pair already exists;
       writes via a .tmp file + os.replace() so an interrupted write can
       never leave a truncated file that a naive skip-check would mistake
       for done.
    4. load_stage2(npz_path) -- the read side, so Stage 3 never needs to
       know the file format.
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
# Dev/test fixture data (LC_V20_H3p5_T8, _staging/) deliberately stays at
# its original location, not moved alongside the code -- see
# See docs/decisions.md, 10.08.2026 folder-reorg session.
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

# Stage 2's .npz cache is large (~9 MB/seed, ~3.6 GB for the full 414-run
# campaign) and fully regenerable from .outb, so it deliberately does NOT
# live in a cloud-synced project tree -- it goes on the external drive,
# next to the .outb data it's derived from.
_DRIVE_MARKER = ".oc4_campaign_drive"
_DRIVE_SUBFOLDER = "OC4_CAMPAIGN"


def find_drive():
    """
    Scans drive letters D:-Z: for the marker file and returns the
    OC4_CAMPAIGN folder on whichever one has it, or None if the external
    drive isn't attached. EXACT same convention as Simulation/config.py's
    own find_drive() -- duplicated here rather than imported (this module
    stays outside Simulation/'s import graph, see module docstring /
    docs/decisions.md) -- and for the identical reason that
    function's own comment gives: the same physical drive enumerates as a
    DIFFERENT letter depending on which machine it is attached to
    (confirmed by moving the drive between machines for a real-campaign run
    -- an earlier version of this file hardcoded "D:" here, which would have
    silently misbehaved the moment the drive was not D:).
    """
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        candidate = Path(f"{letter}:/{_DRIVE_SUBFOLDER}")
        if (candidate / _DRIVE_MARKER).exists():
            return candidate
    return None


# An explicit path always wins -- this is what lets the shipped worked
# example under data/example/ be used without any external drive at all.
_override = os.environ.get("OC4_STAGE2_DIR")
_drive = None if _override else find_drive()
if _override:
    STAGE2_DIR = Path(_override)
elif _drive is not None:
    STAGE2_DIR = _drive / "Postprocessing" / "stage2"
else:
    STAGE2_DIR = POSTPRO_DIR / "_stage2_local"
    print(f"WARNING: external drive not found (checked D:-Z: for a '{_DRIVE_SUBFOLDER}' "
          f"folder containing '{_DRIVE_MARKER}'). Stage 2 cache falling back to local {STAGE2_DIR}")

OUTB_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb"
SD_NAME = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
# endregion


# region --- point table ---
def build_point_table(model, member_ids=None):
    """
    One row per (member, end). member_ids defaults to every member in the
    model (all 112) -- Stage 2 processes ALL of them, unlike the diagnostic
    SCREENING_EXCLUDED_MEMBER_IDS scope (see module docstring).
    """
    member_ids = sorted(member_ids) if member_ids is not None else sorted(model["members"])
    mudline_z = min(z for _, _, z in model["joints"].values())

    rows = []
    point_id = 0
    for mid in member_ids:
        D, t, pid = sdg.member_section(model, mid)
        worst_zone, _touched = sdg.member_zone(model, mid, mudline_z)
        reason = cfg.member_not_assessable_reason(mid)
        for end in (1, 2):
            z = sdg.member_end_z(model, mid, end)
            rows.append(dict(
                point_id=point_id, kind="member", member_id=mid, end=end,
                D=D, t=t, z=z, zone=worst_zone, propset=pid,
                not_assessable=reason is not None,
                not_assessable_reason=reason or "",
            ))
            point_id += 1
    return rows
# endregion


# region --- provenance stamp ---
def build_stamp(outb_path, header, model, case_json, owner_json):
    """
    Everything needed to decide, without touching the .outb again, whether
    a stored .npz is still valid for the current code/config. See the
    module docstring for what's included and why.
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
    )


def _stamp_recompute_key(stamp):
    """
    The subset of the stamp that, if changed, means the STORED ARRAYS are
    stale and must be recomputed -- not the full stamp (e.g. case_json is
    informational, doesn't affect what gets computed from a given .outb).
    """
    return (
        stamp["pipeline_version"], stamp["n_theta"],
        tuple(stamp["bin_edges_mpa"]), tuple(stamp["wohler_exponents"]),
        stamp["transient_cutoff_s"], stamp["subdyn_md5"],
        stamp["outb_size_bytes"], stamp["outb_mtime"],
    )
# endregion


# region --- write side ---
def _npz_json_paths(case_dir, out_root=STAGE2_DIR):
    case_dir = Path(case_dir)
    cond, seed = case_dir.parent.name, case_dir.name
    npz_path = out_root / cond / f"{seed}.npz"
    return npz_path, npz_path.with_suffix(".json")


def process_run(case_dir, out_root=STAGE2_DIR, force=False):
    """
    Compute (or skip, if a stamp-matching .npz/.json pair already exists)
    the Stage 2 histogram file for one run. Returns the .npz path.
    """
    case_dir = Path(case_dir)
    outb_path = case_dir / OUTB_NAME
    sd_path = case_dir / SD_NAME
    case_json = json.loads((case_dir / "case.json").read_text())
    owner_path = case_dir / "owner.json"
    owner_json = json.loads(owner_path.read_text()) if owner_path.exists() else {}

    header = obr.read_outb_header(outb_path)  # raises loud on a corrupt/truncated file
    model = sdg.read_subdyn_model(sd_path)
    stamp = build_stamp(outb_path, header, model, case_json, owner_json)

    npz_path, json_path = _npz_json_paths(case_dir, out_root)

    if not force and npz_path.exists() and json_path.exists():
        try:
            existing_stamp = json.loads(json_path.read_text())
            if _stamp_recompute_key(existing_stamp) == _stamp_recompute_key(stamp):
                return npz_path  # up to date, nothing to do
        except (json.JSONDecodeError, KeyError):
            pass  # sidecar unreadable/incomplete -- fall through and recompute

    point_table = build_point_table(model)
    member_ids = sorted(model["members"])
    n_points = len(point_table)
    n_theta = stress.N_THETA
    n_bins = cfg.N_BINS
    exponents = cfg.WOHLER_EXPONENTS

    counts = np.zeros((n_points, n_theta, n_bins), dtype=np.float64)
    sum_r = {m: np.zeros((n_points, n_theta, n_bins), dtype=np.float64) for m in exponents}
    n_under = np.zeros((n_points, n_theta), dtype=np.float64)
    n_over = np.zeros((n_points, n_theta), dtype=np.float64)

    # ONE bulk read for every fatigue channel of every member (112 x 2 x 3 =
    # 672 columns) -- see bin_range_check.py's performance note on why this
    # matters vs. rereading the file per member-end.
    names = obr.member_end_channels(member_ids, obr.FATIGUE_COMPONENTS)
    t_full, arr = obr.read_channels(outb_path, header, names)
    n_comp = len(obr.FATIGUE_COMPONENTS)

    for i, mid in enumerate(member_ids):
        D, wall_t, pid = sdg.member_section(model, mid)
        for e_idx, end in enumerate((1, 2)):
            point_id = i * 2 + e_idx  # matches build_point_table's row order exactly
            base = point_id * n_comp
            N, Mkx, Mky = arr[:, base], arr[:, base + 1], arr[:, base + 2]
            _t_trim, sigma = stress.member_end_stress_history(t_full, N, Mkx, Mky, D, wall_t)
            for k_theta in range(n_theta):
                cycles = list(rainflow.extract_cycles(sigma[:, k_theta]))
                c, sr, nu, no = rhist.cycles_to_histogram(cycles)
                counts[point_id, k_theta, :] = c
                for m in exponents:
                    sum_r[m][point_id, k_theta, :] = sr[m]
                n_under[point_id, k_theta] = nu
                n_over[point_id, k_theta] = no

    save_kwargs = dict(
        counts=counts, n_under=n_under, n_over=n_over,
        bin_edges_mpa=cfg.BIN_EDGES_MPA, theta_rad=stress.THETA_RAD,
        point_id=np.array([r["point_id"] for r in point_table], dtype=np.int64),
        member_id=np.array([r["member_id"] for r in point_table], dtype=np.int64),
        end=np.array([r["end"] for r in point_table], dtype=np.int64),
        D=np.array([r["D"] for r in point_table], dtype=np.float64),
        t=np.array([r["t"] for r in point_table], dtype=np.float64),
        z=np.array([r["z"] for r in point_table], dtype=np.float64),
        zone=np.array([r["zone"] for r in point_table], dtype="<U16"),
        propset=np.array([r["propset"] for r in point_table], dtype=np.int64),
        not_assessable=np.array([r["not_assessable"] for r in point_table], dtype=bool),
        not_assessable_reason=np.array(
            [r["not_assessable_reason"] for r in point_table], dtype="<U32"),
    )
    for m in exponents:
        save_kwargs[f"sum_r{m}"] = sum_r[m]

    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: .tmp then os.replace(). Without this, killing the
    # process mid-write leaves a truncated .npz that the skip-if-exists
    # check above would happily treat as done forever -- the #1
    # restartability bug per the build plan.
    # np.savez_compressed appends ".npz" to its target unless given an open
    # file object -- pass a handle explicitly, else tmp_npz (which already
    # ends in ".npz.tmp") would be written to "....npz.tmp.npz" instead.
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
def load_stage2(npz_path):
    """
    Read a Stage 2 .npz (+ its .json sidecar) back into a dict:
        counts, sum_r (dict {m: array}), n_under, n_over,
        bin_edges_mpa, theta_rad, point_table (list of dicts), stamp.
    Stage 3 should use this exclusively -- never read the .npz fields directly.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    exponents = sorted(int(k[len("sum_r"):]) for k in data.files if k.startswith("sum_r"))
    sum_r = {m: data[f"sum_r{m}"] for m in exponents}

    n_points = data["counts"].shape[0]
    point_table = [
        dict(
            point_id=int(data["point_id"][i]), kind="member",
            member_id=int(data["member_id"][i]), end=int(data["end"][i]),
            D=float(data["D"][i]), t=float(data["t"][i]), z=float(data["z"][i]),
            zone=str(data["zone"][i]), propset=int(data["propset"][i]),
            not_assessable=bool(data["not_assessable"][i]),
            not_assessable_reason=str(data["not_assessable_reason"][i]),
        )
        for i in range(n_points)
    ]

    json_path = npz_path.with_suffix(".json")
    stamp = json.loads(json_path.read_text()) if json_path.exists() else None

    return dict(
        counts=data["counts"], sum_r=sum_r, n_under=data["n_under"], n_over=data["n_over"],
        bin_edges_mpa=data["bin_edges_mpa"], theta_rad=data["theta_rad"],
        point_table=point_table, stamp=stamp,
    )
# endregion


def _self_check():
    import time

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    assert case_dir.exists(), f"missing fixture: {case_dir}"

    out_root = RESULTS_DIR / "_stage2_selfcheck"
    npz_path, json_path = _npz_json_paths(case_dir, out_root)
    for p in (npz_path, json_path):
        if p.exists():
            p.unlink()

    print(f"processing {case_dir.parent.name}/{case_dir.name} -> {npz_path}")
    t0 = time.time()
    p1 = process_run(case_dir, out_root=out_root)
    dt_first = time.time() - t0
    print(f"  first run: {dt_first:.1f} s")
    assert p1 == npz_path and npz_path.exists() and json_path.exists()

    npz_bytes_1 = npz_path.read_bytes()

    t0 = time.time()
    p2 = process_run(case_dir, out_root=out_root)
    dt_second = time.time() - t0
    print(f"  second run (skip-if-exists): {dt_second:.2f} s")
    assert p2 == npz_path
    assert dt_second < 1.0, "skip-if-exists did not skip -- stamp comparison is wrong"
    assert npz_path.read_bytes() == npz_bytes_1, "skip path must not touch the file at all"

    t0 = time.time()
    p3 = process_run(case_dir, out_root=out_root, force=True)
    dt_force = time.time() - t0
    print(f"  force=True recompute: {dt_force:.1f} s")
    assert p3 == npz_path
    npz_bytes_3 = npz_path.read_bytes()
    assert npz_bytes_3 == npz_bytes_1, "force=True recompute is not byte-identical"

    # Load back and sanity-check shapes/provenance.
    stage2 = load_stage2(npz_path)
    n_points = len(stage2["point_table"])
    print(f"\n  loaded: {n_points} points, counts shape {stage2['counts'].shape}, "
          f"exponents {sorted(stage2['sum_r'].keys())}")
    assert n_points == 112 * 2
    assert stage2["counts"].shape == (n_points, stress.N_THETA, cfg.N_BINS)
    assert stage2["stamp"]["subdyn_md5"] == sdg.KNOWN_CAMPAIGN_MD5

    not_assessable = [r for r in stage2["point_table"] if r["not_assessable"]]
    print(f"  not_assessable points: {len(not_assessable)} "
          f"(expect {len(range(101, 113)) * 2} = 12 members x 2 ends)")
    assert len(not_assessable) == 12 * 2
    reasons = sorted({r["not_assessable_reason"] for r in not_assessable})
    print(f"  reasons present: {reasons}")

    # n_over must be 0 everywhere -- otherwise BIN_HI_MPA is too low (same
    # check rainflow_hist.py's own self-check makes per-case; here it's
    # exercised across the WHOLE run, all points and theta at once).
    assert stage2["n_over"].max() == 0.0, "n_over > 0 somewhere -- BIN_HI_MPA too low"

    # Independent verification: recompute one point's damage directly from
    # the .outb via stress.py + rainflow_hist.py (never touching
    # stage2_histograms.py's own arrays on the way there) and compare
    # against what process_run stored.
    header = obr.read_outb_header(case_dir / OUTB_NAME)
    model = sdg.read_subdyn_model(case_dir / SD_NAME)
    mid, end, k_theta = 4, 1, 7
    t_trim, sigma = stress.compute_member_stress(case_dir / OUTB_NAME, header, model, mid, end)
    signal = sigma[:, k_theta]
    cycles_direct = list(rainflow.extract_cycles(signal))

    row = next(r for r in stage2["point_table"]
               if r["member_id"] == mid and r["end"] == end)
    pid = row["point_id"]

    print(f"\n  independent recheck: member {mid} end {end} theta_idx {k_theta} "
          f"(point_id {pid}):")
    for m in cfg.WOHLER_EXPONENTS:
        d_direct = rhist.damage_from_cycles(cycles_direct, m)
        d_stored = float(np.sum(stage2["sum_r"][m][pid, k_theta, :]))
        rel_diff = abs(d_direct - d_stored) / d_direct if d_direct > 0 else abs(d_direct - d_stored)
        print(f"    m={m}: direct={d_direct:.6e}  stored={d_stored:.6e}  rel.diff={rel_diff:.3e}")
        assert rel_diff < 1e-9, f"m={m}: stored histogram doesn't match direct recompute"

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
