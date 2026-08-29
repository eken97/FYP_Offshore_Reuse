"""
Joint track, Step E -- Stage 2 per-run joint histogram file.

Mirrors stage2_histograms.py's Step 7 (same .npz/.json shape, same
skip-if-stamp-matches / atomic-write discipline), but for the joint track's
brace-to-chord connections instead of member ends. See joint_scf_theory
memory's "SESSION 14.08.2026 (later still) -- Step E architecture talked
through BEFORE coding" section for the full derivation this module
implements -- READ THAT FIRST if picking this up cold.

THE ONE RULE THAT DRIVES THIS WHOLE MODULE'S SHAPE (talked through and
confirmed by the author before any code was written): DNV eqn 3.3.1 assesses
ONE connection = ONE brace-to-chord weld, driven entirely by that BRACE's own
nominal stress. The chord's own bending stress is never independently SCF'd
or rainflow-counted -- it is multiplied by a fixed attachment coefficient and
ADDED into the brace's own crown+intermediate signals (positions 1/2/4/5/6/8
-- see below) BEFORE that one combined signal is rainflow counted (see
stress.hotspot_joint's own docstring for the eqn 6b/7b "safe rearrangement").
So per connection there is exactly ONE brace-driven signal set, not
one-per-member-at-the-node.

EVALUATION COUNT (corrected 14.08.2026, later still -- supersedes an earlier
"3,552" figure, itself a correction of an even earlier "3,680": the
intermediate-point resolution below changed which positions need the
chord-segment split -- see docs/decisions.md section above for the full
derivation table):
  - K family:   64 connections x 2 treatments (K, Y) x 2 sides x 14 signals
                (2 segment-independent PURE-SADDLE positions [3, 7, driven
                by SCF_AS alone, no SCF_AC anywhere in their formula] + 6
                crown/intermediate positions [1,2,4,5,6,8, all of which
                carry a SCF_AC term and therefore the chord-bending
                attachment coefficient] x 2 chord leg-segments)          = 3,584
  - Y/T family: 24 connections x 1 treatment    x 2 sides x 14 signals   =   672
  - X family:   32 connections x 1 treatment    x 2 sides x  8 signals
                (X's crown/intermediate positions have AC_att=0.0 -- no
                chord-bending term at all, so no segment split is needed
                anywhere; chord-segment channels are not even read for X
                connections)                                            =   512
  TOTAL: 4,768 independent rainflow-countable signals per run.

  Why 6 positions double, not just the 2 crown ones: DNV's (3.3.1)
  intermediate-point formula is "0.5*(SCF_AC+SCF_AS)" -- written purely in
  terms of the SYMBOL SCF_AC, which eqn 6b/7b defines (generally, not just
  "at the crown") as base + an attachment term driven by the chord's own
  bending. Since (3.3.1) never distinguishes a narrower "SCF_AC_base"
  concept, the intermediate average must use the FULL SCF_AC, i.e. include
  the chord-bending term -- so positions 2/4/6/8 need it too, same as 1/5.
  Only 3/7 (pure saddle, driven by SCF_AS alone) never touch SCF_AC at all.

DEV-FIXTURE-ONLY STATUS: stress.HOTSPOT_JOINT_VERIFIED is still False. Both
of hotspot_joint()'s open judgement calls (crown-heel sign, intermediate-
point AC_att inclusion) are now RESOLVED (primary-source re-checks + eqn
3.3.1's own structure -- see stress.py and docs/decisions.md) but neither has been
independently reviewed yet. This module builds and self-checks
against the same dev fixture stage2_histograms.py uses
(TestScenario/LC_V20_H3p5_T8/S100001) and is NOT wired into run_pipeline.py
or the real campaign drive -- do not run it against real campaign data until
HOTSPOT_JOINT_VERIFIED is flipped True.

Steps:
    1. build_full_connections(sd_path, sd_sum_path) -- runs the whole
       joint_geometry.py pipeline (Steps 1-4+B) once: DCM, connectivity,
       geometry params, joint axes, chord geometry. Returns (connections,
       model).
    2. build_scf_index(connections) -- scf.compute_all_scf(connections),
       grouped by connection identity so a connection's 1-4 (treatment,
       side) SCF rows can be looked up without a full-table scan per point.
    3. iter_assessment_rows(connections, scf_index) -- the SINGLE source of
       truth for point ordering. Both build_point_table (metadata, no I/O)
       and process_run (the real computation) iterate this same generator,
       so their row orders can never drift apart -- same discipline as
       stage2_histograms.py's own "matches build_point_table's row order
       exactly" comment.
    4. process_run(case_dir) -- per connection: read the brace's own
       (FKze, MKxe, MKye) and (for K/Y-T only) both chord segments' own
       (MKxe, MKye) -- NOT FKze, sigma_BendingChord is bending-only, see
       stress.hotspot_joint's own docstring -- compute nominal stresses,
       rotate into the joint's (mip, mop) frame via
       joint_geometry.rotate_to_joint_axes, call stress.hotspot_joint()
       once per chord segment (K/Y-T) or once with sig_cb=0 (X, exploiting
       AC_att=0.0 rather than computing a discarded zero term), rainflow
       every resulting signal, write counts + sum(R^m) to a stamped .npz.
       Reads are done per-connection, NOT via one global bulk read like the
       member track -- a deliberate simplification, decided by the author:
       .outb reads are cheap memmaps and rainflow counting dominates the
       cost budget, so the batching optimization stage2_histograms.py needs
       (112 members x full campaign) isn't worth the added complexity here
       (120 connections, dev-fixture scope for now).
    5. load_stage2_joints(npz_path) -- the read side.
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
import joint_geometry as jg
import scf

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

# Same D:-Z: scan as stage2_histograms.find_drive() -- duplicated rather
# than imported (this module stays outside Simulation/'s import graph, see
# module docstring / docs/decisions.md). A SEPARATE subfolder
# (stage2_joints, not stage2) from the member track's cache -- different
# point table shape, must never collide.
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
    STAGE2_JOINTS_DIR = _drive / "Postprocessing" / "stage2_joints"
else:
    STAGE2_JOINTS_DIR = RESULTS_DIR / "_stage2_joints_local"
    print(f"WARNING: external drive not found (checked D:-Z: for a '{_DRIVE_SUBFOLDER}' "
          f"folder containing '{_DRIVE_MARKER}'). Joint Stage 2 cache falling back to "
          f"local {STAGE2_JOINTS_DIR}")

OUTB_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb"
SD_NAME = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
SD_SUM_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.SD.sum.yaml"
# endregion


# region --- geometry + SCF assembly ---
def build_full_connections(sd_path, sd_sum_path):
    """Runs joint_geometry.py's whole geometry pipeline (Steps 1-4+B) once.
    Returns (connections, model)."""
    dcm_result = jg.read_member_dcm(sd_sum_path)
    dcm = dcm_result["dcm"]
    model = sdg.read_subdyn_model(sd_path)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    jg.add_joint_axes(connections, model, dcm)
    jg.add_chord_geometry(connections, model, dcm)
    return connections, model


def _connection_key(c):
    """Identity of one brace-to-chord connection -- unique across the 120
    connection rows. Matches the fields scf.compute_all_scf's own output
    rows carry, so scf-rows can be grouped back onto their connection
    without needing compute_all_scf to be changed."""
    return (c["node"], c["sub_joint_id"], c["brace_member"], c["brace_end"],
            c["chord_t_scenario"], c["direction"])


def build_scf_index(connections):
    """{connection_key: [scf-rows]} -- 1 row (TY/X) or up to 4 (K: 2
    treatments x 2 sides) per connection. 368 rows total, matches
    scf.compute_all_scf's own docstring count."""
    rows = scf.compute_all_scf(connections)
    index = {}
    for r in rows:
        key = (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
               r["chord_t_scenario"], r["direction"])
        index.setdefault(key, []).append(r)
    return index
# endregion


# region --- point ordering (single source of truth) ---
def iter_assessment_rows(connections, scf_index):
    """
    Yields (connection, scf_row, variant_positions) for every (connection,
    treatment, side) combination present in scf_index, in a fixed
    deterministic order (connections list order, then scf_index insertion
    order for that connection). variant_positions is an ordered list of
    (position, segment) tuples:
      - X family: [("1",None), ("2",None), ..., ("8",None)] -- 8 entries,
        segment always None (no chord-bending term to split on, AC_att=0.0).
      - K/TY family: [("3",None),("7",None), ("1","a"),("2","a"),("4","a"),
        ("5","a"),("6","a"),("8","a"), ("1","b"),("2","b"),("4","b"),
        ("5","b"),("6","b"),("8","b")] -- 14 entries. Positions 3/7 (pure
        saddle, driven by SCF_AS alone, no SCF_AC anywhere in their
        formula) don't depend on chord segment, built ONCE. Positions
        1/2/4/5/6/8 (crown + intermediate -- ALL of them carry a SCF_AC
        term, hence the chord-bending attachment coefficient, per the
        intermediate-point resolution recorded in stress.hotspot_joint's
        own docstring) get built TWICE, once per chord leg-segment, per the
        "CHORD-SEGMENT HANDLING SUPERSEDED" decision in joint_scf_theory
        memory (never combine the two segments' signals before rainflow).

    Both build_point_table (below) and process_run iterate this SAME
    generator -- guarantees their row orders can never drift apart.
    """
    for c in connections:
        key = _connection_key(c)
        for sr in scf_index.get(key, []):
            if c["family"] == "X":
                # positions are keyed "1".."8" in hotspot_joint's own
                # numbering (DNV Figure 3-6), not by angle.
                variant_positions = [(str(i), None) for i in range(1, 9)]
            else:
                variant_positions = (
                    [(pos, None) for pos in ("3", "7")]
                    + [(pos, "a") for pos in ("1", "2", "4", "5", "6", "8")]
                    + [(pos, "b") for pos in ("1", "2", "4", "5", "6", "8")]
                )
            yield c, sr, variant_positions


def build_point_table(connections, scf_index, model):
    """One row per rainflow-countable joint signal (4,768 for the dev
    fixture / this jacket -- see module docstring). No I/O -- pure
    bookkeeping over already-built connections + scf_index."""
    rows = []
    point_id = 0
    for c, sr, variant_positions in iter_assessment_rows(connections, scf_index):
        z = model["joints"][c["node"]][2]
        for pos, seg in variant_positions:
            rows.append(dict(
                point_id=point_id, node=c["node"], sub_joint_id=str(c["sub_joint_id"]),
                plane_id=c["plane_id"], family=c["family"], type_label=c["type_label"],
                treatment=sr["treatment"], side=sr["side"],
                brace_member=c["brace_member"], brace_end=c["brace_end"],
                chord_t_scenario=c["chord_t_scenario"], direction=c["direction"] or "",
                position=pos, segment=seg or "",
                z=z, brace_D=c["brace_D"], brace_t=c["brace_t"],
                chord_D=c["chord_D"], chord_T=c["chord_T"],
            ))
            point_id += 1
    return rows
# endregion


# region --- provenance stamp ---
def build_stamp(outb_path, header, model, case_json, owner_json, n_points):
    return dict(
        pipeline_version=cfg.PIPELINE_VERSION,
        stage="joint",
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
    """Subset that, if changed, means the stored arrays are stale. Verified
    status (hotspot_joint_verified/scf_equations_verified) is metadata about
    confidence, not about what gets computed -- deliberately excluded."""
    return (
        stamp["pipeline_version"], stamp["n_points"],
        tuple(stamp["bin_edges_mpa"]), tuple(stamp["wohler_exponents"]),
        stamp["transient_cutoff_s"], stamp["subdyn_md5"],
        stamp["outb_size_bytes"], stamp["outb_mtime"],
    )
# endregion


# region --- write side ---
def _npz_json_paths(case_dir, out_root=STAGE2_JOINTS_DIR):
    case_dir = Path(case_dir)
    cond, seed = case_dir.parent.name, case_dir.name
    npz_path = out_root / cond / f"{seed}.npz"
    return npz_path, npz_path.with_suffix(".json")


def _connection_signals(c, outb_path, header, t_cutoff):
    """Per-connection brace + (K/TY only) chord-bending nominal stresses,
    already rotated into the joint frame and transient-trimmed. X-family
    connections never read chord channels -- sig_cb_a/b are zero arrays,
    mathematically inert since AC_att=0.0 for X (see module docstring)."""
    names = [f"M{c['brace_member']}J{c['brace_end']}{comp}" for comp in obr.FATIGUE_COMPONENTS]
    t_full, arr = obr.read_channels(outb_path, header, names)
    N, Mkx, Mky = arr[:, 0], arr[:, 1], arr[:, 2]
    sig_ax, sig_ipb, sig_opb = stress.nominal_components(N, Mkx, Mky, c["brace_D"], c["brace_t"])
    sig_mip, sig_mop = jg.rotate_to_joint_axes(sig_ipb, sig_opb, c["phi_deg"])
    _t_trim, sig_ax = stress.trim_transient(t_full, sig_ax, t_cutoff)
    _t_trim, sig_mip = stress.trim_transient(t_full, sig_mip, t_cutoff)
    _t_trim, sig_mop = stress.trim_transient(t_full, sig_mop, t_cutoff)

    if c["family"] == "X":
        sig_cb_a = np.zeros_like(sig_ax)
        sig_cb_b = np.zeros_like(sig_ax)
    else:
        def _chord_bending(mid, end, D, T, phi_deg):
            cnames = [f"M{mid}J{end}{comp}" for comp in ("MKxe", "MKye")]
            _tc, carr = obr.read_channels(outb_path, header, cnames)
            Mkx_c, Mky_c = carr[:, 0], carr[:, 1]
            _sax, sipb, sopb = stress.nominal_components(
                np.zeros_like(Mkx_c), Mkx_c, Mky_c, D, T)
            smip, _smop = jg.rotate_to_joint_axes(sipb, sopb, phi_deg)
            _t_trim2, smip = stress.trim_transient(t_full, smip, t_cutoff)
            return smip

        sig_cb_a = _chord_bending(c["chord_a_member"], c["chord_a_end"],
                                   c["chord_a_D"], c["chord_a_T"], c["chord_a_phi_deg"])
        sig_cb_b = _chord_bending(c["chord_b_member"], c["chord_b_end"],
                                   c["chord_b_D"], c["chord_b_T"], c["chord_b_phi_deg"])

    return dict(sig_ax=sig_ax, sig_mip=sig_mip, sig_mop=sig_mop,
                sig_cb_a=sig_cb_a, sig_cb_b=sig_cb_b)


def process_run(case_dir, out_root=STAGE2_JOINTS_DIR, force=False):
    """Compute (or skip, if a stamp-matching .npz/.json pair already
    exists) the Stage 2 JOINT histogram file for one run. Returns the .npz
    path. See module docstring for the full per-connection algorithm."""
    case_dir = Path(case_dir)
    outb_path = case_dir / OUTB_NAME
    sd_path = case_dir / SD_NAME
    sd_sum_path = case_dir / SD_SUM_NAME
    case_json = json.loads((case_dir / "case.json").read_text())
    owner_path = case_dir / "owner.json"
    owner_json = json.loads(owner_path.read_text()) if owner_path.exists() else {}

    header = obr.read_outb_header(outb_path)
    connections, model = build_full_connections(sd_path, sd_sum_path)
    scf_index = build_scf_index(connections)
    point_table = build_point_table(connections, scf_index, model)
    n_points = len(point_table)

    stamp = build_stamp(outb_path, header, model, case_json, owner_json, n_points)
    npz_path, json_path = _npz_json_paths(case_dir, out_root)

    if not force and npz_path.exists() and json_path.exists():
        try:
            existing_stamp = json.loads(json_path.read_text())
            if _stamp_recompute_key(existing_stamp) == _stamp_recompute_key(stamp):
                return npz_path  # up to date, nothing to do
        except (json.JSONDecodeError, KeyError):
            pass  # sidecar unreadable/incomplete -- fall through and recompute

    n_bins = cfg.N_BINS
    exponents = cfg.WOHLER_EXPONENTS
    counts = np.zeros((n_points, n_bins), dtype=np.float64)
    sum_r = {m: np.zeros((n_points, n_bins), dtype=np.float64) for m in exponents}
    n_under = np.zeros(n_points, dtype=np.float64)
    n_over = np.zeros(n_points, dtype=np.float64)

    # Per-connection signal cache -- a connection may carry up to 4 scf-rows
    # (K: 2 treatments x 2 sides) that all share the SAME underlying brace/
    # chord signals, only the SCF multipliers differ. Computed once per
    # connection on first use, not once per scf-row.
    signal_cache = {}

    def _signals_for(c):
        key = _connection_key(c)
        if key not in signal_cache:
            signal_cache[key] = _connection_signals(
                c, outb_path, header, cfg.TRANSIENT_CUTOFF_S)
        return signal_cache[key]

    point_id = 0
    for c, sr, variant_positions in iter_assessment_rows(connections, scf_index):
        cs = _signals_for(c)
        AC_base, AC_att = sr["SCF_AC_base"], sr["SCF_AC_att"]
        AS, MIP, MOP = sr["SCF_AS"], sr["SCF_MIP"], sr["SCF_MOP"]

        # Build only the hotspot_joint() calls this row actually needs --
        # one call for X (sig_cb=0, AC_att=0, positions 1-8 all valid), two
        # calls for K/TY (one per chord segment; positions 2/3/4/6/7/8 are
        # identical between the two calls -- see hotspot_joint's own docstring
        # -- reused from the "a" call rather than recomputed).
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

    assert point_id == n_points, (
        f"iter_assessment_rows produced {point_id} signals but build_point_table "
        f"counted {n_points} -- the two must iterate in lockstep, see module docstring"
    )

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

    # Atomic write: .tmp then os.replace() -- same restartability discipline
    # as stage2_histograms.py.
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
def load_stage2_joints(npz_path):
    """Read a Stage 2 joint .npz (+ .json sidecar) back into a dict:
    counts, sum_r (dict {m: array}), n_under, n_over, bin_edges_mpa,
    point_table (list of dicts), stamp. Stage 3 (joint damage, not yet
    built) should use this exclusively."""
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
            z=float(data["z"][i]), brace_D=float(data["brace_D"][i]), brace_t=float(data["brace_t"][i]),
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
    if not stress.HOTSPOT_JOINT_VERIFIED:
        print("NOTE: this run is a build/plumbing self-check only -- the underlying")
        print("hotspot formula has two open judgement calls that have not been")
        print("independently reviewed (see stress.hotspot_joint's docstring). Do NOT treat")
        print("any damage number derived from this output as final.\n")

    out_root = RESULTS_DIR / "_stage2_joints_selfcheck"
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

    stage2 = load_stage2_joints(npz_path)
    n_points = len(stage2["point_table"])
    print(f"\n  loaded: {n_points} points, counts shape {stage2['counts'].shape}, "
          f"exponents {sorted(stage2['sum_r'].keys())}")
    assert n_points == 4768, (
        f"expected 4,768 signals (see module docstring's evaluation-count table), got {n_points}"
    )
    assert stage2["counts"].shape == (n_points, cfg.N_BINS)

    families = {}
    for r in stage2["point_table"]:
        families[r["family"]] = families.get(r["family"], 0) + 1
    print(f"  signals by family: {families} (expect K=3584, TY=672, X=512)")
    assert families == {"K": 3584, "TY": 672, "X": 512}

    print(f"\n  n_over max: {stage2['n_over'].max()} (expect 0.0 -- otherwise "
          f"BIN_HI_MPA={cfg.BIN_HI_MPA} is too low for joint SCF-scaled stresses)")
    assert stage2["n_over"].max() == 0.0, "n_over > 0 somewhere -- BIN_HI_MPA too low"

    # Independent recheck: pick one K connection's chord-side, K-treatment,
    # crown-toe-segment-a signal and recompute it directly, never touching
    # this module's own arrays on the way there.
    connections, model = build_full_connections(
        case_dir / SD_NAME, case_dir / SD_SUM_NAME)
    scf_index = build_scf_index(connections)
    target_row = next(
        r for r in stage2["point_table"]
        if r["family"] == "K" and r["treatment"] == "K" and r["side"] == "chord"
        and r["position"] == "1" and r["segment"] == "a"
    )
    c = next(c for c in connections
             if c["node"] == target_row["node"] and c["brace_member"] == target_row["brace_member"]
             and c["brace_end"] == target_row["brace_end"]
             and c["chord_t_scenario"] == target_row["chord_t_scenario"])
    sr = next(r for r in scf_index[_connection_key(c)]
              if r["treatment"] == "K" and r["side"] == "chord")
    header = obr.read_outb_header(case_dir / OUTB_NAME)
    cs = _connection_signals(c, case_dir / OUTB_NAME, header, cfg.TRANSIENT_CUTOFF_S)
    result_direct = stress.hotspot_joint(
        cs["sig_ax"], cs["sig_mip"], cs["sig_mop"], cs["sig_cb_a"],
        sr["SCF_AC_base"], sr["SCF_AC_att"], sr["SCF_AS"], sr["SCF_MIP"], sr["SCF_MOP"])
    cycles_direct = list(rainflow.extract_cycles(result_direct["1"]))

    print(f"\n  independent recheck: node {c['node']} brace M{c['brace_member']}J{c['brace_end']} "
          f"chord_t_scenario={c['chord_t_scenario']} K-treatment, chord-side, crown toe, segment a "
          f"(point_id {target_row['point_id']}):")
    for m in cfg.WOHLER_EXPONENTS:
        d_direct = rhist.damage_from_cycles(cycles_direct, m)
        d_stored = float(np.sum(stage2["sum_r"][m][target_row["point_id"], :]))
        rel_diff = abs(d_direct - d_stored) / d_direct if d_direct > 0 else abs(d_direct - d_stored)
        print(f"    m={m}: direct={d_direct:.6e}  stored={d_stored:.6e}  rel.diff={rel_diff:.3e}")
        assert rel_diff < 1e-9, f"m={m}: stored histogram doesn't match direct recompute"

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
