"""
Step F -- Stage 3 joint damage.

Joint-track analogue of stage3_damage.py (the member track): turns Stage 2
joint per-run cycle histograms (stage2_joints.py) into a per-connection
fatigue-life table via Miner's rule against the DNV-RP-C203 T-curve
(sn_curves.py), seed-averaged, probability-weighted across load-case bins,
scaled to a 25-year design life. see docs/decisions.md "CANONICAL
END-TO-END PROCEDURE" section (13.08.2026) for the full derivation this
module implements -- READ THAT FIRST if picking this up cold.

DEV-FIXTURE-ONLY STATUS, same as stage2_joints.py: stress.HOTSPOT_JOINT_VERIFIED
is still False (both of hotspot_joint()'s judgement calls are resolved by
reasoning but not independently reviewed). Unlike stage3_damage.py's hard
SN_CONSTANTS_VERIFIED gate (which blocks because a wrong S-N CONSTANT would
silently produce a wrong-but-plausible number -- a data-correctness gate),
this module does NOT hard-block on HOTSPOT_JOINT_VERIFIED: it follows Step
E's own softer convention (print a loud "build/plumbing check only, do not
treat as final" banner, still compute) because the formula itself is fully
built and self-consistent, just not independently reviewed -- a review
gate, not a data gate. Not wired into run_pipeline.py; the hard "refuse to
touch real campaign data" enforcement belongs to Step H's wiring, not here.

THE GROUPING RULE THIS MODULE EXISTS TO GET RIGHT (talked through with the
the author before writing any code): Step E's 4,768 signals/run collapse down to
ONE damage number per (connection, treatment) -- NOT per connection. A
connection's two sides (brace/chord) and, for K/TY, its 14 position/segment
variants ALL collapse together (max-over, same "rainflow first, compare
damage after" rule as the member track's max-over-theta) -- but K-vs-Y
(the treatment) is a genuine "don't pick, report both" uncertainty and must
NEVER be merged into the same group, even though stage2_joints.py's own
internal `_connection_key` (used for its SCF lookups) does not include
treatment. Getting this key one field too coarse would silently envelope
two different modelling assumptions into one number.

EXPECTED OUTPUT SHAPE (verified by this module's own self-check against the
dev fixture's already-verified 4,768-signal breakdown):
  - K family:   64 connections x 2 treatments (K, Y)               = 128 rows
                each row = max over 2 sides x 14 position/segments (28 signals)
  - Y/T family: 24 connections x 1 treatment (no K/Y bound --
                "physically cannot be X or K, nothing to balance against")
                                                                     =  24 rows
                each row = max over 2 sides x 14 position/segments (28 signals)
  - X family:   32 connections x 1 treatment                       =  32 rows
                each row = max over 2 sides x 8 positions (16 signals)
  TOTAL: 184 damage numbers per run. Cross-check: 128*28 + 24*28 + 32*16
  = 3,584 + 672 + 512 = 4,768 -- every one of Step E's signals is consumed
  by exactly one output row, none dropped or double-counted.

A REAL WRINKLE THE MEMBER TRACK NEVER HAD: the T-curve's thickness
correction exponent k depends on the connection's own SCF (k=0.25 if
SCF<=10, else k=0.30 -- see sn_curves.k_for_curve). B1's k_lo==k_hi always,
so stage3_damage.py's point_theta_damage() never needed an scf argument for
real; here it does, sourced from scf.py's own static_scf_max per
(connection, treatment, side) -- built once via build_scf_lookup() below,
not re-derived per point.

ANOTHER REAL WRINKLE: stage2_joints.py's point_table carries `z` but not
`zone`/`environment` (unlike the member track's point_table, which already
has `zone` baked in from Stage 1). This module derives zone itself via
sd_geometry.environment_zone(z, mudline_z) -- same already-verified function
the member track uses, just called directly here since Step E never wired
zone classification into its own output.

GEOMETRY IS CAMPAIGN-CONSTANT, NOT PER-RUN: unlike the brace/chord force
signals (which differ every run), the jacket's connections/SCF table/
mudline_z are identical for every run in the campaign (same SubDyn.dat every
time). build_scf_lookup() and mudline_z are therefore computed ONCE by the
caller and passed down, not rebuilt per seed/bin -- avoids 414x redundant
geometry/SCF recomputation once this is wired into a real campaign run.

Steps:
    1. build_scf_lookup(sd_path, sd_sum_path) / mudline_z_from_sd(sd_path)
       -- campaign-constant geometry/SCF, built once.
    2. connection_theta_damage(...) -- Miner damage for ONE (point_id, ...),
       reusing stage3_damage.point_theta_damage() directly (same physics,
       different caller).
    3. run_connection_damage(stage2, scf_lookup, mudline_z) -- every
       connection-group in one run, max over side/position/segment.
    4. aggregate_condition(cond_dir, scf_lookup, mudline_z) -- every seed in
       one condition folder, mean/std/min/max/n over seeds.
    5. compute_stage3(stage2_root, bins_table, sd_path, sd_sum_path) --
       every condition folder found, probability-weighted, scaled to 25yr.
    6. write_stage3_joint_damage(rows, out_path) -- the deliverable CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_geometry as sdg
import sn_curves as sn
import stage2_joints as s2j
import stage3_damage as s3  # reuse: point_theta_damage, load_bin_probabilities,
                             # load_bin_names, BLOCKS_PER_LIFE, bin_number_from_case_id,
                             # _seed_usable -- same physics/plumbing, different grouping

# region --- paths (mirrors stage3_damage.py) ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

# Real (drive-routed) cache location -- reuses stage2_joints.py's own
# find_drive()-resolved constant rather than duplicating a stale local path
# here (this used to hardcode "_stage2_joints_local", which would have
# silently pointed run_stage3_joints at the wrong folder once
# stage2_joints.py started writing to the drive).
STAGE2_JOINTS_DIR = s2j.STAGE2_JOINTS_DIR
DEV_FIXTURE_STAGE2_JOINTS_DIR = RESULTS_DIR / "_stage2_joints_selfcheck"

BINS_CSV = s3.BINS_CSV
OUTB_NAME = s2j.OUTB_NAME
SD_NAME = s2j.SD_NAME
SD_SUM_NAME = s2j.SD_SUM_NAME
# endregion


# region --- lifetime constants (shared with the member track) ---
DESIGN_LIFE_YEARS = s3.DESIGN_LIFE_YEARS
BLOCKS_PER_LIFE = s3.BLOCKS_PER_LIFE
CATEGORY = "T"  # the joint track always assesses against the T-curve (tubular joints)
# endregion


# region --- campaign-constant geometry/SCF (built once, reused every run) ---
def mudline_z_from_sd(sd_path):
    """min(z) over every SubDyn joint -- same definition sd_geometry.py's
    own member_zone()/environment_zone() callers use. Campaign-constant."""
    model = sdg.read_subdyn_model(sd_path)
    return min(z for _, _, z in model["joints"].values())


def build_scf_lookup(sd_path, sd_sum_path):
    """
    {(node, sub_joint_id, brace_member, brace_end, chord_t_scenario,
      direction, treatment, side): static_scf_max} -- flattens
    stage2_joints.build_scf_index()'s {connection_key: [scf-rows]} (which
    deliberately omits `treatment` from its key, see module docstring) into
    a lookup keyed the same way stage2_joints.py's point_table rows are
    (direction normalized to "" to match the npz round-trip, since
    connections carry direction=None for every non-X family).
    """
    connections, model = s2j.build_full_connections(sd_path, sd_sum_path)
    scf_index = s2j.build_scf_index(connections)
    lookup = {}
    for key, rows in scf_index.items():
        node, sub_joint_id, brace_member, brace_end, chord_t_scenario, direction = key
        for r in rows:
            lookup_key = (node, str(sub_joint_id), brace_member, brace_end,
                          chord_t_scenario, direction or "", r["treatment"], r["side"])
            lookup[lookup_key] = r["static_scf_max"]
    return lookup
# endregion


# region --- one run, every connection-group ---
def _connection_group_key(pt):
    """The grouping key this whole module exists to get right -- see module
    docstring. Includes `treatment`, unlike stage2_joints._connection_key."""
    return (pt["node"], pt["sub_joint_id"], pt["brace_member"], pt["brace_end"],
            pt["chord_t_scenario"], pt["direction"], pt["treatment"])


def run_connection_damage(stage2, scf_lookup, mudline_z):
    """
    stage2: stage2_joints.load_stage2_joints() output for one run. Returns
    one dict per (connection, treatment) group (184 for this jacket -- see
    module docstring): group identity + damage_block (Miner damage for this
    ~600s block, MAX OVER SIDE/POSITION/SEGMENT -- the joint-track analogue
    of the member track's max-over-theta) + which side/position/segment was
    worst.
    """
    bin_edges = stage2["bin_edges_mpa"]
    groups = {}
    for pt in stage2["point_table"]:
        groups.setdefault(_connection_group_key(pt), []).append(pt)

    rows = []
    for key, pts in groups.items():
        (node, sub_joint_id, brace_member, brace_end,
         chord_t_scenario, direction, treatment) = key
        zone = sdg.environment_zone(pts[0]["z"], mudline_z)
        environment = sn.ZONE_TO_ENVIRONMENT[zone]

        best_d, best_pt = -1.0, None
        for pt in pts:
            pid = pt["point_id"]
            t_mm = (pt["brace_t"] if pt["side"] == "brace" else pt["chord_T"]) * 1000.0
            scf_key = (node, sub_joint_id, brace_member, brace_end,
                       chord_t_scenario, direction, treatment, pt["side"])
            scf_val = scf_lookup[scf_key]
            sum_r_point = {m: stage2["sum_r"][m][pid:pid + 1] for m in stage2["sum_r"]}
            d = s3.point_theta_damage(sum_r_point, bin_edges, CATEGORY, environment,
                                       t_mm, scf=scf_val)[0]
            if d > best_d:
                best_d, best_pt = d, pt

        rows.append(dict(
            node=node, sub_joint_id=sub_joint_id, plane_id=best_pt["plane_id"],
            family=best_pt["family"], type_label=best_pt["type_label"],
            brace_member=brace_member, brace_end=brace_end,
            chord_t_scenario=chord_t_scenario, direction=direction,
            treatment=treatment, environment=environment, category=CATEGORY,
            damage_block=float(best_d),
            worst_side=best_pt["side"], worst_position=best_pt["position"],
            worst_segment=best_pt["segment"],
        ))
    return rows
# endregion


# region --- one condition folder, every seed ---
def aggregate_condition(cond_dir, scf_lookup, mudline_z):
    """
    cond_dir: stage2_joints/<COND>/ containing one .npz (+ .json sidecar)
    per seed. Returns (bin_number, [row, ...]) -- one row per connection-
    group, with seed mean/std/min/max/n of damage_block (MEAN OVER SEEDS OF
    THE DAMAGE, never of the stress -- same rule as the member track).
    """
    npz_paths = sorted(Path(cond_dir).glob("*.npz"))
    assert npz_paths, f"no seeds found in {cond_dir}"

    per_seed_damage = []   # [{group_key: damage_block}, ...], usable seeds only
    per_seed_detail = []   # [{group_key: (side, position, segment)}, ...]
    bin_numbers = set()
    meta_by_group = {}
    n_excluded = 0

    for npz_path in npz_paths:
        stage2 = s2j.load_stage2_joints(npz_path)
        stamp = stage2["stamp"]
        if not s3._seed_usable(stamp, stamp["transient_cutoff_s"]):
            n_excluded += 1
            print(f"    excluding {npz_path.name}: duration too short after transient trim")
            continue
        case_id = stamp["case_json"]["case_id"]
        bin_numbers.add(s3.bin_number_from_case_id(case_id))
        rows = run_connection_damage(stage2, scf_lookup, mudline_z)
        group_key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                                r["chord_t_scenario"], r["direction"], r["treatment"])
        per_seed_damage.append({group_key(r): r["damage_block"] for r in rows})
        per_seed_detail.append({group_key(r): (r["worst_side"], r["worst_position"], r["worst_segment"])
                                 for r in rows})
        for r in rows:
            meta_by_group.setdefault(group_key(r), r)

    assert per_seed_damage, f"{cond_dir}: every seed excluded, nothing usable"
    assert len(bin_numbers) == 1, f"{cond_dir}: seeds disagree on bin number: {bin_numbers}"
    bin_number = bin_numbers.pop()
    n_seeds = len(per_seed_damage)

    out_rows = []
    for key, meta in sorted(meta_by_group.items()):
        vals = np.array([seed[key] for seed in per_seed_damage])
        i_dominant = int(np.argmax(vals))
        side_rep, pos_rep, seg_rep = per_seed_detail[i_dominant][key]
        r = {k: v for k, v in meta.items()
             if k not in ("damage_block", "worst_side", "worst_position", "worst_segment")}
        r.update(
            bin_number=bin_number,
            damage_block_mean=float(vals.mean()),
            damage_block_std=float(vals.std(ddof=0)),
            damage_block_min=float(vals.min()),
            damage_block_max=float(vals.max()),
            n_seeds_used=n_seeds,
            n_seeds_excluded=n_excluded,
            worst_side=side_rep, worst_position=pos_rep, worst_segment=seg_rep,
        )
        out_rows.append(r)
    return bin_number, out_rows
# endregion


# region --- every available condition folder, probability-weighted, 25 years ---
def build_per_bin(stage2_root, scf_lookup, mudline_z):
    """Discovers every condition folder under stage2_root, aggregates seeds
    per bin. Returns per_bin: {bin_number: {group_key: row}}."""
    cond_dirs = sorted(p for p in Path(stage2_root).iterdir() if p.is_dir())
    assert cond_dirs, f"no condition folders found under {stage2_root}"

    per_bin = {}
    for i, cond_dir in enumerate(cond_dirs, 1):
        print(f"  [{i}/{len(cond_dirs)}] aggregating {cond_dir.name} ...")
        bin_number, rows = aggregate_condition(cond_dir, scf_lookup, mudline_z)
        key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                          r["chord_t_scenario"], r["direction"], r["treatment"])
        assert bin_number not in per_bin, f"bin {bin_number} appears in more than one condition folder"
        per_bin[bin_number] = {key(r): r for r in rows}
    return per_bin


def compute_stage3(stage2_root, sd_path, sd_sum_path, bins_table=BINS_CSV):
    """
    Discovers every condition folder under stage2_root, aggregates seeds per
    bin, combines bins by the sheet's raw probability, scales to the 25-year
    design life -- direct analogue of stage3_damage.compute_stage3(), just
    grouped by connection instead of by member point.

    Returns (rows, meta) -- meta reports the same "partial-campaign honesty"
    fields as the member track.
    """
    p_bin, raw_total = s3.load_bin_probabilities(bins_table)
    mudline_z = mudline_z_from_sd(sd_path)
    scf_lookup = build_scf_lookup(sd_path, sd_sum_path)
    per_bin = build_per_bin(stage2_root, scf_lookup, mudline_z)
    return stage3_from_per_bin(per_bin, p_bin, raw_total)


def stage3_from_per_bin(per_bin, p_bin, raw_total):
    """Core of compute_stage3(), operating on an already-built per_bin."""
    all_keys = sorted({k for rows in per_bin.values() for k in rows})
    invariant_keys = ("node", "sub_joint_id", "plane_id", "family", "type_label",
                       "brace_member", "brace_end", "chord_t_scenario", "direction",
                       "treatment", "environment", "category")

    out_rows = []
    for gk in all_keys:
        meta = None
        d_life, d_life_var, d_life_min, d_life_max = 0.0, 0.0, 0.0, 0.0
        n_bins_used = 0
        worst_detail_at_max, max_contrib = None, -np.inf

        for bin_number, rows in per_bin.items():
            if gk not in rows:
                continue
            r = rows[gk]
            if meta is None:
                meta = r
            else:
                for key in invariant_keys:
                    assert r[key] == meta[key], (
                        f"connection {gk}: {key} disagrees between bins "
                        f"({r[key]!r} in bin {bin_number} vs {meta[key]!r})"
                    )
            pb = p_bin.get(bin_number)
            assert pb is not None, f"bin {bin_number} (parsed from case_id) not present in bins sheet"

            weight = pb * BLOCKS_PER_LIFE
            contrib_mean = weight * r["damage_block_mean"]
            d_life += contrib_mean
            d_life_var += (weight * r["damage_block_std"]) ** 2
            d_life_min += weight * r["damage_block_min"]
            d_life_max += weight * r["damage_block_max"]
            n_bins_used += 1
            if contrib_mean > max_contrib:
                max_contrib = contrib_mean
                worst_detail_at_max = (r["worst_side"], r["worst_position"], r["worst_segment"])

        row = {k: v for k, v in meta.items()
               if k not in ("bin_number", "damage_block_mean", "damage_block_std",
                             "damage_block_min", "damage_block_max", "n_seeds_used",
                             "n_seeds_excluded", "worst_side", "worst_position", "worst_segment")}
        row.update(
            D_life=d_life,
            D_life_seed_std=float(np.sqrt(d_life_var)),
            D_life_min=d_life_min,
            D_life_max=d_life_max,
            life_years=(DESIGN_LIFE_YEARS / d_life) if d_life > 0 else float("inf"),
            worst_side=worst_detail_at_max[0] if worst_detail_at_max else None,
            worst_position=worst_detail_at_max[1] if worst_detail_at_max else None,
            worst_segment=worst_detail_at_max[2] if worst_detail_at_max else None,
            n_bins_used=n_bins_used,
        )
        out_rows.append(row)

    meta = dict(
        p_bin_raw_sum=raw_total,
        n_bins_available=len(per_bin),
        bins_available=sorted(per_bin),
        n_bins_total_campaign=len(p_bin),
    )
    return out_rows, meta
# endregion


# region --- connection x load-case matrix (overview, not a lifetime estimate) ---
def _full_group_key(r):
    """Same key as _connection_group_key/stage3_from_per_bin's own grouping
    -- (node, sub_joint_id, brace_member, brace_end, chord_t_scenario,
    direction, treatment). NOT collapsed further: both `treatment` (K/Y)
    and `chord_t_scenario` (thick/thin, at the handful of ambiguous nodes --
    see joint_geometry.py) are genuine "don't pick, report both" modeling
    splits, not arbitrary data columns -- collapsing across either would
    silently pick one engineering assumption's answer and present it as THE
    connection's damage, exactly the bug class this module's own docstring
    warns against. Decided by the author: the 184-row full breakdown IS the
    matrix, no further "collapsed" variant."""
    return (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
            r["chord_t_scenario"], r["direction"], r["treatment"])


def matrix_from_per_bin_joints(per_bin, p_bin, bin_names):
    """
    Joint-track analogue of stage3_damage.matrix_from_per_bin() /
    stage3_damage_corrosion.matrix_from_per_bin_corrosion() -- one row per
    connection-group (same 184-row identity as stage3_joint_damage.csv
    itself, see _full_group_key), one column per load-case bin currently in
    per_bin.

    raw      -- damage_block_mean itself, that bin alone -- "which
                conditions are physically harshest for this connection-
                group."
    weighted -- p_bin * BLOCKS_PER_LIFE * damage_block_mean, that bin's real
                share of 25yr D_life -- rows sum to the group's own D_life
                once every campaign bin is present, same partial-campaign
                honesty as the other two matrices.

    Returns (raw_rows, weighted_rows, bin_columns).
    """
    group_bin_rows = {}   # (group_key, bin_number) -> row
    meta_by_key = {}
    for bin_number, rows_by_gk in per_bin.items():
        for _gk, r in rows_by_gk.items():
            key = _full_group_key(r)
            group_bin_rows[(key, bin_number)] = r
            meta_by_key.setdefault(key, r)

    all_keys = sorted(meta_by_key)
    bin_columns = [bin_names[b] for b in sorted(per_bin)]

    raw_rows, weighted_rows = [], []
    for key in all_keys:
        meta = meta_by_key[key]
        id_cols = dict(node=meta["node"], sub_joint_id=meta["sub_joint_id"],
                        brace_member=meta["brace_member"], brace_end=meta["brace_end"],
                        chord_t_scenario=meta["chord_t_scenario"], direction=meta["direction"],
                        treatment=meta["treatment"])

        raw_row = dict(id_cols)
        weighted_row = dict(id_cols)
        worst_col_raw, worst_val_raw = None, -np.inf
        worst_col_w, worst_val_w = None, -np.inf

        for bin_number in sorted(per_bin):
            col = bin_names[bin_number]
            r = group_bin_rows.get((key, bin_number))
            if r is None:
                raw_row[col] = None
                weighted_row[col] = None
                continue

            raw_val = r["damage_block_mean"]
            pb = p_bin.get(bin_number)
            assert pb is not None, f"bin {bin_number} not present in bins sheet"
            weighted_val = pb * BLOCKS_PER_LIFE * raw_val

            raw_row[col] = raw_val
            weighted_row[col] = weighted_val
            if raw_val > worst_val_raw:
                worst_val_raw, worst_col_raw = raw_val, col
            if weighted_val > worst_val_w:
                worst_val_w, worst_col_w = weighted_val, col

        raw_row["worst_bin"] = worst_col_raw
        raw_row["worst_bin_damage"] = worst_val_raw if worst_col_raw is not None else None
        weighted_row["worst_bin"] = worst_col_w
        weighted_row["worst_bin_contribution"] = worst_val_w if worst_col_w is not None else None
        raw_rows.append(raw_row)
        weighted_rows.append(weighted_row)

    return raw_rows, weighted_rows, bin_columns


def write_joint_matrix_csv(rows, bin_columns, worst_col_name, out_path):
    """Fixed column order matching stage3_joint_damage.csv's own identity
    columns, then one column per bin, then worst_bin/worst_col_name."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    id_cols = ["node", "sub_joint_id", "brace_member", "brace_end",
               "chord_t_scenario", "direction", "treatment"]
    cols = id_cols + bin_columns + ["worst_bin", worst_col_name]
    df = pd.DataFrame(rows)[cols].sort_values(
        ["node", "brace_member", "brace_end", "treatment", "chord_t_scenario"]
    ).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def write_joint_matrices(per_bin, p_bin, bin_names, results_dir=RESULTS_DIR):
    """Writes the 2 joint matrix CSVs (raw, weighted). Returns (raw_path, weighted_path)."""
    raw_rows, weighted_rows, bin_columns = matrix_from_per_bin_joints(per_bin, p_bin, bin_names)
    raw_path = results_dir / "joint_damage_matrix_raw.csv"
    weighted_path = results_dir / "joint_damage_matrix_weighted.csv"
    write_joint_matrix_csv(raw_rows, bin_columns, "worst_bin_damage", raw_path)
    write_joint_matrix_csv(weighted_rows, bin_columns, "worst_bin_contribution", weighted_path)
    return raw_path, weighted_path
# endregion


# region --- output ---
def write_stage3_joint_damage(rows, out_path=RESULTS_DIR / "stage3_joint_damage.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ["node", "brace_member", "brace_end", "treatment", "chord_t_scenario"]
    ).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df
# endregion


def _self_check():
    import stress
    import scf as scf_mod
    print(f"HOTSPOT_JOINT_VERIFIED = {stress.HOTSPOT_JOINT_VERIFIED}")
    print(f"SCF_EQUATIONS_VERIFIED = {scf_mod.SCF_EQUATIONS_VERIFIED}")
    if not stress.HOTSPOT_JOINT_VERIFIED:
        print("NOTE: this run is a build/plumbing self-check only -- the underlying")
        print("hotspot formula has two open judgement calls that have not been")
        print("independently reviewed (see stress.hotspot_joint's docstring). Do NOT treat")
        print("any damage number derived from this output as final.")
    print()

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    sd_path = case_dir / SD_NAME
    outb_candidates = list(case_dir.glob("*.SD.sum.yaml"))
    assert outb_candidates, f"no SD.sum.yaml found in {case_dir}"
    sd_sum_path = outb_candidates[0]

    print("1. Campaign-constant geometry/SCF (built once):")
    mudline_z = mudline_z_from_sd(sd_path)
    scf_lookup = build_scf_lookup(sd_path, sd_sum_path)
    print(f"   mudline_z = {mudline_z:.3f}")
    print(f"   scf_lookup has {len(scf_lookup)} entries (expect 368*... wait -- "
          f"see scf.compute_all_scf's own 368-row count x nothing extra, this IS that count)")

    cond_dir = DEV_FIXTURE_STAGE2_JOINTS_DIR / "LC_V20_H3p5_T8"
    if not cond_dir.exists():
        print(f"\n   SKIPPED rest of self-check -- {cond_dir} not found "
              f"(run stage2_joints.process_run for the dev fixture's 6 seeds first)")
        return

    print("\n2. run_connection_damage() on ONE seed -- row count + family split:")
    npz_path = sorted(cond_dir.glob("*.npz"))[0]
    stage2 = s2j.load_stage2_joints(npz_path)
    rows = run_connection_damage(stage2, scf_lookup, mudline_z)
    print(f"   {len(rows)} connection-group rows (expect 184)")
    assert len(rows) == 184

    n_k = sum(1 for r in rows if r["family"] == "K")
    n_ty = sum(1 for r in rows if r["family"] == "TY")
    n_x = sum(1 for r in rows if r["family"] == "X")
    print(f"   family split: K={n_k} (expect 128)  TY={n_ty} (expect 24)  X={n_x} (expect 32)")
    assert (n_k, n_ty, n_x) == (128, 24, 32), \
        f"family split {(n_k, n_ty, n_x)} != expected (128, 24, 32)"

    n_signals_consumed = sum(
        (28 if r["family"] in ("K", "TY") else 16) for r in rows
    )
    print(f"   signals consumed = {n_signals_consumed} (expect 4,768, matching Step E's own count)")
    assert n_signals_consumed == 4768

    print("\n3. Hand-verified regression: node=21, brace=56, end=2, chord_t=single")
    target = [r for r in rows if r["node"] == 21 and r["brace_member"] == 56
              and r["brace_end"] == 2 and r["chord_t_scenario"] == "single"]
    assert len(target) == 2, f"expected exactly 2 rows (K, Y) for this connection, got {len(target)}"
    by_treat = {r["treatment"]: r for r in target}
    d_k = by_treat["K"]["damage_block"]
    d_y = by_treat["Y"]["damage_block"]
    print(f"   K-treatment: D_block={d_k:.6e}  (hand-calc: 5.290144e-06)")
    print(f"   Y-treatment: D_block={d_y:.6e}  (hand-calc: 3.660001e-05)")
    assert abs(d_k - 5.290144e-06) / 5.290144e-06 < 1e-6
    assert abs(d_y - 3.660001e-05) / 3.660001e-05 < 1e-6
    assert by_treat["K"]["worst_side"] == "chord" and by_treat["K"]["worst_position"] == "3"
    assert by_treat["Y"]["worst_side"] == "chord" and by_treat["Y"]["worst_position"] == "3"
    print("   matches the hand walkthrough exactly.")

    print("\n4. aggregate_condition() -- all 6 seeds, row count stable:")
    bin_number, agg_rows = aggregate_condition(cond_dir, scf_lookup, mudline_z)
    print(f"   bin_number={bin_number} (expect 65), {len(agg_rows)} rows (expect 184)")
    assert bin_number == 65
    assert len(agg_rows) == 184
    assert all(r["n_seeds_used"] == 6 for r in agg_rows), "expected all 6 real seeds usable"
    assert all(np.isfinite(r["damage_block_mean"]) for r in agg_rows)

    ranked = sorted(agg_rows, key=lambda r: -r["damage_block_mean"])
    print("   top 5 most-damaged connections this run/bin:")
    for r in ranked[:5]:
        print(f"     node={r['node']:>3} brace={r['brace_member']:>3} end={r['brace_end']} "
              f"treatment={r['treatment']} chord_t={r['chord_t_scenario']:<6} "
              f"D_block={r['damage_block_mean']:.3e}")

    print("\n5. compute_stage3() -- probability-weighted, dev fixture (single bin):")
    p_bin, raw_total = s3.load_bin_probabilities()
    per_bin = {bin_number: {(r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                              r["chord_t_scenario"], r["direction"], r["treatment"]): r
                             for r in agg_rows}}
    stage3_rows, meta = stage3_from_per_bin(per_bin, p_bin, raw_total)
    print(f"   {len(stage3_rows)} rows (expect 184)")
    assert len(stage3_rows) == 184
    assert all(np.isfinite(r["D_life"]) and r["D_life"] >= 0 for r in stage3_rows)

    target_life = [r for r in stage3_rows if r["node"] == 21 and r["brace_member"] == 56
                   and r["brace_end"] == 2 and r["chord_t_scenario"] == "single"]
    for r in sorted(target_life, key=lambda r: r["treatment"]):
        print(f"   treatment={r['treatment']}: D_life (this bin's share) = {r['D_life']:.6e} "
              f"({r['D_life']*100:.4f}% of 25yr)")

    out_path = write_stage3_joint_damage(stage3_rows, RESULTS_DIR / "_stage3_joint_damage_selfcheck.csv")
    print(f"\n   wrote {out_path} ({len(stage3_rows)} rows)")

    print("\n6. Connection x load-case matrix (dev fixture, single bin):")
    bin_names = s3.load_bin_names()
    selfcheck_results_dir = RESULTS_DIR / "_matrix_joints_selfcheck"
    raw_path, weighted_path = write_joint_matrices(per_bin, p_bin, bin_names,
                                                     results_dir=selfcheck_results_dir)
    raw_df = pd.read_csv(raw_path)
    weighted_df = pd.read_csv(weighted_path)
    print(f"   {raw_df.shape[0]} row(s) -> {raw_path.name}, {weighted_path.name}")
    assert raw_df.shape[0] == 184, f"expected 184 connection-groups, got {raw_df.shape[0]}"
    assert set(raw_df["treatment"]) <= {"K", "Y", "TY", "X"} or raw_df["treatment"].notna().all()

    # Cross-check: the matrix's single-bin raw/weighted values for this
    # connection must equal the SEED-AGGREGATED damage_block_mean from
    # agg_rows (step 4) and its D_life weighting -- NOT d_k (step 3's
    # single-seed value, a different number). Proves the matrix's per-cell
    # aggregation/weighting matches stage3_from_per_bin's own math exactly,
    # not just structurally similar code.
    agg_target = [r for r in agg_rows if r["node"] == 21 and r["brace_member"] == 56
                  and r["brace_end"] == 2 and r["chord_t_scenario"] == "single"
                  and r["treatment"] == "K"]
    assert len(agg_target) == 1
    damage_block_mean_agg = agg_target[0]["damage_block_mean"]

    target_row = raw_df[(raw_df["node"] == 21) & (raw_df["brace_member"] == 56)
                         & (raw_df["brace_end"] == 2) & (raw_df["chord_t_scenario"] == "single")
                         & (raw_df["treatment"] == "K")]
    assert len(target_row) == 1
    bin_col = bin_names[bin_number]
    matrix_raw_val = float(target_row.iloc[0][bin_col])
    assert abs(matrix_raw_val - damage_block_mean_agg) / damage_block_mean_agg < 1e-9, \
        f"matrix raw value {matrix_raw_val} != aggregated damage_block_mean {damage_block_mean_agg}"

    target_weighted_row = weighted_df[
        (weighted_df["node"] == 21) & (weighted_df["brace_member"] == 56)
        & (weighted_df["brace_end"] == 2) & (weighted_df["chord_t_scenario"] == "single")
        & (weighted_df["treatment"] == "K")
    ]
    matrix_weighted_val = float(target_weighted_row.iloc[0][bin_col])
    expected_weighted = p_bin[bin_number] * BLOCKS_PER_LIFE * damage_block_mean_agg
    assert abs(matrix_weighted_val - expected_weighted) / expected_weighted < 1e-9
    print(f"   cross-check ok: node=21/brace=56/end=2/K matrix cell matches the seed-aggregated "
          f"damage_block_mean and its D_life weighting exactly")

    print("\n" + "=" * 78)
    print("Reminder: dev-fixture-only, single bin. D_life above reflects ONLY bin 65's")
    print("probability share -- not a lifetime claim. Do not run against the real")
    print("campaign until stress.HOTSPOT_JOINT_VERIFIED is True.")
    print("=" * 78)


if __name__ == "__main__":
    _self_check()
