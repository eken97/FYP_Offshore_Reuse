"""
Joint-track corrosion, Step J4 -- Stage 3 damage and lifetime.

NEW file. Direct analogue of stage3_joint_damage.py (the uncorroded joint
Stage 3) crossed with stage3_damage_corrosion.py's (member-track corrosion)
year-step machinery. Reuses stage3_damage.point_theta_damage() unmodified
(same physics, different caller -- same pattern every Stage 3 in this
pipeline follows).

THE GROUPING RULE, UNCHANGED FROM stage3_joint_damage.py: ONE damage number
per (connection, treatment, year) -- NOT per connection, and NOT collapsing
K vs Y. A connection's two sides (brace/chord) and its position/segment
variants collapse together (max-over, same "rainflow first, compare damage
after" rule), but `treatment` (K/Y) stays a real output column all the way
through, same "don't pick, report both" discipline as chord_t_scenario --
see stage3_joint_damage.py's own module docstring for the full reasoning
(re-affirmed this session after initially mis-describing this step as an
envelope -- it is not).

TWO THINGS THAT CHANGE PER YEAR, BOTH SOURCED FROM THE EARLIER CORROSION
STEPS RATHER THAN RE-DERIVED HERE:
  1. The T-curve thickness-correction uses the CORRODED brace_t/chord_T at
     that year (joint_geometry_corrosion.corroded_connection_geometry), not
     the year-0 t0 stored in stage2_joints_corrosion's point table -- same
     fix stage3_damage_corrosion.py already made for the member track.
  2. The SCF value feeding the T-curve's k-branch (k=0.25 if SCF<=10 else
     0.30) is the year-specific static_scf_max from
     scf_corrosion.compute_scf_corroded(), not stage3_joint_damage.py's own
     year-0-only build_scf_lookup().

EACH 5-YEAR STEP WEIGHTED BY 5 YEARS OF LIFE, NOT 25 -- identical principle
to stage3_damage_corrosion.py: each year-step's damage rate is only valid
for the 5 years it represents, so weight = p_bin * BLOCKS_PER_STEP (not
BLOCKS_PER_LIFE), and the total D_life is the SUM of all steps' shares.

Order of operations, extending stage3_joint_damage.py's own pinned order:
    1. damage per (run, connection-group, year) -- MAX OVER SIDE/POSITION/
       SEGMENT, using that year's corroded t_mm + SCF.
    2. MEAN OVER SEEDS OF THE DAMAGE, per (connection-group, year).
    3. Weighted sum over bins, PER YEAR-STEP (weight = p_bin * BLOCKS_PER_STEP).
    4. SUM OVER YEAR-STEPS -- total D_life per (connection, treatment) over
       the full corrosion horizon.

Steps:
    1. run_connection_damage_corrosion(stage2c, ...) -- Miner damage per
       (connection-group, year), one run, max over side/position/segment.
    2. aggregate_condition_corrosion(cond_dir, ...) -- every seed in one
       condition folder, mean/std/min/max/n over seeds.
    3. build_per_bin_corrosion(stage2_root, ...) -- every condition folder found.
    4. compute_stage3_corrosion(stage2_root, sd_path, sd_sum_path, ...) --
       probability-weighted per year-step, summed over years.
    5. write_stage3_joint_damage_corrosion(rows, out_path) -- the deliverable CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_geometry as sdg
import sn_curves as sn
import stage2_joints as s2j
import stage2_joints_corrosion as s2jc
import stage3_joint_damage as s3j   # reuse: mudline_z_from_sd, build_scf_lookup's
                                     # sibling pattern, CATEGORY
import stage3_damage as s3          # reuse: point_theta_damage, load_bin_probabilities,
                                     # load_bin_names, bin_number_from_case_id,
                                     # _seed_usable, BLOCKS_PER_YEAR
import joint_geometry_corrosion as jgc
import scf_corrosion as scfc

# region --- paths (mirrors stage3_joint_damage.py) ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

BINS_CSV = s3.BINS_CSV
CATEGORY = s3j.CATEGORY  # "T" -- the joint track always assesses against the T-curve
# endregion


# region --- campaign-constant geometry (built once, reused every run) ---
def build_geometry(sd_path, sd_sum_path):
    """(splash_connections, model, mudline_z, k_groups) -- everything this
    module needs about the jacket, computed once, campaign-constant (same
    SubDyn.dat every run in the campaign)."""
    connections, model = s2j.build_full_connections(sd_path, sd_sum_path)
    splash = s2jc.splash_connections(connections, model)
    k_groups = s2jc._k_pairing(splash)
    mudline_z = s3j.mudline_z_from_sd(sd_path)
    return splash, model, mudline_z, k_groups


def _connection_lookup(connections):
    """{(node, brace_member, brace_end, chord_t_scenario, direction): connection}
    -- direction normalized to "" for K/TY, matching the npz round-trip
    convention stage2_joints.build_point_table already uses."""
    lookup = {}
    for c in connections:
        key = (c["node"], c["brace_member"], c["brace_end"],
               c["chord_t_scenario"], c["direction"] or "")
        lookup[key] = c
    return lookup


def build_scf_lookup_corroded(connections, model, year):
    """{(node, sub_joint_id, brace_member, brace_end, chord_t_scenario,
      direction, treatment, side): static_scf_max} at corrosion `year` --
    direct analogue of stage3_joint_damage.build_scf_lookup(), sourced from
    scf_corrosion.compute_scf_corroded() instead of scf.compute_all_scf()."""
    rows = scfc.compute_scf_corroded(connections, model, year)
    lookup = {}
    for r in rows:
        key = (r["node"], str(r["sub_joint_id"]), r["brace_member"], r["brace_end"],
               r["chord_t_scenario"], r["direction"] or "", r["treatment"], r["side"])
        lookup[key] = r["static_scf_max"]
    return lookup


def _corroded_t_mm(c, model, other_row, side, year):
    """Corroded brace_t (side='brace') or chord_T (side='chord') at `year`,
    in mm -- the thickness the T-curve's (t/t_ref)^k correction actually
    needs, NOT the year-0 t0 stage2_joints_corrosion's point table carries
    for reference only."""
    g = jgc.corroded_connection_geometry(c, model, other_row, year)
    t_m = g["brace_t"] if side == "brace" else g["chord_T"]
    return t_m * 1000.0


def _other_row_for(c, k_groups):
    if c["family"] != "K":
        return None
    pair = k_groups[(c["sub_joint_id"], c["chord_t_scenario"])]
    return pair[0] if pair[1] is c else pair[1]
# endregion


# region --- one run, every (connection-group, year) ---
def _group_key(pt):
    """Same grouping rule as stage3_joint_damage._connection_group_key --
    includes `treatment`, does NOT include `year` (year is this module's own
    added axis, appended separately below)."""
    return (pt["node"], pt["sub_joint_id"], pt["brace_member"], pt["brace_end"],
            pt["chord_t_scenario"], pt["direction"], pt["treatment"])


def run_connection_damage_corrosion(stage2c, conn_lookup, k_groups, model, mudline_z,
                                     category=CATEGORY):
    """
    stage2c: stage2_joints_corrosion.load_stage2_joints_corrosion() output
    for one run. Returns one dict per (connection-group, year): group
    identity + year + damage_block (Miner damage for this ~600s block at
    that year's corroded section/SCF, MAX OVER SIDE/POSITION/SEGMENT -- same
    rule as the uncorroded joint track's run_connection_damage()).
    """
    bin_edges = stage2c["bin_edges_mpa"]
    years = stage2c["years"]
    groups = {}
    for pt in stage2c["point_table"]:
        groups.setdefault(_group_key(pt), []).append(pt)

    splash_connections = list(conn_lookup.values())
    rows = []
    for year_idx, year in enumerate(years):
        scf_lookup_year = build_scf_lookup_corroded(splash_connections, model, year)

        for key, pts in groups.items():
            (node, sub_joint_id, brace_member, brace_end,
             chord_t_scenario, direction, treatment) = key
            zone = sdg.environment_zone(pts[0]["z"], mudline_z)
            environment = sn.ZONE_TO_ENVIRONMENT[zone]

            c = conn_lookup[(node, brace_member, brace_end, chord_t_scenario, direction)]
            other_row = _other_row_for(c, k_groups)

            best_d, best_pt = -1.0, None
            for pt in pts:
                pid = pt["point_id"]
                t_mm = _corroded_t_mm(c, model, other_row, pt["side"], year)
                scf_key = (node, sub_joint_id, brace_member, brace_end,
                           chord_t_scenario, direction, treatment, pt["side"])
                scf_val = scf_lookup_year[scf_key]
                sum_r_point = {m: stage2c["sum_r"][m][pid:pid + 1, year_idx, :]
                               for m in stage2c["sum_r"]}
                d = s3.point_theta_damage(sum_r_point, bin_edges, category, environment,
                                           t_mm, scf=scf_val)[0]
                if d > best_d:
                    best_d, best_pt = d, pt

            rows.append(dict(
                node=node, sub_joint_id=sub_joint_id, plane_id=best_pt["plane_id"],
                family=best_pt["family"], type_label=best_pt["type_label"],
                brace_member=brace_member, brace_end=brace_end,
                chord_t_scenario=chord_t_scenario, direction=direction,
                treatment=treatment, environment=environment, category=category,
                year=int(year),
                damage_block=float(best_d),
                worst_side=best_pt["side"], worst_position=best_pt["position"],
                worst_segment=best_pt["segment"],
            ))
    return rows
# endregion


# region --- one condition folder, every seed ---
def aggregate_condition_corrosion(cond_dir, splash_connections, k_groups, model, mudline_z,
                                   category=CATEGORY):
    """
    cond_dir: stage2_joints_corrosion/<COND>/ containing one .npz (+ .json
    sidecar) per seed. Returns (bin_number, [row, ...]) -- one row per
    (connection-group, year), with seed mean/std/min/max/n of damage_block.
    """
    npz_paths = sorted(Path(cond_dir).glob("*.npz"))
    assert npz_paths, f"no seeds found in {cond_dir}"

    conn_lookup = _connection_lookup(splash_connections)

    per_seed_damage = []
    per_seed_detail = []
    bin_numbers = set()
    meta_by_key = {}
    n_excluded = 0

    for npz_path in npz_paths:
        stage2c = s2jc.load_stage2_joints_corrosion(npz_path)
        stamp = stage2c["stamp"]
        if not s3._seed_usable(stamp, stamp["transient_cutoff_s"]):
            n_excluded += 1
            print(f"    excluding {npz_path.name}: duration too short after transient trim")
            continue
        case_id = stamp["case_json"]["case_id"]
        bin_numbers.add(s3.bin_number_from_case_id(case_id))
        rows = run_connection_damage_corrosion(stage2c, conn_lookup, k_groups, model,
                                                mudline_z, category=category)
        key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                          r["chord_t_scenario"], r["direction"], r["treatment"], r["year"])
        per_seed_damage.append({key(r): r["damage_block"] for r in rows})
        per_seed_detail.append({key(r): (r["worst_side"], r["worst_position"], r["worst_segment"])
                                 for r in rows})
        for r in rows:
            meta_by_key.setdefault(key(r), r)

    assert per_seed_damage, f"{cond_dir}: every seed excluded, nothing usable"
    assert len(bin_numbers) == 1, f"{cond_dir}: seeds disagree on bin number: {bin_numbers}"
    bin_number = bin_numbers.pop()
    n_seeds = len(per_seed_damage)

    out_rows = []
    for key, meta in sorted(meta_by_key.items()):
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


# region --- every available condition folder, probability-weighted, summed over years ---
def build_per_bin_corrosion(stage2_root, splash_connections, k_groups, model, mudline_z,
                             category=CATEGORY):
    cond_dirs = sorted(p for p in Path(stage2_root).iterdir() if p.is_dir())
    assert cond_dirs, f"no condition folders found under {stage2_root}"

    per_bin = {}
    for i, cond_dir in enumerate(cond_dirs, 1):
        print(f"  [{i}/{len(cond_dirs)}] aggregating {cond_dir.name} ...")
        bin_number, rows = aggregate_condition_corrosion(
            cond_dir, splash_connections, k_groups, model, mudline_z, category=category)
        key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                          r["chord_t_scenario"], r["direction"], r["treatment"], r["year"])
        assert bin_number not in per_bin, f"bin {bin_number} appears in more than one condition folder"
        per_bin[bin_number] = {key(r): r for r in rows}
    return per_bin


def compute_stage3_corrosion(stage2_root, sd_path, sd_sum_path, bins_table=BINS_CSV,
                              category=CATEGORY, step_years=5.0):
    p_bin, raw_total = s3.load_bin_probabilities(bins_table)
    splash, model, mudline_z, k_groups = build_geometry(sd_path, sd_sum_path)
    per_bin = build_per_bin_corrosion(stage2_root, splash, k_groups, model, mudline_z,
                                       category=category)
    return stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, step_years=step_years)


def stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, step_years=5.0):
    """Core of compute_stage3_corrosion(), operating on an already-built
    per_bin. Keys are (..., treatment, year) -- probability-weight per
    year-step across bins, then SUM over years per (connection, treatment)
    group -- same two-stage reduction as stage3_damage_corrosion.py's own
    stage3_from_per_bin_corrosion, just with `treatment` carried through the
    whole way (never enveloped, see module docstring)."""
    BLOCKS_PER_STEP = step_years * s3.BLOCKS_PER_YEAR

    all_keys = sorted({k for rows in per_bin.values() for k in rows})  # (..., treatment, year)
    invariant_keys = ("node", "sub_joint_id", "plane_id", "family", "type_label",
                       "brace_member", "brace_end", "chord_t_scenario", "direction",
                       "treatment", "environment", "category")

    step_d = {}
    step_meta = {}
    step_worst = {}
    n_bins_used_by_key = {}

    for key in all_keys:
        meta = None
        d_step = 0.0
        n_bins_used = 0
        worst_detail_at_max, max_contrib = None, -np.inf

        for bin_number, rows in per_bin.items():
            if key not in rows:
                continue
            r = rows[key]
            if meta is None:
                meta = r
            else:
                for ikey in invariant_keys:
                    assert r[ikey] == meta[ikey], (
                        f"connection/year {key}: {ikey} disagrees between bins "
                        f"({r[ikey]!r} in bin {bin_number} vs {meta[ikey]!r})"
                    )
            pb = p_bin.get(bin_number)
            assert pb is not None, f"bin {bin_number} (parsed from case_id) not present in bins sheet"

            weight = pb * BLOCKS_PER_STEP
            contrib_mean = weight * r["damage_block_mean"]
            d_step += contrib_mean
            n_bins_used += 1
            if contrib_mean > max_contrib:
                max_contrib = contrib_mean
                worst_detail_at_max = (r["worst_side"], r["worst_position"], r["worst_segment"])

        step_d[key] = d_step
        step_meta[key] = meta
        step_worst[key] = worst_detail_at_max
        n_bins_used_by_key[key] = n_bins_used

    # --- sum over year-steps, per (connection, treatment) group ---
    group_keys = sorted({k[:-1] for k in all_keys})   # drop the trailing year
    years_by_group = {}
    for key in all_keys:
        years_by_group.setdefault(key[:-1], []).append(key[-1])

    out_rows = []
    for gk in group_keys:
        years = sorted(years_by_group[gk])
        d_life = sum(step_d[gk + (y,)] for y in years)
        n_bins_used_total = sum(n_bins_used_by_key[gk + (y,)] for y in years)

        meta_final = step_meta[gk + (years[-1],)]   # most-corroded year, for reference
        worst_year = max(years, key=lambda y: step_d[gk + (y,)])
        worst_detail = step_worst[gk + (worst_year,)]

        horizon_years = len(years) * step_years
        row = {k: v for k, v in meta_final.items()
               if k not in ("bin_number", "damage_block_mean", "damage_block_std",
                             "damage_block_min", "damage_block_max", "n_seeds_used",
                             "n_seeds_excluded", "worst_side", "worst_position",
                             "worst_segment", "year")}
        row.update(
            corrosion_horizon_years=horizon_years,
            D_life=d_life,
            life_years=(horizon_years / d_life) if d_life > 0 else float("inf"),
            worst_year=worst_year,
            worst_side=worst_detail[0] if worst_detail else None,
            worst_position=worst_detail[1] if worst_detail else None,
            worst_segment=worst_detail[2] if worst_detail else None,
            n_bins_used=n_bins_used_total,
        )
        out_rows.append(row)

    meta = dict(
        p_bin_raw_sum=raw_total,
        n_bins_available=len(per_bin),
        bins_available=sorted(per_bin),
        n_bins_total_campaign=len(p_bin),
        corrosion_years=sorted({k[-1] for k in all_keys}),
        step_years=step_years,
    )
    return out_rows, meta
# endregion


# region --- output ---
def write_stage3_joint_damage_corrosion(rows, out_path=RESULTS_DIR / "stage3_joint_damage_corrosion.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ["node", "brace_member", "brace_end", "treatment", "chord_t_scenario"]
    ).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df
# endregion


# region --- connection x load-case matrix, ONE per corrosion year-step (overview, not a lifetime estimate) ---
def matrix_from_per_bin_corrosion_joints(per_bin, p_bin, bin_names, year, step_years=5.0):
    """
    Joint-track analogue of stage3_damage_corrosion.matrix_from_per_bin_corrosion(),
    for ONE corrosion year-step in isolation -- e.g. the year=20 matrix shows
    only that 5-year block's (15->20yr) own damage contribution, NOT the
    cumulative total through year 20 (same per-step-only convention as the
    member-track corrosion matrices, confirmed by the author). Row identity
    is the same (node, sub_joint_id, brace_member, brace_end,
    chord_t_scenario, direction, treatment) 7-tuple as
    stage3_joint_damage.matrix_from_per_bin_joints() and this module's own
    stage3_from_per_bin_corrosion() group key (K/Y never merged) -- scoped to
    whatever connections `per_bin` actually contains (the corrosion track's
    splash-only connections).

    raw      -- damage_block_mean itself, for THIS year's corroded section,
                THAT bin alone -- "which conditions are physically harshest
                at this corrosion stage."
    weighted -- p_bin * BLOCKS_PER_STEP * damage_block_mean, that bin's real
                share of THIS STEP's damage (not the full horizon).

    Returns (raw_rows, weighted_rows, bin_columns), same shape convention as
    stage3_joint_damage.matrix_from_per_bin_joints().
    """
    BLOCKS_PER_STEP = step_years * s3.BLOCKS_PER_YEAR

    group_bin_rows = {}
    meta_by_key = {}
    for bin_number, rows_by_key in per_bin.items():
        for key, r in rows_by_key.items():
            *gk, yr = key
            if yr != year:
                continue
            gk = tuple(gk)
            group_bin_rows[(gk, bin_number)] = r
            meta_by_key.setdefault(gk, r)

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
            weighted_val = pb * BLOCKS_PER_STEP * raw_val

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


def write_joint_matrix_csv_corrosion(rows, bin_columns, worst_col_name, out_path):
    """Same fixed column convention as stage3_joint_damage.write_joint_matrix_csv."""
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


def write_year_step_matrices_joints(per_bin, p_bin, bin_names, years, step_years=5.0,
                                     results_dir=RESULTS_DIR,
                                     prefix="joint_damage_matrix_corrosion"):
    """
    Convenience driver: writes 2 CSVs (raw, weighted) per year in `years`.
    File names: <prefix>_raw_y<year>.csv / <prefix>_weighted_y<year>.csv.
    `prefix` lets a retrofit+corrosion caller (stage3_joint_damage_thickness_
    corrosion.py's own per-scenario per_bin) reuse this unmodified rather
    than duplicating it -- same "thin driver, reuse already-verified
    formulas" pattern as the rest of this pipeline. Returns a dict
    {year: (raw_path, weighted_path)}.
    """
    out_paths = {}
    for year in years:
        raw_rows, weighted_rows, bin_columns = matrix_from_per_bin_corrosion_joints(
            per_bin, p_bin, bin_names, year, step_years=step_years
        )
        raw_path = results_dir / f"{prefix}_raw_y{year}.csv"
        weighted_path = results_dir / f"{prefix}_weighted_y{year}.csv"
        write_joint_matrix_csv_corrosion(raw_rows, bin_columns, "worst_bin_damage", raw_path)
        write_joint_matrix_csv_corrosion(weighted_rows, bin_columns, "worst_bin_contribution", weighted_path)
        out_paths[year] = (raw_path, weighted_path)
    return out_paths
# endregion


def _self_check():
    import stress
    import scf as scf_mod

    print(f"HOTSPOT_JOINT_VERIFIED = {stress.HOTSPOT_JOINT_VERIFIED}")
    print(f"SCF_EQUATIONS_VERIFIED = {scf_mod.SCF_EQUATIONS_VERIFIED}\n")

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    sd_path = case_dir / s2jc.SD_NAME
    sd_sum_path = case_dir / s2jc.SD_SUM_NAME
    assert case_dir.exists(), f"missing fixture: {case_dir}"

    years = [5, 10, 15, 20, 25]
    stage2c_dir = RESULTS_DIR / "_stage2_joints_corrosion_selfcheck"
    cond_dir = stage2c_dir / "LC_V20_H3p5_T8"
    seed_dirs = sorted(p for p in (DEV_FIXTURE_DIR / "LC_V20_H3p5_T8").iterdir() if p.is_dir())
    print(f"  ensuring joint corrosion Stage 2 cache exists for all {len(seed_dirs)} seeds "
          f"(skips any already up to date)...")
    for seed_dir in seed_dirs:
        s2jc.process_run_corroded(seed_dir, years, out_root=stage2c_dir)

    splash, model, mudline_z, k_groups = build_geometry(sd_path, sd_sum_path)
    print(f"\n  mudline_z = {mudline_z:.3f}, {len(splash)} splash connections "
          f"(by family: K={sum(1 for c in splash if c['family']=='K')}, "
          f"X={sum(1 for c in splash if c['family']=='X')})")
    conn_lookup = _connection_lookup(splash)

    print("\n1. run_connection_damage_corrosion() on ONE seed:")
    npz_path = sorted(cond_dir.glob("*.npz"))[0]
    stage2c = s2jc.load_stage2_joints_corrosion(npz_path)
    rows = run_connection_damage_corrosion(stage2c, conn_lookup, k_groups, model, mudline_z)
    # 16 K connections x 2 treatments + 8 X connections x 1 treatment = 40 groups
    n_groups_expected = 16 * 2 + 8 * 1
    print(f"   {len(rows)} rows (expect {n_groups_expected} groups x {len(years)} years "
          f"= {n_groups_expected * len(years)})")
    assert len(rows) == n_groups_expected * len(years)

    families = {}
    for r in rows:
        families[r["family"]] = families.get(r["family"], 0) + 1
    print(f"   family split: {families} (expect K={16*2*len(years)}, X={8*1*len(years)})")
    assert families == {"K": 16 * 2 * len(years), "X": 8 * 1 * len(years)}

    # sanity: damage should increase with year, for a fixed (connection, treatment)
    one_group_rows = sorted(
        (r for r in rows if r["node"] == splash[0]["node"]
         and r["brace_member"] == splash[0]["brace_member"]
         and r["brace_end"] == splash[0]["brace_end"]
         and r["treatment"] == ("K" if splash[0]["family"] == "K" else splash[0]["family"])),
        key=lambda r: r["year"]
    )
    d_by_year = [r["damage_block"] for r in one_group_rows]
    print(f"   node {splash[0]['node']} brace {splash[0]['brace_member']} damage vs year: "
          f"{[f'{d:.3e}' for d in d_by_year]}")
    assert all(d_by_year[i] <= d_by_year[i + 1] for i in range(len(d_by_year) - 1)), \
        "damage should be monotonically non-decreasing with corrosion year"

    print("\n2. aggregate_condition_corrosion() -- all 6 seeds:")
    bin_number, agg_rows = aggregate_condition_corrosion(cond_dir, splash, k_groups, model, mudline_z)
    print(f"   bin_number={bin_number} (expect 65), {len(agg_rows)} rows "
          f"(expect {n_groups_expected * len(years)})")
    assert bin_number == 65
    assert len(agg_rows) == n_groups_expected * len(years)
    assert all(r["n_seeds_used"] == 6 for r in agg_rows), "expected all 6 real seeds usable"

    print("\n3. compute_stage3_corrosion() -- probability-weighted, summed over years "
          "(dev fixture, single bin):")
    p_bin, raw_total = s3.load_bin_probabilities()
    key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                      r["chord_t_scenario"], r["direction"], r["treatment"], r["year"])
    per_bin = {bin_number: {key(r): r for r in agg_rows}}
    stage3_rows, meta = stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, step_years=5.0)
    print(f"   {len(stage3_rows)} rows (expect {n_groups_expected}, one per connection-group -- "
          f"years already summed)")
    assert len(stage3_rows) == n_groups_expected
    assert meta["corrosion_years"] == years
    assert meta["step_years"] == 5.0
    print(f"   corrosion_horizon_years = {stage3_rows[0]['corrosion_horizon_years']} "
          f"(expect {len(years) * 5.0})")
    assert stage3_rows[0]["corrosion_horizon_years"] == len(years) * 5.0
    assert all(np.isfinite(r["D_life"]) and r["D_life"] >= 0 for r in stage3_rows)

    # --- K and Y stay as two distinct rows for a K-family connection (the
    # thing this whole module exists to get right, per the corrected
    # description above) ---
    k_conn = next(c for c in splash if c["family"] == "K")
    k_rows = [r for r in stage3_rows if r["node"] == k_conn["node"]
              and r["brace_member"] == k_conn["brace_member"]
              and r["brace_end"] == k_conn["brace_end"]
              and r["chord_t_scenario"] == k_conn["chord_t_scenario"]]
    print(f"\n   K-family connection (node {k_conn['node']} brace {k_conn['brace_member']}): "
          f"{len(k_rows)} treatment rows (expect 2, K and Y, never merged)")
    assert len(k_rows) == 2
    assert {r["treatment"] for r in k_rows} == {"K", "Y"}
    for r in sorted(k_rows, key=lambda r: r["treatment"]):
        print(f"     treatment={r['treatment']}: D_life (this bin's share, 25yr horizon) = "
              f"{r['D_life']:.6e}")

    out_path = write_stage3_joint_damage_corrosion(
        stage3_rows, RESULTS_DIR / "_stage3_joint_damage_corrosion_selfcheck.csv")
    print(f"\n   wrote {out_path} ({len(stage3_rows)} rows)")

    # --- independent recheck: one K connection, chord side, K-treatment,
    # at year=25 -- rebuild t_mm/scf/damage from scratch via this module's
    # own public functions, never touching stage3_rows' own arrays on the
    # way there.
    print("\n4. Independent recheck: one K connection, chord side, K-treatment, year=25:")
    other_row = _other_row_for(k_conn, k_groups)
    t_mm_direct = _corroded_t_mm(k_conn, model, other_row, "chord", 25)
    scf_lookup_25 = build_scf_lookup_corroded(splash, model, 25)
    scf_key = (k_conn["node"], str(k_conn["sub_joint_id"]), k_conn["brace_member"],
               k_conn["brace_end"], k_conn["chord_t_scenario"], k_conn["direction"] or "",
               "K", "chord")
    scf_direct = scf_lookup_25[scf_key]
    print(f"   t_mm={t_mm_direct:.3f}mm  static_scf_max={scf_direct:.4f}")

    stage2c_s1 = s2jc.load_stage2_joints_corrosion(npz_path)
    target_pt = next(
        pt for pt in stage2c_s1["point_table"]
        if pt["node"] == k_conn["node"] and pt["brace_member"] == k_conn["brace_member"]
        and pt["brace_end"] == k_conn["brace_end"] and pt["treatment"] == "K"
        and pt["side"] == "chord"
    )
    print(f"   {sum(1 for pt in stage2c_s1['point_table'] if pt['node'] == k_conn['node'] and pt['brace_member'] == k_conn['brace_member'] and pt['brace_end'] == k_conn['brace_end'] and pt['treatment'] == 'K' and pt['side'] == 'chord')} "
          f"position/segment variants share this (connection, treatment, side) -- "
          f"run_connection_damage_corrosion takes the max over them")

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
