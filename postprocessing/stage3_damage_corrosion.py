"""
Step C4/C5 -- Corrosion Stage 3 damage and lifetime (member track).

Turns stage2_corrosion.py's per-run, per-year cycle histograms into a
per-(point, corrosion-year-step) fatigue-life table, then combines the
year-steps by Miner's rule into one total-life number per point -- the
direct analogue of stage3_damage.py, with two real differences from the
uncorroded version:

1. THICKNESS CORRECTION USES THE CORRODED THICKNESS, NOT THE STORED t0.
   stage2_corrosion.py's point table only carries D0/t0 (the uncorroded,
   year-0 section) for reference -- the S-N curve's (t/t_ref)^k thickness
   factor must use the ACTUAL corroded thickness at each step, so this
   module recomputes it via sd_geometry.corroded_section(model, mid, year)
   for every (point, year) pair. Cheap (no .outb I/O, campaign-constant
   geometry), same "build once, reuse" pattern as the joint track's
   build_scf_lookup().

2. EACH 5-YEAR STEP IS WEIGHTED BY 5 YEARS OF LIFE, NOT 25.
   stage3_damage.py's D_life formula scales one ~600s block's mean damage
   up by BLOCKS_PER_LIFE (the full 25-year design life), because that
   damage rate is assumed constant for the whole life. Here it is NOT
   constant -- each year-step has its own (higher, as corrosion progresses)
   damage rate, valid for only the 5 years that step represents. So each
   step is scaled by BLOCKS_PER_STEP = step_years * BLOCKS_PER_YEAR, and
   the total life's damage is the SUM of all steps' shares -- plain Miner
   additivity, same principle stage3_damage.py already uses to sum across
   load-case bins, just applied across time instead of across bins.

Order of operations, extending stage3_damage.py's own pinned order:
    1. damage per (run, point, year, theta) -- point_theta_damage(), reused
       unmodified from stage3_damage.py, just called with the corroded t_mm.
    2. MAX OVER THETA, per (point, year) -- same rule, same reason.
    3. MEAN OVER SEEDS OF THE DAMAGE, per (point, year).
    4. Weighted sum over bins, PER YEAR-STEP (weight = p_bin * BLOCKS_PER_STEP,
       not BLOCKS_PER_LIFE) -- this is the "5/25 of life" step.
    5. SUM OVER YEAR-STEPS -- total D_life over the full corrosion horizon
       (len(years) * step_years years, e.g. 25 or 50).

Every splash-zone point's environment is 'free_corrosion' by construction
(stage2_corrosion.py only ever processes splash members) -- no zone lookup
needed here, unlike stage3_damage.py's run_point_damage().

Steps:
    1. run_point_damage_corrosion(stage2c, model, category) -- Miner damage
       per (point, year), one run, max over theta.
    2. aggregate_condition_corrosion(cond_dir, model, category) -- every
       seed in one condition folder, mean/std/min/max/n over seeds, grouped
       by (point_id, year).
    3. build_per_bin_corrosion(stage2_root, model, category) -- every
       condition folder found.
    4. compute_stage3_corrosion(stage2_root, model, bins_table, category,
       step_years) -- probability-weighted per year-step, summed over
       years, scaled to the total corrosion horizon.
    5. write_stage3_damage_corrosion(rows, out_path) -- the deliverable CSV.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_geometry as sdg
import sn_curves as sn
import stage2_corrosion as s2c
import stage3_damage as s3  # reuse: point_theta_damage, load_bin_probabilities,
                             # bin_number_from_case_id, _seed_usable, BLOCKS_PER_YEAR

# region --- paths (mirrors stage3_damage.py) ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

BINS_CSV = s3.BINS_CSV
CATEGORY = "B1"  # member track always assesses against the B1 curve
# endregion


# region --- one run, every (point, year) ---
def run_point_damage_corrosion(stage2c, model, category=CATEGORY):
    """
    stage2c: stage2_corrosion.load_stage2_corrosion() output for one run.
    Returns one dict per (point, year): point/year metadata + damage_block
    (Miner damage for this ~600s block, MAX OVER THETA, at that year's
    corroded section) + worst_theta_deg.
    """
    assert sn.SN_CONSTANTS_VERIFIED, (
        "sn_curves.SN_CONSTANTS_VERIFIED is False -- corrosion Stage 3 "
        "refuses to produce damage numbers until the S-N constants are "
        "signed off, same gate as stage3_damage.py."
    )
    bin_edges = stage2c["bin_edges_mpa"]
    theta_deg = np.degrees(stage2c["theta_rad"])
    years = stage2c["years"]
    environment = "free_corrosion"  # every point here is splash-zone, by construction

    rows = []
    for pt in stage2c["point_table"]:
        pid = pt["point_id"]
        mid = pt["member_id"]
        for y_idx, year in enumerate(years):
            D_corr, t_corr = sdg.corroded_section(model, mid, year)
            t_mm = t_corr * 1000.0
            sum_r_point_year = {m: stage2c["sum_r"][m][pid, y_idx] for m in stage2c["sum_r"]}
            d_theta = s3.point_theta_damage(sum_r_point_year, bin_edges, category, environment, t_mm)
            k_worst = int(np.argmax(d_theta))
            rows.append(dict(
                point_id=pid, member_id=mid, end=pt["end"], member_class=pt["member_class"],
                year=int(year), D_corroded=D_corr, t_corroded=t_corr,
                z=pt["z"], propset=pt["propset"], environment=environment, category=category,
                damage_block=float(d_theta[k_worst]), worst_theta_deg=float(theta_deg[k_worst]),
            ))
    return rows
# endregion


# region --- one condition folder, every seed ---
def aggregate_condition_corrosion(cond_dir, model, category=CATEGORY):
    """
    cond_dir: stage2_corrosion/<COND>/ containing one .npz (+ .json sidecar)
    per seed. Returns (bin_number, [row, ...]) -- one row per (point, year),
    with seed mean/std/min/max/n of damage_block.
    """
    npz_paths = sorted(Path(cond_dir).glob("*.npz"))
    assert npz_paths, f"no seeds found in {cond_dir}"

    per_seed_damage = []   # [{(point_id, year): damage_block}, ...], usable seeds only
    per_seed_theta = []    # [{(point_id, year): worst_theta_deg}, ...], usable seeds only
    bin_numbers = set()
    meta_by_key = {}
    n_excluded = 0

    for npz_path in npz_paths:
        stage2c = s2c.load_stage2_corrosion(npz_path)
        stamp = stage2c["stamp"]
        if not s3._seed_usable(stamp, stamp["transient_cutoff_s"]):
            n_excluded += 1
            print(f"    excluding {npz_path.name}: duration too short after transient trim")
            continue
        case_id = stamp["case_json"]["case_id"]
        bin_numbers.add(s3.bin_number_from_case_id(case_id))
        pt_rows = run_point_damage_corrosion(stage2c, model, category=category)
        key = lambda r: (r["point_id"], r["year"])
        per_seed_damage.append({key(r): r["damage_block"] for r in pt_rows})
        per_seed_theta.append({key(r): r["worst_theta_deg"] for r in pt_rows})
        for r in pt_rows:
            meta_by_key.setdefault(key(r), r)

    assert per_seed_damage, f"{cond_dir}: every seed excluded, nothing usable"
    assert len(bin_numbers) == 1, f"{cond_dir}: seeds disagree on bin number: {bin_numbers}"
    bin_number = bin_numbers.pop()
    n_seeds = len(per_seed_damage)

    out_rows = []
    for key, meta in sorted(meta_by_key.items()):
        vals = np.array([seed[key] for seed in per_seed_damage])
        i_dominant = int(np.argmax(vals))
        theta_rep = per_seed_theta[i_dominant][key]
        r = {k: v for k, v in meta.items() if k not in ("damage_block", "worst_theta_deg")}
        r.update(
            bin_number=bin_number,
            damage_block_mean=float(vals.mean()),
            damage_block_std=float(vals.std(ddof=0)),
            damage_block_min=float(vals.min()),
            damage_block_max=float(vals.max()),
            n_seeds_used=n_seeds,
            n_seeds_excluded=n_excluded,
            worst_theta_deg=float(theta_rep),
        )
        out_rows.append(r)
    return bin_number, out_rows
# endregion


# region --- every available condition folder, probability-weighted, summed over years ---
def build_per_bin_corrosion(stage2_root, model, category=CATEGORY):
    """Discovers every condition folder under stage2_root, aggregates seeds
    per bin. Returns per_bin: {bin_number: {(point_id, year): row}}."""
    cond_dirs = sorted(p for p in Path(stage2_root).iterdir() if p.is_dir())
    assert cond_dirs, f"no condition folders found under {stage2_root}"

    per_bin = {}
    for i, cond_dir in enumerate(cond_dirs, 1):
        print(f"  [{i}/{len(cond_dirs)}] aggregating {cond_dir.name} ...")
        bin_number, rows = aggregate_condition_corrosion(cond_dir, model, category=category)
        assert bin_number not in per_bin, f"bin {bin_number} appears in more than one condition folder"
        per_bin[bin_number] = {(r["point_id"], r["year"]): r for r in rows}
    return per_bin


def compute_stage3_corrosion(stage2_root, sd_path, bins_table=BINS_CSV,
                              category=CATEGORY, step_years=5.0):
    """
    Discovers every condition folder under stage2_root, aggregates seeds per
    bin, then for EACH year-step combines bins by probability (weight =
    p_bin * BLOCKS_PER_STEP, that step's own 5-year life share), and finally
    SUMS across year-steps for each point's total D_life over the full
    corrosion horizon.

    Returns (rows, meta) -- meta reports the same "partial-campaign honesty"
    fields as stage3_damage.py, plus corrosion_horizon_years.
    """
    p_bin, raw_total = s3.load_bin_probabilities(bins_table)
    model = sdg.read_subdyn_model(sd_path)
    per_bin = build_per_bin_corrosion(stage2_root, model, category=category)
    return stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, category, step_years)


def stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, category=CATEGORY, step_years=5.0):
    """Core of compute_stage3_corrosion(), operating on an already-built
    per_bin (see build_per_bin_corrosion())."""
    BLOCKS_PER_STEP = step_years * s3.BLOCKS_PER_YEAR

    all_keys = sorted({k for rows in per_bin.values() for k in rows})  # (point_id, year)
    invariant_keys = ("member_id", "end", "member_class", "D_corroded", "t_corroded",
                       "z", "propset", "environment", "category")

    # --- step 4: per (point, year), probability-weight across bins, scaled
    # to this step's own step_years share of life ---
    step_d = {}       # (point_id, year) -> weighted damage share (this step's years)
    step_meta = {}     # (point_id, year) -> a representative row (for invariant fields)
    step_worst_theta = {}
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
                        f"point/year {key}: {ikey} disagrees between bins "
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
                worst_detail_at_max = r["worst_theta_deg"]

        step_d[key] = d_step
        step_meta[key] = meta
        step_worst_theta[key] = worst_detail_at_max
        n_bins_used_by_key[key] = n_bins_used

    # --- step 5: sum over year-steps, per point ---
    all_points = sorted({pid for pid, _year in all_keys})
    years_by_point = {}
    for pid, year in all_keys:
        years_by_point.setdefault(pid, []).append(year)

    out_rows = []
    for pid in all_points:
        years = sorted(years_by_point[pid])
        d_life = sum(step_d[(pid, y)] for y in years)
        n_bins_used_total = sum(n_bins_used_by_key[(pid, y)] for y in years)

        # representative meta from the LAST (most-corroded) year -- the
        # thickness/geometry columns are year-specific, so report the
        # worst-case (final-year) section for reference, not year-0.
        meta_final = step_meta[(pid, years[-1])]
        worst_year = max(years, key=lambda y: step_d[(pid, y)])

        horizon_years = len(years) * step_years
        row = dict(
            point_id=pid, member_id=meta_final["member_id"], end=meta_final["end"],
            member_class=meta_final["member_class"],
            D_corroded_final=meta_final["D_corroded"], t_corroded_final=meta_final["t_corroded"],
            z=meta_final["z"], propset=meta_final["propset"],
            environment=meta_final["environment"], category=category,
            corrosion_horizon_years=horizon_years,
            D_life=d_life,
            life_years=(horizon_years / d_life) if d_life > 0 else float("inf"),
            worst_year=worst_year,
            worst_theta_deg=step_worst_theta[(pid, worst_year)],
            n_bins_used=n_bins_used_total,
        )
        out_rows.append(row)

    meta = dict(
        p_bin_raw_sum=raw_total,
        n_bins_available=len(per_bin),
        bins_available=sorted(per_bin),
        n_bins_total_campaign=len(p_bin),
        corrosion_years=sorted({y for _pid, y in all_keys}),
        step_years=step_years,
    )
    return out_rows, meta
# endregion


# region --- member x load-case matrix, ONE per corrosion year-step (overview, not a lifetime estimate) ---
def matrix_from_per_bin_corrosion(per_bin, p_bin, bin_names, year, step_years=5.0):
    """
    Direct analogue of stage3_damage.matrix_from_per_bin(), for ONE
    corrosion year-step in isolation -- e.g. the year=20 matrix shows only
    that 5-year block's (15->20yr) own damage contribution, NOT the
    cumulative total through year 20 (confirmed by the author: per-step
    only, not cumulative). Scoped to the 32 splash-zone members (not 112,
    since stage2_corrosion.py never touches non-splash members), max across
    the 2 ends per member (same "worst end" convention as the uncorroded
    matrix).

    raw      -- damage_block_mean itself, for THIS year's corroded section,
                THAT bin alone -- "which conditions are physically harshest
                at this corrosion stage."
    weighted -- p_bin * BLOCKS_PER_STEP * damage_block_mean, i.e. that
                bin's actual share of THIS STEP's damage (not the full
                horizon) -- rows sum to the point's step_d for that year
                once every campaign bin is present, same partial-campaign
                honesty as the uncorroded matrix.

    Returns (raw_rows, weighted_rows, bin_columns), same shape convention as
    stage3_damage.matrix_from_per_bin().
    """
    BLOCKS_PER_STEP = step_years * s3.BLOCKS_PER_YEAR

    member_bin_ends = {}
    class_by_member = {}
    for bin_number, rows_by_key in per_bin.items():
        for (_point_id, yr), r in rows_by_key.items():
            if yr != year:
                continue
            member_bin_ends.setdefault((r["member_id"], bin_number), []).append(r)
            class_by_member.setdefault(r["member_id"], r["member_class"])

    all_members = sorted(class_by_member)
    bin_columns = [bin_names[b] for b in sorted(per_bin)]

    raw_rows, weighted_rows = [], []
    for mid in all_members:
        raw_row = dict(member_id=mid, member_class=class_by_member[mid])
        weighted_row = dict(member_id=mid, member_class=class_by_member[mid])
        worst_col_raw, worst_val_raw = None, -np.inf
        worst_col_w, worst_val_w = None, -np.inf

        for bin_number in sorted(per_bin):
            col = bin_names[bin_number]
            ends = member_bin_ends.get((mid, bin_number))
            if ends is None:
                raw_row[col] = None
                weighted_row[col] = None
                continue

            raw_val = max(r["damage_block_mean"] for r in ends)
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


def write_member_matrix_csv_corrosion(rows, bin_columns, worst_col_name, out_path):
    """Same fixed column convention as stage3_damage.write_member_matrix_csv,
    with member_class instead of zone (every row here is splash-zone, by
    construction -- zone would be a constant column, member_class is the
    informative one for this track)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["member_id", "member_class"] + bin_columns + ["worst_bin", worst_col_name]
    df = pd.DataFrame(rows)[cols].sort_values("member_id").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def write_year_step_matrices(per_bin, p_bin, bin_names, years, step_years=5.0,
                              results_dir=RESULTS_DIR):
    """
    Convenience driver: writes 2 CSVs (raw, weighted) per year in `years` --
    10 files total for the confirmed 5-step/25yr horizon. File names:
    member_damage_matrix_corrosion_raw_y<year>.csv /
    member_damage_matrix_corrosion_weighted_y<year>.csv. Returns a dict
    {year: (raw_path, weighted_path)}.
    """
    out_paths = {}
    for year in years:
        raw_rows, weighted_rows, bin_columns = matrix_from_per_bin_corrosion(
            per_bin, p_bin, bin_names, year, step_years=step_years
        )
        raw_path = results_dir / f"member_damage_matrix_corrosion_raw_y{year}.csv"
        weighted_path = results_dir / f"member_damage_matrix_corrosion_weighted_y{year}.csv"
        write_member_matrix_csv_corrosion(raw_rows, bin_columns, "worst_bin_damage", raw_path)
        write_member_matrix_csv_corrosion(weighted_rows, bin_columns, "worst_bin_contribution", weighted_path)
        out_paths[year] = (raw_path, weighted_path)
    return out_paths
# endregion


# region --- output ---
def write_stage3_damage_corrosion(rows, out_path=RESULTS_DIR / "stage3_damage_corrosion.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["member_id", "end"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df
# endregion


def _self_check():
    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    sd_path = case_dir / s2c.SD_NAME
    assert case_dir.exists(), f"missing fixture: {case_dir}"

    print(f"SN_CONSTANTS_VERIFIED = {sn.SN_CONSTANTS_VERIFIED}\n")

    years = [5, 10, 15, 20, 25]
    stage2c_dir = RESULTS_DIR / "_stage2_corrosion_selfcheck"
    cond_dir = stage2c_dir / "LC_V20_H3p5_T8"
    # process_run_corroded() has its own skip-if-stamp-matches check, so
    # calling it for all 6 seeds is cheap for whichever ones are already
    # cached (e.g. S100001 from stage2_corrosion.py's own self-check) --
    # NOT a "does anything exist" check, which would wrongly skip building
    # the other 5 seeds if only one happened to be present already.
    seed_dirs = sorted(p for p in (DEV_FIXTURE_DIR / "LC_V20_H3p5_T8").iterdir() if p.is_dir())
    print(f"  ensuring corrosion Stage 2 cache exists for all {len(seed_dirs)} seeds "
          f"(skips any already up to date)...")
    for seed_dir in seed_dirs:
        s2c.process_run_corroded(seed_dir, years, out_root=stage2c_dir)

    model = sdg.read_subdyn_model(sd_path)

    print("\n1. run_point_damage_corrosion() on ONE seed:")
    npz_path = sorted(cond_dir.glob("*.npz"))[0]
    stage2c = s2c.load_stage2_corrosion(npz_path)
    rows = run_point_damage_corrosion(stage2c, model)
    print(f"   {len(rows)} rows (expect 64 points x {len(years)} years = {64 * len(years)})")
    assert len(rows) == 64 * len(years)

    # sanity: damage should increase with year, for a fixed point -- same
    # check as stage2_corrosion.py's self-check, now at the DAMAGE level
    # (post-Miner, post-thickness-correction) rather than the raw power sum.
    pid0_rows = sorted((r for r in rows if r["point_id"] == 0), key=lambda r: r["year"])
    d_by_year = [r["damage_block"] for r in pid0_rows]
    print(f"   point 0 damage vs year: {[f'{d:.3e}' for d in d_by_year]}")
    assert all(d_by_year[i] <= d_by_year[i + 1] for i in range(len(d_by_year) - 1)), \
        "damage should be monotonically non-decreasing with corrosion year"

    print("\n2. aggregate_condition_corrosion() -- all 6 seeds:")
    bin_number, agg_rows = aggregate_condition_corrosion(cond_dir, model)
    print(f"   bin_number={bin_number} (expect 65), {len(agg_rows)} rows (expect {64 * len(years)})")
    assert bin_number == 65
    assert len(agg_rows) == 64 * len(years)
    assert all(r["n_seeds_used"] == 6 for r in agg_rows), "expected all 6 real seeds usable"

    print("\n3. compute_stage3_corrosion() -- probability-weighted, summed over years "
          "(dev fixture, single bin):")
    p_bin, raw_total = s3.load_bin_probabilities()
    per_bin = {bin_number: {(r["point_id"], r["year"]): r for r in agg_rows}}
    stage3_rows, meta = stage3_from_per_bin_corrosion(per_bin, p_bin, raw_total, step_years=5.0)
    print(f"   {len(stage3_rows)} rows (expect 64, one per point -- years already summed)")
    assert len(stage3_rows) == 64
    assert meta["corrosion_years"] == years
    assert meta["step_years"] == 5.0
    print(f"   corrosion_horizon_years = {stage3_rows[0]['corrosion_horizon_years']} "
          f"(expect {len(years) * 5.0})")
    assert stage3_rows[0]["corrosion_horizon_years"] == len(years) * 5.0
    assert all(np.isfinite(r["D_life"]) and r["D_life"] >= 0 for r in stage3_rows)

    # Cross-check against the UNCORRODED stage3_damage.py result on the same
    # bin/points, same dev fixture -- corrosion damage must be >= the
    # uncorroded damage's SHARE for the same horizon (corrosion can only add
    # damage, never remove it, at fixed load). Compare like-for-like: scale
    # the uncorroded single-bin D_life (which represents a 25yr share) down
    # to this corrosion run's horizon for a fair ratio.
    real_stage2_dir = DEV_FIXTURE_DIR / "stage2" / "LC_V20_H3p5_T8"
    if real_stage2_dir.exists():
        print("\n4. Cross-check vs. UNCORRODED stage3_damage.py (same bin, same points):")
        _bn, unc_rows = s3.aggregate_condition(real_stage2_dir, category="B1")
        # Match by (member_id, end), NOT point_id -- stage2_corrosion.py
        # re-indexes point_id 0..63 over just the 32 splash members, while
        # stage2_histograms.py indexes point_id 0..223 over all 112 members.
        # The same point_id number means a DIFFERENT physical point in each
        # file; (member_id, end) is the only key genuinely shared by both.
        unc_by_member_end = {(r["member_id"], r["end"]): r for r in unc_rows}
        horizon = stage3_rows[0]["corrosion_horizon_years"]
        ratios = []
        for r in stage3_rows:
            unc = unc_by_member_end.get((r["member_id"], r["end"]))
            if unc is None or unc["damage_block_mean"] <= 0:
                continue
            # uncorroded per-block damage rate, scaled to the SAME horizon
            # (not stage3_damage.py's own 25yr D_life, to compare fairly
            # against a possibly-different corrosion horizon):
            weight_horizon = p_bin[bin_number] * horizon * s3.BLOCKS_PER_YEAR
            unc_d_life_at_horizon = weight_horizon * unc["damage_block_mean"]
            if unc_d_life_at_horizon > 0:
                ratios.append(r["D_life"] / unc_d_life_at_horizon)
        print(f"   corroded/uncorroded D_life ratio at this bin, same {horizon:.0f}yr horizon: "
              f"min={min(ratios):.3f}  median={np.median(ratios):.3f}  max={max(ratios):.3f}")
        assert min(ratios) >= 1.0 - 1e-9, (
            "corrosion should never REDUCE damage relative to the uncorroded case "
            "at the same load and horizon"
        )
        assert np.median(ratios) > 1.0, "expected corrosion to measurably increase damage on average"

    ranked = sorted(stage3_rows, key=lambda r: -r["D_life"])
    print("\n   top 5 most-damaged points (this bin's probability share, "
          f"{stage3_rows[0]['corrosion_horizon_years']:.0f}yr horizon):")
    for r in ranked[:5]:
        print(f"     member {r['member_id']:>3} end {r['end']} class={r['member_class']:<6} "
              f"worst_year={r['worst_year']:>2}  D_life={r['D_life']:.3e}")

    out_path = write_stage3_damage_corrosion(stage3_rows, RESULTS_DIR / "_stage3_damage_corrosion_selfcheck.csv")
    print(f"\n   wrote {out_path} ({len(stage3_rows)} rows)")

    print("\n5. Per-year-step member x load-case matrices (dev fixture, single bin):")
    bin_names = s3.load_bin_names()
    selfcheck_results_dir = RESULTS_DIR / "_matrix_corrosion_selfcheck"
    out_paths = write_year_step_matrices(per_bin, p_bin, bin_names, years, step_years=5.0,
                                          results_dir=selfcheck_results_dir)
    for year, (raw_path, weighted_path) in out_paths.items():
        raw_df = pd.read_csv(raw_path)
        weighted_df = pd.read_csv(weighted_path)
        print(f"   year {year}: {raw_df.shape[0]} member(s) -> {raw_path.name}, {weighted_path.name}")
        assert raw_df.shape[0] == 32, f"expected 32 splash members, got {raw_df.shape[0]}"
        assert set(raw_df["member_class"]) <= {"leg", "brace"}
    assert len(out_paths) == len(years) == 5

    # Cross-check: a year's matrix weighted-row sum for a member (over the
    # single available bin) must equal that member's own step_d contribution
    # in stage3_rows -- i.e. matrix_from_per_bin_corrosion's per-step weight
    # (BLOCKS_PER_STEP) matches stage3_from_per_bin_corrosion's internal
    # step_d exactly, not just structurally similar code.
    year_check = years[2]  # year 15
    _raw_rows_chk, weighted_rows_chk, bin_columns_chk = matrix_from_per_bin_corrosion(
        per_bin, p_bin, bin_names, year_check, step_years=5.0
    )
    assert len(bin_columns_chk) == 1, "dev fixture has exactly 1 bin"
    col = bin_columns_chk[0]
    weighted_by_member = {r["member_id"]: r[col] for r in weighted_rows_chk}
    # rebuild the same step from stage3's own per-point step_d for a direct match
    step_d_check = {}
    for bin_number, rows_by_key in per_bin.items():
        for (pid, yr), r in rows_by_key.items():
            if yr != year_check:
                continue
            weight = p_bin[bin_number] * 5.0 * s3.BLOCKS_PER_YEAR
            contrib = weight * r["damage_block_mean"]
            key = r["member_id"]
            step_d_check[key] = max(step_d_check.get(key, -np.inf), contrib)  # max over 2 ends
    n_checked = 0
    for mid, matrix_val in weighted_by_member.items():
        if matrix_val is None:
            continue
        rel_diff = abs(matrix_val - step_d_check[mid]) / step_d_check[mid] if step_d_check[mid] > 0 else 0.0
        assert rel_diff < 1e-9, f"member {mid} year {year_check}: matrix={matrix_val} vs direct={step_d_check[mid]}"
        n_checked += 1
    print(f"   cross-check ok: {n_checked} members' year-{year_check} weighted matrix value "
          f"matches an independent max-over-ends recompute exactly")

    print("\n" + "=" * 78)
    print("Reminder: dev-fixture-only, single bin. D_life reflects ONLY this bin's")
    print("probability share over the corrosion horizon above -- not a lifetime claim")
    print("until the real campaign is processed (Step H-equivalent for corrosion).")
    print("=" * 78)


if __name__ == "__main__":
    _self_check()
