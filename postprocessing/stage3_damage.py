"""
Step 9 -- Stage 3 damage and lifetime.

Turns Stage 2's per-run cycle histograms (stage2_histograms.py) into a
per-member fatigue-life table: Miner's rule against the signed-off
DNV-RP-C203 curves (sn_curves.py), seed-averaged, probability-weighted
across load-case bins, and scaled to a 25-year design life. This is the
priority deliverable of the whole pipeline (see fatigue-postpro-design
memory) -- per-member damage broken down by load case.

GATED on sn_curves.SN_CONSTANTS_VERIFIED: this module refuses to produce a
damage number while that flag is False (see "Values you verify, not me"
in the build plan -- the sign-off discipline is enforced here, not just
documented).

Order of operations -- this is where damage/lifetime calculations are most
likely to be silently wrong, so it is pinned explicitly and tested:
    1. damage per (run, point, theta), for one ~600s block, using the
       stored power sums -- exact within a bin, see point_theta_damage().
    2. MAX OVER THETA -- at the very end, on the DAMAGES, never on the
       stress and never per-timestep (that would be enveloping, see
       fatigue-postpro-docs/decisions.md "Enveloping is NOT conservative").
    3. MEAN OVER THE 6 SEEDS OF THE DAMAGE, not of the stress -- damage is
       a cubic-ish (m=3..5) function of stress, so averaging stress first
       and computing damage once would systematically underestimate
       (Jensen's inequality). Keep seed std as an uncertainty band.
    4. Weighted sum over however many bins currently have Stage 2 data,
       joined on the bin number parsed from case_id (LC65_S3 -> bin 65),
       using OC4_Final_Bins.xlsx's own Probability column AS-IS (raw sheet
       sums to 1.00046, not renormalized -- see load_bin_probabilities()'s
       docstring for why that 0.046% is not worth correcting).
       Re-implemented standalone here -- NOT by importing
       Simulation/config.py, which is under docs/decisions.md and
       has import-time side effects (detect_laptop(), reading the same
       Excel at import time).
    5. Scale by BLOCKS_PER_LIFE = 25yr * 365.25 * 24 * 6 ten-minute blocks
       = 1,314,900 (confirmed exact, asserted below).

**Partial-campcampaign honesty**: while the 414-run campaign is still
running, D_life only reflects the AVAILABLE bins' probability-weighted
share, not a claim about the structure's real remaining life. That only
becomes meaningful once every bin has Stage 2 data. The per-point RANKING
and the seed-scatter band are meaningful now, on however many bins exist.

**The exponent-set bug found while building this step**: fatigue_config.py
originally set WOHLER_EXPONENTS = (3, 5), assumed uniform across every S-N
curve. Building point_theta_damage() below required reading sn_curves.py's
actual SN_CURVES table to know which m1/m2 values a given (category,
environment) pair needs -- and B1's air/seawater_cp curves (which cover
every atmospheric AND submerged-zone member, 80 of 112) use m1=4.0, not
3.0. Only B1's free-corrosion curve and all of T's curves use m1=3. That
means the ORIGINAL (3, 5) exponent set would have silently forced 80
members' high-stress-branch damage through the wrong stored power sum (an
m=3 sum used where DNV's own table says m=4) -- a real, invisible bug: the
output would have looked like a perfectly plausible damage number, just
wrong by whatever the m=3-vs-m=4 mismatch happens to be at that member's
actual stress level. Caught by cross-referencing WOHLER_EXPONENTS against
sn_curves.py's own table before writing this module's core function, not
by inspection. Fixed: WOHLER_EXPONENTS = (3, 4, 5) (fatigue_config.py),
PIPELINE_VERSION bumped 1->2 so any stale (3, 5)-only Stage 2 file is
rejected by stage2_histograms.py's own stamp check, not silently reused.
point_theta_damage() below also asserts loudly if a curve ever needs an
exponent Stage 2 didn't store, rather than producing a wrong number.

**Section-loss corrosion is explicitly OUT of this step's default output.**
The 5-year-step thin-plate-scalar method is fully designed (see
docs/decisions.md) and its core arithmetic
IS built and tested here (corrosion_multiplier(), apply_corrosion_scalar())
-- the self-check proves c=1.10 gives exactly x1.331 at m=3, one of this
step's own listed verification criteria. But it is NOT wired into
compute_stage3()'s output, for two independent reasons: (a) the method is
explicitly flagged TO-VERIFY -- it remains open
before treating it as final, not a formality; (b) it requires identifying
which members are LEGS (for the doubled flooded allowance), and no
function in sd_geometry.py currently distinguishes legs from braces by
geometry -- inventing that classification here, unreviewed, would bury a
second unverified assumption inside an already-unverified one. Once both
are resolved, apply_corrosion_scalar() plugs directly into
point_theta_damage()'s sum_r arrays with no other change.

Steps:
    1. bin_number_from_case_id(case_id) / load_bin_probabilities(table_path)
       -- the probability-weighting join.
    2. point_theta_damage(sum_r, bin_edges_mpa, category, environment,
       t_mm, scf) -- Miner damage per theta for ONE point, ONE run.
    3. run_point_damage(stage2, category) -- every point in one run, max
       over theta.
    4. aggregate_condition(cond_dir, category) -- every seed in one
       condition folder, mean/std/min/max/n over seeds.
    5. compute_stage3(stage2_root, bins_table, category) -- every condition
       folder found, probability-weighted, scaled to 25 years.
    6. corrosion_multiplier(t0_mm, delta_t_mm) / apply_corrosion_scalar(...)
       -- tested utilities, not wired into the default output (see above).
    7. write_stage3_damage(rows, out_path) -- the deliverable CSV.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

import sn_curves as sn
import stage2_histograms as s2

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# Dev/test fixture data (LC_V20_H3p5_T8, _staging/) stays at its original
# location, not moved alongside the code -- see fatigue_pipeline_build
# memory, 10.08.2026 folder-reorg session. Its own Stage 2 cache
# (LC_V20_H3p5_T8's 6 seeds) also stayed there when the code moved.
DEV_FIXTURE_DIR = PROJECT / "data" / "example"
DEV_FIXTURE_STAGE2_DIR = DEV_FIXTURE_DIR / "stage2"

# Same D:-drive convention as stage2_histograms.py (large, regenerable,
# deliberately kept off the cloud-synced tree).
STAGE2_DIR = s2.STAGE2_DIR
BINS_CSV = PROJECT / "data" / "oc4_k13_bins.csv"
# endregion


# region --- lifetime constants ---
DESIGN_LIFE_YEARS = 25.0
BLOCK_SECONDS = 600.0  # usable seconds per run: 700s campaign run - 100s transient
BLOCKS_PER_YEAR = 365.25 * 24 * 6.0  # six 10-minute blocks per hour
BLOCKS_PER_LIFE = DESIGN_LIFE_YEARS * BLOCKS_PER_YEAR
assert BLOCKS_PER_LIFE == 1_314_900.0, "block arithmetic drifted -- see docs/decisions.md"
# endregion


# region --- bin number / probability join ---
CASE_ID_RE = re.compile(r"^LC(\d+)_S\d+$")


def bin_number_from_case_id(case_id):
    """'LC65_S3' -> 65. Fails loudly on anything that doesn't match this
    real-campaign naming (e.g. old test data's 'TS09') -- a run whose bin
    number can't be determined must never be silently skipped or silently
    assigned bin 0."""
    m = CASE_ID_RE.match(case_id)
    assert m, f"case_id {case_id!r} doesn't match the real-campaign LC<nn>_S<n> pattern"
    return int(m.group(1))


def load_bin_probabilities(table_path=BINS_CSV):
    """
    {bin_number (1-based, matching case_id's LC<nn>): p_bin} -- the sheet's
    OWN Probability column, used as-is. The raw sheet sums to 1.00046, not
    1.0 exactly (see docs/decisions.md) -- a 0.046% correction,
    decided NOT worth renormalizing away: it is two orders of magnitude
    below the seed-to-seed scatter this pipeline already reports
    (~45% coefficient of variation on real data, see stage3_damage.py's own
    self-check), so dividing every value by 1.00046 would add a line of
    math without changing any conclusion. Returns (probabilities_dict,
    raw_sum) so callers can still report/inspect the raw total.
    """
    df = pd.read_csv(table_path)
    p_bin = {idx: float(p) for idx, p in enumerate(df["Probability"], start=1)}
    raw_sum = sum(p_bin.values())
    return p_bin, raw_sum


def load_bin_names(table_path=BINS_CSV):
    """{bin_number (1-based, matching load_bin_probabilities' keys): NAME column
    string, e.g. 'LC_V20_H3p5_T8'} -- same row-order convention as
    load_bin_probabilities(), so the two dicts always share keys."""
    df = pd.read_csv(table_path)
    return {idx: str(name) for idx, name in enumerate(df["NAME"], start=1)}
# endregion


# region --- per-bin S-N branch selection ---
def _bin_branch(bin_edges_mpa, category, environment, t_mm, scf=None):
    """
    Which S-N branch (m1/log_a1 vs m2/log_a2) applies to each of the 256
    global bins, decided ONCE per bin from its geometric-midpoint stress
    (log-spaced, ~4.7% wide -- see module docstring on why this is a small
    approximation, not a per-cycle one) rather than per cycle. Returns
    (row, factor, mask_hi) -- mask_hi True where the bin's effective stress
    is on the m1 (>= knee) branch; row["n_knee"] is None for free-corrosion
    curves, in which case mask_hi is all True (single slope, no branch).
    """
    row = sn.SN_CURVES[(category, environment)]
    k = sn.k_for_curve(row, scf)
    factor = sn.thickness_factor(t_mm, sn.T_REF_MM[category], k)
    mid = np.sqrt(bin_edges_mpa[:-1] * bin_edges_mpa[1:])
    ds_eff = mid * factor
    if row["n_knee"] is None:
        mask_hi = np.ones_like(mid, dtype=bool)
    else:
        ds_knee = sn.knee_stress(category, environment)
        mask_hi = ds_eff >= ds_knee
    return row, factor, mask_hi


def point_theta_damage(sum_r, bin_edges_mpa, category, environment, t_mm, scf=None):
    """
    sum_r: {m: array shape (n_theta, n_bins)} for ONE assessment point (one
    member end, one run). Returns Miner damage per theta, shape (n_theta,),
    for this one ~600s block -- EXACT within a bin (the whole reason Stage
    2 stores power sums instead of counts), given the per-bin branch
    selection above.

    Miner increment for a bin's cycles at effective range Delta_sigma_eff:
        n / N(Delta_sigma_eff) = n * Delta_sigma_eff^m / 10^log_a
    summed over a bin (constant m within that bin):
        sum(n_i * (Delta_sigma_i*factor)^m) / 10^log_a
      = factor^m * sum_r[m][bin] / 10^log_a
    """
    row, factor, mask_hi = _bin_branch(bin_edges_mpa, category, environment, t_mm, scf)
    m1, log_a1 = row["m1"], row["log_a1"]
    assert m1 in sum_r, (
        f"S-N curve ({category}, {environment}) needs m1={m1} but Stage 2 "
        f"only stored exponents {sorted(sum_r)} -- fatigue_config.WOHLER_EXPONENTS "
        f"is missing an exponent this curve needs; recompute Stage 2 after fixing it."
    )
    d = (factor ** m1 / 10.0 ** log_a1) * sum_r[m1][:, mask_hi].sum(axis=1)
    if row["n_knee"] is not None:
        m2, log_a2 = row["m2"], row["log_a2"]
        assert m2 in sum_r, (
            f"S-N curve ({category}, {environment}) needs m2={m2} but Stage 2 "
            f"only stored exponents {sorted(sum_r)} -- fatigue_config.WOHLER_EXPONENTS "
            f"is missing an exponent this curve needs; recompute Stage 2 after fixing it."
        )
        mask_lo = ~mask_hi
        d = d + (factor ** m2 / 10.0 ** log_a2) * sum_r[m2][:, mask_lo].sum(axis=1)
    return d
# endregion


# region --- one run, every point ---
def run_point_damage(stage2, category="B1"):
    """
    stage2: stage2_histograms.load_stage2() output for one run. Returns one
    dict per assessment point (ALL of them -- not_assessable points get a
    real number too, just flagged, per "all 112 members get a damage
    number eventually"): point metadata + damage_block (Miner damage for
    this ~600s block, MAX OVER THETA -- step 2 of the module docstring's
    pinned order) + worst_theta_deg.
    """
    assert sn.SN_CONSTANTS_VERIFIED, (
        "sn_curves.SN_CONSTANTS_VERIFIED is False -- Stage 3 refuses to "
        "produce damage numbers until the S-N constants are signed off "
        "(see the build plan's 'Values you verify, not me')."
    )
    bin_edges = stage2["bin_edges_mpa"]
    theta_deg = np.degrees(stage2["theta_rad"])
    rows = []
    for pt in stage2["point_table"]:
        pid = pt["point_id"]
        environment = sn.ZONE_TO_ENVIRONMENT[pt["zone"]]
        sum_r_point = {m: stage2["sum_r"][m][pid] for m in stage2["sum_r"]}
        t_mm = pt["t"] * 1000.0  # sd_geometry stores t in metres
        d_theta = point_theta_damage(sum_r_point, bin_edges, category, environment, t_mm)
        k_worst = int(np.argmax(d_theta))
        rows.append(dict(
            point_id=pid, member_id=pt["member_id"], end=pt["end"],
            D=pt["D"], t=pt["t"], z=pt["z"], zone=pt["zone"], propset=pt["propset"],
            environment=environment, category=category,
            not_assessable=pt["not_assessable"], not_assessable_reason=pt["not_assessable_reason"],
            damage_block=float(d_theta[k_worst]), worst_theta_deg=float(theta_deg[k_worst]),
        ))
    return rows
# endregion


# region --- seed usability / exclusion ---
def _seed_usable(stamp, transient_cutoff_s):
    """
    True unless the run's own recorded duration leaves no usable signal
    after the transient trim -- e.g. the known-crashed QA fixture
    (LC_V23_H4p0_T10/S654321, a 0.77s tower-strike abort). Its .outb header
    is structurally self-consistent (Stage 0's finding), so a size/format
    check alone would NOT catch it; must check duration against the
    transient cutoff, same pattern as bin_range_check.py's scan_run().
    Excluding here (not upstream) means Stage 3 stays safe even before a
    batch driver (Step 10) exists to filter crashed runs out beforehand --
    the plan's own Step 9 verify criterion: "the crashed run must be
    excluded, not contributing 0.75s of damage multiplied by 1.3 million
    blocks."
    """
    duration_s = (stamp["n_t"] - 1) * stamp["dt_s"]
    return duration_s > transient_cutoff_s
# endregion


# region --- one condition folder, every seed ---
def aggregate_condition(cond_dir, category="B1"):
    """
    cond_dir: stage2/<COND>/ containing one .npz (+ .json sidecar) per seed.
    Returns (bin_number, [row, ...]) -- one row per assessment point, with
    seed mean/std/min/max/n of damage_block (step 3 of the pinned order:
    MEAN OVER SEEDS OF THE DAMAGE, never of the stress).
    """
    npz_paths = sorted(cond_dir.glob("*.npz"))
    assert npz_paths, f"no seeds found in {cond_dir}"

    per_seed_damage = []   # [{point_id: damage_block}, ...], usable seeds only
    per_seed_theta = []    # [{point_id: worst_theta_deg}, ...], usable seeds only
    bin_numbers = set()
    meta_by_point = {}
    n_excluded = 0

    for npz_path in npz_paths:
        stage2 = s2.load_stage2(npz_path)
        stamp = stage2["stamp"]
        if not _seed_usable(stamp, stamp["transient_cutoff_s"]):
            n_excluded += 1
            print(f"    excluding {npz_path.name}: duration too short after transient trim")
            continue
        case_id = stamp["case_json"]["case_id"]
        bin_numbers.add(bin_number_from_case_id(case_id))
        pt_rows = run_point_damage(stage2, category=category)
        per_seed_damage.append({r["point_id"]: r["damage_block"] for r in pt_rows})
        per_seed_theta.append({r["point_id"]: r["worst_theta_deg"] for r in pt_rows})
        for r in pt_rows:
            meta_by_point.setdefault(r["point_id"], r)

    assert per_seed_damage, f"{cond_dir}: every seed excluded, nothing usable"
    assert len(bin_numbers) == 1, f"{cond_dir}: seeds disagree on bin number: {bin_numbers}"
    bin_number = bin_numbers.pop()
    n_seeds = len(per_seed_damage)

    out_rows = []
    for pid, meta in sorted(meta_by_point.items()):
        vals = np.array([seed[pid] for seed in per_seed_damage])
        i_dominant = int(np.argmax(vals))  # the seed whose damage dominates this point's mean
        theta_rep = per_seed_theta[i_dominant][pid]
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


# region --- every available condition folder, probability-weighted, 25 years ---
def build_per_bin(stage2_root, category):
    """
    Discovers every condition folder under stage2_root and aggregates seeds
    per bin -- the shared discovery+aggregation work behind both
    compute_stage3() (probability-weighted 25yr life) and
    compute_member_matrix() (per-bin member x load-case matrix), so a given
    campaign snapshot is only read off disk once regardless of which
    output(s) a caller wants from it.

    Returns per_bin: {bin_number: {point_id: row}}.
    """
    cond_dirs = sorted(p for p in Path(stage2_root).iterdir() if p.is_dir())
    assert cond_dirs, f"no condition folders found under {stage2_root}"

    per_bin = {}
    for i, cond_dir in enumerate(cond_dirs, 1):
        print(f"  [{i}/{len(cond_dirs)}] aggregating {cond_dir.name} ...")
        bin_number, rows = aggregate_condition(cond_dir, category=category)
        assert bin_number not in per_bin, f"bin {bin_number} appears in more than one condition folder"
        per_bin[bin_number] = {r["point_id"]: r for r in rows}
    return per_bin


def compute_stage3(stage2_root=STAGE2_DIR, bins_table=BINS_CSV, category="B1"):
    """
    Discovers every condition folder under stage2_root, aggregates seeds
    per bin, then combines bins by the sheet's raw probability (not
    renormalized, see load_bin_probabilities()) and scales to the 25-year
    design life (step 4/5 of the pinned order). See the module docstring's
    "partial-campaign honesty" note: D_life only reflects whichever bins
    currently have Stage 2 data.

    Thin wrapper around stage3_from_per_bin() -- builds its own per_bin.
    Callers that also want compute_member_matrix() from the SAME campaign
    snapshot should call build_per_bin() once themselves and pass it to
    both stage3_from_per_bin() and matrix_from_per_bin() directly, so the
    (potentially ~GB-scale) Stage 2 cache is only read off disk once (see
    run_pipeline.py's run_stage3()).

    Returns (rows, meta) where meta reports p_bin's raw sum (for reference
    only, no correction applied) and which/how-many of the 69 bins actually
    contributed.
    """
    p_bin, raw_total = load_bin_probabilities(bins_table)
    per_bin = build_per_bin(stage2_root, category)
    return stage3_from_per_bin(per_bin, p_bin, raw_total, category)


def stage3_from_per_bin(per_bin, p_bin, raw_total, category="B1"):
    """Core of compute_stage3(), operating on an already-built per_bin (see
    build_per_bin()) instead of discovering/aggregating it itself."""
    all_point_ids = sorted({pid for rows in per_bin.values() for pid in rows})
    invariant_keys = ("member_id", "end", "D", "t", "z", "zone", "propset",
                       "environment", "category", "not_assessable", "not_assessable_reason")

    out_rows = []
    for pid in all_point_ids:
        meta = None
        d_life = 0.0
        d_life_var = 0.0   # error-propagated variance, treating bins/seeds as independent
        d_life_min = 0.0
        d_life_max = 0.0
        n_bins_used = 0
        theta_at_max_contrib = None
        max_contrib = -np.inf

        for bin_number, rows in per_bin.items():
            if pid not in rows:
                continue
            r = rows[pid]
            if meta is None:
                meta = r
            else:
                for key in invariant_keys:
                    assert r[key] == meta[key], (
                        f"point {pid}: {key} disagrees between bins "
                        f"({r[key]!r} in bin {bin_number} vs {meta[key]!r})"
                    )
            pb = p_bin.get(bin_number)
            assert pb is not None, f"bin {bin_number} (parsed from case_id) not present in {bins_table}"

            weight = pb * BLOCKS_PER_LIFE
            contrib_mean = weight * r["damage_block_mean"]
            d_life += contrib_mean
            d_life_var += (weight * r["damage_block_std"]) ** 2
            d_life_min += weight * r["damage_block_min"]
            d_life_max += weight * r["damage_block_max"]
            n_bins_used += 1
            if contrib_mean > max_contrib:
                max_contrib = contrib_mean
                theta_at_max_contrib = r["worst_theta_deg"]

        row = {k: v for k, v in meta.items()
               if k not in ("bin_number", "damage_block_mean", "damage_block_std",
                             "damage_block_min", "damage_block_max", "n_seeds_used",
                             "n_seeds_excluded", "worst_theta_deg")}
        row.update(
            sn_curve=category,
            D_life=d_life,
            D_life_seed_std=float(np.sqrt(d_life_var)),
            D_life_min=d_life_min,
            D_life_max=d_life_max,
            life_years=(DESIGN_LIFE_YEARS / d_life) if d_life > 0 else float("inf"),
            worst_theta_deg=theta_at_max_contrib,
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


# region --- member x load-case matrix (high-level overview, not a lifetime estimate) ---
def compute_member_matrix(stage2_root=STAGE2_DIR, bins_table=BINS_CSV, category="B1"):
    """
    One row per member (112, both ends collapsed to whichever is worse),
    one column per load-case bin currently in stage2_root -- a wide
    overview of which members/conditions are most affected, NOT a
    replacement for compute_stage3()'s per-point, per-end, probability-
    weighted 25yr life numbers. Confirmed with the author 10.08.2026: max
    across the 2 ends (not sum/average), column headers are the sheet's
    descriptive NAME (not bin number), two separate metrics wanted rather
    than one:

      raw       -- damage_block_mean itself (seed-averaged damage per
                   10-min block for THAT bin alone), comparable across bins
                   on equal footing -- answers "which conditions are
                   physically harshest," independent of how often they
                   occur.
      weighted  -- p_bin * BLOCKS_PER_LIFE * damage_block_mean, i.e. that
                   bin's actual share of 25yr D_life -- rows sum to the
                   member's real D_life once every campaign bin is present
                   (a cheap cross-check against compute_stage3()'s own
                   D_life column, not asserted here since a partial
                   campaign won't sum to the same total stage3 reports for
                   a different point set).

    A missing bin (not yet in stage2_root) leaves that column as None for
    every member, same "partial-campaign honesty" as compute_stage3().

    Thin wrapper around matrix_from_per_bin() -- builds its own per_bin. See
    compute_stage3()'s docstring for why a caller wanting both outputs from
    the same snapshot should call build_per_bin() once and use both
    matrix_from_per_bin()/stage3_from_per_bin() directly instead.

    Returns (raw_rows, weighted_rows, bin_columns) -- bin_columns is the
    ordered list of column names actually used (bin-number order), so a
    caller can build a DataFrame with a stable column order even though
    dict key order alone isn't guaranteed to survive every pandas version.
    """
    p_bin, _raw_total = load_bin_probabilities(bins_table)
    bin_names = load_bin_names(bins_table)
    per_bin = build_per_bin(stage2_root, category)
    return matrix_from_per_bin(per_bin, p_bin, bin_names)


def matrix_from_per_bin(per_bin, p_bin, bin_names):
    """Core of compute_member_matrix(), operating on an already-built
    per_bin (see build_per_bin())."""
    # (member_id, bin_number) -> list of that member's per-end rows
    member_bin_ends = {}
    zone_by_member = {}
    for bin_number, rows_by_pid in per_bin.items():
        for r in rows_by_pid.values():
            member_bin_ends.setdefault((r["member_id"], bin_number), []).append(r)
            zone_by_member.setdefault(r["member_id"], r["zone"])  # same both ends

    all_members = sorted(zone_by_member)
    bin_columns = [bin_names[b] for b in sorted(per_bin)]

    raw_rows, weighted_rows = [], []
    for mid in all_members:
        raw_row = dict(member_id=mid, zone=zone_by_member[mid])
        weighted_row = dict(member_id=mid, zone=zone_by_member[mid])
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
            assert pb is not None, f"bin {bin_number} not present in {bins_table}"
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


def write_member_matrix_csv(rows, bin_columns, worst_col_name, out_path):
    """Fixed column order: member_id, zone, one column per bin_columns (in
    campaign order, not dict/insertion order), then worst_bin/worst_col_name."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["member_id", "zone"] + bin_columns + ["worst_bin", worst_col_name]
    df = pd.DataFrame(rows)[cols].sort_values("member_id").reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df
# endregion


# region --- section-loss corrosion: tested utilities, NOT wired in (see module docstring) ---
def corrosion_multiplier(t0_mm, delta_t_mm):
    """
    c = t0 / (t0 - delta_t) -- the thin-plate stress-scaling factor for a
    uniform wall-thickness loss delta_t on a starting thickness t0. Applying
    c to an already-stored power sum (apply_corrosion_scalar) reproduces
    exactly what re-running stress recovery on a thinned section would give,
    under the thin-wall approximation quantified (and flagged TO-VERIFY) in
    fatigue-postpro-docs/decisions.md "Section loss" section.
    """
    assert delta_t_mm < t0_mm, f"corrosion loss {delta_t_mm} >= starting thickness {t0_mm}"
    return t0_mm / (t0_mm - delta_t_mm)


def apply_corrosion_scalar(sum_r_m, c, m):
    """
    sum_r_m: a stored sum(range^m) array (any shape). Rainflow ranges scale
    linearly under the thin-wall approximation (R' = c*R), so
    sum(R'^m) = c^m * sum(R^m) -- exact given that approximation, no
    re-rainflowing. See corrosion_multiplier()'s docstring for the caveat.
    """
    return (c ** m) * sum_r_m
# endregion


# region --- output ---
def write_stage3_damage(rows, out_path=RESULTS_DIR / "stage3_damage.csv"):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["member_id", "end"]).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df
# endregion


def _self_check():
    import rainflow
    import rainflow_hist as rhist

    print(f"SN_CONSTANTS_VERIFIED = {sn.SN_CONSTANTS_VERIFIED}")
    print(f"BLOCKS_PER_LIFE = {BLOCKS_PER_LIFE:,.0f} (25yr x 365.25 x 24 x 6)\n")

    # --- 1. Synthetic single-bin histogram: D = n/N_allowed to 1e-14, and
    # doubling the cycle count exactly doubles D (Miner's rule is linear in
    # n, for fixed range/curve). Use a free-corrosion curve (single m1
    # branch, no knee) so the whole bin sits unambiguously on one branch.
    print("1. Synthetic single-bin Miner check (B1, free_corrosion, one bin, no knee):")
    bin_edges = np.geomspace(0.01, 1000.0, 257)
    mid_bins = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    k_bin = 100  # an arbitrary interior bin
    ds_mid = mid_bins[k_bin]
    n_cycles = 37.0
    sum_r3 = np.zeros((1, 256))
    sum_r3[0, k_bin] = n_cycles * ds_mid ** 3
    sum_r = {3: sum_r3}
    d = point_theta_damage(sum_r, bin_edges, "B1", "free_corrosion", t_mm=25.0)[0]
    row = sn.SN_CURVES[("B1", "free_corrosion")]
    N_allowed = 10.0 ** row["log_a1"] / ds_mid ** row["m1"]
    expect = n_cycles / N_allowed
    rel_diff = abs(d - expect) / expect
    print(f"   n={n_cycles} cycles at Delta_sigma={ds_mid:.4f} MPa: "
          f"D={d:.6e}  n/N_allowed={expect:.6e}  rel.diff={rel_diff:.2e}")
    assert rel_diff < 1e-12

    sum_r3_double = {3: sum_r3 * 2.0}
    d_double = point_theta_damage(sum_r3_double, bin_edges, "B1", "free_corrosion", t_mm=25.0)[0]
    print(f"   doubled n -> D={d_double:.6e}  ratio={d_double / d:.6f} (expect exactly 2.0)")
    assert abs(d_double / d - 2.0) < 1e-12

    # --- 2. Corrosion multiplier: c=1.10 at m=3 -> exactly x1.331.
    print("\n2. Corrosion multiplier (utility function, not wired into compute_stage3 -- see module docstring):")
    c = corrosion_multiplier(t0_mm=20.0, delta_t_mm=20.0 - 20.0 / 1.10)
    print(f"   t0=20.0mm, delta_t chosen so c={c:.6f} (target 1.10)")
    assert abs(c - 1.10) < 1e-9
    scaled = apply_corrosion_scalar(np.array([5.0]), c, m=3)[0]
    ratio = scaled / 5.0
    print(f"   apply_corrosion_scalar(5.0, c=1.10, m=3) / 5.0 = {ratio:.6f} (expect exactly 1.331)")
    assert abs(ratio - 1.331) < 1e-9

    # --- 3. max-over-theta != damage-at-arbitrary-theta: build a SINGLE-
    # PLANE bending fixture (Mkx only, Mky=0 -- NOT a rotating constant-
    # magnitude moment: sigma(theta,t)=A/W*cos(theta)*sin(wt) for that case
    # has IDENTICAL amplitude at every theta, just phase-shifted, since
    # cos(wt)*cos(th)-sin(wt)*sin(th)=cos(wt+th) -- a genuinely flat-damage
    # fixture by construction, not a useful check here; that symmetry is
    # exactly why the member track is theta-origin-invariant, see stress.py).
    # Single-plane bending instead gives amplitude proportional to
    # |cos(theta)|, so damage is strongly theta-dependent, and MAX OVER
    # THETA (step 2 of the pinned order) is doing real, necessary work.
    print("\n3. Single-plane bending fixture -- theta genuinely matters (max-over-theta is real work):")
    n_t = 20000
    t_axis = np.arange(n_t) * 0.05
    omega = 2 * np.pi * 0.2
    Mkx = 5.0e5 * np.sin(omega * t_axis)
    Mky = np.zeros(n_t)
    D_test, t_test = 0.8, 0.02
    import stress
    sig_ax, sig_ipb, sig_opb = stress.nominal_components(
        np.zeros(n_t), Mkx, Mky, D_test, t_test)
    sigma = stress.hotspot_member(sig_ax, sig_ipb, sig_opb)
    bin_edges_fine = bin_edges
    sum_r_theta = {3: np.zeros((stress.N_THETA, 256)), 5: np.zeros((stress.N_THETA, 256))}
    for k_theta in range(stress.N_THETA):
        cycles = list(rainflow.extract_cycles(sigma[:, k_theta]))
        counts, sr, n_under, n_over = rhist.cycles_to_histogram(cycles, bin_edges_fine, (3, 5))
        for m in (3, 5):
            sum_r_theta[m][k_theta, :] = sr[m]
    d_theta = point_theta_damage(sum_r_theta, bin_edges_fine, "B1", "free_corrosion", t_mm=t_test * 1000.0)
    # theta=90/270deg see zero bending amplitude (pure single-plane bending,
    # cos(90deg)=0) -> exactly zero damage there by construction -- report
    # the min over the NONZERO thetas instead of dividing by the true zero.
    d_nonzero_min = d_theta[d_theta > 0].min()
    print(f"   damage(theta) range: min={d_theta.min():.4e} (theta with zero bending amplitude, "
          f"expected exactly 0)  max={d_theta.max():.4e}  "
          f"ratio max/min_nonzero={d_theta.max() / d_nonzero_min:.3f}")
    assert d_theta.min() == 0.0, "expected exactly zero damage at the zero-amplitude theta"
    assert d_theta.max() / d_nonzero_min > 1.5, "theta-dependence unexpectedly weak -- check the fixture"
    k_worst = int(np.argmax(d_theta))
    print(f"   worst theta = {np.degrees(stress.THETA_RAD[k_worst]):.1f} deg -- confirms max-over-theta "
          f"is selecting a real material point, not defaulting to theta=0")

    # --- 4. Crashed-run exclusion: the known-crashed QA fixture (0.77s,
    # structurally self-consistent header) must be excluded from an
    # aggregate, not silently contribute near-zero damage x 1.3M blocks.
    print("\n4. Crashed-run exclusion (LC_V23_H4p0_T10/S654321, 0.77s tower-strike abort):")
    crashed_dir = DEV_FIXTURE_DIR / "_staging" / "TestScenario" / "LC_V23_H4p0_T10" / "S654321"
    assert crashed_dir.exists(), f"missing QA fixture: {crashed_dir}"
    tmp_out = RESULTS_DIR / "_stage3_selfcheck_crashtest"
    npz_path = s2.process_run(crashed_dir, out_root=tmp_out)
    stage2_crashed = s2.load_stage2(npz_path)
    usable = _seed_usable(stage2_crashed["stamp"], stage2_crashed["stamp"]["transient_cutoff_s"])
    duration_s = ((stage2_crashed["stamp"]["n_t"] - 1) * stage2_crashed["stamp"]["dt_s"])
    print(f"   duration={duration_s:.2f}s vs transient_cutoff={stage2_crashed['stamp']['transient_cutoff_s']}s "
          f"-> usable={usable} (expect False)")
    assert usable is False

    print("\n  all synthetic/utility checks passed.")

    # --- 5. End-to-end on the real bin (LC65, 6 seeds) -----------------
    print("\n5. End-to-end on real Stage 2 data (LC_V20_H3p5_T8, bin 65):")
    # Uses the dev fixture's OWN Stage 2 cache (left at its original
    # location, TestScenario/stage2/), not the module-default STAGE2_DIR
    # (which now points at the D: drive for real-campaign processing).
    real_cond_dir = DEV_FIXTURE_STAGE2_DIR / "LC_V20_H3p5_T8"
    if not real_cond_dir.exists():
        print(f"   SKIPPED -- {real_cond_dir} not found (run stage2_histograms.process_run "
              f"for all 6 real seeds into stage2/ first)")
        return

    bin_number, rows = aggregate_condition(real_cond_dir, category="B1")
    print(f"   bin_number={bin_number} (expect 65), {len(rows)} points "
          f"(expect 224 = 112 members x 2 ends)")
    assert bin_number == 65
    assert len(rows) == 224
    assert all(np.isfinite(r["damage_block_mean"]) for r in rows), "NaN/Inf in per-run damage"
    assert all(r["n_seeds_used"] == 6 for r in rows), "expected all 6 real seeds usable"

    not_assessable = [r for r in rows if r["not_assessable"]]
    print(f"   not_assessable points: {len(not_assessable)} (expect 24 = 12 members x 2 ends), "
          f"still carry a damage number: {all(np.isfinite(r['damage_block_mean']) for r in not_assessable)}")
    assert len(not_assessable) == 24

    # environment sanity: environment assignment must match the zone->env
    # table exactly (not e.g. accidentally inverted).
    for r in rows:
        assert r["environment"] == sn.ZONE_TO_ENVIRONMENT[r["zone"]]

    # seed scatter: report, don't assert a specific number (this varies
    # with the actual campaign data) -- just confirm it's finite and
    # generally much smaller than the mean (sanity, not a hard bound).
    cov = [r["damage_block_std"] / r["damage_block_mean"] for r in rows
           if r["damage_block_mean"] > 0]
    print(f"   seed coefficient of variation (std/mean) across assessable points: "
          f"median={np.median(cov):.3f}  max={np.max(cov):.3f}")

    ranked = sorted(rows, key=lambda r: -r["damage_block_mean"])
    print("   top 5 most-damaged points this run/bin (member_id, end, zone, damage_block_mean):")
    for r in ranked[:5]:
        print(f"     member {r['member_id']:>3} end {r['end']} zone={r['zone']:<11} "
              f"D_block={r['damage_block_mean']:.3e}")

    # --- 6. Full compute_stage3 on whatever bins currently exist ---
    # Explicitly against the dev fixture's own stage2 cache, not the
    # module-default STAGE2_DIR (D: drive) -- this self-check must stay
    # reproducible regardless of how much of the real campaign has been
    # processed on D: at the time it's run.
    print("\n6. compute_stage3() over all available bins (dev fixture only):")
    stage3_rows, meta = compute_stage3(stage2_root=DEV_FIXTURE_STAGE2_DIR, category="B1")
    print(f"   p_bin raw sum = {meta['p_bin_raw_sum']:.5f} (used as-is, not renormalized -- "
          f"see load_bin_probabilities() docstring)")
    print(f"   bins available: {meta['bins_available']} of {meta['n_bins_total_campaign']} total campaign bins")
    print(f"   {len(stage3_rows)} points in output (expect 224)")
    assert len(stage3_rows) == 224
    assert all(np.isfinite(r["D_life"]) and r["D_life"] >= 0 for r in stage3_rows)
    assert all(np.isfinite(r["life_years"]) or r["life_years"] == float("inf") for r in stage3_rows)

    # physical sanity: braces (PropSet 1, D=0.8/t=0.02) should generally be
    # more damaged than legs (D=1.2, thicker) at comparable load -- report,
    # don't hard-assert (true "in general", not guaranteed for every pair).
    member_max = {}
    for r in stage3_rows:
        member_max[r["member_id"]] = max(member_max.get(r["member_id"], 0.0), r["D_life"])
    ranked_members = sorted(member_max.items(), key=lambda kv: -kv[1])
    print("   top 5 most-damaged MEMBERS (25yr, this bin's probability share only):")
    for mid, d in ranked_members[:5]:
        print(f"     member {mid:>3}: D_life={d:.4e}")

    out_path = write_stage3_damage(stage3_rows)
    print(f"\n   wrote {out_path} ({len(stage3_rows)} rows)")

    # --- 7. Member x load-case matrix (added 10.08.2026, dev fixture only) ---
    print("\n7. compute_member_matrix() -- member x load-case overview (dev fixture only):")
    raw_rows, weighted_rows, bin_columns = compute_member_matrix(
        stage2_root=DEV_FIXTURE_STAGE2_DIR, category="B1")
    print(f"   {len(raw_rows)} member(s) x {len(bin_columns)} bin column(s): {bin_columns}")
    assert len(raw_rows) == 112 and len(weighted_rows) == 112
    assert bin_columns == ["LC_V20_H3p5_T8"], (
        "dev fixture is a single real bin (LC65) -- if this fails, the bins "
        "sheet's NAME column no longer matches, or a second dev-fixture bin "
        "was added without updating this self-check")

    raw_by_member = {r["member_id"]: r for r in raw_rows}
    weighted_by_member = {r["member_id"]: r for r in weighted_rows}
    # cross-check against stage3_rows computed above (step 6, same category,
    # same single-bin snapshot) -- member 22's worst end should agree with
    # the matrix's max-over-ends value to the same D_life/weighted number,
    # since with only ONE bin present, weighted == that member's full D_life.
    m22_stage3 = max(r["D_life"] for r in stage3_rows if r["member_id"] == 22)
    m22_matrix_weighted = weighted_by_member[22]["LC_V20_H3p5_T8"]
    rel_diff = abs(m22_stage3 - m22_matrix_weighted) / m22_stage3
    print(f"   member 22 cross-check: stage3 D_life={m22_stage3:.4e}  "
          f"matrix weighted={m22_matrix_weighted:.4e}  rel.diff={rel_diff:.2e}")
    assert rel_diff < 1e-9, "matrix's weighted value should exactly match stage3's D_life (single bin)"

    # not-assessable members still get a number (0.0), never dropped
    print(f"   not-assessable members still present (not silently excluded), "
          f"e.g. member 101 (TP interface stub): raw={raw_by_member[101]['LC_V20_H3p5_T8']:.3e}")

    raw_df = write_member_matrix_csv(
        raw_rows, bin_columns, "worst_bin_damage", RESULTS_DIR / "_member_matrix_selfcheck_raw.csv")
    weighted_df = write_member_matrix_csv(
        weighted_rows, bin_columns, "worst_bin_contribution",
        RESULTS_DIR / "_member_matrix_selfcheck_weighted.csv")
    print(f"   wrote self-check matrices ({len(raw_df)} rows each) -- "
          f"real output goes through run_pipeline.py's run_stage3(), not this self-check")

    print("\n" + "=" * 78)
    print("Reminder: D_life/life_years above reflect ONLY the bins currently in")
    print("stage2/ -- meaningful for ranking members now, NOT a claim about total")
    print("remaining life until every campaign bin has Stage 2 data.")
    print("=" * 78)


if __name__ == "__main__":
    _self_check()
