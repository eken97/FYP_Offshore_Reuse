"""
Final member-track results: one CSV, one row per member, one D/life_years
pair per time step (0, 5, 10, 15, 20, 25 yr) -- decided by the author
16.08.2026 (see docs/decisions.md).

Selection convention, confirmed by the author 16.08.2026: for EVERY load
case independently, pick the worst end, THEN sum the per-load-case worst
values across load cases and time steps -- never pick a single globally-
worst end first and use its own already-fully-aggregated total. Both
D_0 and every corrosion step are built this same way, reusing the
already-written member_damage_matrix_weighted.csv (baseline) and
member_damage_matrix_corrosion_weighted_y{N}.csv (each step) -- both
already have the per-load-case worst-end reduction baked in.

Step 0 (no corrosion) = the full 25yr projection at year-0 section
properties. Steps 5..25 are CUMULATIVE damage through that year (not each
step's contribution in isolation) -- running sum of each step's own
contribution. life_years at each step = elapsed_years / D_cum(elapsed_years),
the same "life implied by the damage accrued so far" idea used for the
no-corrosion case (life_years = 25yr / D_0 there, just anchored to a fixed
25yr window instead of the elapsed one).

Corrosion columns are NaN for the 80 non-splash members -- corrosion was
never modelled for them (deliberate scope decision, see
docs/decisions.md "MEMBER-TRACK CORROSION LOOP" session), not
a missing value.

Self-check below compares D_25 (this file's cumulative total) against
stage3_damage_corrosion.csv's own D_life -- built under the OTHER
convention (single globally-worst end, its already-fully-aggregated
total). D_25 here is expected to be >= that reference (per-load-case
worst-end selection is more conservative by construction, a real
rearrangement-inequality effect, not a bug) -- the check reports the size
of that gap, not an exact-match assertion.
"""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
REAL_RESULTS_DIR = RESULTS_DIR / "real_campaign"

DESIGN_LIFE_YEARS = 25.0
CORROSION_YEARS = [5, 10, 15, 20, 25]

NON_LC_COLUMNS = {"member_id", "member_class", "zone", "worst_bin",
                   "worst_bin_damage", "worst_bin_contribution"}


def load_baseline(results_dir=REAL_RESULTS_DIR):
    """member_id, zone, D_0, life_years_0 -- SAME convention as the
    corrosion steps: worst end picked PER LOAD CASE (already baked into
    member_damage_matrix_weighted.csv), then summed across load cases.
    D_0 is therefore generally >= stage3_damage.csv's own D_life (which
    picks one globally-worst end and uses its already-fully-aggregated
    total) -- an expected, not-a-bug consequence of the per-load-case
    reduction being more conservative (see module docstring)."""
    df = pd.read_csv(results_dir / "member_track" / "member_damage_matrix_weighted.csv")
    lc_cols = [c for c in df.columns if c not in NON_LC_COLUMNS]
    n_missing = df[lc_cols].isna().sum().sum()
    if n_missing:
        print(f"  WARNING baseline: {n_missing} missing bin values -- treated as 0 contribution")
    d0 = df[lc_cols].fillna(0.0).sum(axis=1)
    out = pd.DataFrame({
        "member_id": df["member_id"],
        "zone": df["zone"],
        "D_0": d0,
        "life_years_0": np.where(d0 > 0, DESIGN_LIFE_YEARS / d0, np.inf),
    })
    return out.reset_index(drop=True)


def load_step_contribution(year, results_dir=REAL_RESULTS_DIR):
    """member_id -> that 5-year step's own total weighted damage
    contribution (sum across every load-case column). Reuses the already-
    written member_damage_matrix_corrosion_weighted_y{year}.csv, not a
    fresh Stage 2/3 recompute."""
    path = results_dir / "member_track" / "corrosion" / f"member_damage_matrix_corrosion_weighted_y{year}.csv"
    df = pd.read_csv(path)
    lc_cols = [c for c in df.columns if c not in NON_LC_COLUMNS]
    n_missing = df[lc_cols].isna().sum().sum()
    if n_missing:
        print(f"  WARNING year={year}: {n_missing} missing bin values (should be 0 "
              f"for a complete campaign) -- treated as 0 contribution")
    contrib = df[lc_cols].fillna(0.0).sum(axis=1)
    return pd.Series(contrib.values, index=df["member_id"].values, name=year)


def build_cumulative_corrosion(years=CORROSION_YEARS, results_dir=REAL_RESULTS_DIR):
    """member_id x year -> cumulative D through that year (running sum of
    each step's own contribution)."""
    running = None
    cum_by_year = {}
    for y in sorted(years):
        step = load_step_contribution(y, results_dir)
        running = step.copy() if running is None else running.add(step, fill_value=0.0)
        cum_by_year[y] = running.copy()
    cum = pd.DataFrame(cum_by_year)
    cum.index.name = "member_id"
    return cum.reset_index()


def build_final_results_member(results_dir=REAL_RESULTS_DIR, years=CORROSION_YEARS):
    baseline = load_baseline(results_dir)
    cum = build_cumulative_corrosion(years, results_dir)

    out = baseline.merge(cum, on="member_id", how="left")
    for y in years:
        d_col, life_col = f"D_{y}", f"life_years_{y}"
        out = out.rename(columns={y: d_col})
        out[life_col] = np.where(out[d_col] > 0, y / out[d_col], np.inf)
        out.loc[out[d_col].isna(), life_col] = np.nan  # non-splash members: stay NaN, not inf

    ordered_cols = ["member_id", "zone", "D_0", "life_years_0"]
    for y in years:
        ordered_cols += [f"D_{y}", f"life_years_{y}"]
    out = out[ordered_cols].sort_values("member_id").reset_index(drop=True)
    return out


def self_check(out, results_dir=REAL_RESULTS_DIR):
    print("--- self-check: D_25 (this file, per-load-case worst-end) vs "
          "stage3_damage_corrosion.csv's own D_life (single globally-worst-end) ---")
    ref = pd.read_csv(results_dir / "member_track" / "corrosion" / "stage3_damage_corrosion.csv")
    ref_worst = ref.groupby("member_id")["D_life"].max()

    mine = out.set_index("member_id")["D_25"].dropna()
    common = mine.index.intersection(ref_worst.index)
    assert len(common) == len(ref_worst), (
        f"member-set mismatch: {len(common)} common vs {len(ref_worst)} in reference "
        f"({sorted(set(ref_worst.index) - set(common))} missing from cumulative build)"
    )

    # per-load-case worst-end sum can never be less than a single fixed-end sum
    # (term-by-term max_e >= any fixed e*) -- allow float noise only (~1e-10 rel.)
    signed_gap = mine.loc[common] - ref_worst.loc[common]
    rel_signed = signed_gap / ref_worst.loc[common]
    n_below = (rel_signed < -1e-9).sum()
    assert n_below == 0, (
        f"{n_below} members have D_25 measurably BELOW the reference (beyond float "
        f"noise) -- per-load-case worst-end selection should never be less conservative "
        f"than the single-worst-end total; this would indicate a real bug"
    )

    diff = (mine.loc[common] - ref_worst.loc[common]).abs()
    rel = diff / ref_worst.loc[common]
    print(f"  {len(common)} splash-zone members compared, D_25 >= reference in all of them (as expected)")
    print(f"  max rel.diff = {rel.max():.4%}, median rel.diff = {rel.median():.4%}")
    worst_member = rel.idxmax()
    print(f"  largest-gap member: {worst_member} "
          f"(this file's D_25={mine[worst_member]:.6e}, "
          f"single-worst-end reference={ref_worst[worst_member]:.6e})")


if __name__ == "__main__":
    result = build_final_results_member()
    self_check(result)

    out_path = REAL_RESULTS_DIR / "final_results_member.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} -- {len(result)} rows, {len(result.columns)} columns")
    print(f"  {result['D_5'].notna().sum()} members with corrosion columns populated "
          f"(splash-zone only), {result['D_5'].isna().sum()} non-splash (NaN by design)")
