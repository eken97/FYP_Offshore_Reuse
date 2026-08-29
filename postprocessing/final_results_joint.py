"""
Final joint-track results: one CSV, one row per physical joint (node),
column groups for all 10 scenarios decided by the author 16-17.08.2026 (see
docs/decisions.md). Column naming uses the thesis's collapsed
Table 14 scenario numbering (S1-S5) with an explicit -K/-Y treatment suffix,
e.g. D_S1-K / D_S1-Y (renamed 18.08.2026 from the original flat S1-S10
numbering, which conflated the scenario axis with the K/Y treatment axis).

Selection convention, SAME rule as final_results_member.py: for each load
case independently, pick the worst row among every row that survives this
scenario's own filters (chord_t_scenario != 'thin' always; treatment
restricted to the scenario's K-or-Y choice, TY and X rows always included
since they have no K/Y duality), take ITS damage, and sum those per-load-
case worst values across load cases -- never pick a single row's own
already-fully-aggregated total.

The "worst row per node" collapse folds together whatever rows physically
share that node under this scenario's filters:
  - K-family nodes: 2 planes x 2 braces each = up to 4 rows (thin dropped
    at the 4 bottom-K nodes, so exactly the rows for the chosen K/Y
    treatment survive).
  - TY-family nodes: 2 planes x 1 brace each = 2 rows.
  - X-family nodes: 1 crossing, 2 rows from the chord/brace ambiguity
    (A_as_chord / B_as_chord).

Scenarios 1-4 and 7-8 (no corrosion) report one D/life_years pair (25yr
projection). Scenarios 5-6 and 9-10 (corrosion) report the FULL cumulative
trajectory through 5/10/15/20/25yr, same cumulative-sum-of-per-year-step
convention already used and verified in final_results_member.py.

Scenario table (see docs/decisions.md for the full derivation):
    S1: baseline,   no corrosion       (-K / -Y)
    S2: retrofit A, no corrosion       (-K / -Y)
    S3: retrofit A, corrosion          (-K / -Y)
    S4: retrofit B, no corrosion       (-K / -Y)
    S5: retrofit B, corrosion          (-K / -Y)
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

NON_LC_COLUMNS = {"node", "sub_joint_id", "brace_member", "brace_end",
                   "chord_t_scenario", "direction", "treatment",
                   "worst_bin", "worst_bin_contribution"}

# scenario-treatment -> (weighted-matrix filename, K-or-Y treatment filter)
SCENARIOS_UNCORRODED = {
    "S1-K": ("joint_track/joint_damage_matrix_weighted.csv", "K"),
    "S1-Y": ("joint_track/joint_damage_matrix_weighted.csv", "Y"),
    "S2-K": ("joint_track/joint_damage_matrix_thickness_A_weighted.csv", "K"),
    "S2-Y": ("joint_track/joint_damage_matrix_thickness_A_weighted.csv", "Y"),
    "S4-K": ("joint_track/joint_damage_matrix_thickness_B_weighted.csv", "K"),
    "S4-Y": ("joint_track/joint_damage_matrix_thickness_B_weighted.csv", "Y"),
}
# scenario-treatment -> (per-year-step weighted-matrix filename PREFIX, treatment filter)
SCENARIOS_CORRODED = {
    "S3-K": ("joint_track/corrosion/joint_damage_matrix_thickness_corrosion_A_weighted", "K"),
    "S3-Y": ("joint_track/corrosion/joint_damage_matrix_thickness_corrosion_A_weighted", "Y"),
    "S5-K": ("joint_track/corrosion/joint_damage_matrix_thickness_corrosion_B_weighted", "K"),
    "S5-Y": ("joint_track/corrosion/joint_damage_matrix_thickness_corrosion_B_weighted", "Y"),
}
# scenario-treatment -> reference aggregated CSV (for the self-check only)
REFERENCE_UNCORRODED = {
    "S1-K": "joint_track/stage3_joint_damage.csv", "S1-Y": "joint_track/stage3_joint_damage.csv",
    "S2-K": "joint_track/stage3_joint_damage_thickness_A.csv", "S2-Y": "joint_track/stage3_joint_damage_thickness_A.csv",
    "S4-K": "joint_track/stage3_joint_damage_thickness_B.csv", "S4-Y": "joint_track/stage3_joint_damage_thickness_B.csv",
}
REFERENCE_CORRODED = {
    "S3-K": "joint_track/corrosion/stage3_joint_damage_thickness_corrosion_A.csv",
    "S3-Y": "joint_track/corrosion/stage3_joint_damage_thickness_corrosion_A.csv",
    "S5-K": "joint_track/corrosion/stage3_joint_damage_thickness_corrosion_B.csv",
    "S5-Y": "joint_track/corrosion/stage3_joint_damage_thickness_corrosion_B.csv",
}


def load_node_family_map(results_dir=REAL_RESULTS_DIR):
    """node -> family ('K'/'TY'/'X'), fixed regardless of treatment --
    sourced from stage3_joint_damage.csv's own explicit family column."""
    df = pd.read_csv(results_dir / "joint_track" / "stage3_joint_damage.csv")
    return df.drop_duplicates("node").set_index("node")["family"]


def filter_scenario_rows(df, treatment_filter):
    """Drop bottom-K thin rows (always) and restrict to this scenario's
    K-or-Y treatment, keeping TY/X rows unconditionally (no duality)."""
    df = df[df["chord_t_scenario"] != "thin"]
    df = df[df["treatment"].isin([treatment_filter, "TY", "X"])]
    return df


def node_worst_per_lc(df, lc_cols):
    """Worst (max) row per node, PER LOAD CASE independently."""
    return df.groupby("node")[lc_cols].max()


def scenario_D_uncorroded(file, treatment_filter, results_dir=REAL_RESULTS_DIR):
    df = pd.read_csv(results_dir / file)
    lc_cols = [c for c in df.columns if c not in NON_LC_COLUMNS]
    df_f = filter_scenario_rows(df, treatment_filter)
    node_lc_max = node_worst_per_lc(df_f, lc_cols)
    n_missing = node_lc_max.isna().sum().sum()
    if n_missing:
        print(f"  WARNING {file}/{treatment_filter}: {n_missing} missing bin values "
              f"-- treated as 0 contribution")
    return node_lc_max.fillna(0.0).sum(axis=1)


def scenario_step_contribution(prefix, year, treatment_filter, results_dir=REAL_RESULTS_DIR):
    df = pd.read_csv(results_dir / f"{prefix}_y{year}.csv")
    lc_cols = [c for c in df.columns if c not in NON_LC_COLUMNS]
    df_f = filter_scenario_rows(df, treatment_filter)
    node_lc_max = node_worst_per_lc(df_f, lc_cols)
    n_missing = node_lc_max.isna().sum().sum()
    if n_missing:
        print(f"  WARNING {prefix} y={year}/{treatment_filter}: {n_missing} missing "
              f"bin values -- treated as 0 contribution")
    return node_lc_max.fillna(0.0).sum(axis=1)


def scenario_cumulative_corroded(prefix, treatment_filter, results_dir=REAL_RESULTS_DIR,
                                  years=CORROSION_YEARS):
    running = None
    cum_by_year = {}
    for y in sorted(years):
        step = scenario_step_contribution(prefix, y, treatment_filter, results_dir)
        running = step.copy() if running is None else running.add(step, fill_value=0.0)
        cum_by_year[y] = running.copy()
    return cum_by_year


def build_final_results_joint(results_dir=REAL_RESULTS_DIR):
    family = load_node_family_map(results_dir)
    out = pd.DataFrame({"node": sorted(family.index)})
    out["family"] = out["node"].map(family)

    for s, (file, tf) in SCENARIOS_UNCORRODED.items():
        print(f"scenario {s}: {file} / treatment={tf}")
        D = scenario_D_uncorroded(file, tf, results_dir)
        d_col, life_col = f"D_{s}", f"life_years_{s}"
        out[d_col] = out["node"].map(D)
        out[life_col] = np.where(out[d_col] > 0, DESIGN_LIFE_YEARS / out[d_col], np.inf)
        out.loc[out[d_col].isna(), life_col] = np.nan

    for s, (prefix, tf) in SCENARIOS_CORRODED.items():
        print(f"scenario {s}: {prefix} / treatment={tf}")
        cum = scenario_cumulative_corroded(prefix, tf, results_dir)
        for y in CORROSION_YEARS:
            d_col, life_col = f"D_{s}_{y}", f"life_years_{s}_{y}"
            out[d_col] = out["node"].map(cum[y])
            out[life_col] = np.where(out[d_col] > 0, y / out[d_col], np.inf)
            out.loc[out[d_col].isna(), life_col] = np.nan

    ordered = ["node", "family"]
    for s in ("S1-K", "S1-Y", "S2-K", "S2-Y"):
        ordered += [f"D_{s}", f"life_years_{s}"]
    for s in ("S3-K", "S3-Y"):
        for y in CORROSION_YEARS:
            ordered += [f"D_{s}_{y}", f"life_years_{s}_{y}"]
    for s in ("S4-K", "S4-Y"):
        ordered += [f"D_{s}", f"life_years_{s}"]
    for s in ("S5-K", "S5-Y"):
        for y in CORROSION_YEARS:
            ordered += [f"D_{s}_{y}", f"life_years_{s}_{y}"]

    out = out[ordered].sort_values("node").reset_index(drop=True)
    return out


def _reference_max(agg_file, treatment_filter, results_dir=REAL_RESULTS_DIR):
    df = pd.read_csv(results_dir / agg_file)
    df = filter_scenario_rows(df, treatment_filter)
    return df.groupby("node")["D_life"].max()


def self_check(out, results_dir=REAL_RESULTS_DIR):
    print("\n--- self-check: this file's D (per-load-case worst-row) vs each scenario's own "
          "aggregated CSV (single worst-row-on-the-total) ---")
    for s, agg_file in REFERENCE_UNCORRODED.items():
        tf = SCENARIOS_UNCORRODED[s][1]
        ref = _reference_max(agg_file, tf, results_dir)
        mine = out.set_index("node")[f"D_{s}"]
        common = mine.index.intersection(ref.index)
        assert len(common) == len(ref), f"{s}: node-set mismatch vs reference"
        signed_rel = (mine.loc[common] - ref.loc[common]) / ref.loc[common]
        n_below = (signed_rel < -1e-9).sum()
        assert n_below == 0, f"{s}: {n_below} nodes measurably BELOW reference -- real bug"
        print(f"  {s} ({agg_file}, {tf}): {len(common)} nodes, "
              f"max rel.diff={signed_rel.max():.4%}, median={signed_rel.median():.4%}")

    for s, agg_file in REFERENCE_CORRODED.items():
        tf = SCENARIOS_CORRODED[s][1]
        ref = _reference_max(agg_file, tf, results_dir)
        mine = out.set_index("node")[f"D_{s}_25"].dropna()
        common = mine.index.intersection(ref.index)
        assert len(common) == len(ref), (
            f"{s} y25: node-set mismatch ({len(common)} vs {len(ref)} in reference)"
        )
        signed_rel = (mine.loc[common] - ref.loc[common]) / ref.loc[common]
        n_below = (signed_rel < -1e-9).sum()
        assert n_below == 0, f"{s} y25: {n_below} nodes measurably BELOW reference -- real bug"
        print(f"  {s} y25 ({agg_file}, {tf}): {len(common)} nodes, "
              f"max rel.diff={signed_rel.max():.4%}, median={signed_rel.median():.4%}")


if __name__ == "__main__":
    result = build_final_results_joint()
    self_check(result)

    out_path = REAL_RESULTS_DIR / "final_results_joint.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} -- {len(result)} rows, {len(result.columns)} columns")
    for s in ("S3-K", "S3-Y", "S5-K", "S5-Y"):
        n = result[f"D_{s}_5"].notna().sum()
        print(f"  scenario {s}: {n} nodes with corrosion data (splash-zone joints only)")
