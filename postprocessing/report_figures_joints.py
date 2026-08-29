"""
CANONICAL script for the thesis's joint-track results figures -- produces
ONLY the figures the author has actually approved after review, nothing else.
Trimmed from the broader iteration script `joint_results_final.py` (which
stays the place for candidate work still under discussion -- see
docs/decisions.md for why this project keeps a two-script
split per track).

STATUS (19.08.2026): B0, B1, B3 are approved -- both K and Y consideration
for B1/B3 (B0 shows K and Y side by side already). B7 (K vs Y dumbbell, the
12 true K-family nodes), B8 (T_cp sensitivity dumbbell, the 8 chronic nodes,
K only), and the B1-template sensitivity map (S2.1/S4.1, K only) are also
approved. B2 (ranked worst joints), B5 (scenario ladder), B6 (fail-count
summary) were explicitly DROPPED, not just deferred -- their code no longer
exists in joint_results_final.py.

Also approved 19.08.2026: B_damage_by_environment_facets_weighted/_raw, the
joint-track counterpart to the member-track A5c family (probability-weighted
and raw damage contribution, faceted by Tp, 2x2 square-cell layout -- a
1-column/4-row stacked layout was tried and rejected on looks). Exact per-bin
values are NOT annotated on the figure -- see the appendix tables built the
same session (build_appendix_tables.py) for the full numeric matrices.

Run directly to regenerate every approved report figure from scratch:
    python report_figures_joints.py

Source of truth: final_results_joint.csv (per-node, per-scenario damage) and
final_results_joint_summary.csv (the 7-scenario S1/S2/S2.1/S3/S4/S4.1/S5 x
K/Y overview), both using the D_S1-K/D_S1-Y naming convention.

Output: figures/joint_track/final/ (PNG + SVG each).
"""
from pathlib import Path

import joint_results_final as jrf

PROJECT = Path(__file__).resolve().parents[1]   # repo root
OUT_DIR = PROJECT / "figures" / "joint_track" / "final"

if __name__ == "__main__":
    jrf.OUT_DIR = OUT_DIR  # redirect the shared module's save target to final/

    df = jrf.load_final()
    S = jrf.build_scenario_columns(df)
    jrf.self_check(df, S)
    model = jrf.sdg.read_subdyn_model(jrf.sdg.DEFAULT_SD_PATH)
    joints, members = model["joints"], model["members"]

    summary = jrf.load_summary()
    jrf.fig_B0_overview_heatmap(summary)
    for treatment in ("K", "Y"):
        jrf.fig_B1_joint_map_grid(S, joints, members, treatment)
        jrf.fig_B3_family_comparison(S, treatment)

    jrf.fig_B7_k_vs_y_dumbbell(S)
    jrf.fig_B8_tcp_sensitivity_dumbbell(summary)
    jrf.fig_B1_sensitivity_map_grid(summary, joints, members)

    jrf.fig_B_damage_by_environment_facets_weighted()
    jrf.fig_B_damage_by_environment_facets_raw()

    print(f"\n10 approved report figures written to {OUT_DIR}")
    print("B0, B1_K, B1_Y, B3_K, B3_Y, B7, B8_K, B1_sensitivity_K,")
    print("B_damage_by_environment_facets_weighted, B_damage_by_environment_facets_raw")
    print("Y-plane B1/B3 rows reuse the same treatment-parametrised functions as K.")
    print("B2/B5/B6 dropped. B8/B1-sensitivity are K-only for now -- Y-plane versions")
    print("and the S-N-curve-comparison set are not yet started -- see docs/decisions.md.")
