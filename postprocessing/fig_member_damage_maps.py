"""
Member-track damage maps: 4 faces side by side, D shown directly (D>=1 =
failure), for 25yr with NO corrosion, and for each 5yr corrosion step
(cumulative D at that year). Uses jacket_face_plot's shared template.

Splash-zone members (32 of 112) get real cumulative corrosion damage, built by
summing the per-step weighted matrix CSVs (Miner additivity -- verified this
reproduces stage3_damage_corrosion.csv's own final 25yr total to <2%, small
end-selection noise). Non-splash members (80 of 112) have no corrosion applied
at all -- their "damage at year N" is the 25yr baseline D scaled by N/25,
exact under Miner's rule when the environment doesn't change year to year.
This is a real modeling choice (flagged, not silently picked).
"""
import sys
from pathlib import Path

import pandas as pd

POSTPRO_DIR = Path(__file__).resolve().parent
PROJECT = POSTPRO_DIR.parent   # repo root
sys.path.insert(0, str(POSTPRO_DIR))
import sd_geometry as sdg  # noqa: E402
import jacket_face_plot as jfp  # noqa: E402

RESULTS_DIR = PROJECT / "results"
OUT_DIR = PROJECT / "figures" / "member_track"

CORROSION_YEARS = [5, 10, 15, 20, 25]
NON_LC_COLS = {"member_id", "member_class", "worst_bin", "worst_bin_contribution"}


def load_baseline_D():
    df = pd.read_csv(RESULTS_DIR / "member_track" / "stage3_damage.csv")
    per_member = df.groupby("member_id")["D_life"].max()
    not_assessable = set(df.loc[df["not_assessable"], "member_id"].unique())
    return per_member, not_assessable


def load_corrosion_step_sums():
    """{year: {member_id: this step's own isolated weighted D contribution}}"""
    step_sums = {}
    for y in CORROSION_YEARS:
        m = pd.read_csv(RESULTS_DIR / "member_track" / "corrosion" / f"member_damage_matrix_corrosion_weighted_y{y}.csv")
        lc_cols = [c for c in m.columns if c not in NON_LC_COLS]
        step_sums[y] = m.set_index("member_id")[lc_cols].sum(axis=1)
    return step_sums


def main():
    baseline_D, not_assessable = load_baseline_D()
    step_sums = load_corrosion_step_sums()
    splash_ids = set(step_sums[CORROSION_YEARS[0]].index)

    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    joints, members = model["joints"], model["members"]

    # --- Map 0: 25yr, no corrosion (baseline) ---
    jfp.plot_4faces(joints, members, baseline_D.to_dict(), not_assessable,
                     OUT_DIR, "02_member_damage_maps_baseline_25yr")
    print("wrote baseline map")

    # --- Maps 1-5: cumulative D at each corrosion year step ---
    cumulative = pd.Series(0.0, index=step_sums[CORROSION_YEARS[0]].index)
    for y in CORROSION_YEARS:
        cumulative = cumulative.add(step_sums[y], fill_value=0.0)
        year_D = {}
        for mid, base_d in baseline_D.items():
            if mid in splash_ids:
                year_D[mid] = cumulative.get(mid, base_d)
            else:
                year_D[mid] = base_d * (y / 25.0)  # linear-in-time assumption, see docstring
        jfp.plot_4faces(joints, members, year_D, not_assessable,
                         OUT_DIR, f"0{2+CORROSION_YEARS.index(y)+1}_member_damage_maps_y{y}")
        print(f"wrote year {y} map")


if __name__ == "__main__":
    main()
