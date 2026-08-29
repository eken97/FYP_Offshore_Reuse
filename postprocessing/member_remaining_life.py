"""
Combines the two independent life estimates for each splash-zone member into
one governing "remaining life":

  - fatigue life: solves the fitted power law D(T) = power_A * T^power_p
    (member_power_law_fit.py) for the year T at which D = 1
  - static life: the year at which the corroded section first fails
    tension/compression/buckling (member_static_life_check.py)

  governing_life_years = min(fatigue_life_years, static_life_years)

Runs both underlying scripts fresh (so this is always up to date with the
current force CSV / damage trajectories), then joins and compares.

Output: results/real_campaign/member_remaining_life.csv
"""
from pathlib import Path

import pandas as pd

import member_power_law_fit
import member_static_life_check
import sd_geometry as sg

PROJECT = Path(__file__).resolve().parents[1]   # repo root
OUT_PATH = PROJECT / "results" / "member_remaining_life.csv"
TRAJECTORY_OUT_PATH = PROJECT / "results" / "member_remaining_life_trajectory.csv"

# Common year range/resolution for the per-year trajectory, shared across all
# members so they can be plotted/faceted together later. Covers every
# member's static cutoff (max ~120yr) with headroom to also show a bit of
# the "would-be" fatigue curve past that point, same idea as the dotted
# past-cutoff segment in the schematic figure.
TRAJECTORY_MAX_YEAR = 150
TRAJECTORY_STEP = 1.0


def fatigue_life_years(power_A, power_p):
    """Solve power_A * T^power_p = 1 for T."""
    if pd.isna(power_A) or pd.isna(power_p) or power_p == 0:
        return float("nan")
    return (1.0 / power_A) ** (1.0 / power_p)


def build_trajectory(combined):
    """Long-format per-year table: for every splash member, the fatigue
    damage D(T) from the fitted power law and the corroded wall thickness,
    at each year T from 0 to TRAJECTORY_MAX_YEAR. `past_static_cutoff` flags
    years beyond that member's static failure year -- the fatigue number is
    still computed there (so the curve can be drawn dotted/invalid past the
    cutoff, same convention as the schematic figure), but is not physically
    meaningful past that point."""
    model = sg.read_subdyn_model(sg.DEFAULT_SD_PATH)
    all_years = [y * TRAJECTORY_STEP for y in range(int(TRAJECTORY_MAX_YEAR / TRAJECTORY_STEP) + 1)]

    rows = []
    for _, r in combined.iterrows():
        mid = int(r["member_id"])
        power_A, power_p = r["power_A"], r["power_p"]
        static_life = r["static_life_years"]

        # Cap at 99% of full wall-thickness consumption for this member's
        # class -- legs (0.60 mm/yr total, both surfaces at the general
        # rate) run out well before TRAJECTORY_MAX_YEAR; corroded_section()
        # asserts t>0 so this must be respected per-member, not with one
        # shared horizon. General (not fatigue-halved) rate here so the
        # trajectory is consistent with the static cutoff it exists to
        # visualise (see member_static_life_check.py).
        _D0, t0, _pid = sg.member_section(model, mid)
        rate = sg.GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE[r["member_class"]]
        total_rate_mm_per_yr = rate["ext"] + rate["int"]
        full_consumption_year = (t0 * 1000.0) / total_rate_mm_per_yr
        member_max_year = min(TRAJECTORY_MAX_YEAR, 0.99 * full_consumption_year)
        years = [y for y in all_years if y <= member_max_year]

        for year in years:
            D_fatigue = power_A * year**power_p if year > 0 else 0.0
            # t_mm tracks the static-check section (general 0.30 rate);
            # D_fatigue is the fatigue power-law curve and is unaffected.
            _D_m, t_m = sg.corroded_section(
                model, mid, year,
                rates=sg.GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE,
            )
            rows.append(dict(
                member_id=mid,
                member_class=r["member_class"],
                year=year,
                D_fatigue=D_fatigue,
                t_mm=t_m * 1000.0,
                past_static_cutoff=(not pd.isna(static_life)) and year > static_life,
            ))

    traj = pd.DataFrame(rows)
    TRAJECTORY_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    traj.to_csv(TRAJECTORY_OUT_PATH, index=False)
    print(f"Wrote {len(traj)} rows ({traj['member_id'].nunique()} members x "
          f"{len(years)} years) to {TRAJECTORY_OUT_PATH}")
    return traj


def main():
    print("--- running member_power_law_fit ---")
    fatigue_df = member_power_law_fit.main()
    if fatigue_df is None:
        fatigue_df = pd.read_csv(member_power_law_fit.OUT_PATH)

    print("\n--- running member_static_life_check ---")
    member_static_life_check.main()
    static_df = pd.read_csv(member_static_life_check.OUT_CSV)

    fatigue_df = fatigue_df[["member_id", "member_class", "power_p", "power_A", "power_R2"]].copy()
    fatigue_df["fatigue_life_years"] = fatigue_df.apply(
        lambda r: fatigue_life_years(r["power_A"], r["power_p"]), axis=1
    )

    static_df = static_df[[
        "member_id", "static_life_years", "governing_mode",
        "t_at_failure_mm", "D_at_failure_mm",
    ]].rename(columns={"governing_mode": "static_governing_mode"})

    combined = fatigue_df.merge(static_df, on="member_id", how="inner")

    def combine_row(r):
        fat = r["fatigue_life_years"]
        stat = r["static_life_years"]
        if pd.isna(stat):
            return pd.Series([fat, "fatigue"])
        if fat <= stat:
            return pd.Series([fat, "fatigue"])
        return pd.Series([stat, f"static:{r['static_governing_mode']}"])

    combined[["governing_life_years", "governing_mechanism"]] = combined.apply(combine_row, axis=1)

    combined = combined.sort_values("member_id")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nWrote {len(combined)} rows to {OUT_PATH}")
    print(combined[[
        "member_id", "member_class", "fatigue_life_years", "static_life_years",
        "governing_life_years", "governing_mechanism",
    ]].to_string(index=False))
    print("\ngoverning mechanism counts:")
    print(combined["governing_mechanism"].value_counts())

    print("\n--- building per-year fatigue-damage trajectory ---")
    build_trajectory(combined)


if __name__ == "__main__":
    main()
