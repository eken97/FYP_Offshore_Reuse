"""
Fit a power law D(T) = A * T^p to each splash-zone member's damage trajectory
(D_5..D_25 from final_results_member.csv), in log-log space (matches Excel's
Power trendline: regression of ln(D) vs ln(T)). T=0 excluded (D=0 -> ln undefined).

Output: results/member_power_law_fit.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import sd_geometry

PROJECT = Path(__file__).resolve().parents[1]   # repo root
IN_PATH = PROJECT / "results" / "final_results_member.csv"
OUT_PATH = PROJECT / "results" / "member_power_law_fit.csv"

YEARS = [5, 10, 15, 20, 25]


def fit_power_law(years, damages):
    years = np.asarray(years, dtype=float)
    damages = np.asarray(damages, dtype=float)
    mask = damages > 0
    years, damages = years[mask], damages[mask]
    if len(years) < 2:
        return np.nan, np.nan, np.nan, mask.sum()

    x = np.log(years)
    y = np.log(damages)
    p, ln_a = np.polyfit(x, y, 1)
    a = np.exp(ln_a)

    y_pred = p * x + ln_a
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return p, a, r2, mask.sum()


def main():
    df = pd.read_csv(IN_PATH)
    splash = df[df["zone"] == "splash"].copy()

    model = sd_geometry.read_subdyn_model(sd_geometry.DEFAULT_SD_PATH)

    rows = []
    for _, row in splash.iterrows():
        mid = row["member_id"]
        damages = [row[f"D_{y}"] for y in YEARS]
        p, a, r2, n_pts = fit_power_law(YEARS, damages)
        rows.append({
            "member_id": mid,
            "zone": row["zone"],
            "member_class": sd_geometry.member_class(model, mid),
            "power_p": p,
            "power_A": a,
            "power_R2": r2,
            "n_points_fitted": n_pts,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out.describe(include="all"))


if __name__ == "__main__":
    main()
