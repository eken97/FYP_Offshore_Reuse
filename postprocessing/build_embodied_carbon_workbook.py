"""
Builds an Excel workbook joining per-member reuse classification results
(stage4_reuse_classification.py output) with member geometry (length, D, t,
density, mass) so the embodied-carbon / carbon-credit maths can be done by
hand in Excel. This script does NOT compute any carbon numbers -- no
emission factor has been sourced/agreed yet (see docs/decisions.md). It only assembles the input data.

Length is the FULL joint-to-joint member length (no can cut-off applied --
see note in the "Length" sheet). Cutting the joint cans off each end is left
to the author.
"""
import math
from pathlib import Path

import pandas as pd

import sd_geometry as sg

RESULTS_DIR = Path(__file__).parent / "results" / "real_campaign"
CLASSIFICATION_CSV = RESULTS_DIR / "reuse_classification.csv"
OUT_XLSX = RESULTS_DIR / "embodied_carbon_inputs.xlsx"


def build_member_geometry(model):
    rows = []
    for mid, m in model["members"].items():
        D, t, propset = sg.member_section(model, mid)
        rho = model["circ_props"][propset]["rho"]
        j1, j2 = m["j1"], m["j2"]
        x1, y1, z1 = model["joints"][j1]
        x2, y2, z2 = model["joints"][j2]
        length_m = math.dist((x1, y1, z1), (x2, y2, z2))
        area_m2 = math.pi * t * (D - t)
        mass_kg = area_m2 * length_m * rho
        rows.append(dict(
            member_id=mid,
            j1=j1, j2=j2,
            z1_m=z1, z2_m=z2,
            D_m=D, t_m=t, rho_kgm3=rho,
            length_m=length_m,
            area_m2=area_m2,
            mass_kg=mass_kg,
        ))
    return pd.DataFrame(rows).sort_values("member_id")


def main():
    model = sg.read_subdyn_model(sg.DEFAULT_SD_PATH)
    geom_df = build_member_geometry(model)
    reuse_df = pd.read_csv(CLASSIFICATION_CSV)

    merged = reuse_df.merge(geom_df, on="member_id", how="left", validate="one_to_one")

    front_cols = [
        "member_id", "zone", "member_class",
        "length_m", "D_m", "t_m", "rho_kgm3", "area_m2", "mass_kg",
        "reuse_level_A", "reuse_category_A",
        "reuse_level_B", "reuse_category_B",
    ]
    other_cols = [c for c in merged.columns if c not in front_cols]
    merged = merged[front_cols + other_cols]

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="members_full", index=False)

        summary_cols = [
            "member_id", "zone", "member_class", "length_m", "D_m", "t_m", "mass_kg",
            "reuse_level_A", "reuse_category_A",
            "reuse_level_B", "reuse_category_B",
        ]
        merged[summary_cols].to_excel(writer, sheet_name="carbon_input", index=False)

        notes = pd.DataFrame({
            "Note": [
                "length_m is the FULL joint-to-joint member length (SubDyn node-to-node "
                "distance). It does NOT subtract joint-can material at either end -- "
                "cutting the can off is left to you, per your own request.",
                "Members ending at a boundary condition (mudline/pile transition) rather "
                "than a second real joint only lose a can at ONE end, not two -- check "
                "j1/j2 against model['reaction_joints']/['interface_joints'] if you need this.",
                "mass_kg = area_m2 x length_m x rho_kgm3, using each member's own PropSet "
                "density (not a single assumed steel density).",
                "reuse_level/category_A and _B are TWO INDEPENDENT verdicts (Retrofit A vs "
                "Retrofit B) -- do not AND them together. See stage4_reuse_classification.py "
                "and docs/reuse-criteria.md for the full L0-L4 definitions.",
                "No emission factor, credit basis (vs. primary vs. recycled steel), or "
                "carbon-accounting boundary has been chosen yet -- that is the open "
                "methodology decision this workbook is feeding into.",
            ]
        })
        notes.to_excel(writer, sheet_name="notes", index=False)

    print(f"Wrote {OUT_XLSX} ({len(merged)} members)")


if __name__ == "__main__":
    main()
