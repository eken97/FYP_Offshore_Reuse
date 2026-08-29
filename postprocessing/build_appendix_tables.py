"""
Appendix tables -- full per-environment-bin numeric backup to the A5c
(member-track) / B_damage_by_environment_facets (joint-track) heatmap
figures. The figures are colour-only (no cell text, per the 19.08.2026 call
-- "I think it's fine to just read it off the colour... report all numbers
as tables in the appendices"); this script produces the numbers.

One row per one of the 69 real campaign bins. Columns: bin name, Vw/Hs/Tp
(so a reader can still cross-reference a row against the heatmap), the
bin's own occurrence probability (straight from OC4_Final_Bins.xlsx, not
re-derived), the RAW (unweighted) D contribution summed over all
members/joints, the probability-WEIGHTED D contribution summed over all
members/joints, and each contribution's % share of that track's own grand
total. Sorted by weighted contribution descending -- ranks bins by what
actually eats fatigue life first, matching the figures' primary/original
variant.

Joint rows are pre-filtered to one row per physical joint hotspot before
summing (see joint_results_final._one_row_per_joint) -- the raw matrix
otherwise double-counts K-family joints (K/Y treatment duality) and
X-family joints (A_as_chord/B_as_chord duality).
"""
import sys
from pathlib import Path

import pandas as pd

POSTPRO_DIR = Path(__file__).resolve().parent
PROJECT = POSTPRO_DIR.parent   # repo root
sys.path.insert(0, str(POSTPRO_DIR))
import stage3_damage as s3d  # noqa: E402
import fatigue_style as fs  # noqa: E402
from joint_results_final import _one_row_per_joint, NON_LC_JOINT  # noqa: E402

RESULTS_DIR = PROJECT / "results"
OUT_DIR = PROJECT / "figures" / "appendix_tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NON_LC_MEMBER = {"member_id", "zone", "worst_bin", "worst_bin_contribution"}


def _bin_environment_table():
    """{LC name -> (Vw, Hs, Tp, probability)}, keyed off the campaign's own
    bin sheet (stage3_damage.load_bin_probabilities/load_bin_names) so the
    probability column is exact, not re-derived from a matrix CSV."""
    probs, _raw_sum = s3d.load_bin_probabilities()
    names = s3d.load_bin_names()
    out = {}
    for idx, name in names.items():
        parsed = fs.parse_lc_name(name)
        if parsed is None:
            continue
        v, h, t = parsed
        out[name] = (v, h, t, probs[idx])
    return out


def _build_table(raw_csv, weighted_csv, non_lc_cols, filter_fn=None):
    raw = pd.read_csv(raw_csv)
    weighted = pd.read_csv(weighted_csv)
    if filter_fn is not None:
        raw, weighted = filter_fn(raw), filter_fn(weighted)
    lc_cols_raw = [c for c in raw.columns if c not in non_lc_cols]
    lc_cols_w = [c for c in weighted.columns if c not in non_lc_cols]
    raw_sum = raw[lc_cols_raw].sum(axis=0)
    w_sum = weighted[lc_cols_w].sum(axis=0)

    env = _bin_environment_table()
    rows = []
    for name in w_sum.index:
        if name not in env:
            continue
        v, h, t, p = env[name]
        rows.append({
            "bin": name, "Vw_m_s": v, "Hs_m": h, "Tp_s": t, "probability": p,
            "D_raw": raw_sum.get(name, float("nan")),
            "D_weighted": w_sum[name],
        })
    table = pd.DataFrame(rows)
    table["D_weighted_pct"] = 100 * table["D_weighted"] / table["D_weighted"].sum()
    table["D_raw_pct"] = 100 * table["D_raw"] / table["D_raw"].sum()
    table = table.sort_values("D_weighted", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    return table


def build_member_table():
    table = _build_table(
        RESULTS_DIR / "member_track" / "member_damage_matrix_raw.csv",
        RESULTS_DIR / "member_track" / "member_damage_matrix_weighted.csv",
        NON_LC_MEMBER,
    )
    out_path = OUT_DIR / "member_damage_by_environment.csv"
    table.to_csv(out_path, index=False)
    return table, out_path


def build_joint_table():
    table = _build_table(
        RESULTS_DIR / "joint_track" / "joint_damage_matrix_raw.csv",
        RESULTS_DIR / "joint_track" / "joint_damage_matrix_weighted.csv",
        NON_LC_JOINT,
        filter_fn=_one_row_per_joint,
    )
    out_path = OUT_DIR / "joint_damage_by_environment.csv"
    table.to_csv(out_path, index=False)
    return table, out_path


if __name__ == "__main__":
    m_table, m_path = build_member_table()
    j_table, j_path = build_joint_table()

    assert len(m_table) == 69, f"expected 69 member bins, got {len(m_table)}"
    assert len(j_table) == 69, f"expected 69 joint bins, got {len(j_table)}"
    assert abs(m_table["probability"].sum() - 1.0) < 0.01, m_table["probability"].sum()
    assert abs(j_table["probability"].sum() - 1.0) < 0.01, j_table["probability"].sum()
    assert abs(m_table["D_weighted_pct"].sum() - 100) < 1e-6
    assert abs(j_table["D_weighted_pct"].sum() - 100) < 1e-6

    pd.set_option("display.width", 160)
    print(f"Member table -> {m_path} ({len(m_table)} rows)")
    print(m_table.head(10).to_string(index=False))
    print(f"\nJoint table -> {j_path} ({len(j_table)} rows)")
    print(j_table.head(10).to_string(index=False))
    print("\nOK: self-check passed (69 bins each, probabilities sum to ~1.0, "
          "weighted pct sums to 100)")
