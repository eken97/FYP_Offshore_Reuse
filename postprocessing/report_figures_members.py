"""
CANONICAL script for the member-track results figures -- produces exactly
the 11 figures + 1 summary table chosen for the report, nothing else.
Trimmed from a broader iteration script that also held several rejected
variants (a ranked-bar chart, an all-112-member spaghetti plot, a jittered
scatter) -- none of those are reproduced here; this file is deliberately
the approved subset only.

Run directly to regenerate every report figure from scratch:
    python report_figures_members.py

Source of truth: `results/real_campaign/final_results_member.csv` (the
corrected, final per-load-case-worst-summed convention -- see
docs/decisions.md memory). M1 = D_0 (no corrosion, full 25yr).
M2 = D_5..D_25 (cumulative corrosion, 5yr steps).

Output: figures/member_track/final/ (PNG + SVG each).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

POSTPRO_DIR = Path(__file__).resolve().parent
PROJECT = POSTPRO_DIR.parent   # repo root
sys.path.insert(0, str(POSTPRO_DIR))
import sd_geometry as sdg  # noqa: E402
import fatigue_style as fs  # noqa: E402
import jacket_face_plot as jfp  # noqa: E402


RESULTS_DIR = PROJECT / "results"
OUT_DIR = PROJECT / "figures" / "member_track" / "final"

CORROSION_YEARS = [5, 10, 15, 20, 25]
NOT_ASSESSABLE = set(range(101, 113))  # confirmed against stage3_damage.csv
NON_LC_MEMBER = {"member_id", "zone", "worst_bin", "worst_bin_contribution"}


def load_final():
    return pd.read_csv(RESULTS_DIR / "final_results_member.csv").set_index("member_id")


def assessable(df):
    return df[~df.index.isin(NOT_ASSESSABLE)]


def member_trajectory(df, mid, years):
    """Cumulative D at each year, starting genuinely at 0 (year 0 = no time
    elapsed yet -- NOT the D_0 column, which means '0 years of corrosion,
    full 25yr no-corrosion total'). Splash members use the real cumulative
    D_5..D_25 columns; non-splash members have no intermediate data
    (corrosion isn't modelled there), so a linear 0->D_0 approximation is
    used instead."""
    is_splash = df.loc[mid, "zone"] == "splash"
    d0 = df.loc[mid, "D_0"]
    out = [0.0]
    for y in years[1:]:
        out.append(df.loc[mid, f"D_{y}"] if is_splash else d0 * (y / 25.0))
    return out


# ---------------------------------------------------------------------------
# A1 -- baseline (M1) jacket map
# ---------------------------------------------------------------------------

def fig_A1_member_map(df, joints, members):
    a = assessable(df)
    vmax = a["D_0"].max()
    jfp.plot_4faces(joints, members, df["D_0"].to_dict(), NOT_ASSESSABLE,
                     OUT_DIR, "A1_member_map_baseline", scale="linear", vmax=vmax)


# ---------------------------------------------------------------------------
# A2 -- corrosion-year jacket map grid (M2)
# ---------------------------------------------------------------------------

def fig_A2_member_map_grid(df, joints, members):
    """5 corrosion-year snapshots (5/10/15/20/25) -- baseline dropped since
    A1 already establishes the no-corrosion case."""
    a = assessable(df)
    vmax = a["D_25"].max()
    datasets = [(f"Year {y}", df[f"D_{y}"].to_dict()) for y in CORROSION_YEARS]
    jfp.plot_4faces_grid(joints, members, datasets, NOT_ASSESSABLE,
                          OUT_DIR, "A2_member_map_grid", scale="linear", vmax=vmax,
                          target_height_frac=0.9)


# ---------------------------------------------------------------------------
# A3 -- top-3 worst-member trajectories
# ---------------------------------------------------------------------------

def fig_A3_worst_member_trajectories(df):
    a = assessable(df).sort_values("D_25", ascending=False)
    worst_ids = a.head(3).index.tolist()
    years = [0] + CORROSION_YEARS

    fig, ax = plt.subplots(figsize=fs.usable_figsize(width_frac=0.62, aspect=0.7))
    all_vals = []
    for i, mid in enumerate(worst_ids):
        color = fs.CATEGORICAL[i]
        corroded = member_trajectory(df, mid, years)
        d0 = df.loc[mid, "D_0"]
        no_corrosion = [d0 * (y / 25.0) for y in years]
        all_vals.extend(corroded)
        all_vals.extend(no_corrosion)
        ax.plot(years, corroded, color=color, linewidth=1.4,
                label=f"Member {mid} (with corrosion)")
        ax.plot(years, no_corrosion, color=color, linestyle="--", linewidth=1.0, alpha=0.6,
                label=f"Member {mid} (no-corrosion, linear)")
    # D=1 (failure) is ~9x above anything plotted here -- capping ylim to the
    # real data range instead of autoscaling to include a D=1 reference line.
    ax.set_ylim(0, max(all_vals) * 1.15)
    ax.set_xlabel("Year", color=fs.INK_SECONDARY)
    ax.set_ylabel("D (cumulative)", color=fs.INK_SECONDARY)
    ax.tick_params(colors=fs.INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper left")
    fig.tight_layout()
    fs.save_fig(fig, OUT_DIR, "A3_worst_member_trajectories")


# ---------------------------------------------------------------------------
# A3c -- splash-only corrosion vs. no-corrosion spaghetti
# ---------------------------------------------------------------------------

def fig_A3c_splash_corrosion_vs_none(df):
    """Splash-zone members only (n=32, the only members where corrosion is
    modelled) -- every member gets two thin lines: solid = real cumulative
    trajectory with corrosion, dashed = the same member's linear
    no-corrosion projection. One legend entry per family, not per member."""
    splash = assessable(df)
    splash = splash[splash["zone"] == "splash"]
    years = [0] + CORROSION_YEARS

    fig, ax = plt.subplots(figsize=fs.usable_figsize(width_frac=0.75, aspect=0.7))
    labeled = {"corroded": False, "none": False}
    for mid in splash.index:
        d0 = df.loc[mid, "D_0"]
        corroded = member_trajectory(df, mid, years)
        no_corrosion = [d0 * (y / 25.0) for y in years]
        ax.plot(years, corroded, color=fs.CAT_ORANGE, linewidth=0.8, alpha=0.55, zorder=2,
                label="With corrosion" if not labeled["corroded"] else None)
        ax.plot(years, no_corrosion, color=fs.CAT_BLUE, linewidth=0.8, alpha=0.45,
                linestyle="--", zorder=1,
                label="No corrosion (linear)" if not labeled["none"] else None)
        labeled["corroded"] = labeled["none"] = True
    ax.set_xlabel("Year", color=fs.INK_SECONDARY)
    ax.set_ylabel(f"D (cumulative) -- splash-zone members only (n={len(splash)})",
                  color=fs.INK_SECONDARY)
    ax.tick_params(colors=fs.INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(fs.BASELINE)
    leg = ax.legend(fontsize=7, frameon=False, loc="upper left")
    for line in leg.get_lines():
        line.set_linewidth(2.0)
        line.set_alpha(1.0)
    fig.tight_layout()
    fs.save_fig(fig, OUT_DIR, "A3c_splash_corrosion_vs_none")


# ---------------------------------------------------------------------------
# A6 -- zone breakdown (boxplot + strip), M1 and M2, log and linear
# ---------------------------------------------------------------------------

def _boxplot_with_strip(ax, groups, values_by_group, colors, ylabel, seed=0, log=True, ylim=None):
    data_list = [values_by_group[g] for g in groups]
    positions = list(range(len(groups)))
    box_colors = [colors[g] for g in groups]
    fs.styled_boxplot(ax, data_list, positions, box_colors, width=0.5)
    rng = np.random.default_rng(seed)
    for i, g in enumerate(groups):
        vals = values_by_group[g]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[g], s=12, alpha=0.55,
                   edgecolor=fs.MARKER_EDGE, linewidth=0.3, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7, rotation=30, ha="right")
    if log:
        ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        fs.add_log_gridlines(ax, subs=range(2, 10))
    else:
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", which="major", color=fs.GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
    ax.set_ylabel(ylabel, color=fs.INK_SECONDARY)
    ax.tick_params(colors=fs.INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(fs.BASELINE)


def fig_A6_zone_breakdown(df, scale="log", value_col="D_0", suffix="", short_label="M1", ylim=None):
    """width_frac=0.47 so two of these sit side by side on a page. ylim:
    pass a shared (0, max) so the M1/M2 linear pair are directly comparable."""
    a = assessable(df)
    zones = ["Atmospheric", "Submerged", "Splash"]
    zone_key = {"Atmospheric": "atmospheric", "Submerged": "submerged", "Splash": "splash"}
    colors = dict(zip(zones, fs.CATEGORICAL))
    values_by_group = {z: a.loc[a["zone"] == zone_key[z], value_col].values for z in zones}
    fig, ax = plt.subplots(figsize=fs.usable_figsize(width_frac=0.47, aspect=0.95))
    _boxplot_with_strip(ax, zones, values_by_group, colors, f"D ({short_label})",
                         log=(scale == "log"), ylim=ylim)
    fig.tight_layout()
    scale_tag = "" if scale == "log" else "_LINEAR"
    fs.save_fig(fig, OUT_DIR, f"A6_zone_breakdown{suffix}{scale_tag}")


# ---------------------------------------------------------------------------
# A5c -- damage-by-environment, faceted by Tp
# ---------------------------------------------------------------------------

def _facet_heatmap_by_tp(value_by_name, cbar_label, out_name):
    """Shared engine behind A5c and its two companions: one small heatmap
    panel per Tp value (5/6/7/8s -- the only 4 real values in the campaign),
    square cells, one shared colour scale. `value_by_name`: {LC column name
    -> value}, works identically whether the value is probability-weighted
    damage, raw damage, or plain occurrence probability."""
    rows = []
    for name, val in value_by_name.items():
        parsed = fs.parse_lc_name(name)
        if parsed is None:
            continue
        v, h, t = parsed
        rows.append((v, h, t, val))
    plot_df = pd.DataFrame(rows, columns=["v", "h", "t", "d"])

    VW_STEP, HS_STEP = 2.0, 0.5
    vw_vals = sorted(plot_df["v"].unique())
    hs_vals = sorted(plot_df["h"].unique())
    tp_vals = sorted(plot_df["t"].unique())
    v_bounds = [vw_vals[0] - VW_STEP / 2] + [v + VW_STEP / 2 for v in vw_vals]
    h_bounds = [hs_vals[0] - HS_STEP / 2] + [h + HS_STEP / 2 for h in hs_vals]

    cmap = LinearSegmentedColormap.from_list("seq_blue", fs.SEQUENTIAL_BLUE)
    vmax = plot_df["d"].max()

    fig, axes = plt.subplots(2, 2, figsize=fs.usable_figsize(width_frac=0.85, aspect=0.58),
                              sharex=True, sharey=True, constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.03, wspace=0.03)
    mesh = None
    for ax, tp in zip(axes.flat, tp_vals):
        grid = np.full((len(hs_vals), len(vw_vals)), np.nan)
        sub = plot_df[plot_df["t"] == tp]
        for _, r in sub.iterrows():
            grid[hs_vals.index(r["h"]), vw_vals.index(r["v"])] = r["d"]
        mesh = ax.pcolormesh(v_bounds, h_bounds, grid, cmap=cmap, vmin=0, vmax=vmax,
                              edgecolors=fs.GRIDLINE, linewidth=0.5)
        # Vw cells (2 m/s) and Hs cells (0.5 m) are different physical units
        # -- force the display aspect so each cell renders as a visual square.
        ax.set_aspect(VW_STEP / HS_STEP, adjustable="box")
        ax.set_title(f"Tp = {tp:g} s", fontsize=7, color=fs.INK_PRIMARY, loc="left")
        ax.set_xticks(vw_vals)
        ax.set_yticks(hs_vals)
        ax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supxlabel("Wind speed Vw [m/s]", color=fs.INK_SECONDARY, fontsize=7)
    fig.supylabel("Sig. wave height Hs [m]", color=fs.INK_SECONDARY, fontsize=7)
    cbar = fig.colorbar(mesh, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label(cbar_label, color=fs.INK_SECONDARY, fontsize=7)
    cbar.ax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)

    fs.save_fig(fig, OUT_DIR, out_name, tight=True)


def fig_A5c_damage_by_environment_facets(df):
    """Probability-WEIGHTED damage contribution -- each bin's real share of
    25yr lifetime D (p_bin x block-count x mean damage), summed over all 112
    members. This is what actually eats fatigue life."""
    m = pd.read_csv(RESULTS_DIR / "member_track" / "member_damage_matrix_weighted.csv")
    lc_cols = [c for c in m.columns if c not in NON_LC_MEMBER]
    contribution = m[lc_cols].sum(axis=0)
    _facet_heatmap_by_tp(contribution.to_dict(), "Total D contribution (all members, weighted)",
                         "A5c_damage_by_environment_facets")


def fig_A5c_raw_damage_facets(df):
    """RAW per-bin severity (mean damage per block, NOT multiplied by that
    bin's occurrence probability) -- 'which conditions are physically
    harshest', independent of how often they occur."""
    m = pd.read_csv(RESULTS_DIR / "member_track" / "member_damage_matrix_raw.csv")
    lc_cols = [c for c in m.columns if c not in NON_LC_MEMBER]
    contribution = m[lc_cols].sum(axis=0)
    _facet_heatmap_by_tp(contribution.to_dict(), "Total D contribution (all members, raw/unweighted)",
                         "A5c_raw_damage_facets")


def fig_A5c_probability_facets():
    """Plain occurrence probability per load case (from the campaign's own
    binned scatter table), independent of members/damage entirely."""
    import stage3_damage as s3
    p_bin, _ = s3.load_bin_probabilities()
    names = s3.load_bin_names()
    value_by_name = {names[k]: p_bin[k] for k in p_bin}
    _facet_heatmap_by_tp(value_by_name, "Occurrence probability (per load case)",
                         "A5c_probability_facets")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(df, n=10):
    a = assessable(df).sort_values("D_25", ascending=False).head(n)
    table = a[["zone", "D_0", "life_years_0", "D_25", "life_years_25"]].copy()
    table.insert(0, "member_id", table.index)
    table = table.rename(columns={
        "D_0": "D_M1", "life_years_0": "life_years_M1",
        "D_25": "D_M2", "life_years_25": "life_years_M2",
    })
    out_path = RESULTS_DIR / "member_results_summary_table.csv"
    table.to_csv(out_path, index=False)
    return table, out_path


def self_check(df):
    a = assessable(df)
    assert (a["D_25"] >= a["D_0"] - 1e-12).all(), "corrosion must never DECREASE cumulative damage"
    assert len(df) == 112, f"expected 112 members, got {len(df)}"
    non_splash = a[a["zone"] != "splash"]
    assert np.allclose(non_splash["D_5"], non_splash["D_0"]), "non-splash members should be corrosion-flat"
    print(f"OK: self-check passed ({len(df)} members, {len(a)} assessable)")


if __name__ == "__main__":
    df = load_final()
    self_check(df)

    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    joints, members = model["joints"], model["members"]

    fig_A1_member_map(df, joints, members)
    fig_A2_member_map_grid(df, joints, members)
    fig_A3_worst_member_trajectories(df)
    fig_A3c_splash_corrosion_vs_none(df)
    fig_A5c_damage_by_environment_facets(df)
    fig_A5c_raw_damage_facets(df)
    fig_A5c_probability_facets()
    fig_A6_zone_breakdown(df, scale="log", value_col="D_0", suffix="_M1", short_label="M1")
    fig_A6_zone_breakdown(df, scale="linear", value_col="D_0", suffix="_M1", short_label="M1",
                           ylim=(0, 0.12))
    fig_A6_zone_breakdown(df, scale="log", value_col="D_25", suffix="_M2", short_label="M2")
    fig_A6_zone_breakdown(df, scale="linear", value_col="D_25", suffix="_M2", short_label="M2",
                           ylim=(0, 0.12))

    table, table_path = build_summary_table(df)
    print(f"\nWrote summary table -> {table_path}")
    print(table.to_string(index=False))

    print(f"\nAll {11} report figures written to {OUT_DIR}")
