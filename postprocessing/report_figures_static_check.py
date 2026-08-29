"""
Results-section figures for the static (EC3 + DNV-ST-0126) member life check --
see docs/decisions.md. Two figures, both sourced directly from
member_static_life_check.py / member_remaining_life.py's real output, nothing
illustrative:

1. static_vs_fatigue_life_dotplot -- paired dot plot, one row per splash
   member, static life vs fatigue-extrapolated life. Carries the headline
   "static governs for all 32/32 splash members" claim.
2. damage_trajectory_static_cutoff -- real per-year D_fatigue(T) trajectory
   for the closest-call member in each class (leg: 30, brace: 79), with the
   static-failure year marked and the (never-reached, off-chart) fatigue
   D=1 year annotated -- the real-data version of the earlier illustrative
   schematic_damage_growth_zones figure.

Run directly to regenerate both:
    python report_figures_static_check.py

Output: figures/static_check/final/ (PNG + SVG each).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

POSTPRO_DIR = Path(__file__).resolve().parent
PROJECT = POSTPRO_DIR.parent   # repo root
sys.path.insert(0, str(POSTPRO_DIR))
import fatigue_style as fs  # noqa: E402

RESULTS_DIR = PROJECT / "results"
OUT_DIR = PROJECT / "figures" / "static_check" / "final"


def load_remaining_life():
    return pd.read_csv(RESULTS_DIR / "member_remaining_life.csv")


def load_trajectory():
    return pd.read_csv(RESULTS_DIR / "member_remaining_life_trajectory.csv")


# ---------------------------------------------------------------------------
# Figure 1 -- paired dot plot, static vs fatigue life, all 32 splash members
# ---------------------------------------------------------------------------

def fig_static_vs_fatigue_dotplot(df):
    class_order = df["member_class"].map({"leg": 0, "brace": 1})
    df = (df.assign(_class_order=class_order)
            .sort_values(["_class_order", "static_life_years"])
            .drop(columns="_class_order")
            .reset_index(drop=True))
    n = len(df)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=fs.usable_figsize(width_frac=1.0, aspect=0.8))

    # Shade the brace-group columns so the two classes read as visually
    # separate blocks without needing a second axis or a legend entry.
    n_leg = int((df["member_class"] == "leg").sum())
    ax.axvspan(n_leg - 0.5, n - 0.5, color=fs.GRIDLINE, alpha=0.35, zorder=0)

    for xi, (_, row) in zip(x, df.iterrows()):
        ax.plot([xi, xi], [row.static_life_years, row.fatigue_life_years],
                color=fs.BASELINE, linewidth=0.9, zorder=1)
    ax.scatter(x, df.static_life_years, s=22, color=fs.CAT_BLUE,
               edgecolor=fs.MARKER_EDGE, linewidth=fs.MARKER_EDGE_WIDTH, zorder=3,
               label="Static life (governs)")
    ax.scatter(x, df.fatigue_life_years, s=22, color=fs.CAT_ORANGE,
               edgecolor=fs.MARKER_EDGE, linewidth=fs.MARKER_EDGE_WIDTH, zorder=3,
               label="Fatigue life (power-law extrapolation)")

    ax.axhline(25, color=fs.STATUS_GOOD, linestyle="--", linewidth=1.0, zorder=2,
               label="25 yr design life")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r.member_id)}" for _, r in df.iterrows()],
                        fontsize=6, rotation=90)
    ax.set_xlim(-0.5, n - 0.5)

    # Group labels along the top, centred over each block.
    ax.text((n_leg - 1) / 2, 1.02, f"Legs (n={n_leg})", transform=ax.get_xaxis_transform(),
            va="bottom", ha="center", fontsize=7, color=fs.INK_SECONDARY)
    ax.text(n_leg + (n - n_leg - 1) / 2, 1.02, f"Braces (n={n - n_leg})",
            transform=ax.get_xaxis_transform(), va="bottom", ha="center",
            fontsize=7, color=fs.INK_SECONDARY)

    ax.set_ylabel("Life (years)", color=fs.INK_SECONDARY)
    ax.set_xlabel("Member ID", color=fs.INK_SECONDARY)
    ax.tick_params(colors=fs.INK_SECONDARY)
    ax.grid(axis="y", which="major", color=fs.GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(fs.BASELINE)
    ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.tight_layout()
    fs.save_fig(fig, OUT_DIR, "static_vs_fatigue_life_dotplot")


# ---------------------------------------------------------------------------
# Figure 2 -- real damage trajectory with the static-failure cutoff marked,
# worst-case (closest-call) member per class.
# ---------------------------------------------------------------------------

def fig_trajectory_static_cutoff(traj, life):
    targets = [
        ("leg", 30, fs.CAT_BLUE),
        ("brace", 79, fs.CAT_ORANGE),
    ]
    fig, axes = plt.subplots(1, 2, figsize=fs.usable_figsize(width_frac=1.0, aspect=0.5))

    for ax, (cls, mid, color) in zip(axes, targets):
        sub = traj[traj.member_id == mid].sort_values("year")
        row = life[life.member_id == mid].iloc[0]
        static_yr = row.static_life_years
        fatigue_yr = row.fatigue_life_years

        before = sub[sub.year <= static_yr]
        after = sub[sub.year >= static_yr]
        ax.plot(before.year, before.D_fatigue, color=color, linewidth=1.4, zorder=3,
                label="Cumulative Damage")
        ax.plot(after.year, after.D_fatigue, color=color, linewidth=1.1, linestyle=":",
                alpha=0.7, zorder=2, label="Cumulative Damage - hypothetical")

        label_bbox = dict(facecolor=fs.SURFACE, edgecolor="none", alpha=0.85, pad=2.0)

        ax.axvline(static_yr, color=fs.INK_PRIMARY, linewidth=1.0, linestyle="--", zorder=2)
        ax.annotate(f"Static failure\nT = {static_yr:.0f} yr", xy=(static_yr, 0),
                    xytext=(static_yr - sub.year.max() * 0.04, sub.D_fatigue.max() * 1.15),
                    ha="right", va="top", fontsize=6.5, color=fs.INK_SECONDARY, bbox=label_bbox)

        y_top = sub.D_fatigue.max() * 1.35
        ax.set_ylim(0, y_top)
        ax.set_xlim(0, sub.year.max())
        ax.annotate(f"Fatigue D=1 not reached\nuntil T = {fatigue_yr:.0f} yr (off-chart)",
                    xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                    fontsize=6.5, color=fs.INK_SECONDARY, style="italic", bbox=label_bbox)

        ax.set_xlabel("Year", color=fs.INK_SECONDARY)
        ax.set_ylabel("D", color=fs.INK_SECONDARY)
        ax.set_title(f"Member {mid}", fontsize=8, color=fs.INK_PRIMARY, loc="left")
        ax.tick_params(colors=fs.INK_SECONDARY)
        ax.grid(axis="y", which="major", color=fs.GRIDLINE, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines["left"].set_color(fs.BASELINE)
        ax.legend(fontsize=6.5, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.16), ncol=1)

    fig.tight_layout()
    fs.save_fig(fig, OUT_DIR, "damage_trajectory_static_cutoff")


def main():
    fs.apply_style()
    life = load_remaining_life()
    traj = load_trajectory()

    fig_static_vs_fatigue_dotplot(life)
    fig_trajectory_static_cutoff(traj, life)
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
