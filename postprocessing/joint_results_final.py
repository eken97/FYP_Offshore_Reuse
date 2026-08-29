"""
Joint-track results figures -- iteration/candidate script, the joint-track
analogue of the two-script convention documented in docs/decisions.md
(member and static-check each have the same split). Writes every candidate
to figures/joint_track/final_candidates/ for review; report_figures_joints.py
imports the approved subset of functions from here and writes the trimmed
set straight to figures/joint_track/final/. The member-track's own iteration
script was removed from this repository once its canonical figures were
finalised -- this one stays because report_figures_joints.py still imports
from it directly (see docs/pipeline.md).

Source of truth: results/real_campaign/final_results_joint.csv (40 rows, one
per physical joint node -- the correct per-load-case-worst-summed
convention, NOT the gallery prototype's stage3_joint_damage.csv single-
worst-row convention). Columns already use the thesis's Table 14 scenario
numbering (S1-S5) with an explicit -K/-Y treatment suffix, e.g. D_S1-K
(renamed 18.08.2026 in final_results_joint.py itself, see
docs/decisions.md for the scenario-table derivation):
    S1-K/S1-Y   -- baseline, no retrofit, no corrosion
    S2-K/S2-Y   -- Resize A, no corrosion
    S3-K/S3-Y   -- Resize A + corrosion, 25yr cumulative (D_S3-K_25 etc);
                   falls back to the uncorroded S2-K/S2-Y value for the 32
                   non-splash nodes (corrosion is only modelled at the 8
                   splash joints -- same fillna convention as the member
                   track's non-splash D_5..D_25)
    S4-K/S4-Y   -- Resize B, no corrosion
    S5-K/S5-Y   -- Resize B + corrosion, 25yr cumulative, same fillna as S3

"Resize A/B" (not "Retrofit A/B" -- renamed 18.08.2026 by the author: retrofit
implies a repair to something already built, but this is a can-wall-thickness
design choice; "Retrofit" also isn't used anywhere else, so no collision there,
but the report's own "Scenario A/B" would have collided with this script's
unrelated "S1-S5" scenario axis, hence "Resize" not "Scenario").
S2.1/S4.1 (T_cp sensitivity) are discussion-section-only per the author's
explicit call -- not plotted here.

K and Y are treated as two entirely separate figure sets (by the author
18.08.2026): fig_B1_joint_map_grid/fig_B3_family_comparison take an explicit
`treatment` arg ("K" or "Y") and are called once per treatment in __main__,
writing separate `_K`/`_Y` files. Only B0 keeps K and Y side by side.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter

POSTPRO_DIR = Path(__file__).resolve().parent
PROJECT = POSTPRO_DIR.parent   # repo root
sys.path.insert(0, str(POSTPRO_DIR))
import sd_geometry as sdg  # noqa: E402
import fatigue_style as fs  # noqa: E402
import jacket_face_plot as jfp  # noqa: E402


RESULTS_DIR = PROJECT / "results"
OUT_DIR = PROJECT / "figures" / "joint_track" / "final_candidates"
NON_LC_JOINT = {"node", "sub_joint_id", "brace_member", "brace_end", "chord_t_scenario",
                "direction", "treatment", "worst_bin", "worst_bin_contribution"}

SCENARIO_TITLES = {
    "S1": "S1 -- Baseline", "S2": "S2 -- Resize A",
    "S3": "S3 -- Resize A + corrosion, 25yr", "S4": "S4 -- Resize B",
    "S5": "S5 -- Resize B + corrosion, 25yr",
}
SCENARIO_ORDER = ["S1", "S2", "S3", "S4", "S5"]
CHRONIC_NODES = [22, 26, 30, 34, 45, 46, 47, 48]  # K-top / X-upper-mid -- fail in every scenario


def load_final():
    return pd.read_csv(RESULTS_DIR / "final_results_joint.csv").set_index("node")


def build_scenario_columns(df):
    """Returns a DataFrame indexed by node with columns S1_K..S5_Y -- the 5
    reported scenarios x 2 treatments, corrosion scenarios (S3/S5) already
    fallen back to the uncorroded value for the 32 nodes where corrosion
    isn't modelled (matches the member-track's own non-splash convention).
    Source columns already use the S1-K/S1-Y naming (final_results_joint.py,
    renamed 18.08.2026) -- this just drops the hyphen for valid Python
    identifiers/f-string use downstream."""
    out = pd.DataFrame(index=df.index)
    out["S1_K"], out["S1_Y"] = df["D_S1-K"], df["D_S1-Y"]
    out["S2_K"], out["S2_Y"] = df["D_S2-K"], df["D_S2-Y"]
    out["S3_K"] = df["D_S3-K_25"].fillna(df["D_S2-K"])
    out["S3_Y"] = df["D_S3-Y_25"].fillna(df["D_S2-Y"])
    out["S4_K"], out["S4_Y"] = df["D_S4-K"], df["D_S4-Y"]
    out["S5_K"] = df["D_S5-K_25"].fillna(df["D_S4-K"])
    out["S5_Y"] = df["D_S5-Y_25"].fillna(df["D_S4-Y"])
    out["family"] = df["family"]
    return out


# ---------------------------------------------------------------------------
# B0 -- overview table/heatmap, all 40 nodes x 7 scenarios, K/Y blocks side
# by side -- first figure in the joint section: the full node-level picture
# before zooming into any one story. Mirrors the author's own hand-built Excel
# table (final_results_joint_summary.csv, D_S1-K..D_S5-Y incl. the S2.1/S4.1
# T_cp columns), same node order (ascending, not grouped by family) so the
# two are directly comparable. One shared Axes (not two subplots) so the K
# and Y blocks can sit flush against each other -- separate subplots always
# leave a visible gutter even at wspace=0.
# ---------------------------------------------------------------------------

SUMMARY_SCENARIOS = ["S1", "S2", "S2.1", "S3", "S4", "S4.1", "S5"]
SPLASH_TEXT_COLOR = fs.CAT_ORANGE


def _blend(hex_a, hex_b, frac):
    """frac=0 -> hex_a, frac=1 -> hex_b. Used to derive pastel fills / darker
    text tones from the project's own CAT_BLUE/CAT_ORANGE rather than
    introducing a new green/amber scheme just for this one figure."""
    a = tuple(int(hex_a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    b = tuple(int(hex_b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    mixed = tuple(a[k] * (1 - frac) + b[k] * frac for k in range(3))
    return "#" + "".join(f"{int(round(c)):02x}" for c in mixed)


PASS_FILL = _blend(fs.CAT_BLUE, "#ffffff", 0.82)
PASS_TEXT = _blend(fs.CAT_BLUE, "#000000", 0.12)  # close to pure CAT_BLUE -- vivid, not muddied
FAIL_FILL = _blend(fs.CAT_ORANGE, "#ffffff", 0.78)
FAIL_TEXT = _blend(fs.CAT_ORANGE, "#000000", 0.12)


def load_summary():
    return pd.read_csv(RESULTS_DIR / "final_results_joint_summary.csv").set_index("node").sort_index()


def fig_B0_overview_heatmap(summary):
    nodes = summary.index.tolist()
    n = len(nodes)
    n_scen = len(SUMMARY_SCENARIOS)
    NODE_COL_W, FAMILY_COL_W = 1.5, 1.3
    x_K = NODE_COL_W + FAMILY_COL_W
    x_Y = x_K + n_scen
    total_w = x_Y + n_scen

    fig, ax = plt.subplots(figsize=(fs.USABLE_WIDTH_IN, fs.USABLE_HEIGHT_IN * 0.92))

    for i, node in enumerate(nodes):
        y = n - 1 - i
        fam = summary.loc[node, "family"]
        splash = bool(summary.loc[node, "in_splash_zone"])
        ax.text(NODE_COL_W / 2, y + 0.5, f"N{node}", ha="center", va="center", fontsize=6.5,
                fontweight="bold" if splash else "normal",
                color=SPLASH_TEXT_COLOR if splash else fs.INK_PRIMARY)
        ax.text(NODE_COL_W + FAMILY_COL_W / 2, y + 0.5, fam, ha="center", va="center",
                 fontsize=6.5, color=fs.INK_PRIMARY)
        for j, s in enumerate(SUMMARY_SCENARIOS):
            for x0, treatment in [(x_K, "K"), (x_Y, "Y")]:
                val = summary.loc[node, f"D_{s}-{treatment}"]
                fail = val >= jfp.D_FAILURE
                fill = FAIL_FILL if fail else PASS_FILL
                text_color = FAIL_TEXT if fail else PASS_TEXT
                x = x0 + j
                ax.add_patch(Rectangle((x, y), 1, 1, facecolor=fill, edgecolor="white", linewidth=0.5))
                ax.text(x + 0.5, y + 0.5, f"{val:.2f}", ha="center", va="center",
                         fontsize=6.2, fontweight="bold", color=text_color)

    ax.axvline(x_K, color=fs.INK_PRIMARY, linewidth=1.0, zorder=5)
    ax.axvline(x_Y, color=fs.INK_PRIMARY, linewidth=1.0, zorder=5)

    header_y = n + 0.7
    superheader_y = n + 1.7
    ax.text(NODE_COL_W / 2, header_y, "Node", ha="center", fontsize=8.5, fontweight="bold",
            color=fs.INK_PRIMARY)
    ax.text(NODE_COL_W + FAMILY_COL_W / 2, header_y, "Family", ha="center", fontsize=8.5,
             fontweight="bold", color=fs.INK_PRIMARY)
    for j, s in enumerate(SUMMARY_SCENARIOS):
        ax.text(x_K + j + 0.5, header_y, s, ha="center", fontsize=8.5, color=fs.INK_SECONDARY)
        ax.text(x_Y + j + 0.5, header_y, s, ha="center", fontsize=8.5, color=fs.INK_SECONDARY)
    ax.text(x_K + n_scen / 2, superheader_y, "K consideration", ha="center", fontsize=12,
             fontweight="bold", color=fs.INK_PRIMARY)
    ax.text(x_Y + n_scen / 2, superheader_y, "Y consideration", ha="center", fontsize=12,
             fontweight="bold", color=fs.INK_PRIMARY)
    ax.axhline(n + 0.25, xmin=0.0, xmax=1.0, color=fs.INK_PRIMARY, linewidth=1.0, zorder=5)

    ax.set_xlim(0, total_w)
    ax.set_ylim(0, n + 2.3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=PASS_FILL, edgecolor="white", label="D < 1 (pass)"),
        Patch(facecolor=FAIL_FILL, edgecolor="white", label="D >= 1 (fail)"),
        Line2D([0], [0], color=SPLASH_TEXT_COLOR, linewidth=3, label="Splash-zone node (node no.)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.97, bottom=0.05, left=0.01, right=0.99)
    fs.save_fig(fig, OUT_DIR, "B0_joint_overview_heatmap")


# ---------------------------------------------------------------------------
# B1 -- joint face-map grid (S1 baseline vs S5 retrofit-B+corrosion), K-treatment
# ---------------------------------------------------------------------------

RED_LIGHT = _blend(fs.STATUS_CRITICAL, "#ffffff", 0.75)
RED_DARK = _blend(fs.STATUS_CRITICAL, "#000000", 0.15)


def _draw_joint_face_panel(ax, joints, members, node_D, color_fn, face_name, test_fn,
                            coord_idx, show_xlabel, marker_size=22):
    ax.axhline(jfp.SPLASH_ZMIN, color=fs.BASELINE, linewidth=0.7, linestyle=":", zorder=1)
    ax.axhline(jfp.SPLASH_ZMAX, color=fs.BASELINE, linewidth=0.7, linestyle=":", zorder=1)
    ax.axhline(0, color=fs.INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)

    face_ids = {jid for jid, (x, y, z) in joints.items() if test_fn(x, y)}
    face_members = {mid: d for mid, d in members.items()
                     if d["j1"] in face_ids and d["j2"] in face_ids}
    for mid, d in face_members.items():
        p1, p2 = joints[d["j1"]], joints[d["j2"]]
        ax.plot([p1[coord_idx], p2[coord_idx]], [p1[2], p2[2]], color=fs.GRIDLINE,
                 linewidth=1.2, solid_capstyle="butt", zorder=1)

    for jid in sorted(face_ids):
        if jid not in node_D:
            continue
        x, y, z = joints[jid]
        color = color_fn(node_D[jid])
        ax.scatter([x if coord_idx == 0 else y], [z], color=color, s=marker_size, zorder=2,
                   edgecolor=fs.MARKER_EDGE, linewidth=fs.MARKER_EDGE_WIDTH)

    if show_xlabel:
        ax.set_xlabel(face_name, color=fs.INK_PRIMARY, fontsize=8, labelpad=6)
    ax.set_xticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_joint_4faces_grid(joints, members, datasets, out_dir, name, vmax=1.0, fail_vmax=5.0,
                            row_height_in=1.57):
    """datasets: list of (row_label, {node: D}). Two-tone diverging scale, seam
    at D=1 (the failure threshold): pass_cmap (blue) covers [0, vmax], fail_cmap
    (red) covers [1, fail_vmax] -- every failed joint's OWN magnitude now shows
    (darker red = further past the threshold), not a flat red like before.
    fail_vmax caps the red ramp (extend='max' folds anything past it into the
    darkest red) so a few extreme baseline outliers (D up to ~74) don't wash
    out the more informative near-threshold range where retrofit comparisons
    actually matter.
    Colorbar: two horizontal bars, widths proportional to their value range so
    the seam at 1 sits at the true physical break, placed under the grid (not
    a vertical bar eating into page width).
    Sizing: fig_h = row_height_in * n_rows + a FIXED absolute-inch chrome
    budget (colorbar + legend + gaps), NOT a fraction of the page height --
    the original target_height_frac approach fixed the page-height fraction
    but let the chrome (colorbar/legend text, fixed in POINTS) eat a smaller
    absolute slice as fig_h shrank, so a 2-row call (row_height_in unchanged,
    fewer rows) rendered with overlapping colorbar/legend text at small
    fig_h. Absolute-inch chrome keeps the same physical space regardless of
    row count, so this one function works for both the 5-row S1-S5 grid and
    a 2-row S2.1/S4.1 sensitivity grid without retuning."""
    fs.apply_style()
    pass_cmap = LinearSegmentedColormap.from_list("seq_blue", fs.SEQUENTIAL_BLUE)
    pass_norm = Normalize(vmin=0.0, vmax=vmax)
    pass_sm = ScalarMappable(norm=pass_norm, cmap=pass_cmap)
    fail_cmap = LinearSegmentedColormap.from_list("seq_red", [RED_LIGHT, RED_DARK])
    fail_norm = Normalize(vmin=1.0, vmax=fail_vmax)
    fail_sm = ScalarMappable(norm=fail_norm, cmap=fail_cmap)

    def color_fn(d_val):
        if d_val >= 1.0:
            return fail_cmap(fail_norm(min(d_val, fail_vmax)))
        return pass_cmap(pass_norm(max(d_val, 0.0)))

    # fixed ABSOLUTE-INCH chrome budget (see docstring) -- derived from the
    # known-good 5-row, fig_h=9.69in build so this reproduces it exactly at
    # n_rows=5, and stays legible at any other row count.
    CHROME_BOTTOM_IN = 1.65
    CHROME_TOP_IN = 0.19
    GAP_GRID_TO_CBAR_IN = 0.27
    CBAR_H_IN = 0.17
    GAP_CBAR_TO_LEGEND_IN = 0.68

    n_rows = len(datasets)
    fig_w = fs.USABLE_WIDTH_IN
    fig_h = row_height_in * n_rows + CHROME_BOTTOM_IN + CHROME_TOP_IN
    fig, axes = plt.subplots(n_rows, 4, figsize=(fig_w, fig_h), sharey=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, (row_label, node_D) in enumerate(datasets):
        is_bottom = r == n_rows - 1
        for c, (face_name, test_fn, coord_idx) in enumerate(jfp.FACES):
            ax = axes[r, c]
            _draw_joint_face_panel(ax, joints, members, node_D, color_fn, face_name,
                                    test_fn, coord_idx, show_xlabel=is_bottom)
            ax.tick_params(axis="y", left=False, labelleft=False)
        jfp._style_yaxis(axes[r, 0], label_fontsize=6, show_splash_ticks=False)
        axes[r, 0].set_title(row_label, loc="left", fontsize=7, color=fs.INK_PRIMARY,
                              fontweight="bold", pad=3)

    bottom_frac = CHROME_BOTTOM_IN / fig_h
    top_frac = 1 - CHROME_TOP_IN / fig_h
    fig.subplots_adjust(wspace=0.06, hspace=0.28, top=top_frac, bottom=bottom_frac,
                         left=0.05, right=0.98)
    grid_bottom = axes[-1, 0].get_position().y0

    # two colorbar segments, widths proportional to their value range so the
    # seam sits at the physically correct spot -- pass covers width `vmax`,
    # fail covers width `fail_vmax - vmax` of the same continuous D axis.
    bar_x0, bar_x1 = 0.30, 0.70
    bar_w = bar_x1 - bar_x0
    total_range = fail_vmax
    pass_w = bar_w * (vmax / total_range)
    fail_w = bar_w * ((fail_vmax - vmax) / total_range)
    cbar_h = CBAR_H_IN / fig_h
    cbar_top = grid_bottom - GAP_GRID_TO_CBAR_IN / fig_h
    cbar_bottom = cbar_top - cbar_h

    pass_cax = fig.add_axes([bar_x0, cbar_bottom, pass_w, cbar_h])
    fail_cax = fig.add_axes([bar_x0 + pass_w, cbar_bottom, fail_w, cbar_h])
    fig.colorbar(pass_sm, cax=pass_cax, orientation="horizontal")
    fig.colorbar(fail_sm, cax=fail_cax, orientation="horizontal", extend="max")
    pass_cax.set_xticks([0, vmax / 2, vmax])
    fail_cax.set_xticks(np.linspace(1, fail_vmax, 3)[1:])
    for cax in (pass_cax, fail_cax):
        cax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)
    fig.text((bar_x0 + bar_x1) / 2, cbar_bottom - 0.24 / fig_h,
              rf"$D$  (blue = pass 0-{vmax:.2g}, red = fail 1-{fail_vmax:.2g}+, seam at $D{{=}}1$)",
              ha="center", va="top", fontsize=7, color=fs.INK_SECONDARY)

    handles = [
        Line2D([0], [0], color=fs.BASELINE, linewidth=0.7, linestyle=":",
               label=f"Splash zone ({jfp.SPLASH_ZMIN:.2f} to {jfp.SPLASH_ZMAX:.2f} m MSL)"),
        Line2D([0], [0], color=fs.INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)), label="MSL"),
    ]
    legend_y = cbar_bottom - GAP_CBAR_TO_LEGEND_IN / fig_h
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, legend_y))
    return fs.save_fig(fig, out_dir, name, tight=True)


def fig_B1_joint_map_grid(S, joints, members, treatment):
    """All 5 scenarios, one row each -- same template as the member track's A2
    (5 corrosion-year rows). Two-tone diverging colour scale (see
    plot_joint_4faces_grid): fail_vmax=5.0 chosen so the near-threshold
    Resize-comparison range gets the visible gradient, extend='max' folds the
    handful of much larger baseline outliers (up to D~74) into the same
    darkest red -- same fixed vmax/fail_vmax for BOTH treatments so K and Y
    versions are directly colour-comparable.
    No "(K-plane)"/"(K consideration)" suffix on the row titles -- there is no
    such thing as a "K-plane", that phrasing conflated the K/Y TREATMENT axis
    with a literal geometric plane; which consideration this whole figure is
    will be stated in the report's own figure caption/name instead."""
    datasets = [(SCENARIO_TITLES[s], S[f"{s}_{treatment}"].to_dict()) for s in SCENARIO_ORDER]
    plot_joint_4faces_grid(joints, members, datasets, OUT_DIR, f"B1_joint_map_grid_{treatment}",
                            vmax=1.0, fail_vmax=5.0)


# ---------------------------------------------------------------------------
# B3 -- family comparison (X / K / TY), boxplot + jittered strip (same combo
# as the member track's A6 zone breakdown), faceted across S1->S5 (same
# faceting pattern as A5c's per-Tp panels) so the retrofit effect on each
# family's whole distribution -- not just its worst point -- is visible in
# one figure. 3+2 grid (not 1x5) so each panel is bigger; the 6th cell (2nd
# row, 3rd column) is left empty.
# ---------------------------------------------------------------------------

def _boxplot_with_strip(ax, groups, values_by_group, colors, ylabel, seed=0):
    data_list = [values_by_group[g] for g in groups]
    positions = list(range(len(groups)))
    box_colors = [colors[g] for g in groups]
    fs.styled_boxplot(ax, data_list, positions, box_colors, width=0.5)
    rng = np.random.default_rng(seed)
    for i, g in enumerate(groups):
        vals = values_by_group[g]
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color=colors[g], s=10, alpha=0.55,
                   edgecolor=fs.MARKER_EDGE, linewidth=0.3, zorder=1)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=7)
    ax.set_ylabel(ylabel, color=fs.INK_SECONDARY)
    ax.tick_params(colors=fs.INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(fs.BASELINE)


def fig_B3_family_comparison(S, treatment):
    families = ["X", "K", "TY"]
    colors = dict(zip(families, fs.CATEGORICAL))

    fig, axes = plt.subplots(2, 3, figsize=fs.usable_figsize(width_frac=0.95, aspect=0.65),
                              sharey=True)
    axes[1, 2].axis("off")
    panel_axes = [axes[0, 0], axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]]
    left_col_axes = {axes[0, 0], axes[1, 0]}

    for i, (ax, s) in enumerate(zip(panel_axes, SCENARIO_ORDER)):
        values_by_group = {fam: S.loc[S["family"] == fam, f"{s}_{treatment}"].values
                            for fam in families}
        _boxplot_with_strip(ax, families, values_by_group, colors,
                             "D" if ax in left_col_axes else "", seed=1)
        ax.set_title(s, loc="left", fontsize=8, color=fs.INK_PRIMARY, fontweight="bold", pad=3)
        ax.set_yscale("log")
        fs.add_log_gridlines(ax, subs=range(2, 10))
        ax.axhline(1.0, color=fs.STATUS_CRITICAL, linewidth=0.9, linestyle=":", zorder=3)
        if ax not in left_col_axes:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

    fig.tight_layout()

    # centre the bottom row (S4/S5) under the full 3-panel row above, instead
    # of leaving it left-aligned in columns 0-1 of the 2x3 grid -- tight_layout
    # must run first (it resets any custom position), so this repositioning
    # happens as a final pass after it.
    pos_top_left = axes[0, 0].get_position()
    pos_top_right = axes[0, 2].get_position()
    pos_bot_0 = axes[1, 0].get_position()
    pos_bot_1 = axes[1, 1].get_position()
    panel_w = pos_bot_0.width
    gap = pos_bot_1.x0 - pos_bot_0.x1
    row_left = pos_top_left.x0 + (pos_top_right.x1 - pos_top_left.x0 - (2 * panel_w + gap)) / 2
    axes[1, 0].set_position([row_left, pos_bot_0.y0, panel_w, pos_bot_0.height])
    axes[1, 1].set_position([row_left + panel_w + gap, pos_bot_1.y0, panel_w, pos_bot_1.height])

    fs.save_fig(fig, OUT_DIR, f"B3_family_comparison_facets_{treatment}")


# ---------------------------------------------------------------------------
# Shared dumbbell helper -- "method A vs method B, per node": a thin
# connecting line plus two dots, node on the y-axis, D on a log x-axis.
# Reused for both the K-vs-Y comparison (B7) and the T_cp sensitivity
# comparison (B8) -- same underlying question ("does changing one
# calculation choice change the answer, and for which nodes"), just a
# different pair of columns and a different node population.
# ---------------------------------------------------------------------------

def _dumbbell_plot(ax, labels, left_vals, right_vals, left_color, right_color,
                    left_label, right_label):
    """Sorts by left_vals descending (worst-under-left-method at top) so the
    story reads top-to-bottom. Returns nothing -- caller sets title/axis."""
    order = np.argsort(left_vals)[::-1]
    labels = [labels[i] for i in order]
    left_vals = np.asarray(left_vals)[order]
    right_vals = np.asarray(right_vals)[order]

    y = np.arange(len(labels))[::-1]  # top row = first/worst label
    for yi, lv, rv in zip(y, left_vals, right_vals):
        ax.plot([lv, rv], [yi, yi], color=fs.GRIDLINE, linewidth=1.2, zorder=1)
    ax.scatter(left_vals, y, color=left_color, s=42, zorder=3, label=left_label,
               edgecolor=fs.MARKER_EDGE, linewidth=fs.MARKER_EDGE_WIDTH)
    ax.scatter(right_vals, y, color=right_color, s=42, zorder=3, label=right_label,
               edgecolor=fs.MARKER_EDGE, linewidth=fs.MARKER_EDGE_WIDTH)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.set_xscale("log")
    fs.add_log_gridlines(ax, axis="x", subs=range(2, 10))
    ax.axvline(1.0, color=fs.STATUS_CRITICAL, linewidth=0.9, linestyle=":", zorder=2)
    ax.set_xlabel("D")
    ax.tick_params(colors=fs.INK_SECONDARY)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color(fs.BASELINE)


# ---------------------------------------------------------------------------
# B7 -- K vs Y dumbbell, the 12 true K-family nodes only (the only nodes
# where K/Y treatment actually differs -- every X/TY row is numerically
# identical between K and Y, confirmed from B0). S1 baseline.
# ---------------------------------------------------------------------------

def fig_B7_k_vs_y_dumbbell(S):
    k_nodes = S.loc[S["family"] == "K"].index.tolist()
    labels = [f"N{n}" for n in k_nodes]
    left_vals = S.loc[k_nodes, "S1_K"].values
    right_vals = S.loc[k_nodes, "S1_Y"].values

    fig, ax = plt.subplots(figsize=fs.usable_figsize(width_frac=0.62, aspect=1.05))
    _dumbbell_plot(ax, labels, left_vals, right_vals, fs.CAT_BLUE, fs.CAT_ORANGE,
                   "K consideration", "Y consideration")
    ax.set_title(SCENARIO_TITLES["S1"], loc="left", fontsize=8.5, color=fs.INK_PRIMARY,
                  fontweight="bold", pad=6)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fs.save_fig(fig, OUT_DIR, "B7_k_vs_y_dumbbell")


# ---------------------------------------------------------------------------
# B8 -- T_cp sensitivity dumbbell, the 8 chronic splash nodes only (S2.1/S4.1
# are numerically identical to S2/S4 everywhere else -- confirmed directly
# from final_results_joint_summary.csv). K consideration only, per the author's
# explicit scope for this figure. Two panels: Resize A (S2 vs S2.1) and
# Resize B (S4 vs S4.1).
# ---------------------------------------------------------------------------

def fig_B8_tcp_sensitivity_dumbbell(summary):
    nodes = CHRONIC_NODES
    labels = [f"N{n}" for n in nodes]
    idx = summary.set_index("node") if summary.index.name != "node" else summary

    fig, axes = plt.subplots(1, 2, figsize=fs.usable_figsize(width_frac=0.85, aspect=0.75))
    pairs = [
        (axes[0], "S2 -- Resize A", "D_S2-K", "D_S2.1-K", "standard", "T_cp sensitivity"),
        (axes[1], "S4 -- Resize B", "D_S4-K", "D_S4.1-K", "standard", "T_cp sensitivity"),
    ]
    for ax, title, left_col, right_col, left_label, right_label in pairs:
        _dumbbell_plot(ax, labels, idx.loc[nodes, left_col].values, idx.loc[nodes, right_col].values,
                       fs.CAT_BLUE, fs.CAT_ORANGE, left_label, right_label)
        ax.set_title(title, loc="left", fontsize=8.5, color=fs.INK_PRIMARY, fontweight="bold", pad=6)

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="lower center", ncol=2, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fs.save_fig(fig, OUT_DIR, "B8_tcp_sensitivity_dumbbell_K")


# ---------------------------------------------------------------------------
# B1 template, sensitivity scenarios -- same jacket-face grid as B1, S2.1 and
# S4.1 instead of S1-S5, K consideration only. Reads final_results_joint_
# summary.csv (the only source with S2.1/S4.1 columns) rather than
# build_scenario_columns()'s S DataFrame, which only carries S1-S5.
# ---------------------------------------------------------------------------

def fig_B1_sensitivity_map_grid(summary, joints, members):
    idx = summary.set_index("node") if summary.index.name != "node" else summary
    datasets = [
        ("S2.1 -- Resize A, T_cp sensitivity", idx["D_S2.1-K"].to_dict()),
        ("S4.1 -- Resize B, T_cp sensitivity", idx["D_S4.1-K"].to_dict()),
    ]
    plot_joint_4faces_grid(joints, members, datasets, OUT_DIR, "B1_sensitivity_map_grid_K",
                            vmax=1.0, fail_vmax=5.0)


# ---------------------------------------------------------------------------
# Damage-by-environment facets (joint-track counterpart to member-track A5c)
# ---------------------------------------------------------------------------

def _one_row_per_joint(m):
    """Each physical joint/brace hotspot must appear exactly once before
    summing across the LC columns, but the raw matrix carries two kinds of
    duplicate scoring: (1) K-family rows are scored twice, once as
    treatment=K and once as treatment=Y (same physical joint, two SCF
    formulas -- see docs/decisions.md), and (2) X-family rows
    are scored twice, once per chord/brace assignment (direction=A_as_chord
    vs B_as_chord). Baseline (chord_t_scenario=='single', i.e. no can-thickness
    resize) + treatment != 'Y' + direction != 'B_as_chord' keeps exactly one
    row per physical hotspot: 32 K + 24 TY + 16 X = 72 rows."""
    keep = (m["chord_t_scenario"] == "single") & (m["treatment"] != "Y") \
        & (m["direction"] != "B_as_chord")
    return m[keep]


def fig_B_damage_by_environment_facets_weighted(m=None):
    """Joint-track counterpart to member-track A5c (probability-weighted).
    Same environment bins, same LC column format, same 2x2 square-cell
    layout as the member version -- kept visually consistent across the
    report (a 1-column/4-row stack was tried and rejected: the author preferred
    the 2x2 on looks alone). Exact per-bin values go in an appendix table,
    not baked into the figure as cell text."""
    if m is None:
        m = pd.read_csv(RESULTS_DIR / "joint_track" / "joint_damage_matrix_weighted.csv")
    m = _one_row_per_joint(m)
    lc_cols = [c for c in m.columns if c not in NON_LC_JOINT]
    contribution = m[lc_cols].sum(axis=0)
    _facet_heatmap_by_tp(contribution.to_dict(),
                          "Total D contribution (all joints, weighted)",
                          "B_damage_by_environment_facets_weighted")


def fig_B_damage_by_environment_facets_raw(m=None):
    """Joint-track counterpart to member-track A5c_raw (physical severity,
    ignoring bin occurrence probability)."""
    if m is None:
        m = pd.read_csv(RESULTS_DIR / "joint_track" / "joint_damage_matrix_raw.csv")
    m = _one_row_per_joint(m)
    lc_cols = [c for c in m.columns if c not in NON_LC_JOINT]
    contribution = m[lc_cols].sum(axis=0)
    _facet_heatmap_by_tp(contribution.to_dict(),
                          "Total D contribution (all joints, raw/unweighted)",
                          "B_damage_by_environment_facets_raw")


def _facet_heatmap_by_tp(value_by_name, cbar_label, out_name):
    """Same engine as the member track's own facet-heatmap helper (2x2 grid,
    one panel per Tp value, square cells, shared colour scale) --
    duplicated rather than shared, since the two tracks' iteration scripts
    don't share a module."""
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

    # Fixed ABSOLUTE layout (no constrained_layout) so the grid and colorbar
    # sit at the identical figure-fraction position regardless of how wide
    # this call's colorbar tick labels happen to be -- weighted's ticks are
    # short ("10"/"20"/"30"/"40") but raw's are long decimals ("0.00025"
    # etc.), and constrained_layout was reflowing the whole grid+colorbar
    # block sideways to make room for whichever one needed more space. That
    # made the two companion figures' plot areas/colorbars land at different
    # x-positions once pasted into Word (caught in review from a screenshot:
    # "the legends... should be directly below each other and that's not the
    # case"). A fixed rect is immune to that -- see the same fix already
    # applied to plot_joint_4faces_grid's chrome for the same root cause.
    fig = plt.figure(figsize=fs.usable_figsize(width_frac=0.85, aspect=0.58))
    GRID_RECT = (0.09, 0.10, 0.76, 0.84)  # left, bottom, width, height
    CBAR_RECT = (0.87, 0.14, 0.025, 0.76)
    gs = fig.add_gridspec(2, 2, left=GRID_RECT[0], bottom=GRID_RECT[1],
                           right=GRID_RECT[0] + GRID_RECT[2], top=GRID_RECT[1] + GRID_RECT[3],
                           hspace=0.30, wspace=0.10)
    axes = gs.subplots(sharex=True, sharey=True)
    mesh = None
    for ax, tp in zip(axes.flat, tp_vals):
        grid = np.full((len(hs_vals), len(vw_vals)), np.nan)
        sub = plot_df[plot_df["t"] == tp]
        for _, r in sub.iterrows():
            grid[hs_vals.index(r["h"]), vw_vals.index(r["v"])] = r["d"]
        mesh = ax.pcolormesh(v_bounds, h_bounds, grid, cmap=cmap, vmin=0, vmax=vmax,
                              edgecolors=fs.GRIDLINE, linewidth=0.5)
        # data units differ (2 m/s per Vw cell vs 0.5 m per Hs cell) -- force
        # the display aspect so each cell renders as a visual square rather
        # than a 4x-wide rectangle
        ax.set_aspect(VW_STEP / HS_STEP, adjustable="box")
        ax.set_title(f"Tp = {tp:g} s", fontsize=7, color=fs.INK_PRIMARY, loc="left")
        ax.set_xticks(vw_vals)
        ax.set_yticks(hs_vals)
        ax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.supxlabel("Wind speed Vw [m/s]", color=fs.INK_SECONDARY, fontsize=7,
                  x=GRID_RECT[0] + GRID_RECT[2] / 2, y=0.015)
    fig.supylabel("Sig. wave height Hs [m]", color=fs.INK_SECONDARY, fontsize=7,
                  x=0.015, y=GRID_RECT[1] + GRID_RECT[3] / 2)

    cax = fig.add_axes(CBAR_RECT)
    cbar = fig.colorbar(mesh, cax=cax)
    # scientific notation with a shared exponent (e.g. "x10^-3") keeps tick
    # labels roughly the same width whether the underlying values are ~1e-3
    # (raw) or ~1e1 (weighted) -- also just more readable than "0.00025".
    # Must assign to cbar.formatter itself (not cbar.ax.yaxis's formatter --
    # a separate object Colorbar.update_ticks() doesn't look at) for the
    # powerlimits to actually take effect.
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    cbar.formatter = fmt
    cbar.update_ticks()
    cbar.ax.yaxis.get_offset_text().set(color=fs.INK_SECONDARY, fontsize=6)
    cbar.set_label(cbar_label, color=fs.INK_SECONDARY, fontsize=7)
    cbar.ax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)

    # tight=False: bbox_inches="tight" auto-crops to each figure's own
    # rendered content, which would undo the fixed-rect layout above (it
    # re-introduces exactly the per-figure size drift this was meant to
    # kill). The fixed rect already guarantees nothing sits outside the
    # canvas, so no auto-crop is needed.
    fs.save_fig(fig, OUT_DIR, out_name, tight=False)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary_table(S, n=12):
    top = S.sort_values("S1_K", ascending=False).head(n)
    table = top[["family", "S1_K", "S2_K", "S3_K", "S4_K", "S5_K"]].copy()
    table.insert(0, "node", table.index)
    out_path = RESULTS_DIR / "joint_results_summary_table.csv"
    table.to_csv(out_path, index=False)
    return table, out_path


def self_check(df, S):
    assert len(df) == 40, f"expected 40 joint rows, got {len(df)}"
    # corrosion must never decrease cumulative damage vs. the uncorroded retrofit value
    corroded_mask = S["S3_K"].notna() & df["D_S2-K"].notna()
    assert (S.loc[corroded_mask, "S3_K"] >= df.loc[corroded_mask, "D_S2-K"] - 1e-9).all()
    # the 8 chronic nodes are exactly the ones with real (non-fallback) corrosion data
    real_corrosion_nodes = set(df.loc[df["D_S3-K_25"].notna()].index)
    assert real_corrosion_nodes == set(CHRONIC_NODES), \
        f"chronic-node list mismatch: {real_corrosion_nodes} vs {CHRONIC_NODES}"
    print(f"OK: self-check passed ({len(df)} joints, {len(CHRONIC_NODES)} chronic splash nodes)")


if __name__ == "__main__":
    df = load_final()
    S = build_scenario_columns(df)
    self_check(df, S)

    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    joints, members = model["joints"], model["members"]

    summary = load_summary()
    fig_B0_overview_heatmap(summary)
    for treatment in ("K", "Y"):
        fig_B1_joint_map_grid(S, joints, members, treatment)
        fig_B3_family_comparison(S, treatment)

    fig_B7_k_vs_y_dumbbell(S)
    fig_B8_tcp_sensitivity_dumbbell(summary)
    fig_B1_sensitivity_map_grid(summary, joints, members)

    fig_B_damage_by_environment_facets_weighted()
    fig_B_damage_by_environment_facets_raw()

    table, table_path = build_summary_table(S)
    print(f"\nWrote summary table -> {table_path}")
    print(table.to_string(index=False))

    print(f"\nAll candidate figures written to {OUT_DIR}")
