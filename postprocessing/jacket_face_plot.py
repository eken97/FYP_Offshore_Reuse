"""
Reusable "4 faces side by side" jacket elevation plot, coloured by a damage
value D. Two entry points:

- plot_4faces(): one map, full page width -- the detailed/appendix version.
- plot_4faces_grid(): several maps (e.g. baseline + corrosion years) stacked
  as rows in ONE figure, sharing a single colorbar and legend instead of
  repeating that chrome per map -- the compact "several per page" version.
  (A naive per-map size shrink was tried first and broke: fixed-point-size
  labels/legend/colorbar don't scale down with the figure, so they just
  collided. A shared-chrome grid is the actual fix, not a smaller version of
  the same layout.)

Colour rule: D < vmax uses the sequential blue ramp (log or linear, caller's
choice). D >= 1 (Miner's-rule failure) is ALWAYS rendered in status-critical
red, independent of where the scale's own vmax sits -- this matches the
the author's framing: show D directly, D>=1 means failure.

Used by fig_member_damage_maps.py today; designed so the joint-track map
(harder -- K/Y treatment spread, chord-thickness split) can reuse the same
face-splitting/geometry/colour-scale code without duplicating it.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

import fatigue_style as fs
import sd_geometry as sdg

NOT_ASSESSABLE_COLOR = "#b8b6ad"
SPLASH_ZMIN = sdg.ZONE_SPLASH_ZMIN
SPLASH_ZMAX = sdg.ZONE_SPLASH_ZMAX

# (name, membership test, in-plane horizontal coordinate index into (x,y,z))
# North/South faces (legs at constant y) vary in x; East/West faces (legs at
# constant x) vary in y -- plotting the wrong one collapses the face into a
# near-straight line (real bug, caught by looking at the actual rendered PNG).
FACES = [
    ("North", lambda x, y: y > 0.01, 0),
    ("East", lambda x, y: x > 0.01, 1),
    ("South", lambda x, y: y < -0.01, 0),
    ("West", lambda x, y: x < -0.01, 1),
]

D_FAILURE = 1.0  # Miner's-rule failure threshold -- ALWAYS the red/critical cutoff,
                 # independent of where the colour scale's own top (vmax) sits


def _face_members(joints, members, test_fn):
    face_ids = {jid for jid, (x, y, z) in joints.items() if test_fn(x, y)}
    return {
        mid: d for mid, d in members.items()
        if d["j1"] in face_ids and d["j2"] in face_ids
    }


def build_cmap_norm(scale="log", vmax=D_FAILURE, vmin_floor=1e-6):
    """vmax: top of the colour scale -- pass the real observed max (shared
    across a comparison series), NOT hardcoded to 1.0, so the scale actually
    uses its dynamic range. vmin_floor: only used for the log scale's lower
    bound (log can't start at exactly 0).
    """
    cmap = LinearSegmentedColormap.from_list("seq_blue", fs.SEQUENTIAL_BLUE)
    if scale == "log":
        norm = LogNorm(vmin=vmin_floor, vmax=vmax)
    elif scale == "linear":
        norm = Normalize(vmin=0.0, vmax=vmax)
    else:
        raise ValueError(f"unknown scale {scale!r}")
    return cmap, norm


def _color_for(d_value, not_assessable, cmap, norm):
    if not_assessable or d_value is None or (isinstance(d_value, float) and np.isnan(d_value)):
        return NOT_ASSESSABLE_COLOR
    if d_value >= D_FAILURE:
        return fs.STATUS_CRITICAL
    if isinstance(norm, LogNorm):
        if d_value <= 0:
            return cmap(0.0)
        return cmap(norm(max(d_value, norm.vmin)))
    return cmap(norm(max(d_value, 0.0)))


def _draw_face_panel(ax, joints, members, member_D, not_assessable_ids, cmap, norm,
                      face_name, test_fn, coord_idx, linewidth, show_xlabel,
                      xlabel_fontsize=8):
    ax.axhline(SPLASH_ZMIN, color=fs.BASELINE, linewidth=0.7, linestyle=":", zorder=1)
    ax.axhline(SPLASH_ZMAX, color=fs.BASELINE, linewidth=0.7, linestyle=":", zorder=1)
    ax.axhline(0, color=fs.INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)), zorder=1)

    face_members = _face_members(joints, members, test_fn)
    for mid, d in face_members.items():
        p1 = joints[d["j1"]]
        p2 = joints[d["j2"]]
        d_value = member_D.get(mid)
        color = _color_for(d_value, mid in not_assessable_ids, cmap, norm)
        # solid_capstyle="butt" -- "round" left a small filled dot at every
        # joint where members meet (visible, distracting; flagged in review)
        ax.plot([p1[coord_idx], p2[coord_idx]], [p1[2], p2[2]], color=color,
                 linewidth=linewidth, solid_capstyle="butt", zorder=2)

    if show_xlabel:
        ax.set_xlabel(face_name, color=fs.INK_PRIMARY, fontsize=xlabel_fontsize, labelpad=6)
    ax.set_xticks([])
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_yaxis(ax, label_fontsize=7, show_splash_ticks=True):
    """0 -> 'MSL'. show_splash_ticks additionally shows the splash-zone bounds
    as their real numeric elevation on the axis -- fine at full row height,
    but at grid-row height (~3.5cm covering the full 70m span) "4.84" sits too
    close to "MSL" to both be legible (real collision, caught by looking at
    the rendered PNG) -- grid rows pass False and rely on the thin boundary
    lines plus a single caption for the whole figure instead.
    ylim is locked BEFORE adding the extra non-round ticks, otherwise
    matplotlib re-picks 'nice' limits around them and pads the figure with
    empty space top/bottom (a separate real bug, also caught the same way)."""
    ax.set_ylabel("z [m]", color=fs.INK_SECONDARY)
    ax.tick_params(axis="y", left=True, labelleft=True, colors=fs.INK_SECONDARY,
                    labelsize=label_fontsize)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(fs.BASELINE)

    if show_splash_ticks:
        original_ylim = ax.get_ylim()
        auto_ticks = ax.get_yticks()
        all_ticks = sorted(set(auto_ticks.tolist()) | {SPLASH_ZMIN, SPLASH_ZMAX})
        ax.set_yticks(all_ticks)
        ax.set_ylim(original_ylim)

    def _ytick_fmt(v, pos):
        if v == 0:
            return "MSL"
        if show_splash_ticks and (abs(v - SPLASH_ZMIN) < 1e-6 or abs(v - SPLASH_ZMAX) < 1e-6):
            return f"{v:.2f}"
        return f"{v:g}"

    ax.yaxis.set_major_formatter(FuncFormatter(_ytick_fmt))


def plot_4faces(joints, members, member_D, not_assessable_ids, out_dir, name,
                 legend_extra=None, scale="log", vmax=D_FAILURE, vmin_floor=1e-6,
                 width_frac=1.0, aspect=0.72, linewidth=2.0):
    """One map, all 4 faces in a row. Full detail -- appendix/standalone use.
    vmax: top of the colour scale (pass the real series max for a comparable
    set of plots, not the default 1.0, or linear scaling shows nothing).
    """
    fs.apply_style()
    cmap, norm = build_cmap_norm(scale=scale, vmax=vmax, vmin_floor=vmin_floor)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(1, 4, figsize=fs.usable_figsize(width_frac=width_frac, aspect=aspect),
                              sharey=True)

    for ax, (face_name, test_fn, coord_idx) in zip(axes, FACES):
        _draw_face_panel(ax, joints, members, member_D, not_assessable_ids, cmap, norm,
                          face_name, test_fn, coord_idx, linewidth, show_xlabel=True)
        ax.tick_params(axis="y", left=False, labelleft=False)

    _style_yaxis(axes[0])

    cbar = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.02, extend="max")
    scale_note = "log scale" if isinstance(norm, LogNorm) else "linear scale"
    cbar.set_label(rf"$D$  ({scale_note}, $D{{=}}1$ = failure)", color=fs.INK_SECONDARY, fontsize=7)
    cbar.ax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)

    handles = [
        Line2D([0], [0], color=fs.STATUS_CRITICAL, linewidth=2.0, label=r"$D \geq 1$ (failed)"),
        Line2D([0], [0], color=NOT_ASSESSABLE_COLOR, linewidth=2.0, label="Not assessable"),
    ]
    if legend_extra:
        handles.extend(legend_extra)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(wspace=0.15)
    return fs.save_fig(fig, out_dir, name, tight=True)


def plot_4faces_grid(joints, members, datasets, not_assessable_ids, out_dir, name,
                      scale="linear", vmax=D_FAILURE, vmin_floor=1e-6,
                      target_height_frac=0.85, linewidth=1.4):
    """datasets: list of (row_label, member_D_dict), one row per snapshot (e.g.
    baseline + each corrosion year). ONE shared colorbar + legend for the
    whole figure instead of repeating them per row -- the fix for fitting
    several maps on one page without the per-map chrome colliding.
    target_height_frac: total figure height as a fraction of the PAGE's
    usable height (not per-row) -- a dense multi-row figure is worth using
    most of the page for (the author's call: 70-90%, leaving room for a caption),
    so this scales the whole grid to hit that regardless of row count, rather
    than a fixed per-row size that undersells how much page is available.
    """
    fs.apply_style()
    cmap, norm = build_cmap_norm(scale=scale, vmax=vmax, vmin_floor=vmin_floor)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    n_rows = len(datasets)
    fig_w = fs.USABLE_WIDTH_IN
    fig_h = fs.USABLE_HEIGHT_IN * target_height_frac
    fig, axes = plt.subplots(n_rows, 4, figsize=(fig_w, fig_h), sharey=True, sharex=False)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for r, (row_label, member_D) in enumerate(datasets):
        is_bottom_row = r == n_rows - 1
        for c, (face_name, test_fn, coord_idx) in enumerate(FACES):
            ax = axes[r, c]
            _draw_face_panel(ax, joints, members, member_D, not_assessable_ids, cmap, norm,
                              face_name, test_fn, coord_idx, linewidth,
                              show_xlabel=is_bottom_row, xlabel_fontsize=7)
            ax.tick_params(axis="y", left=False, labelleft=False)
        _style_yaxis(axes[r, 0], label_fontsize=6, show_splash_ticks=False)
        axes[r, 0].set_title(row_label, loc="left", fontsize=7, color=fs.INK_PRIMARY,
                              fontweight="bold", pad=3)

    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.015, extend="max")
    scale_note = "log scale" if isinstance(norm, LogNorm) else "linear scale"
    cbar.set_label(rf"$D$  ({scale_note}, $D{{=}}1$ = failure)", color=fs.INK_SECONDARY, fontsize=7)
    cbar.ax.tick_params(colors=fs.INK_SECONDARY, labelsize=6)

    handles = [
        Line2D([0], [0], color=fs.STATUS_CRITICAL, linewidth=2.0, label=r"$D \geq 1$ (failed)"),
        Line2D([0], [0], color=NOT_ASSESSABLE_COLOR, linewidth=2.0, label="Not assessable"),
        Line2D([0], [0], color=fs.BASELINE, linewidth=0.7, linestyle=":",
               label=f"Splash zone ({SPLASH_ZMIN:.2f} to {SPLASH_ZMAX:.2f} m MSL)"),
        Line2D([0], [0], color=fs.INK_MUTED, linewidth=0.8, linestyle=(0, (2, 2)), label="MSL"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))

    fig.subplots_adjust(wspace=0.15, hspace=0.45)
    return fs.save_fig(fig, out_dir, name, tight=True)
