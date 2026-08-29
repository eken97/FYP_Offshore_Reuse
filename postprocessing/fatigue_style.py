"""
Shared figure style for the fatigue-chapter report figures: font, page-fit sizing,
DPI, and the validated colour palette. Import this in every fatigue_report_figures*
script instead of setting rcParams locally, so every figure in the chapter matches
by construction.

Page geometry: A4, 2.54 cm margins all round (confirmed against the author's real
Word doc, 16.08.2026) -> usable width 15.92 cm.

Colour palette: colour-vision-deficiency-validated set (OKLab CVD-simulation
checks), assigned by JOB not by taste -- categorical (identity), sequential
(magnitude), diverging (before/after), status (pass/fail vs 25yr design life).
Values as documented, not re-derived here.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------
CM_PER_IN = 2.54
PAGE_MARGIN_CM = 2.54
PAGE_WIDTH_CM = 21.0  # A4
PAGE_HEIGHT_CM = 29.7  # A4
USABLE_WIDTH_CM = PAGE_WIDTH_CM - 2 * PAGE_MARGIN_CM  # 15.92 cm
USABLE_WIDTH_IN = USABLE_WIDTH_CM / CM_PER_IN  # ~6.27 in
USABLE_HEIGHT_CM = PAGE_HEIGHT_CM - 2 * PAGE_MARGIN_CM  # 24.62 cm
USABLE_HEIGHT_IN = USABLE_HEIGHT_CM / CM_PER_IN  # ~9.69 in

DPI = 300

# ---------------------------------------------------------------------------
# Font -- Segoe UI (closest installed relative of the thesis body font, Aptos;
# Aptos itself is not installed as a standalone system font on this machine)
# ---------------------------------------------------------------------------
FONT_FAMILY = "Segoe UI"


def _register_segoe_ui():
    """Register Segoe UI's actual files with matplotlib's font manager.

    Setting rcParams["font.family"] alone is not reliable if the font isn't in
    matplotlib's own cache yet -- register the real Windows font files directly.
    """
    windir = Path(r"C:\Windows\Fonts")
    for fname in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "segoeuiz.ttf"):
        fpath = windir / fname
        if fpath.exists():
            font_manager.fontManager.addfont(str(fpath))


def apply_style():
    """Call once at the top of any figure script."""
    _register_segoe_ui()
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": BASELINE,
        "axes.grid": False,
    })


def usable_figsize(width_frac=1.0, aspect=0.65):
    """Figure size in inches for a given fraction of the page's usable width.

    aspect = height / width. Returns (w_in, h_in).
    """
    w_in = USABLE_WIDTH_IN * width_frac
    return (w_in, w_in * aspect)


# ---------------------------------------------------------------------------
# Colour palette -- see dataviz skill's palette.md for full validation detail.
# Surface here is white (Word page), not the skill's off-white chart surface --
# close enough that the documented pass/fail results carry over unchanged.
# ---------------------------------------------------------------------------

# Categorical: identity (e.g. Baseline / Retrofit A / Retrofit B; or zone).
# Fixed order -- never reassign per-chart, never cycle past slot 3 without
# folding extra series into "Other" or faceting (all-pairs CVD floor).
CAT_BLUE = "#2a78d6"
CAT_ORANGE = "#eb6834"
CAT_AQUA = "#1baf7a"
CATEGORICAL = [CAT_BLUE, CAT_ORANGE, CAT_AQUA]

# Sequential: magnitude (damage severity maps / heatmaps). Light -> dark, one hue.
SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec",
    "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
    "#184f95", "#104281", "#0d366b",
]

# Diverging: before/after retrofit change. Blue = damage reduced, red = increased,
# grey = no change. Equal steps per arm.
DIVERGING_COOL = "#2a78d6"   # improved (less damage)
DIVERGING_WARM = "#e34948"   # worsened (more damage)
DIVERGING_MID = "#f0efec"    # no change

# Status: fixed, reserved meaning -- life vs the 25yr design threshold. Always
# pair with an icon/label, never rely on colour alone (esp. "warning"/"serious"
# are sub-3:1 contrast on white by design).
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

# Chrome / ink (not the skill's off-white -- pure white to match a Word page)
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

# Marker outline for scatter/dot points -- NOT white. White-on-white (the page
# surface) is invisible in practice; a thin dark edge is what actually reads
# as an outline and separates overlapping same-hue points (caught from a real
# reference figure the author referred to -- their dots have a visible dark rim,
# ours didn't).
MARKER_EDGE = INK_SECONDARY
MARKER_EDGE_WIDTH = 0.6


def add_log_gridlines(ax, axis="y", subs=(2,)):
    """Light reference lines at each major decade PLUS intermediate lines at
    `subs`x each decade (default just 2x, e.g. ...0.02, 0.2, 2, 20...; pass
    subs=range(2,10) for a full log-paper grid at every step 2-9x each
    decade), behind the data.

    Drawn directly (axhline/axvline) rather than through matplotlib's minor-
    tick system: ax.grid(which="minor") + a custom LogLocator silently
    produced ZERO minor ticks on our real axes (5-8 decades span) even though
    the locator's own .tick_values() returned the correct numbers when called
    directly -- minorticks_on()'s internal log-scale heuristic appears to
    suppress minor ticks past some decade-count, and setting a custom locator
    afterward didn't override that suppression. Root-caused by bisecting
    (isolated LogLocator call worked; same call through the real boxplot with
    real data didn't) rather than guessed. Drawing the lines directly sidesteps
    the whole locator/minorticks machinery, so there's nothing left to fight.
    """
    import math
    ax.grid(axis=axis, which="major", color=GRIDLINE, linewidth=0.7, zorder=0)
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    if lo <= 0:
        return
    dec_lo = math.floor(math.log10(lo))
    dec_hi = math.ceil(math.log10(hi))
    line_fn = ax.axhline if axis == "y" else ax.axvline
    for dec in range(dec_lo, dec_hi + 1):
        for s in subs:
            v = s * (10.0 ** dec)
            if lo <= v <= hi:
                line_fn(v, color=GRIDLINE, linewidth=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def save_fig(fig, out_dir, name, tight=True):
    """Save PNG (300dpi) + SVG, no in-image title per the project's own
    graphics convention -- caption/number comes from the surrounding Word doc.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {"bbox_inches": "tight"} if tight else {}
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.savefig(png_path, dpi=DPI, **kwargs)
    fig.savefig(svg_path, **kwargs)
    return png_path, svg_path


# ---------------------------------------------------------------------------
# Small plotting helpers shared by more than one figure builder. Kept here
# rather than in a standalone module so callers only need one shared import.
# ---------------------------------------------------------------------------

LC_PATTERN = re.compile(r"LC_V(?P<v>\d+(?:p\d+)?)_H(?P<h>\d+p\d+)_T(?P<t>\d+(?:p\d+)?)")


def parse_lc_name(name):
    """LC_V18_H2p0_T7 -> (Vw=18.0, Hs=2.0, Tp=7.0). All 69 real campaign bins
    follow this exact naming convention (verified: no _PARKED-suffixed or
    other variant columns exist in the real matrix CSVs)."""
    m = LC_PATTERN.match(name)
    if not m:
        return None
    v = float(m.group("v").replace("p", "."))
    h = float(m.group("h").replace("p", "."))
    t = float(m.group("t").replace("p", "."))
    return v, h, t


def styled_boxplot(ax, data_list, positions, colors, width=0.5):
    """colors: one colour, or a list matching len(data_list) (one per box)."""
    if isinstance(colors, str):
        colors = [colors] * len(data_list)
    bp = ax.boxplot(data_list, positions=positions, widths=width, patch_artist=True,
                    showfliers=False, whis=(0, 100), zorder=2)
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_alpha(0.30)
        box.set_edgecolor(MARKER_EDGE)
        box.set_linewidth(MARKER_EDGE_WIDTH)
    for key in ("whiskers", "caps"):
        for line in bp[key]:
            line.set_color(MARKER_EDGE)
            line.set_linewidth(1.0)
    for median in bp["medians"]:
        median.set_color(INK_PRIMARY)
        median.set_linewidth(1.5)
    return bp


if __name__ == "__main__":
    # Self-check: confirm Segoe UI actually resolves, not silently falling
    # back to DejaVu Sans (matplotlib does this silently on a font-name miss).
    apply_style()
    resolved = font_manager.FontProperties(family=FONT_FAMILY).get_name()
    assert "Segoe UI" in resolved, (
        f"Segoe UI did not resolve (got {resolved!r}) -- font registration failed"
    )
    print(f"OK: font resolves to {resolved!r}")
    print(f"OK: usable width = {USABLE_WIDTH_CM:.2f} cm = {USABLE_WIDTH_IN:.3f} in")
    print(f"OK: figsize(1.0, 0.65) = {usable_figsize()}")
