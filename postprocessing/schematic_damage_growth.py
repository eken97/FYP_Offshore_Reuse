"""
Schematic (not real-data) figure for the Methodology section: motivates the
corrosion-extrapolation step by showing that a non-corroding member's damage
grows linearly in time, while a splash-zone (corroding) member's damage grows
super-linearly (power law), and that this projection is only meaningful up to
a real structural cutoff (compression/buckling failure of the corroded
section), not out to D=1 on the power-law curve itself.

Not fitted to real data -- illustrative shapes only, arbitrary axis units.
"""
import numpy as np
import matplotlib.pyplot as plt

import fatigue_style as fs

OUT_DIR = "../figures/methodology"
FIG_NAME = "schematic_damage_growth_zones"

T_MAX = 10.0
T_POWER_REF = 8.0   # power curve reaches the linear curve's endpoint value here, then keeps rising
T_CUTOFF = 7.0       # vertical cutoff line, placed after the crossover so orange is visibly above blue
POWER_EXPONENT = 1.5


def main():
    fs.apply_style()

    t = np.linspace(0, T_MAX, 300)
    d_linear = t / T_MAX
    d_power = (t / T_POWER_REF) ** POWER_EXPONENT

    fig, ax = plt.subplots(figsize=fs.usable_figsize(width_frac=0.65, aspect=0.75))

    ax.grid(True, color=fs.GRIDLINE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)

    ax.plot(t, d_linear, color=fs.CAT_BLUE, linewidth=1.8)
    ax.plot(t, d_power, color=fs.CAT_ORANGE, linewidth=1.8)

    ax.axvline(T_CUTOFF, color="#d03b3b", linewidth=1.4, linestyle="--")

    ax.set_xlim(0, T_MAX)
    ax.set_ylim(0, max(d_linear.max(), d_power.max()) * 1.05)
    ax.set_xlabel("Time")
    ax.set_ylabel("$D$")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(fs.GRIDLINE)

    fs.save_fig(fig, OUT_DIR, FIG_NAME)
    plt.close(fig)
    print(f"Saved to {OUT_DIR}/{FIG_NAME}.(png|svg)")


if __name__ == "__main__":
    main()
