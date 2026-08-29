"""
Step 5 -- Invariant tests.

Plain asserts on synthetic signals with a KNOWN correct answer, run against
the real stress.py (Step 4) + rainflow package before either is trusted on
real campaign data. Each check here exercises actual pipeline code (not a
standalone derivation) and would fail loudly if stress.py's formula, the
theta discretisation, or the rainflow call were wrong -- catching a class
of bug that is otherwise invisible in the output (a wrong damage number
looks exactly as plausible as a right one).

One check considered and deliberately DROPPED (per 06.08.2026 review):
SRSS-vs-16-point. The pipeline never implements SRSS anywhere, so a
comparison against it doesn't exercise or protect our own code -- it's a
methodology-justification talking point for the thesis text, not a
regression test. Skipped here; can be added back as a standalone note if
the thesis needs the number.

Checks kept, in order:
    1. Range-vs-amplitude trap: rainflow on a pure sine of amplitude 50
       must report range=100, not 50 -- the classic 8x-damage mixup
       (m=3: (100/50)^3 = 8).
    2. max(range) <= np.ptp(signal): no rainflow cycle can exceed the
       signal's own peak-to-peak.
    3. 8 vs 16 theta discretisation bound: a fixed-direction (non-rotating)
       bending signal peaking at theta=337.5 deg (chosen so it lands
       exactly on a 16-point grid node but exactly BETWEEN two 8-point
       grid nodes) -- this is the number that actually justifies
       n_theta=16 over a coarser grid, not just an aesthetic choice.
    4. DC offset invariance: adding a large constant mean stress changes
       no rainflow range, hence no damage -- rainflow must be blind to
       absolute level, only ranges.

No pytest -- matches this pipeline's other steps (assert + a self-check
entry point).
"""
import sys
from pathlib import Path

import numpy as np
import rainflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/

import stress

# region --- damage proxy ---
# A real S-N damage number needs sn_curves.py (Step 8, not built yet, and
# gated on the author's sign-off -- see docs/decisions.md). These
# checks don't need a real S-N curve, only something MONOTONIC in range
# with an exponent representative of DNV's m=3 branch, since the whole
# point is to compare RATIOS between two discretisations/signals, not to
# produce a real life estimate. sum(count * range^m) is exactly
# proportional to Miner damage for any single S-N branch (same log_a, same
# thickness correction) -- the proportionality constant cancels in every
# ratio taken below.
DAMAGE_EXPONENT_M = 3


def damage_proxy(signal, m=DAMAGE_EXPONENT_M):
    """sum(count * range^m) over rainflow.extract_cycles(signal)."""
    d = 0.0
    for rng, mean, count, i_start, i_end in rainflow.extract_cycles(signal):
        if rng > 0:
            d += count * rng ** m
    return d


def cycle_ranges(signal):
    return [rng for rng, mean, count, i_start, i_end in rainflow.extract_cycles(signal)]
# endregion


# region --- 1. range vs amplitude ---
def check_range_vs_amplitude():
    """Pure sine, amplitude 50 -> rainflow range must be 100, not 50."""
    t = np.linspace(0.0, 10.0, 10001)
    sig = 50.0 * np.sin(2 * np.pi * 1.0 * t)
    ranges = cycle_ranges(sig)
    max_range = max(ranges)
    print(f"  [1] pure sine +-50 MPa: max rainflow range = {max_range:.6f} "
          f"(expect 100.000000, i.e. peak-to-trough, not amplitude)")
    assert abs(max_range - 100.0) < 1e-6, (
        f"got {max_range}, expected 100.0 -- range/amplitude mixup would "
        f"give ~50.0 here and silently under-damage by 2^3=8x at m=3"
    )
# endregion


# region --- 2. max range <= ptp ---
def check_range_bounded_by_ptp():
    """No rainflow cycle range can exceed the signal's own peak-to-peak."""
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=5000))
    ranges = cycle_ranges(walk)
    max_range = max(ranges)
    ptp = np.ptp(walk)  # np.ptp(x), not x.ptp() -- removed in numpy 2
    print(f"  [2] random walk: max rainflow range = {max_range:.6f}, "
          f"signal ptp = {ptp:.6f}, ratio = {max_range/ptp:.6f}")
    assert max_range <= ptp + 1e-9, (
        f"rainflow reported a cycle range ({max_range}) larger than the "
        f"signal's own peak-to-peak ({ptp}) -- impossible, something is "
        f"structurally wrong with the cycle extraction"
    )
# endregion


# region --- 3. theta discretisation bound ---
def _unidirectional_bending_signal(t, phi_deg, amplitude=100.0):
    """
    Pure bending (sig_ax=0) with FIXED direction phi and oscillating
    magnitude -- not a rotating moment. Via stress.hotspot_member's
    sig_ipb*cos(theta) - sig_opb*sin(theta), setting
    sig_ipb=A*cos(phi)*wave, sig_opb=A*sin(phi)*wave collapses to
    A*wave(t)*cos(theta+phi), which peaks at theta = -phi (mod 360).
    """
    phi = np.radians(phi_deg)
    wave = np.sin(2 * np.pi * 1.0 * t)
    sig_ax = np.zeros_like(t)
    sig_ipb = amplitude * np.cos(phi) * wave
    sig_opb = amplitude * np.sin(phi) * wave
    return sig_ax, sig_ipb, sig_opb


def check_theta_resolution():
    """
    8 vs 16 theta points on a fixed-direction bending signal peaking at
    theta=337.5 deg -- exactly on a 16-point grid node, exactly midway
    between two 8-point grid nodes (315, 0/360). At m=3 the undercount
    from missing the true peak by 22.5 deg is cos(22.5deg)^3 = 0.788675.
    """
    t = np.linspace(0.0, 20.0, 4001)
    peak_theta_deg = 337.5  # -phi with phi=22.5deg
    sig_ax, sig_ipb, sig_opb = _unidirectional_bending_signal(t, phi_deg=22.5)

    results = {}
    for n_theta in (8, 16):
        theta = stress.theta_grid(n_theta)
        sigma = stress.hotspot_member(sig_ax, sig_ipb, sig_opb, theta)  # (n_t, n_theta)
        damage_per_theta = [damage_proxy(sigma[:, k]) for k in range(n_theta)]
        results[n_theta] = max(damage_per_theta)

    ratio = results[8] / results[16]
    expected_ratio = np.cos(np.radians(22.5)) ** DAMAGE_EXPONENT_M
    print(f"  [3] true peak direction: theta={peak_theta_deg} deg (on the "
          f"16-pt grid, 22.5 deg off every 8-pt grid node)")
    print(f"      max-over-theta damage: n=8 -> {results[8]:.6e}   "
          f"n=16 -> {results[16]:.6e}   ratio(8/16) = {ratio:.6f}")
    print(f"      expected ratio = cos(22.5deg)^{DAMAGE_EXPONENT_M} = {expected_ratio:.6f}")
    assert abs(ratio - expected_ratio) < 1e-3, (
        f"n=8/n=16 damage ratio {ratio:.6f} doesn't match the geometric "
        f"prediction {expected_ratio:.6f} -- this is the number that "
        f"justifies n_theta=16, so it must hold to before trusting the grid choice"
    )
    # n=16 must land closer to the true peak (ratio 1.0 at its own maximum)
    # than n=8 does -- confirms n=16 isn't accidentally worse.
    assert results[16] > results[8]
# endregion


# region --- 4. DC offset invariance ---
def check_dc_offset_invariance():
    """A large constant mean-stress offset must not change rainflow damage
    at all -- rainflow only ever sees ranges, never absolute level.

    Uses a multi-harmonic, incommensurate-frequency bending signal (NOT the
    single-phi signal from check_theta_resolution) specifically so no theta
    in the grid degenerates to near-zero amplitude: a single sin(wt) at
    fixed phi is exactly zero at two grid angles (theta = 90-phi and
    270-phi) up to float noise (~1e-14), and adding 500 to a ~1e-14 signal
    is swamped by float64's ~500*1e-16 absolute precision at that
    magnitude -- a real floating-point artifact, not a stress.py bug. Two
    incommensurate frequencies can't both cancel at the same theta for all
    t, so every grid angle keeps a real, non-degenerate amplitude.
    """
    t = np.linspace(0.0, 50.0, 8001)
    sig_ax = np.zeros_like(t)
    sig_ipb = 80.0 * np.sin(2 * np.pi * 0.37 * t) + 30.0 * np.sin(2 * np.pi * 0.91 * t + 0.5)
    sig_opb = 60.0 * np.sin(2 * np.pi * 0.53 * t + 1.1) + 20.0 * np.sin(2 * np.pi * 1.21 * t)
    theta = stress.THETA_RAD

    sigma_0 = stress.hotspot_member(sig_ax, sig_ipb, sig_opb, theta)
    sigma_offset = stress.hotspot_member(sig_ax + 500.0, sig_ipb, sig_opb, theta)

    d0 = np.array([damage_proxy(sigma_0[:, k]) for k in range(len(theta))])
    d1 = np.array([damage_proxy(sigma_offset[:, k]) for k in range(len(theta))])
    assert d0.min() > 1e3, "test signal degenerated to near-zero amplitude at some theta -- fix the signal, not the tolerance"
    max_rel_diff = float(np.max(np.abs(d0 - d1) / d0))
    print(f"  [4] DC offset +500 MPa: max relative damage difference across "
          f"{len(theta)} theta = {max_rel_diff:.3e}  (all theta amplitudes "
          f"well clear of zero: min damage_proxy = {d0.min():.3e})")
    assert max_rel_diff < 1e-9, (
        f"a constant mean-stress offset changed damage by a relative "
        f"{max_rel_diff:.3e} -- rainflow should be exactly blind to "
        f"absolute level, only ranges matter"
    )
# endregion


def _self_check():
    print("Step 5 -- invariant tests\n")
    check_range_vs_amplitude()
    check_range_bounded_by_ptp()
    check_theta_resolution()
    check_dc_offset_invariance()
    print("\n  all invariants hold.")


if __name__ == "__main__":
    _self_check()
