"""
Step 6 -- Log binning and the histogram.

Turns a raw rainflow cycle list (rainflow.extract_cycles output: one
(range, mean, count, i_start, i_end) tuple per cycle, count in {0.5, 1.0})
into the compressed per-run storage format the design settled on: for each
of the 256 global log-spaced bins (fatigue_config.BIN_EDGES_MPA), store
counts PLUS sum(range^m) for every exponent in
fatigue_config.WOHLER_EXPONENTS -- not counts alone.

Why power sums, not counts alone: Miner damage for a bin of cycles at a
given range is proportional to range^m. Approximating every cycle in a bin
by its bin midpoint loses precision that GROWS with m and with bin width.
Storing the exact sum of range^m per bin makes damage recovery from the
histogram EXACT for that exponent, at the cost of one extra float64 array
per exponent (which mostly compresses away, since most of the 256 bins are
empty for any one theta/member-end).

The exponent SET is a config value (fatigue_config.WOHLER_EXPONENTS,
currently (3, 5) -- DNV-RP-C203's typical welded-joint bilinear pattern,
UNVERIFIED against a primary copy of the standard, see that file's
docstring), not hardcoded here as named sum_r3/sum_r5 fields. This matters
because, unlike log_a/knee constants Stage 3 will read later, the exponent
is baked into what gets STORED -- a wrong exponent means every Stage 2
file must be recomputed from raw .outb data, not just re-read. Keeping it
as a config-driven list means a correction is a one-line edit + recompute,
not a code change.

Steps:
    1. cycles_to_histogram(cycles, bin_edges, exponents) -- bin every cycle
       by range into (counts, sum_r[m] for each m), plus separately
       tracked n_under (range below the lowest edge, or exactly zero) and
       n_over (range at or above the highest edge -- should be 0 on real
       data; see fatigue_config.py's docstring on why the range was chosen
       wide).
    2. Two self-check damage proxies -- damage_from_cycles (ground truth,
       straight from the raw cycle list) and damage_from_histogram (using
       the stored power sums) -- must agree exactly, for every exponent in
       WOHLER_EXPONENTS. A counts-only variant is also computed, to show
       the error it WOULD have introduced.
"""
from pathlib import Path

import numpy as np
import rainflow

import fatigue_config as cfg
import stress

BIN_EDGES_MPA = cfg.BIN_EDGES_MPA
N_BINS = cfg.N_BINS
WOHLER_EXPONENTS = cfg.WOHLER_EXPONENTS


# region --- histogram builder ---
def cycles_to_histogram(cycles, bin_edges=BIN_EDGES_MPA, exponents=WOHLER_EXPONENTS):
    """
    cycles: iterable of (range, mean, count, i_start, i_end), as yielded by
    rainflow.extract_cycles. count is a half-integer (0.5 for a half
    cycle in the residue) -- NEVER cast to int, anywhere in this function.

    Returns (counts, sum_r, n_under, n_over):
        counts  : float64 array, shape (len(bin_edges)-1,)
        sum_r   : dict {m: float64 array, shape (len(bin_edges)-1,)} --
                  one array per exponent in `exponents`, sum_r[m][k] =
                  sum(count * range**m) over cycles landing in bin k.
        n_under : float, total count with range <= bin_edges[0] (includes
                  range == 0 exactly -- log10(0) is undefined, so a
                  zero-range cycle is routed here rather than binned)
        n_over  : float, total count with range >= bin_edges[-1]
    """
    n_bins = len(bin_edges) - 1
    counts = np.zeros(n_bins, dtype=np.float64)
    sum_r = {m: np.zeros(n_bins, dtype=np.float64) for m in exponents}
    n_under = 0.0
    n_over = 0.0

    for rng, mean, count, i_start, i_end in cycles:
        if rng <= bin_edges[0]:
            n_under += count
            continue
        if rng >= bin_edges[-1]:
            n_over += count
            continue
        # edges[k] <= rng < edges[k+1]
        k = int(np.searchsorted(bin_edges, rng, side="right") - 1)
        counts[k] += count
        for m in exponents:
            sum_r[m][k] += count * rng ** m

    return counts, sum_r, n_under, n_over
# endregion


# region --- damage proxies (self-check only -- real S-N is Step 8) ---
def damage_from_cycles(cycles, m):
    """Ground truth: sum(count * range^m) straight from the raw cycle list."""
    return sum(count * rng ** m for rng, mean, count, i_start, i_end in cycles if rng > 0)


def damage_from_histogram(sum_r, m):
    """Exact recovery from the stored power sums -- only valid for m in sum_r's keys."""
    if m not in sum_r:
        raise ValueError(f"power sum not stored for m={m} -- available: {sorted(sum_r)}")
    return float(np.sum(sum_r[m]))


def damage_from_counts_only(counts, bin_edges, m):
    """
    What Stage 3 would get if only counts (bin midpoints) were stored --
    computed here purely to quantify the error the power-sum design
    avoids, never used for a real damage number.
    """
    mid = np.sqrt(bin_edges[:-1] * bin_edges[1:])  # geometric midpoint, matches log spacing
    return float(np.sum(counts * mid ** m))
# endregion


def _self_check():
    project = Path(__file__).resolve().parent.parent.parent   # .../OpenFast
    outb_path = (project / "TestScenario" / "LC_V20_H3p5_T8" / "S100001" /
                 "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb")
    sd_path = (project / "TestScenario" / "LC_V20_H3p5_T8" / "S100001" /
               "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat")

    import outb_reader as obr
    import sd_geometry as sdg

    header = obr.read_outb_header(outb_path)
    model = sdg.read_subdyn_model(sd_path)

    print(f"bin edges: {N_BINS} bins, {BIN_EDGES_MPA[0]:.4f} .. {BIN_EDGES_MPA[-1]:.1f} MPa, log-spaced")
    print(f"Wohler exponents (fatigue_config.WOHLER_EXPONENTS): {WOHLER_EXPONENTS}\n")

    # Real data, several member/end/theta combinations -- not just one, so
    # a pass here isn't luck on a single easy case.
    cases = [(4, 1, 0), (4, 1, 7), (41, 2, 3), (101, 1, 0)]  # includes a
    # degenerate member (101, near-zero force -- flagged not_assessable
    # downstream, but the histogram code must still handle it cleanly:
    # near-constant signal -> near-zero/zero-range cycles -> n_under, not
    # a crash).

    for mid, end, k_theta in cases:
        t_trim, sigma = stress.compute_member_stress(outb_path, header, model, mid, end)
        signal = sigma[:, k_theta]
        cycles = list(rainflow.extract_cycles(signal))
        total_cycle_count = sum(c[2] for c in cycles)

        counts, sum_r, n_under, n_over = cycles_to_histogram(cycles)

        print(f"member {mid} end {end} theta_idx {k_theta} "
              f"(theta={np.degrees(stress.THETA_RAD[k_theta]):.1f} deg):")
        print(f"  {len(cycles)} cycles, total count (incl. 0.5-residue) = {total_cycle_count}")

        # 1. Count conservation: everything binned + n_under + n_over must
        # equal the total cycle count exactly (counts stay float throughout,
        # so this is an exact float equality, not an approximate one).
        conserved = counts.sum() + n_under + n_over
        print(f"  binned={counts.sum()}  n_under={n_under}  n_over={n_over}  "
              f"sum={conserved}  (expect exactly {total_cycle_count})")
        assert abs(conserved - total_cycle_count) < 1e-9, "count not conserved -- a cycle was dropped or double-counted"

        # 2. n_over must be 0 on real data -- if not, BIN_HI_MPA is too low.
        assert n_over == 0.0, f"n_over={n_over} -- BIN_HI_MPA={cfg.BIN_HI_MPA} is too low for this signal"

        # 3. Exact damage recovery, for every configured exponent: histogram
        # vs raw list. If every cycle fell below BIN_LO_MPA (counts.sum()==0
        # -- e.g. member 101, a near-rigid interface stub whose "cycles" are
        # pure numerical noise well under 0.01 MPa), d_raw is itself noise
        # and d_hist is correctly exactly 0 -- a relative-diff comparison
        # against noise is meaningless, so check exact-zero instead.
        for m in WOHLER_EXPONENTS:
            d_raw = damage_from_cycles(cycles, m)
            d_hist = damage_from_histogram(sum_r, m)
            if counts.sum() == 0.0:
                print(f"  m={m}: all cycles below BIN_LO_MPA -- raw(noise)={d_raw:.3e}  histogram={d_hist}")
                assert d_hist == 0.0
                continue
            rel_diff = abs(d_raw - d_hist) / d_raw if d_raw > 0 else 0.0
            print(f"  m={m}: raw={d_raw:.6e}  from histogram={d_hist:.6e}  rel.diff={rel_diff:.3e}")
            # 1e-9, not 1e-15 -- summing ~700 float64 terms grouped by bin
            # vs. in the raw list's original order accumulates rounding
            # differently (order-dependent float addition, not a bug).
            # Still 3+ orders of magnitude tighter than the counts-only
            # error this design avoids (see check 4 below).
            assert rel_diff < 1e-9, f"m={m}: histogram damage doesn't match raw cycle list -- power-sum bug"

        # 4. Counts-only would have been approximate -- quantify what the
        # power-sum design bought, don't just assert it away.
        if counts.sum() > 0:
            for m in WOHLER_EXPONENTS:
                d_raw = damage_from_cycles(cycles, m)
                d_counts_only = damage_from_counts_only(counts, BIN_EDGES_MPA, m)
                if d_raw > 0:
                    err_pct = 100.0 * abs(d_raw - d_counts_only) / d_raw
                    print(f"  m={m}: counts-only-midpoint approx error = {err_pct:.3f}%")
        print()

    print("all checks passed.")


if __name__ == "__main__":
    _self_check()
