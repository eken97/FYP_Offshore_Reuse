"""
Single source of truth for every ASSUMED number in the fatigue pipeline --
the point of this file is that you can open it alone and see everything
that's being taken as given, with its verification status, rather than
hunt through stress.py/rainflow_hist.py/sn_curves.py for scattered
constants.

Every value below falls into one of two categories:
  VERIFIED  -- derived from the model itself (nothing here yet -- section
              properties in sd_geometry.py are the closest example, and
              those are signed off per-run against SubDyn.dat, not stored
              as a fixed constant here).
  ASSUMED   -- a modelling choice or a literature value, not measured from
              this specific structure. Marked UNVERIFIED until you've
              independently checked it. sn_curves.py (Step 8) will add the
              S-N constants (log_a, knee, thickness exponent) to this list
              with the same discipline once it's built.

Why this matters more than a normal "config file": some of these values
are baked into STORED data (Stage 2 .npz files, ~1-3 GB across the
campaign), not just read at the point of use. If WOHLER_EXPONENTS turns
out wrong, every Stage 2 file must be recomputed from raw .outb data
(expensive) -- unlike, say, a wrong log_a, which only requires Stage 3 to
re-read data that's already correct. See WOHLER_EXPONENTS below for the
detail. This is exactly why the exponents were pulled out of
rainflow_hist.py's hardcoded sum_r3/sum_r5 fields into this file
(06.08.2026 review) -- so a correction is a one-line edit + recompute, not
a code change.

If PIPELINE_VERSION changes, every Stage 2/3 output must be recomputed
(see run_pipeline.py, Step 10) -- consumers should stamp this value and
refuse to trust a file whose stamp doesn't match.
"""
import numpy as np

PIPELINE_VERSION = 2


# region --- ASSUMED: stress recovery (stress.py, Step 4) ---
# Number of circumferential points sampled per member end. Provably
# irrelevant WHICH 16 (theta-origin invariant, see stress.py) for the
# member track; the COUNT itself (16, not 8) is justified by
# test_invariants.py's theta-resolution check (Step 5): n=8 undercounts
# damage by a factor cos(22.5deg)^3 = 0.7886 relative to n=16 in the worst
# case. Changing this changes Stage 2's array shape -- bump
# PIPELINE_VERSION and recompute if it ever changes.
N_THETA = 16

# Drop the OpenFAST startup transient before rainflow counting. Chosen
# once, applied by TIME VALUE (t >= cutoff), never by a hard-coded sample
# index -- see stress.py's trim_transient.
TRANSIENT_CUTOFF_S = 100.0
# endregion


# region --- ASSUMED: rainflow bin range (rainflow_hist.py, Step 6) ---
# Bin edges must be identical across every run, seed, and member -- Stage 3
# sums histograms across seeds and bins, which only works if every Stage 2
# file used the same edges. Log-spaced because stress ranges in this
# problem span several orders of magnitude, and damage is governed by the
# upper tail, where log spacing gives useful resolution without an
# unmanageable bin count.
#
# Range 0.01-1000 MPa: 0.01 MPa is far below anything that could
# meaningfully accumulate fatigue damage on steel; 1000 MPa is comfortable
# headroom above this jacket's real stress ranges as seen on ONE
# moderate-severity run (LC65, V=20/Hs=3.5 -- Step 4's self-check saw
# sigma itself in the tens of MPa). NOT YET independently checked against
# the campaign's most severe (highest wind/wave) bins -- see the range-
# validation discussion, 06.08.2026: n_over==0 has only been confirmed on
# one mid-severity condition so far, not the full envelope. Do that check
# before trusting this range across all 414 runs.
BIN_LO_MPA = 0.01
BIN_HI_MPA = 1000.0
N_BINS = 256

# 257 edges -> 256 bins, log-spaced. Frozen: do not change without bumping
# PIPELINE_VERSION and recomputing every Stage 2 file.
BIN_EDGES_MPA = np.geomspace(BIN_LO_MPA, BIN_HI_MPA, N_BINS + 1)
# endregion


# region --- Wohler (S-N) exponents actually used by the SIGNED-OFF sn_curves.py (rainflow_hist.py, sn_curves.py) ---
# The power-law exponents rainflow_hist.py stores exact power sums for
# (sum_r[m] = sum(count * range**m) per bin, for every m in this tuple).
# CORRECTED 09.08.2026 while building Stage 3 (stage3_damage.py, Step 9):
# the original guess here was (3, 5), assumed uniform across all curves.
# That is wrong for the category this pipeline actually uses. Read directly
# off sn_curves.py's now-VERIFIED SN_CURVES table: B1's air and
# seawater_cp curves (which cover 80 of 112 members -- every atmospheric
# and submerged-zone member) use m1=4.0 above their knee, not m1=3. Only
# B1's free-corrosion curve and all of T's curves use m1=3. m2=5.0
# everywhere there is a second branch. So the exponent SET actually needed
# for exact power-sum recovery is {3, 4, 5}, not {3, 5} -- (3, 5) alone
# would have silently forced every atmospheric/submerged member's
# high-stress-branch damage through the wrong exponent's stored data (an
# m=3 sum used where the curve says m=4), a real bug that would have been
# invisible in the output (a plausible-looking damage number, just wrong).
# Caught by cross-checking WOHLER_EXPONENTS against sn_curves.py's actual
# (category, environment) table before Stage 3 could use it -- not found
# by inspection alone.
#
# UNLIKE log_a/knee/thickness-exponent (which Stage 3 reads fresh from
# Stage 2's already-correct histograms), a WRONG exponent here means the
# stored sum_r[m] arrays are for the wrong power entirely and cannot be
# corrected by re-reading -- every Stage 2 .npz file must be recomputed
# from the raw .outb data. This correction bumped PIPELINE_VERSION (1->2)
# so any Stage 2 file written under the old (3, 5) set is detected as
# stale and rejected, not silently reused.
WOHLER_EXPONENTS = (3, 4, 5)
# endregion


# region --- ASSUMED: not-assessable member classification (stage2_histograms.py, Step 7) ---
# Members treated as "not a normal structural member" for fatigue
# REPORTING purposes -- identified during Step 2/3 (see sd_geometry.py,
# docs/decisions.md). Stress recovery, rainflow counting, and a
# damage number are still computed for these exactly like every other
# member (see stage2_histograms.py's build_point_table) -- nothing is
# silently dropped. This list only controls the reason string attached to
# each one, so a reader of the final output knows not to trust that number
# as a real fatigue-life estimate:
#   101-104  TP interface stubs, elastic force ~1.3e-8 N -- pure numerical
#            noise, not a real structural signal.
#   105-108  grouted equivalent tube, MatDens=3339 kg/m3 -- not real steel.
#   109-112  piles, buried -- real members, just outside this pipeline's
#            splash/corrosion zoning logic as currently scoped.
NOT_ASSESSABLE_REASONS = [
    (101, 104, "interface_degenerate"),
    (105, 108, "grouted_equivalent"),
    (109, 112, "buried_pile"),
]


def member_not_assessable_reason(mid):
    """None for an ordinary member; else the reason string from the table above."""
    for lo, hi, reason in NOT_ASSESSABLE_REASONS:
        if lo <= mid <= hi:
            return reason
    return None
# endregion


# region --- diagnostic/QA scan scope (NOT the real Stage 2/3 pipeline) ---
# Members to SKIP ENTIRELY in screening/diagnostic tools like
# bin_range_check.py, purely to save scan time on members that carry no
# meaningful signal. Derived from NOT_ASSESSABLE_REASONS above so the two
# lists can never silently drift apart.
#
# IMPORTANT: this is NOT the same mechanism as the not_assessable flag
# above. Stage 2/3 still compute and report a damage number for ALL 112
# members (see docs/decisions.md, "all 112 members get a
# damage number eventually"). This list only controls what ad-hoc
# QA/screening scripts bother looking at, to keep them fast; it must never
# be imported by stage2_histograms.py or stage3_damage.py.
SCREENING_EXCLUDED_MEMBER_IDS = tuple(
    mid for lo, hi, _reason in NOT_ASSESSABLE_REASONS for mid in range(lo, hi + 1)
)
# endregion
