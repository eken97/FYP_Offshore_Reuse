"""
Step 8 -- S-N curves (DNV-RP-C203).

Converts a stress-range histogram bin (Delta_sigma, MPa) into a cycles-to-
failure N, using DNV-RP-C203's bilinear (or, for free corrosion, single-
slope) S-N curve, with the thickness correction folded into the applied
stress range. This is the material-property half of Miner's rule; Stage 3
(Step 9) supplies the n (cycle count) half and does the actual damage sum.

    N = 10^log_a1 / Delta_sigma_eff^m1     for Delta_sigma_eff >= knee stress
    N = 10^log_a2 / Delta_sigma_eff^m2     for Delta_sigma_eff <  knee stress
    Delta_sigma_eff = Delta_sigma * (max(t, t_ref)/t_ref)^k

THIS WAS THE SECOND SIGN-OFF STEP (see the plan's "Values you verify, not
me" -- the first was sd_geometry.py's section_properties_CHECK.csv). The author
checked fatigue_results/sn_curves_CHECK.csv against their own copy of
DNV-RP-C203 and confirmed it correct (08.08.2026); SN_CONSTANTS_VERIFIED is
now True. SN_CONSTANTS_VERIFIED gates nothing by itself here -- it is read
and enforced by Stage 3 (Step 9, not yet built), which must still check
this flag itself and refuse to produce damage numbers if it is ever False
(e.g. after a future edit to SN_CURVES that hasn't been re-signed-off).

Two detail categories only -- see docs/decisions.md, "Detail
category -- RESOLVED 08.08.2026":
    B1  member/reuse track. Non-welded parent material (Table A-9) -- the
        physically correct detail once a member is cut from its joint and
        the cut edge ground smooth for reuse. Pairs with stress.py's
        SCF=1 member track. k=0 in every environment, so the thickness
        correction is a structural non-issue for this curve.
    T   joint track. Dedicated tubular-joint curve, meant to pair with
        hot-spot stress (nominal x SCF at 8 positions) -- NOT built yet
        (joint SCF/hot-spot method is still parked). k depends on SCF
        (0.25 if SCF<=10, else 0.30) -- a value that does not exist until
        that method exists, so T's k is exposed as two branches here and
        left for Stage 3/joint-stage to select, not resolved in this file.

Three environments per category (six curves total), all values read
directly off DNV-RP-C203 Tables 2-1/2-2/2-3, transcribed by the author from a licensed copy of the standard -- not typed
from memory. Real gotcha caught while cross-checking these against the
tables' own printed columns: the knee moves between tables (N=1e7 for air,
N=1e6 for seawater+CP; free corrosion has no knee at all -- single m=3
line, "for all cycles"), and the printed "Fatigue limit at 10^7 cycles"
column is NOT always the knee stress -- for Table 2-1 (air) it is (knee IS
at 1e7), but for Table 2-2 (seawater+CP) it's a DIFFERENT, further-out
point on the m2 branch evaluated at N=1e7, two decades past the real knee
at N=1e6. See the module's own self-check for both, computed and checked
separately, not conflated.

t_ref: 25 mm for B1 (welded connections other than tubular joints, and the
general default); 32 mm for T (tubular joints). Also confirmed directly
from the same document (the defining clause, not a table): t = "thickness
through which a crack will most likely grow"; t = t_ref is used whenever
the actual thickness is less than t_ref (a clamp, never a bonus for thin
sections). For a joint-track point, which physical thickness (chord side
or brace side) is passed as t is a joint-classification question -- one t
per side, not one t per joint -- deferred with the rest of the joint track.

DFF (Design Fatigue Factor) / splash-zone substitution -- CONSIDERED AND
REJECTED, 08.08.2026 (see docs/decisions.md for the full
discussion). DNV's commentary (D.6, on 2.4.9) allows substituting the
seawater+CP curve for splash-zone JOINTS instead of free corrosion, but
only when paired with a high Design Fatigue Factor and an assumed-intact
coating -- a reliability framework this pipeline does not implement.
Adopting the optimistic curve without its paired safety margin would
borrow DNV's assumption without what justifies it. Splash zone therefore
uses free corrosion for BOTH categories, by an explicit decision by the author -- a
stated, deliberate conservative assumption, not an oversight. State this
directly in the write-up: real remaining life at splash-zone members/
joints may be more favourable than reported, since a DFF-justified
substitution exists in the standard but was not applied here.

Final zone -> environment mapping (unambiguous, both categories):
    atmospheric -> air
    submerged   -> seawater_cp
    splash      -> free_corrosion   (see DFF decision above)
    buried      -> air              (no electrolyte cycling)

Steps:
    1. SN_CURVES -- the six (category, environment) rows, each carrying
       its own source citation and verification status.
    2. knee_stress(category, environment) -- Delta_sigma where the two
       branches meet, computed from (log_a1, m1, n_knee), independent of
       thickness (the correction is applied to the stress being evaluated,
       not to the curve's own geometry).
    3. thickness_factor(t_mm, t_ref_mm, k) -- (max(t,t_ref)/t_ref)^k.
    4. cycles_to_failure(delta_sigma_mpa, category, environment, t_mm,
       scf=None) -- the end-to-end N(Delta_sigma) Stage 3 will call once
       per histogram bin.
    5. write_sn_curves_check(out_path) -- the sign-off artifact.
"""
from pathlib import Path

import numpy as np

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# endregion


# region --- sign-off gate ---
# The author checked sn_curves_CHECK.csv against their own copy of DNV-RP-C203
# and confirmed it correct (08.08.2026) -- same sign-off discipline as
# section_properties_CHECK.csv in sd_geometry.py. Stage 3 (Step 9) must
# still check this flag itself and refuse to run if it is ever False.
SN_CONSTANTS_VERIFIED = True
# endregion


# region --- reference thickness, mm ---
T_REF_MM = {
    "B1": 25.0,   # welded connections other than tubular joints (general default)
    "T": 32.0,    # tubular joints
}
# endregion


# region --- the six curves ---
DOC_SOURCE = "DNV-RP-C203 Tables 2-1/2-2/2-3 (licensed copy; not redistributed)"

# n_knee/log_a2/m2 are None for free-corrosion rows -- single m1 slope,
# "for all cycles", no second branch. k_lo == k_hi for every B1 row (k=0
# always); T's k_lo/k_hi differ and require SCF to choose between them
# (see _k_for_curve).
SN_CURVES = {
    ("B1", "air"): dict(
        m1=4.0, log_a1=15.117, n_knee=1.0e7, log_a2=17.146, m2=5.0,
        k_lo=0.0, k_hi=0.0,
        source=f"DNV-RP-C203 Table 2-1 (air), sec 2.4.4 -- {DOC_SOURCE}",
        status="VERIFIED",
    ),
    ("B1", "seawater_cp"): dict(
        m1=4.0, log_a1=14.917, n_knee=1.0e6, log_a2=17.146, m2=5.0,
        k_lo=0.0, k_hi=0.0,
        source=f"DNV-RP-C203 Table 2-2 (seawater+CP), sec 2.4.5 -- {DOC_SOURCE}",
        status="VERIFIED",
    ),
    ("B1", "free_corrosion"): dict(
        m1=3.0, log_a1=12.436, n_knee=None, log_a2=None, m2=None,
        k_lo=0.0, k_hi=0.0,
        source=f"DNV-RP-C203 Table 2-3 (free corrosion), sec 2.4.9 -- {DOC_SOURCE}",
        status="VERIFIED",
    ),
    ("T", "air"): dict(
        m1=3.0, log_a1=12.164, n_knee=1.0e7, log_a2=15.606, m2=5.0,
        k_lo=0.25, k_hi=0.30,
        source=f"DNV-RP-C203 Table 2-1 (air), sec 2.4.4/2.4.6 -- {DOC_SOURCE}",
        status="VERIFIED",
    ),
    ("T", "seawater_cp"): dict(
        m1=3.0, log_a1=11.764, n_knee=1.0e6, log_a2=15.606, m2=5.0,
        k_lo=0.25, k_hi=0.30,
        source=f"DNV-RP-C203 Table 2-2 (seawater+CP), sec 2.4.5/2.4.6 -- {DOC_SOURCE}",
        status="VERIFIED",
    ),
    ("T", "free_corrosion"): dict(
        m1=3.0, log_a1=11.687, n_knee=None, log_a2=None, m2=None,
        k_lo=0.25, k_hi=0.30,
        source=f"DNV-RP-C203 Table 2-3 (free corrosion), sec 2.4.9 -- {DOC_SOURCE}",
        status="VERIFIED",
    ),
}

# Zone -> environment, per the DFF decision above. Not category-dependent
# -- both B1 and T points at the same z-zone use the same environment.
ZONE_TO_ENVIRONMENT = {
    "atmospheric": "air",
    "submerged": "seawater_cp",
    "splash": "free_corrosion",
    "buried": "air",
}
# endregion


# region --- curve evaluation ---
def knee_stress(category, environment):
    """
    Delta_sigma (MPa) where the two branches meet, from (log_a1, m1,
    n_knee) alone -- independent of thickness. None for free-corrosion
    rows (no second branch, so no knee).
    """
    row = SN_CURVES[(category, environment)]
    if row["n_knee"] is None:
        return None
    return 10.0 ** ((row["log_a1"] - np.log10(row["n_knee"])) / row["m1"])


def fatigue_limit_at_1e7(category, environment):
    """
    The stress on the LOW branch (log_a2, m2) at N=1e7 -- what DNV's own
    "Fatigue limit at 10^7 cycles" column means. Equal to knee_stress only
    when n_knee itself is 1e7 (true for the air table, NOT for seawater+CP
    or free corrosion -- see the module docstring). None where there is no
    second branch.
    """
    row = SN_CURVES[(category, environment)]
    if row["m2"] is None:
        return None
    return 10.0 ** ((row["log_a2"] - 7.0) / row["m2"])


def k_for_curve(row, scf):
    """
    k_lo == k_hi for every B1 row (always 0) -- returned directly, no SCF
    needed. T's k_lo/k_hi differ and require scf to pick a branch (0.25 if
    SCF<=10, else 0.30) -- see the module docstring. Public (not
    underscore-prefixed): stage3_damage.py (Step 9) needs the same
    thickness factor sn_curves.py uses internally, to decide per-bin which
    S-N branch (m1/log_a1 vs m2/log_a2) a stored histogram bin falls on --
    reuses this rather than re-deriving the k-selection rule a second time.
    """
    if row["k_lo"] == row["k_hi"]:
        return row["k_lo"]
    assert scf is not None, "SCF required to select k for this curve (k_lo != k_hi)"
    return row["k_hi"] if scf > 10.0 else row["k_lo"]


def thickness_factor(t_mm, t_ref_mm, k):
    """(max(t, t_ref)/t_ref)^k -- t=t_ref clamp, never a bonus for t<t_ref."""
    return (max(t_mm, t_ref_mm) / t_ref_mm) ** k


def cycles_to_failure(delta_sigma_mpa, category, environment, t_mm, scf=None):
    """
    N(Delta_sigma) for one histogram bin, MPa in, cycles out. delta_sigma_mpa
    may be scalar or array. t_mm is the wall thickness the crack would grow
    through at this point (member's own thickness for B1; chord- or brace-
    side thickness for T, one call per side). scf is required only when the
    looked-up row's k_lo != k_hi (T only).
    """
    row = SN_CURVES[(category, environment)]
    k = k_for_curve(row, scf)
    t_ref = T_REF_MM[category]
    factor = thickness_factor(t_mm, t_ref, k)
    ds_eff = np.asarray(delta_sigma_mpa, dtype=float) * factor

    if row["n_knee"] is None:
        return 10.0 ** row["log_a1"] / ds_eff ** row["m1"]

    ds_knee = knee_stress(category, environment)
    N1 = 10.0 ** row["log_a1"] / ds_eff ** row["m1"]
    N2 = 10.0 ** row["log_a2"] / ds_eff ** row["m2"]
    return np.where(ds_eff >= ds_knee, N1, N2)
# endregion


# region --- sign-off artifact ---
def write_sn_curves_check(out_path):
    """
    One row per (category, environment) -- the six curves, every constant
    traceable to its DNV-RP-C203 table via the source column.
    """
    rows = []
    for (cat, env), row in SN_CURVES.items():
        ds_knee = knee_stress(cat, env)
        fl_1e7 = fatigue_limit_at_1e7(cat, env)
        rows.append(dict(
            category=cat, environment=env,
            m1=row["m1"], log_a1=row["log_a1"],
            n_knee=row["n_knee"] if row["n_knee"] is not None else "",
            log_a2=row["log_a2"] if row["log_a2"] is not None else "",
            m2=row["m2"] if row["m2"] is not None else "",
            delta_sigma_knee_mpa=f"{ds_knee:.3f}" if ds_knee is not None else "",
            fatigue_limit_at_1e7_mpa=f"{fl_1e7:.3f}" if fl_1e7 is not None else "",
            k_lo=row["k_lo"], k_hi=row["k_hi"],
            t_ref_mm=T_REF_MM[cat],
            source=row["source"], status=row["status"],
        ))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    return rows
# endregion


def _self_check():
    print(f"SN_CONSTANTS_VERIFIED = {SN_CONSTANTS_VERIFIED}\n")

    # --- 1. knee self-consistency: both branches must meet at the knee, up
    # to the rounding DNV's own published tables carry. log_a1/log_a2 are
    # each independently rounded to 3 decimals in the standard, so N1/N2
    # can only agree to ~0.1%, not machine precision -- verified this IS
    # rounding, not a bug, by hand: propagating +/-0.0005 through
    # 10^log_a gives ~+/-0.12% on N, matching the ~0.06% actually seen.
    # 1e-9 would be the right tolerance for a knee DERIVED to full float
    # precision (as knee_stress() itself is); it is the wrong tolerance
    # for two independently 3-decimal-rounded SOURCE constants meeting --
    # loosen to 1% here and say so, rather than silently tightening the
    # source data by picking values that happen to agree better.
    print("Knee self-consistency (both branches evaluated at the curve's own "
          "n_knee should agree to within DNV's 3-decimal log_a rounding, ~0.1%):")
    for (cat, env), row in SN_CURVES.items():
        if row["n_knee"] is None:
            continue
        ds_knee = knee_stress(cat, env)
        N1 = 10.0 ** row["log_a1"] / ds_knee ** row["m1"]
        N2 = 10.0 ** row["log_a2"] / ds_knee ** row["m2"]
        diff = abs(N1 - N2) / N1
        print(f"  {cat:>2} {env:<15} n_knee={row['n_knee']:.0e}  "
              f"delta_sigma_knee={ds_knee:.3f} MPa  "
              f"N1={N1:.4e}  N2={N2:.4e}  rel.diff={diff:.2e}")
        assert diff < 3e-3, f"branches disagree at knee for {cat}/{env} by more than rounding explains"

    # --- 2. Cross-check against the tables' own printed columns directly
    # (not just internal self-consistency) -- catches a transcription typo
    # without either of us needing to be right.
    print("\nCross-check against DNV-RP-C203's own printed columns:")
    printed_knee_air = {"B1": 106.97, "T": 52.63}       # Table 2-1's own "fatigue limit" col IS the knee (n_knee=1e7 there)
    printed_fl1e7_cp = {"B1": 106.97, "T": 52.63}        # Table 2-2's printed col, NOT its knee (n_knee=1e6 there)
    # Relative, not absolute, tolerance -- same rounding source as the knee
    # check above (each log_a independently rounded to 3 decimals in the
    # published table), sized from what that rounding actually produces
    # here (largest observed: 0.015 MPa / 106.97 MPa = 1.4e-4).
    REL_TOL = 2e-3
    for cat in ("B1", "T"):
        ds_knee_air = knee_stress(cat, "air")
        print(f"  {cat} air: knee={ds_knee_air:.3f} MPa vs table's printed "
              f"{printed_knee_air[cat]:.3f} MPa")
        assert abs(ds_knee_air - printed_knee_air[cat]) / printed_knee_air[cat] < REL_TOL

        fl_cp = fatigue_limit_at_1e7(cat, "seawater_cp")
        ds_knee_cp = knee_stress(cat, "seawater_cp")
        print(f"  {cat} seawater_cp: fatigue_limit_at_1e7={fl_cp:.3f} MPa vs "
              f"table's printed {printed_fl1e7_cp[cat]:.3f} MPa "
              f"(NOTE: this is NOT the knee -- knee={ds_knee_cp:.3f} MPa at "
              f"N=1e6, two decades earlier)")
        assert abs(fl_cp - printed_fl1e7_cp[cat]) / printed_fl1e7_cp[cat] < REL_TOL
        assert ds_knee_cp > fl_cp, "CP knee stress should sit above the 1e7 reference point"

    # --- 3. Thickness clamp: t < t_ref must give factor exactly 1.0, not a
    # bonus. Use T (nonzero k) -- B1's k=0 makes this check trivial there.
    print("\nThickness clamp (t=20mm < t_ref=32mm for T -> factor must be 1.0):")
    factor = thickness_factor(20.0, T_REF_MM["T"], k=0.25)
    print(f"  thickness_factor(t=20, t_ref=32, k=0.25) = {factor:.6f}")
    assert factor == 1.0

    print("  thickness_factor(t=40, t_ref=32, k=0.25) "
          f"= {thickness_factor(40.0, 32.0, 0.25):.6f} (t > t_ref -> real correction, no clamp)")
    assert thickness_factor(40.0, 32.0, 0.25) > 1.0

    # --- 4. Monotonic N(Delta_sigma): life must strictly decrease as
    # stress range increases, for every curve.
    print("\nMonotonicity (N must strictly decrease as Delta_sigma increases):")
    ds_sweep = np.geomspace(10.0, 500.0, 200)
    for (cat, env) in SN_CURVES:
        t_mm = T_REF_MM[cat]  # no clamp/correction noise in this check
        scf = 5.0 if cat == "T" else None
        N = cycles_to_failure(ds_sweep, cat, env, t_mm, scf)
        assert np.all(np.diff(N) < 0), f"N not monotonic for {cat}/{env}"
    print("  OK for all 6 curves")

    # --- 5. Environment ordering: free_corrosion < seawater_cp <= air life,
    # over the stress range that is actually PHYSICALLY RELEVANT to this
    # jacket -- NOT a universal claim across all possible Delta_sigma.
    #
    # Two real, confirmed-from-the-tables reasons the blanket version of
    # this check is wrong, found while building it (neither is a pipeline
    # bug):
    #  (a) cp==air (not cp<air) below both curves' knees: B1 and T both
    #      carry IDENTICAL log_a2/m2 on their low-stress branch across the
    #      air and seawater+CP tables (log_a2=17.146/m2=5.0 for B1,
    #      15.606/5.0 for T, in BOTH tables) -- DNV only differentiates
    #      air vs CP in the high-stress (m1) region for these two curves.
    #  (b) B1's free-corrosion curve CROSSES seawater+CP above ~302.7 MPa:
    #      free corrosion is forced to a single m=3 slope (Table 2-3's own
    #      header: "log a, for all cycles, m=3.0"), but B1's air/CP tables
    #      use m1=4 above their knee -- a shallower exponent eventually
    #      wins at high enough stress despite a lower log_a. Crossover at
    #      Delta_sigma = 10^(log_a1_cp - log_a1_fc) = 10^(14.917-12.436) =
    #      302.7 MPa, verified by hand. This is a genuine feature of DNV's
    #      published constants, not a computation error.
    # Neither matters here: the extracted document's own sec 2.4.4 says
    # "for offshore structures subjected to typical wave and wind loading
    # the main contribution to fatigue damage is in the region N > 10^6
    # cycles" (i.e. moderate stress, well below any B1 crossover), and
    # bin_range_check.py already measured this jacket's actual max cycle
    # range at ~70 MPa across the real campaign data -- 4x below the
    # crossover, comfortably inside the region DNV itself says governs.
    # Check the physically relevant range only; state the limit rather
    # than silently narrowing it to make the assertion pass.
    ORDERING_CHECK_MAX_MPA = 200.0  # comfortably above ~70 MPa measured, below 302.7 MPa crossover
    ds_ordering = ds_sweep[ds_sweep <= ORDERING_CHECK_MAX_MPA]
    print(f"\nEnvironment ordering (free_corrosion < seawater_cp <= air life, "
          f"for Delta_sigma <= {ORDERING_CHECK_MAX_MPA:.0f} MPa -- the physically relevant "
          f"range, see comment for why NOT the full sweep):")
    for cat in ("B1", "T"):
        t_mm = T_REF_MM[cat]
        scf = 5.0 if cat == "T" else None
        N_air = cycles_to_failure(ds_ordering, cat, "air", t_mm, scf)
        N_cp = cycles_to_failure(ds_ordering, cat, "seawater_cp", t_mm, scf)
        N_fc = cycles_to_failure(ds_ordering, cat, "free_corrosion", t_mm, scf)
        ok = np.all(N_fc < N_cp) and np.all(N_cp <= N_air)
        n_equal = np.sum(N_cp == N_air)
        print(f"  {cat}: free_corrosion < seawater_cp <= air, ok={ok} "
              f"({n_equal}/{len(ds_ordering)} points where cp==air, all below both knees)")
        assert ok

    # --- 6. T's SCF-dependent k actually changes the result (k_lo != k_hi
    # must matter, not be silently ignored).
    print("\nT's SCF-dependent k branch actually changes N (k_lo=0.25 vs k_hi=0.30, t=40mm>t_ref):")
    N_lo = cycles_to_failure(100.0, "T", "air", 40.0, scf=5.0)   # SCF<=10 -> k_lo
    N_hi = cycles_to_failure(100.0, "T", "air", 40.0, scf=15.0)  # SCF>10  -> k_hi
    print(f"  SCF=5 (k=0.25): N={N_lo:.4e}   SCF=15 (k=0.30): N={N_hi:.4e}")
    assert N_lo != N_hi

    # --- write the sign-off artifact ---
    out_path = RESULTS_DIR / "sn_curves_CHECK.csv"
    rows = write_sn_curves_check(out_path)
    print(f"\nwrote {out_path}  ({len(rows)} rows)")

    print("\n" + "=" * 78)
    if SN_CONSTANTS_VERIFIED:
        print("SN_CONSTANTS_VERIFIED = True -- signed off. Stage 3 (Step 9) may use")
        print("these curves. If SN_CURVES is ever edited, set this back to False")
        print("until the new values are re-checked against DNV-RP-C203.")
    else:
        print("SIGN-OFF REQUIRED: open sn_curves_CHECK.csv and verify every log_a, m,")
        print("n_knee, k, t_ref against your own copy of DNV-RP-C203 before flipping")
        print("SN_CONSTANTS_VERIFIED = True. Stage 3 (Step 9) must refuse to run while")
        print("it is False.")
    print("=" * 78)


if __name__ == "__main__":
    _self_check()
