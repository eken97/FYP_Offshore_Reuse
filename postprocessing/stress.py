"""
Step 4 -- Stress recovery.

Converts the internal force/moment channels outb_reader.py (Step 1) reads
-- axial force N and the two bending moments Mkx, Mky -- into a nominal
stress time history around the tube's circumference, at 16 fixed angles.
This is the raw material rainflow counting (Step 6) will need. It is never
written to disk -- one member-end's full-length stress-at-16-theta array
for one run is the largest transient object in the whole pipeline, and
Stage 2 (Step 7) only ever needs its rainflow histogram, not the signal
itself.

    sigma(theta, t) = N(t)/A + (Mkx(t)/W)*cos(theta) - (Mky(t)/W)*sin(theta)
    where W = I/R (section modulus). Form per Pacheco, J.; Pimenta, F.;
    Pereira, S.; Cunha, A.; Magalhaes, F. "Fatigue Assessment of Wind
    Turbine Towers: Review of Processing Strategies with Illustrative Case
    Study." Energies 2022, 15(13), 4782. doi:10.3390/en15134782 -- their
    eq. for sigma_z(theta_i) = (Mx/W)*cos(theta_i) - (My/W)*sin(theta_i) +
    FN/A. Metadata verified via Crossref; swapped in from the sin/-cos form
    06.08.2026 by the author (they have their own copy of this
    paper for the thesis). Exact quoted equation not independently
    verified against the PDF beyond what's shown above -- check the
    author's own copy before citing in the thesis text (see
    docs/decisions.md).

Per the joint seam (see the build plan and docs/decisions.md),
the three nominal components are kept SEPARATE until hotspot_member
combines them -- nominal_components() alone is what a future joint track
would reuse with per-component SCFs, since SCF*(A+B) != SCF_A*A + SCF_B*B
and rainflow is not linear. The fork must happen before rainflow; this
module is that fork point for the member track (SCF=1 everywhere).

theta=0 origin and the cos/-sin sign convention are pinned here and will
be stamped into every Stage 2 file (Step 7). For the MEMBER track this
choice is provably irrelevant: a uniform 16-point grid maps onto itself
under 90-degree rotation and reflection, so all sign/axis permutations of
this formula (including the sin/-cos form checked during design) give an
identical SET of 16 sigma values, just relabeled by theta -- verified
numerically to 2.7e-15 when this formula was swapped in (06.08.2026), so
switching form changes nothing about the resulting damage. That invariance
does NOT hold once joints bring in per-position SCFs, which is why the
convention is pinned and recorded now rather than left implicit.

Steps:
    1. nominal_components(N, Mkx, Mky, D, t) -- axial + two bending nominal
       stresses, in MPa (DNV S-N constants assume MPa; converting once here
       means nothing downstream has to remember to).
    2. hotspot_member(sig_ax, sig_ipb, sig_opb, theta) -- combine into
       sigma(theta) at SCF=1, broadcasting over time and theta together.
    3. trim_transient(t, arr, t_cutoff) -- drop the startup transient by
       TIME VALUE (t >= t_cutoff), not by index -- dt is fixed within a run
       but must never be assumed equal across runs.
    4. member_end_stress_history(...) / compute_member_stress(...) -- the
       end-to-end per-member-end stress recovery used by Step 6 onward.
"""
import math
from pathlib import Path

import numpy as np

import fatigue_config as cfg
import outb_reader as obr
import sd_geometry as sdg

# region --- pinned constants ---
# N_THETA and TRANSIENT_CUTOFF_S are ASSUMED values -- see fatigue_config.py
# for the single source of truth and their verification status. Kept here
# only as local aliases so the rest of this module reads naturally.
def theta_grid(n_theta):
    """n_theta equally spaced angles in [0, 2*pi), radians. theta=0 origin
    per the pinned convention above."""
    return np.linspace(0.0, 2 * np.pi, n_theta, endpoint=False)


N_THETA = cfg.N_THETA
THETA_RAD = theta_grid(N_THETA)
TRANSIENT_CUTOFF_S = cfg.TRANSIENT_CUTOFF_S
# endregion


# region --- nominal components ---
def nominal_components(N, Mkx, Mky, D, t):
    """
    N (axial force, N), Mkx/Mky (bending moments, N*m) -> three SEPARATE
    nominal stress components in MPa: (sig_ax, sig_ipb, sig_opb).

    sig_ax  = N / A
    sig_ipb = Mkx * R / I   (bending about local x)
    sig_opb = Mky * R / I   (bending about local y)

    D, t here are the member's section (from sd_geometry.member_section),
    NOT time -- named to match section_properties' own signature.
    Kept as three separate arrays rather than combined here -- see the
    module docstring on the joint seam.
    """
    props = sdg.section_properties(D, t)
    A, R, I = props["A"], props["R"], props["I"]
    sig_ax = (np.asarray(N) / A) / 1e6
    sig_ipb = (np.asarray(Mkx) * R / I) / 1e6
    sig_opb = (np.asarray(Mky) * R / I) / 1e6
    return sig_ax, sig_ipb, sig_opb


def hotspot_member(sig_ax, sig_ipb, sig_opb, theta=THETA_RAD):
    """
    Combine the three nominal components into sigma(theta, t) for the
    MEMBER track (SCF=1 at every position -- a joint track would apply a
    separate SCF per component here instead, see the module docstring).

        sigma(theta) = sig_ax + sig_ipb*cos(theta) - sig_opb*sin(theta)

    sig_ax/sig_ipb/sig_opb: shape (n_t,), MPa.
    theta: shape (n_theta,), radians.
    Returns: shape (n_t, n_theta), MPa.
    """
    sig_ax = np.asarray(sig_ax)[:, None]
    sig_ipb = np.asarray(sig_ipb)[:, None]
    sig_opb = np.asarray(sig_opb)[:, None]
    theta = np.asarray(theta)[None, :]
    return sig_ax + sig_ipb * np.cos(theta) - sig_opb * np.sin(theta)
# endregion


# region --- Step D: joint hotspot stress (DNV eqn 3.3.1, 8-point crown/
# saddle/intermediate superposition) ---
#
# Pure maths only, same seam as nominal_components/hotspot_member above:
# takes already-computed nominal stresses and already-computed SCFs
# (scf.py, Step C) as ARGUMENTS, does not read any channel or geometry
# itself. Reading the real per-connection chord-bending signal from a
# .outb (both chord segments, per joint_geometry.py's Step B chord_a_W/
# chord_b_W/chord_a_phi_deg/chord_b_phi_deg + rotate_to_joint_axes) is
# Step E's job (stage2_joints.py, not yet built) -- exactly parallel to
# how nominal_components()/hotspot_member() are pure maths and
# compute_member_stress() is the separate I/O wrapper around them.
#
# Per the module docstring's "THE FORK IS HERE, NOT LATER" note above:
# this function starts from THREE separate brace nominal components
# (sig_ax, sig_mip, sig_mop -- already rotated into the joint's own
# in-plane/out-of-plane frame via joint_geometry.rotate_to_joint_axes,
# a DIFFERENT rotation from the member-track's raw Mkx/Mky) plus the
# chord's own nominal bending signal, never from hotspot_member()'s
# combined output.

THETA_8_DEG = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
THETA_8_RAD = np.radians(THETA_8_DEG)

HOTSPOT_JOINT_VERIFIED = True  # the author's own sign-off, 14.08.2026 -- see docstring below


def hotspot_joint(sig_ax, sig_mip, sig_mop, sig_cb,
                   SCF_AC_base, SCF_AC_att, SCF_AS, SCF_MIP, SCF_MOP):
    """
    DNV (3.3.1)'s 8-point crown/saddle/intermediate superposition for ONE
    brace-to-chord connection (call once per side already resolved by
    scf.py -- this function does not know chord vs brace, only takes the
    SCF set for whichever side it's given).

    sig_ax, sig_mip, sig_mop: BRACE nominal stress components, MPa, shape
        (n_t,) -- sig_ax from nominal_components() unchanged, sig_mip/
        sig_mop from rotate_to_joint_axes(sig_ipb, sig_opb, phi_deg) using
        the BRACE's own phi_deg (joint_geometry Step 4), NOT the member-
        track's raw sig_ipb/sig_opb.
    sig_cb: CHORD nominal bending stress, MPa, shape (n_t,) -- DNV's
        "sigma_BendingChord" (eqn 6b/7b "Alternatively" box), evaluated at
        the CROWN TOE (theta=0deg, the direction pointing from the chord's
        own cross-section toward the brace -- joint_geometry's
        chord_e_par_axis / chord_a_phi_deg or chord_b_phi_deg, ONE chord
        segment at a time per the "never combine chord segments before
        rainflow" rule). The crown HEEL (theta=180deg) uses this SAME
        signal, SAME sign, unflipped -- see the CROWN DERIVATION section
        below for why (RESOLVED 14.08.2026, the author's own primary-source
        re-check + eqn 3.3.1's own printed structure -- was previously a
        working assumption with a sign flip, now corrected).
    SCF_AC_base, SCF_AC_att, SCF_AS, SCF_MIP, SCF_MOP: static per-
        connection, per-side SCFs, straight out of scf.py's _side_result
        dict (AC_base/AC_att are the crown's two pieces -- see scf.py's
        module docstring for why AC_att cannot be pre-multiplied into a
        single crown SCF; the rest are plain static SCFs, unchanged).

    Returns a dict of 8 arrays, shape (n_t,) each, keyed "1".."8" (DNV's
    own Figure 3-6 numbering: 1=crown toe, 3=saddle, 5=crown heel,
    7=saddle, 2/4/6/8=intermediate, going around at 45deg per hotspot_
    member's own theta=0/cos/-sin convention).

    CROWN DERIVATION (the "safe rearrangement" flagged in scf.py's module
    docstring). DNV's literal crown formula is

        sigma_1 = SCF_AC * sig_ax + SCF_MIP * sig_mip
        SCF_AC  = AC_base + AC_att * sig_cb / sig_ax        (eqn 6b/7b)

    which divides by sig_ax -- unsafe in a time-domain rainflow signal,
    where a brace axial force genuinely crosses zero. Substituting SCF_AC
    into sigma_1 and cancelling sig_ax is an EXACT algebraic identity
    everywhere sig_ax != 0, and (unlike the ratio form) the resulting
    expression is also correctly defined AT sig_ax=0:

        sigma_1 = AC_base*sig_ax + AC_att*sig_cb + SCF_MIP*sig_mip

    -- this is what the code below actually evaluates, DNV's literal box
    form is never computed. Same identity for sigma_5 (crown heel), with
    ONLY sig_mip negated (theta=180: cos=-1) -- NOT sig_cb.

    CROWN HEEL SIGN -- RESOLVED 14.08.2026 (was an open, unverified working
    assumption; see docs/decisions.md session record for the
    full exchange). Eqn 3.3.1's own printed crown formulas are
        sigma_1 = SCF_AC*sx + SCF_MIP*smy
        sigma_5 = SCF_AC*sx - SCF_MIP*smy
    -- the `SCF_AC*sx` term is IDENTICAL, UNFLIPPED, between toe and heel;
    only the bending term SCF_MIP*smy flips. This is physically required:
    sx (brace axial stress) does not vary around the chord's circumference
    at all (uniform tension/compression), so nothing multiplying it should
    either. Eqn 6b's "Alternatively" box does not add a new term alongside
    SCF_AC -- it REDEFINES what SCF_AC itself equals (SCF_Cc = geometry
    terms + (sigma_BendingChord/sigma_Axialbrace)*SCFatt), and the box
    itself gives no separate toe/heel instruction (confirmed by re-checking
    the primary source specifically for this -- no note saying the ratio's
    sign should differ by position, it is stated once, generically). Since
    eqn 3.3.1 never flips SCF_AC*sx's sign between the two crown positions,
    and 6b's own text doesn't introduce a new sign rule either, the
    attachment term (part of SCF_AC) must carry the SAME sign at both
    crown positions -- the earlier working assumption (flip sig_cb at the
    heel, treating it like a bending-type term) was wrong; sig_cb behaves
    like the axial term it modifies, not like sig_mip.

    INTERMEDIATE POINTS (2/4/6/8) -- RESOLVED 14.08.2026 (later still): DNV's
    (3.3.1) averages the axial SCF at intermediate points as
    "0.5*(SCF_AC+SCF_AS)", written purely in terms of the SYMBOL SCF_AC --
    (3.3.1) does not know about a separate "SCF_AC_base"/"SCF_AC_att" split,
    that distinction exists only in this codebase as a computational device
    for keeping the ratio form safe at sig_ax=0. Per the SAME reasoning that
    resolved the crown-heel sign above (the author's own re-read of the
    "Alternatively" box's surrounding commentary: 6b/7b, in general
    recommended used, simply IS SCF_AC -- not a bonus term layered on a
    narrower "base" concept), the intermediate-point average must use the
    FULL SCF_AC, i.e. INCLUDE AC_att -- the earlier working assumption
    (drop AC_att here, treat the attachment effect as crown-local) was
    wrong, same failure class as the heel-sign guess: an unstated,
    unverified physical narrowing of a symbol DNV defines once, generally.

    Same safe-rearrangement identity as the crown, extended one step:
        0.5*(SCF_AC+SCF_AS)*sig_ax
      = 0.5*(AC_base + AC_att*sig_cb/sig_ax + SCF_AS)*sig_ax
      = 0.5*(AC_base+SCF_AS)*sig_ax + 0.5*AC_att*sig_cb
    -- division cancels safely, exact algebraic identity, correctly defined
    at sig_ax=0, same as the crown's own derivation above.

    CONSEQUENCE FOR CALLERS: positions 2/4/6/8 now depend on sig_cb, same as
    1/5 -- only 3/7 (pure saddle, SCF_AS*sig_ax +/- SCF_MOP*sig_mop, no
    SCF_AC anywhere in their formula) remain segment-independent for K/TY
    connections. stage2_joints.py's iter_assessment_rows() must build 6 of
    the 8 positions twice (once per chord leg-segment), not just the 2
    crown positions -- see that module's own docstring for the resulting
    signal-count change (3,552 -> 4,768).

    G3 SIGN-OFF: with SCF_AC_base=SCF_AS=SCF_MIP=SCF_MOP=1 and SCF_AC_att=0,
    this function must reproduce hotspot_member(sig_ax, sig_mip, sig_mop,
    THETA_8_RAD) exactly at all 8 points (a regression that the joint
    formula collapses to the already-signed-off member formula in the
    trivial case) -- see _self_check() below, which asserts this to machine
    precision. Both open judgement calls (crown-heel sign, intermediate-
    point AC_att inclusion) were RESOLVED via primary-source re-checks +
    eqn 3.3.1's own structure -- neither is "silently guessed."

    HOTSPOT_JOINT_VERIFIED = True as of 14.08.2026 -- this is the AUTHOR'S
    OWN sign-off on the reasoning above, not an independent
    confirmation. The 3 findings (crown-heel sign, SCF_ATT=1.00,
    intermediate-point AC_att inclusion) still need independent
    review before treating this as externally validated --
    see docs/decisions.md session handoff for the full record of
    that decision and what's still pending.
    """
    sig_ax = np.asarray(sig_ax, dtype=np.float64)
    sig_mip = np.asarray(sig_mip, dtype=np.float64)
    sig_mop = np.asarray(sig_mop, dtype=np.float64)
    sig_cb = np.asarray(sig_cb, dtype=np.float64)

    sqrt2_2 = math.sqrt(2.0) / 2.0
    # Safe rearrangement of 0.5*(SCF_AC+SCF_AS)*sig_ax with SCF_AC = eqn
    # 6b/7b substituted in -- see docstring. AC_att now INCLUDED (was
    # excluded before this session's resolution).
    ac_avg_base = 0.5 * (SCF_AC_base + SCF_AS)
    ac_avg_att = 0.5 * SCF_AC_att * sig_cb

    return {
        "1": SCF_AC_base * sig_ax + SCF_AC_att * sig_cb + SCF_MIP * sig_mip,
        "2": ac_avg_base * sig_ax + ac_avg_att + sqrt2_2 * SCF_MIP * sig_mip - sqrt2_2 * SCF_MOP * sig_mop,
        "3": SCF_AS * sig_ax - SCF_MOP * sig_mop,
        "4": ac_avg_base * sig_ax + ac_avg_att - sqrt2_2 * SCF_MIP * sig_mip - sqrt2_2 * SCF_MOP * sig_mop,
        "5": SCF_AC_base * sig_ax + SCF_AC_att * sig_cb - SCF_MIP * sig_mip,
        "6": ac_avg_base * sig_ax + ac_avg_att - sqrt2_2 * SCF_MIP * sig_mip + sqrt2_2 * SCF_MOP * sig_mop,
        "7": SCF_AS * sig_ax + SCF_MOP * sig_mop,
        "8": ac_avg_base * sig_ax + ac_avg_att + sqrt2_2 * SCF_MIP * sig_mip + sqrt2_2 * SCF_MOP * sig_mop,
    }


def _self_check_hotspot_joint():
    """G3 regression only (see hotspot_joint's own docstring for what this
    does and does NOT prove). Synthetic data -- no .outb read, since a real
    per-connection chord-bending signal is Step E's job, not this one."""
    print(f"HOTSPOT_JOINT_VERIFIED = {HOTSPOT_JOINT_VERIFIED}")
    rng = np.random.default_rng(0)
    n_t = 500
    sig_ax = rng.normal(-30.0, 15.0, n_t)     # crosses zero -- the case the
    sig_mip = rng.normal(0.0, 8.0, n_t)       # safe-rearrangement derivation
    sig_mop = rng.normal(0.0, 8.0, n_t)       # is specifically for.
    sig_cb = rng.normal(0.0, 5.0, n_t)

    # unit SCFs, zero attachment term -> must reduce EXACTLY to hotspot_member
    unit = hotspot_joint(sig_ax, sig_mip, sig_mop, sig_cb,
                          SCF_AC_base=1.0, SCF_AC_att=0.0, SCF_AS=1.0,
                          SCF_MIP=1.0, SCF_MOP=1.0)
    member_ref = hotspot_member(sig_ax, sig_mip, sig_mop, theta=THETA_8_RAD)
    max_err = 0.0
    for i, key in enumerate("12345678"):
        max_err = max(max_err, np.max(np.abs(unit[key] - member_ref[:, i])))
    print(f"max |hotspot_joint(unit SCF, AC_att=0) - hotspot_member| over "
          f"{n_t} samples x 8 positions: {max_err:.2e} (expect ~0)")
    assert max_err < 1e-9

    # AC_att>0 must change 6 of the 8 positions (1,2,4,5,6,8 -- crown +
    # intermediate, since AC_att is now INCLUDED in the intermediate
    # average per this session's resolution) and leave ONLY the pure-saddle
    # positions (3, 7) untouched -- catches the attachment term leaking
    # into the wrong position, updated from the earlier (now superseded)
    # "only 1/5 change" version of this check.
    with_att = hotspot_joint(sig_ax, sig_mip, sig_mop, sig_cb,
                              SCF_AC_base=1.0, SCF_AC_att=1.27, SCF_AS=1.0,
                              SCF_MIP=1.0, SCF_MOP=1.0)
    for key in ("3", "7"):
        assert np.array_equal(unit[key], with_att[key]), (
            f"AC_att leaked into pure-saddle position {key} -- should never affect 3/7")
    for key in ("1", "2", "4", "5", "6", "8"):
        assert not np.array_equal(unit[key], with_att[key]), (
            f"AC_att did NOT affect position {key} -- expected it to, per this "
            f"session's intermediate-point resolution")
    print("AC_att isolated to crown+intermediate positions (1,2,4,5,6,8), "
          "pure-saddle (3,7) untouched: confirmed")

    # AC_att's contribution at the toe must be IDENTICAL to its contribution
    # at the heel (RESOLVED 14.08.2026 -- see docstring's "CROWN HEEL SIGN"
    # section: eqn 3.3.1's own SCF_AC*sx term is unflipped between toe/heel,
    # and AC_att is part of SCF_AC, so it carries the same sign at both --
    # this used to assert the OPPOSITE (a same-magnitude sign flip), which
    # was the now-corrected working assumption).
    toe_att_contribution = with_att["1"] - unit["1"]
    heel_att_contribution = with_att["5"] - unit["5"]
    assert np.allclose(toe_att_contribution, heel_att_contribution)

    # Every intermediate position's AC_att contribution must be exactly
    # HALF the crown's own contribution (0.5*AC_att*sig_cb vs AC_att*sig_cb,
    # per the "0.5*(SCF_AC+SCF_AS)" average) -- and all four intermediate
    # positions (2,4,6,8) must agree with each other, since none of them
    # apply a directional sign to sig_cb (only sig_mip/sig_mop pick up the
    # +/- sqrt2/2 pattern).
    crown_att_contribution = toe_att_contribution
    for key in ("2", "4", "6", "8"):
        intermediate_att_contribution = with_att[key] - unit[key]
        assert np.allclose(intermediate_att_contribution, 0.5 * crown_att_contribution), (
            f"position {key}'s AC_att contribution should be exactly half the crown's"
        )
    print("intermediate positions' AC_att contribution is exactly half the crown's, "
          "consistent across 2/4/6/8: confirmed")
    print("crown toe/heel AC_att contributions are identical (unflipped "
          "sign, per eqn 3.3.1's own SCF_AC*sx structure): confirmed")
    print("\n" + "=" * 78)
    if HOTSPOT_JOINT_VERIFIED:
        print("HOTSPOT_JOINT_VERIFIED = True -- signed off. Step E may build on this.")
    else:
        print("SIGN-OFF REQUIRED: this is a G3 REGRESSION check only (unit-SCF")
        print("reduction to hotspot_member). Both open judgement calls -- the sig_cb")
        print("heel-sign convention, and the intermediate-point AC_att inclusion --")
        print("were RESOLVED 14.08.2026 (primary-source re-checks + eqn 3.3.1's own")
        print("structure, see docstring). Neither has been independently reviewed")
        print("yet -- do that before flipping HOTSPOT_JOINT_VERIFIED = True.")
    print("=" * 78)
# endregion


# region --- transient trim ---
def trim_transient(t, arr, t_cutoff=TRANSIENT_CUTOFF_S):
    """
    Keep only samples with t >= t_cutoff. Trims by TIME VALUE, never by a
    hard-coded sample-count offset -- dt is 0.05 s on every run seen so far
    but must never be assumed fixed across runs (see outb_reader.py).
    arr's first axis must be time-aligned with t.
    """
    mask = t >= t_cutoff
    return t[mask], arr[mask]
# endregion


# region --- end-to-end per-member-end recovery ---
def member_end_stress_history(t, N, Mkx, Mky, D, wall_t,
                               theta=THETA_RAD, t_cutoff=TRANSIENT_CUTOFF_S):
    """
    Full pipeline for one member end: raw (N, Mkx, Mky) time series -> nominal
    components -> sigma(theta, t) -> transient trim.
    Returns (t_trim, sigma_trim) with sigma_trim shape (n_t_trim, n_theta), MPa.
    """
    sig_ax, sig_ipb, sig_opb = nominal_components(N, Mkx, Mky, D, wall_t)
    sigma = hotspot_member(sig_ax, sig_ipb, sig_opb, theta)
    return trim_transient(t, sigma, t_cutoff)


def compute_member_stress(outb_path, header, model, mid, end,
                           theta=THETA_RAD, t_cutoff=TRANSIENT_CUTOFF_S):
    """
    Read one member end's (FKze, MKxe, MKye) straight out of a .outb and
    return its trimmed sigma(theta, t), MPa. end: 1 (J1) or 2 (J2).
    """
    D, wall_t, pid = sdg.member_section(model, mid)
    names = [f"M{mid}J{end}{c}" for c in obr.FATIGUE_COMPONENTS]
    t, arr = obr.read_channels(outb_path, header, names)
    N, Mkx, Mky = arr[:, 0], arr[:, 1], arr[:, 2]
    return member_end_stress_history(t, N, Mkx, Mky, D, wall_t, theta, t_cutoff)
# endregion


def _self_check():
    project = Path(__file__).resolve().parent.parent.parent   # .../OpenFast
    outb_path = (project / "TestScenario" / "LC_V20_H3p5_T8" / "S100001" /
                 "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb")
    sd_path = (project / "TestScenario" / "LC_V20_H3p5_T8" / "S100001" /
               "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat")
    assert outb_path.exists(), f"missing fixture: {outb_path}"

    header = obr.read_outb_header(outb_path)
    model = sdg.read_subdyn_model(sd_path)

    print(f"theta grid ({N_THETA} pts, deg): {np.degrees(THETA_RAD).round(1)}")
    print(f"transient cutoff: t >= {TRANSIENT_CUTOFF_S} s\n")

    # Member 4: a leg, near-vertical, previously confirmed to carry real
    # compressive axial load (~-5.4 MN, matching platform self-weight) --
    # see outb_reader.py's Step 1 verification notes.
    mid, end = 4, 1
    D, wall_t, pid = sdg.member_section(model, mid)
    print(f"member {mid} end {end}: propset {pid}, D={D:.4f} t={wall_t:.4f}")

    t_trim, sigma = compute_member_stress(outb_path, header, model, mid, end)
    n_t_full = header["n_t"]
    print(f"  full record: {n_t_full} samples ({(n_t_full-1)*header['t_incr']:.1f} s)")
    print(f"  trimmed: {sigma.shape[0]} samples x {sigma.shape[1]} theta "
          f"(t[0]={t_trim[0]:.2f}  t[-1]={t_trim[-1]:.2f})")
    assert t_trim[0] >= TRANSIENT_CUTOFF_S
    assert t_trim[0] - header["t_incr"] < TRANSIENT_CUTOFF_S  # first sample AT/just past cutoff
    assert sigma.shape == (n_t_full - int(round(TRANSIENT_CUTOFF_S / header["t_incr"])), N_THETA)

    print(f"\n  sigma(theta,t) stats over trimmed record, MPa:")
    print(f"    min={sigma.min():.3f}  max={sigma.max():.3f}  mean={sigma.mean():.3f}")
    print(f"    per-theta mean (should all be close -- axial dominates a leg): "
          f"{sigma.mean(axis=0).round(2)}")

    # Hand-check one (t, theta) point directly from the raw channels, fully
    # independent of nominal_components/hotspot_member's own internals.
    names = [f"M{mid}J{end}{c}" for c in obr.FATIGUE_COMPONENTS]
    t_raw, arr_raw = obr.read_channels(outb_path, header, names)
    k = len(t_raw) - sigma.shape[0]  # index into raw arrays matching sigma's row 0
    assert abs(t_raw[k] - t_trim[0]) < 1e-9
    N_k, Mkx_k, Mky_k = arr_raw[k]
    props = sdg.section_properties(D, wall_t)
    theta0 = THETA_RAD[3]  # an arbitrary non-trivial angle, not 0/90/180/270
    W = props["I"] / props["R"]
    expect = (N_k / props["A"] / 1e6
              + (Mkx_k / W / 1e6) * np.cos(theta0)
              - (Mky_k / W / 1e6) * np.sin(theta0))
    got = sigma[0, 3]
    print(f"\n  hand-check at t={t_trim[0]:.2f}s theta={np.degrees(theta0):.1f}deg: "
          f"formula={expect:.6f}  code={got:.6f}  diff={abs(expect-got):.2e}")
    assert abs(expect - got) < 1e-9

    # Rough magnitude sanity: axial alone for this member is
    # N/A ~= -5.4e6 N / 0.1807 m2 ~= -30 MPa, using THIS member's actual
    # section (D=1.2, t=0.05, A=pi*(1.2^2-1.1^2)/4=0.1807 m2) -- the
    # -5.4 MN axial force itself is the same number outb_reader.py's Step 1
    # notes cite for this member. Order of magnitude only -- confirms no
    # stray Pa-vs-MPa or m-vs-mm slip, not a precision check.
    sig_ax, _, _ = nominal_components(arr_raw[:, 0], arr_raw[:, 1], arr_raw[:, 2], D, wall_t)
    print(f"\n  axial-only sigma mean over full record: {sig_ax.mean():.2f} MPa "
          f"(order-of-magnitude check against ~-30 MPa expected)")
    assert -100 < sig_ax.mean() < -5, "axial stress not in the expected order of magnitude"

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
    print("\n")
    _self_check_hotspot_joint()
