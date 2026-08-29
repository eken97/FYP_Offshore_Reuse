"""
Joint track, Step C -- DNV-RP-C203 Appendix B parametric SCF equations.

REWORK of the earlier Step 5 build (13.08.2026) -- see joint_scf_theory
memory's "STEP 5 BUILT, THEN LARGELY SUPERSEDED" section for the full
derivation this rework implements. Two things changed:

1. CROWN moved from eqn 6a/7a (fixed C=0.7 chord-end-fixity assumption) to
   eqn 6b/7b. This was forced, not a preference: eqn 6a/7a's alpha term is
   evaluated far outside its stated validity range at every K joint on this
   jacket once alpha reverted to full-span (Step A). 6b/7b removes the
   alpha upper limit, the C=0.7 assumption, and the off-midspan chord-
   length judgement -- three stated assumptions dropped at once.

   SOURCE FOR 6b/7b, TRANSCRIBED DIRECTLY FROM DNV, not derived or taken
   from any secondary source (author-supplied photo of the actual Table B-1
   "Alternatively" box, DNVGL-RP-0005:2014-06, 14.08.2026 --
   supersedes this module's earlier citation of Lotsberg (2011) as the
   formula's source; that paper is now background/history only, see
   docs/decisions.md, NOT what these equations are transcribed
   from):

       SCF_Cc = gamma^0.2*tau*(2.65+5*(beta-0.65)^2) - 3*tau*beta*sin(theta)
                + (sigma_BendingChord / sigma_Axialbrace) * SCFatt          (6b)

       SCF_Bc = 3 + gamma^1.2*(0.12*exp(-4*beta)+0.011*beta^2-0.045) - 1.2*beta*tau
                + (0.4 * sigma_BendingChord / sigma_Axialbrace) * SCFatt    (7b)

       where sigma_BendingChord = nominal bending stress in the chord
             sigma_Axialbrace   = nominal axial stress in the brace
             SCFatt             = stress concentration factor for an
                                   attachment = 1.27

   SCFatt VALUE CORRECTED 14.08.2026 (later still): the "= 1.27" printed in
   the box above is DNV's own WORKED EXAMPLE value, not a universal
   constant -- DNV's surrounding commentary defines SCFatt generally as
   "the structural stress concentration embedded in the detail (S-N
   class)" (Table 2-1's own column of that name), and explicitly derives
   the box's 1.27 from an F/E-curve illustration ("For long chords the
   brace can be considered as an attachment to the chord... This would
   give detail category F... which corresponds to SCFatt=1.27 from Table
   2-1"). This project's joint track uses the T-curve (tubular joints, see
   sn_curves.py -- same Table 2-1, already primary-source verified there,
   SN_CONSTANTS_VERIFIED=True), whose row gives structural-SCF-embedded =
   1.00. SCF_ATT below is therefore 1.00, not the box's own illustrative
   1.27 -- see SCF_ATT's own constant definition for the full derivation.

   NOTE THE 0.4: the brace equation's attachment term carries an explicit
   0.4 factor that the chord equation's does NOT -- a real, easy-to-miss
   asymmetry (caught and fixed 14.08.2026 after a review asked to see the
   crown numbers cross-checked and re-derived; the code before this fix
   wrongly applied the bare SCFatt=1.27 to BOTH sides, over-weighting the
   brace-side attachment contribution by 2.5x). Kept as two separate named
   constants below (SCF_ATT, SCF_ATT_BRACE_FACTOR) rather than folded into
   one number, so the 0.4 stays visibly traceable to the (7b) box above.

   THESE EQUATIONS AS PRINTED ARE SCFs (dimensionless): SCF_Cc/SCF_Bc are
   multipliers on sigma_Axialbrace, and the attachment term itself is a
   dimensionless stress ratio (sigma_BendingChord/sigma_Axialbrace) times
   the dimensionless constant SCFatt. This module transcribes and computes
   ONLY the geometry-only part of each equation (the terms before the "+"
   -- a real, complete, dimensionless SCF sub-value on its own, callable
   with no stress signal at all: eqn6b_chord_crown_base /
   eqn7b_brace_crown_base below). The attachment term genuinely cannot be
   evaluated here -- it needs a real per-timestep chord bending stress,
   which does not exist until Step D reads the chord's own MKxe/MKye
   (joint_geometry.py's Step B chord_a_W/chord_b_W supplies the section
   modulus). This module hands back that base value plus the SCFatt-based
   coefficient (AC_att, see _side_result) unevaluated, exactly as DNV
   prints them -- Step D is where sigma_crown = SCF_Cc/SCF_Bc * sigma_ax
   finally gets computed, using DNV's own literal formula, not a
   rearranged one (see hotspot_joint's own docstring, once built, for the
   one-line algebraic identity that makes that multiplication safe when
   sigma_ax passes through zero in a time-domain signal -- shown there as
   a checkable derivation of DNV's own printed formula, not a
   substitution).

   AC_att is 0.0 for X: Table B-2's own crown equations (13/15) already
   have NO chord-bending/SCFatt term (physically -- an X joint's axial
   load passes straight through to the opposing brace, there is no chord
   reaction to react a bending moment from), so X's crown stays the plain
   single-number formula it always was; adding an attachment term on top
   of it would add a term that formula was never missing. Proven
   algebraically this session's predecessor: eqn13 (X chord crown) IS
   eqn6a (old T/Y chord crown) evaluated at alpha=0 -- and the new base_AC
   term below (chord side) is the SAME formula again, just now understood
   as "6b's geometry-only part" rather than "6a with alpha zeroed".
   Brace-side base_AC is NOT eqn15 (Table B-2's own X brace crown, which is
   missing a -1.2*beta*tau term relative to (7b)'s own base) -- kept as a
   separate function, do not conflate the two.

2. CLASSIFICATION dropped the old "run every connection through X as a
   conservative bound, then again through its real family" sweep entirely
   -- that global X pass is now WRONG (the 13.08.2026 "THE CONTRADICTION"
   session proved X actively UNDER-predicts the crown at K/TY joints, since
   X's own crown formula has no chord-bending term to under-predict with).
   Per the final classification scheme (docs/decisions.md):
     - TY family  -> scf_TY only.
     - X family   -> scf_X only (both directions, already separate rows).
     - K family   -> BOTH scf_K_pair (real K-specific axial saddle, eqn
       20/21) AND scf_TY evaluated on that SAME brace's own geometry (the
       "Y treatment" -- physically, "assume zero K-balancing benefit, all
       load reacted by chord beam shear", endorsed directly by Lotsberg:
       "geometrical K-joints can be defined as T-joints based on force flow
       consideration in frame structures"). Both are kept as separate
       output rows tagged by `treatment` ("K" vs "Y") -- the envelope
       (worse of the two) is taken at the DAMAGE stage after rainflow, not
       here (never combine/pick before rainflow -- same rule as every other
       fork in this pipeline, see docs/decisions.md).

Everything else is unchanged from the Step 5 build: pure maths module, same
design pattern as sn_curves.py -- every function takes its geometry as
ARGUMENTS (beta, gamma, tau, theta[, alpha][, zeta]), never a module
constant, so the same code is callable again per corrosion-year-step with
thinned D/T/t without redesign. SADDLE formulas (eqn 1/3/5/12/14/20/21) and
IPB/OPB formulas (eqn 8/9/10/11/16/17/23/24) are untouched -- the general-
fixity chord-end constant CHORD_FIXITY_C=0.7 (and C1, its only remaining
consumer, eqn5) now applies ONLY to the saddle, not the crown, and the
K-saddle's own alpha=41-57 extrapolation (eqn 5/3 reused for the Y
treatment) is unchanged from before -- kept, flagged, per the author's
existing call (docs/decisions.md).

SOURCE: DNVGL-RP-0005:2014-06, Appendix B, read via
PyMuPDF-rendered PNGs (scanned, no text layer) 13.08.2026, cross-checked
against the full-text 2006-04 edition PDF for the crown/saddle ambiguity;
eqn 6b/7b specifically re-transcribed 14.08.2026 directly from a author-
supplied photo of the Table B-1 "Alternatively" box (see above) -- DNV is
the sole formula source for this module. Lotsberg (2011), Marine
Structures 24:60-69, doi:10.1016/j.marstruc.2011.01.002, is background/
history only (it independently proposed the same equations DNV later
adopted -- see docs/decisions.md) and is NOT cited as the
source of any transcribed formula here, to avoid mixing two presentations
of the same equations and losing exact traceability to the one code being
designed to. DOC_SOURCE below is quoted into every G2 CSV row so the
traceability isn't only in this docstring.

Every formula below is transcribed with its DNV equation number in both the
function name and a per-call `eqn_*` label carried into the output row, so
your hand-check (G2 sign-off, same discipline as sn_curves_CHECK.csv) can
trace every number straight back to one equation on one page.
"""
import math
from pathlib import Path

import numpy as np

import sd_geometry as sdg
import joint_geometry as jg

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# endregion


# region --- sign-off gate ---
# Same discipline as sn_curves.SN_CONSTANTS_VERIFIED: this module produces
# no damage numbers on its own, but every downstream stage (hotspot_joint,
# stage2_joints, stage3_joint_damage) must check this flag itself and
# refuse to run while it is False.
#
# G2 SIGNED OFF 14.08.2026: the author hand-calculated node 29 (a K-plane, brace
# M66J2/M83J1) against DNVGL-RP-0005 App. B directly -- eqn 20 (chord saddle) and eqn 21
# (brace saddle) both checked out. Along the way caught a REAL error in the
# the author's own spreadsheet, not this code: their eqn20 formula referenced the
# wrong zeta cell (zeta_ab, a partial gap to an unused 3rd-brace template
# slot) instead of zeta_full (the real A-C gap, matching this code's own
# 0.296269 to within rounding) -- resolved by cross-computing both
# candidates against this module's real output and identifying the match.
SCF_EQUATIONS_VERIFIED = True
DOC_SOURCE = "DNVGL-RP-0005:2014-06 Appendix B (licensed copy; not redistributed), incl. eqn 6b/7b re-transcribed 14.08.2026"
# endregion


# region --- chord-end fixity -- SADDLE ONLY now, see module docstring ---
CHORD_FIXITY_C = 0.7
C1 = 2.0 * (CHORD_FIXITY_C - 0.5)
# endregion

# DNV-RP-C203's own "stress concentration factor for an attachment" --
# applied to the chord's own bending-stress contribution to the crown for
# TY/K (not X, which needs no addition -- see module docstring).
#
# VALUE CORRECTED 14.08.2026 (later still): SCFatt is NOT a universal
# constant -- DNV's own commentary (Section 2, immediately after the eqn
# 6b/7b box) defines it as "the structural stress concentration embedded in
# the detail (S-N class)", i.e. Table 2-1's own "Structural stress
# concentration embedded in the detail (S-N class)" column, read off
# whichever S-N curve the assessment actually uses. DNV's own worked
# illustration uses an F/E-curve example (giving 1.27) -- that is an
# EXAMPLE of the principle, not a universal value, and was wrongly
# transcribed as if it were fixed. This project's joint track uses the
# T-curve (tubular joints, see sn_curves.py / fatigue_postpro_design
# memory, SN_CONSTANTS_VERIFIED=True, same Table 2-1 already used there --
# printed_knee_air["T"]=52.63 there matches the T row's fatigue-limit
# column on the same table the author re-checked this against). Table 2-1's
# T row gives structural-SCF-embedded = 1.00. Was 1.27, now 1.00 -- a real,
# ~21% reduction in every K/TY crown's chord-bending contribution,
# project-wide.
SCF_ATT = 1.00

# eqn 7b's OWN attachment term carries an explicit 0.4 factor that eqn 6b's
# does NOT: "+ 0.4*sigma_BendingChord/sigma_Axialbrace * SCFatt" (brace) vs
# "+ sigma_BendingChord/sigma_Axialbrace * SCFatt" (chord, implicit 1.0).
# Kept as its own named constant, not folded into SCF_ATT, so it stays
# visibly traceable to the (7b) box it comes from -- a real bug (bare
# SCF_ATT applied to BOTH sides, over-weighting the brace attachment term
# by 2.5x) was caught and fixed here 14.08.2026.
SCF_ATT_BRACE_FACTOR = 0.4

ALPHA_SHORT_CHORD_THRESHOLD = 12.0


def _sin_deg(x_deg):
    return math.sin(math.radians(x_deg))


# region --- Table B-1: simple tubular T/Y joints (DNVGL-RP-0005 App. B p.111-112) ---
def eqn1_chord_saddle_fixed(beta, gamma, tau, theta_deg):
    return gamma * tau**1.1 * (1.11 - 3.0 * (beta - 0.52)**2) * _sin_deg(theta_deg)**1.6


def eqn3_brace_saddle_fixed(beta, gamma, tau, theta_deg, alpha):
    return (1.3 + gamma * tau**0.52 * alpha**0.1
            * (0.187 - 1.25 * beta**1.1 * (beta - 0.96))
            * _sin_deg(theta_deg)**(2.7 - 0.01 * alpha))


def eqn5_chord_saddle_general(beta, gamma, tau, theta_deg, alpha):
    return (eqn1_chord_saddle_fixed(beta, gamma, tau, theta_deg)
            + C1 * (0.8 * alpha - 6.0) * tau * beta**2 * (1.0 - beta**2)**0.5
            * _sin_deg(2.0 * theta_deg)**2)


def eqn8_chord_crown_ipb(beta, gamma, tau, theta_deg):
    return 1.45 * beta * tau**0.85 * gamma**(1.0 - 0.68 * beta) * _sin_deg(theta_deg)**0.7


def eqn9_brace_crown_ipb(beta, gamma, tau, theta_deg):
    return (1.0 + 0.65 * beta * tau**0.4 * gamma**(1.09 - 0.77 * beta)
            * _sin_deg(theta_deg)**(0.06 * gamma - 1.16))


def eqn10_chord_saddle_opb(beta, gamma, tau, theta_deg):
    return gamma * tau * beta * (1.7 - 1.05 * beta**3) * _sin_deg(theta_deg)**1.6


def eqn11_brace_saddle_opb(beta, gamma, tau, theta_deg):
    return (tau**-0.54 * gamma**-0.05 * (0.99 - 0.47 * beta + 0.08 * beta**4)
            * eqn10_chord_saddle_opb(beta, gamma, tau, theta_deg))


def F1(beta, gamma, alpha):
    return 1.0 - (0.83 * beta - 0.56 * beta**2 - 0.02) * gamma**0.23 * math.exp(-0.21 * gamma**-1.16 * alpha**2.5)


def F2(beta, gamma, alpha):
    return 1.0 - (1.43 * beta - 0.97 * beta**2 - 0.03) * gamma**0.04 * math.exp(-0.71 * gamma**-1.38 * alpha**2.5)


def F3(beta, gamma, alpha):
    return 1.0 - 0.55 * beta**1.8 * gamma**0.16 * math.exp(-0.49 * gamma**-0.89 * alpha**1.8)


def F4(beta, gamma, alpha):
    return 1.0 - 1.07 * beta**1.88 * math.exp(-0.16 * gamma**-1.06 * alpha**2.4)


def _apply_short_chord(value, factor_fn, beta, gamma, alpha):
    """Returns (corrected_value, factor_used, applied_bool). Correction only
    fires for alpha < ALPHA_SHORT_CHORD_THRESHOLD (=12, NOT the alpha=4
    validity floor) -- see module docstring."""
    if alpha < ALPHA_SHORT_CHORD_THRESHOLD:
        f = factor_fn(beta, gamma, alpha)
        return value * f, f, True
    return value, 1.0, False
# endregion


# region --- Table B-1: crown, eqn 6b/7b (DNV, transcribed directly -- see module docstring) ---
def eqn6b_chord_crown_base(beta, gamma, tau, theta_deg):
    """Geometry-only part of DNV eqn 6b (the part before "+"). Proven
    algebraically identical to eqn13_chord_crown_x (Table B-2, X joints) --
    physically because an X joint's own crown formula already has no
    chord-bending term to be missing (see module docstring) -- so this is a
    thin, documented alias, not a duplicated formula."""
    return eqn13_chord_crown_x(beta, gamma, tau, theta_deg)


def eqn7b_brace_crown_base(beta, gamma, tau):
    """Geometry-only part of DNV eqn 7b (the part before "+"). NOT the same
    as eqn15_brace_crown_x (Table B-2's X brace crown is missing the
    -1.2*beta*tau term this has) -- kept as its own function deliberately,
    do not conflate the two (see module docstring)."""
    return (3.0 + gamma**1.2 * (0.12 * math.exp(-4.0 * beta) + 0.011 * beta**2 - 0.045)
            - 1.2 * beta * tau)
# endregion


# region --- Table B-2: simple X tubular joints (DNVGL-RP-0005 App. B p.113-114) ---
def eqn12_chord_saddle_x(beta, gamma, tau, theta_deg):
    return 3.87 * gamma * tau * beta * (1.10 - beta**1.8) * _sin_deg(theta_deg)**1.7


def eqn13_chord_crown_x(beta, gamma, tau, theta_deg):
    return gamma**0.2 * tau * (2.65 + 5.0 * (beta - 0.65)**2) - 3.0 * tau * beta * _sin_deg(theta_deg)


def eqn14_brace_saddle_x(beta, gamma, tau, theta_deg):
    return 1.0 + 1.9 * gamma * tau**0.5 * beta**0.9 * (1.09 - beta**1.7) * _sin_deg(theta_deg)**2.5


def eqn15_brace_crown_x(beta, gamma, tau):
    return 3.0 + gamma**1.2 * (0.12 * math.exp(-4.0 * beta) + 0.011 * beta**2 - 0.045)


def eqn16_chord_saddle_opb_x(beta, gamma, tau, theta_deg):
    return gamma * tau * beta * (1.56 - 1.34 * beta**4) * _sin_deg(theta_deg)**1.6


def eqn17_brace_saddle_opb_x(beta, gamma, tau, theta_deg):
    return (tau**-0.54 * gamma**-0.05 * (0.99 - 0.47 * beta + 0.08 * beta**4)
            * eqn16_chord_saddle_opb_x(beta, gamma, tau, theta_deg))
# endregion


# region --- Table B-3: simple tubular K joints and overlap K joints (DNVGL-RP-0005 App. B p.115) ---
def eqn20_chord_saddle_k(beta, gamma, tau, theta_deg, beta_max, beta_min, theta_max_deg, theta_min_deg, zeta):
    """tau, beta, theta here are the brace UNDER CONSIDERATION's own values
    (DNV's own note); beta_max/min, theta_max/min, zeta are joint-level,
    shared by both braces of the K plane."""
    return (tau**0.9 * gamma**0.5 * (0.67 - beta**2 + 1.16 * beta) * _sin_deg(theta_deg)
            * (_sin_deg(theta_max_deg) / _sin_deg(theta_min_deg))**0.30
            * (beta_max / beta_min)**0.30
            * (1.64 + 0.29 * beta**-0.38 * math.atan(8.0 * zeta)))


def eqn21_brace_saddle_k(beta, gamma, tau, theta_deg, beta_max, beta_min, theta_max_deg, theta_min_deg,
                          zeta, C_overlap):
    """C_overlap: 0 for gap joints, 1 for the through brace, 0.5 for the
    overlapping brace (Table B-3's own definition). This jacket's K planes
    all have a real positive gap (zeta=0.25-0.34, see joint_geometry's
    corrected zeta formula) -- gap joints, C_overlap=0, everywhere."""
    term1 = (1.0 + (1.97 - 1.57 * beta**0.25) * tau**-0.14 * _sin_deg(theta_deg)**0.7
             * eqn20_chord_saddle_k(beta, gamma, tau, theta_deg, beta_max, beta_min,
                                     theta_max_deg, theta_min_deg, zeta))
    term2 = (_sin_deg(theta_max_deg + theta_min_deg)**1.8
             * (0.131 - 0.084 * math.atan(14.0 * zeta + 4.2 * beta))
             * C_overlap * beta**1.5 * gamma**0.5 * tau**-1.22)
    return term1 + term2


def eqn23_chord_saddle_opb_k(beta_a, gamma, tau_a, theta_a_deg, beta_b, tau_b, theta_b_deg, beta_max, zeta):
    """Chord saddle SCF adjacent to brace A. (Eqn.10) is called with the
    RAW (uncorrected) inputs for each brace, per DNV's own layout (the F4
    short-chord correction is applied to eqn23's and eqn24's own results
    separately downstream, not baked into this internal call -- same
    pattern as eqn10/eqn11 in Table B-1)."""
    x = 1.0 + zeta * _sin_deg(theta_a_deg) / beta_a
    eqn10_a = eqn10_chord_saddle_opb(beta_a, gamma, tau_a, theta_a_deg)
    eqn10_b = eqn10_chord_saddle_opb(beta_b, gamma, tau_b, theta_b_deg)
    term1 = eqn10_a * (1.0 - 0.08 * (beta_b * gamma)**0.5 * math.exp(-0.8 * x))
    term2 = (eqn10_b * (1.0 - 0.08 * (beta_a * gamma)**0.5 * math.exp(-0.8 * x))
             * (2.05 * beta_max**0.5 * math.exp(-1.3 * x)))
    return term1 + term2


def eqn24_brace_saddle_opb_k(beta_a, gamma, tau_a, eqn23_value):
    return (tau_a**-0.54 * gamma**-0.05 * (0.99 - 0.47 * beta_a + 0.08 * beta_a**4)
            * eqn23_value)
# endregion


# region --- per-family dispatchers: 4 SCFs per side (chord, brace) ---
def _side_result(AC_base, AC_att, AS, MIP, MOP, eqn_AC, eqn_AS, eqn_MIP, eqn_MOP,
                  F_AS=None, F_MOP=None):
    """AC_base: geometry-only crown SCF (a real, static number). AC_att:
    the coefficient (0.0, SCF_ATT for chord, or SCF_ATT_BRACE_FACTOR*SCF_ATT
    for brace -- eqn 7b's OWN 0.4 factor, NOT the same value as chord, see
    module docstring) that multiplies the chord's own bending stress -- NOT
    resolved to a single crown SCF here, since that needs a real chord-
    bending stress signal this module doesn't have. AS/MIP/MOP are
    unchanged, plain static SCFs, exactly as before."""
    return dict(AC_base=AC_base, AC_att=AC_att, AS=AS, MIP=MIP, MOP=MOP,
                eqn_AC=eqn_AC, eqn_AS=eqn_AS, eqn_MIP=eqn_MIP, eqn_MOP=eqn_MOP,
                F_AS=F_AS, F_MOP=F_MOP)


def scf_TY(beta, gamma, tau, theta_deg, alpha):
    """Table B-1: saddle general-fixity family (eqn 5/3, C=0.7), crown eqn
    6b/7b (base + attachment coefficient -- SCF_ATT for chord,
    SCF_ATT_BRACE_FACTOR*SCF_ATT for brace, see module docstring). Returns
    {"chord": {...}, "brace": {...}}, each an 8-field dict per
    _side_result."""
    chord_AS_raw = eqn5_chord_saddle_general(beta, gamma, tau, theta_deg, alpha)
    chord_AS, f2_c, _ = _apply_short_chord(chord_AS_raw, F2, beta, gamma, alpha)
    chord_AC_base = eqn6b_chord_crown_base(beta, gamma, tau, theta_deg)
    chord_MIP = eqn8_chord_crown_ipb(beta, gamma, tau, theta_deg)
    chord_MOP_raw = eqn10_chord_saddle_opb(beta, gamma, tau, theta_deg)
    chord_MOP, f3_c, _ = _apply_short_chord(chord_MOP_raw, F3, beta, gamma, alpha)

    brace_AS_raw = eqn3_brace_saddle_fixed(beta, gamma, tau, theta_deg, alpha)
    brace_AS, f2_b, _ = _apply_short_chord(brace_AS_raw, F2, beta, gamma, alpha)
    brace_AC_base = eqn7b_brace_crown_base(beta, gamma, tau)
    brace_MIP = eqn9_brace_crown_ipb(beta, gamma, tau, theta_deg)
    brace_MOP_raw = eqn11_brace_saddle_opb(beta, gamma, tau, theta_deg)
    brace_MOP, f3_b, _ = _apply_short_chord(brace_MOP_raw, F3, beta, gamma, alpha)

    return dict(
        chord=_side_result(chord_AC_base, SCF_ATT, chord_AS, chord_MIP, chord_MOP,
                            "6b", "5(+F2 if a<12)", "8", "10(+F3 if a<12)", f2_c, f3_c),
        brace=_side_result(brace_AC_base, SCF_ATT_BRACE_FACTOR * SCF_ATT, brace_AS, brace_MIP, brace_MOP,
                            "7b", "3(+F2 if a<12)", "9", "11(+F3 if a<12)", f2_b, f3_b),
    )


def scf_X(beta, gamma, tau, theta_deg, alpha):
    """Table B-2, balanced axial load family (eqn 12-17). Crown (13/15)
    already has no chord-bending term -- AC_att=0.0, no addition needed."""
    chord_AS_raw = eqn12_chord_saddle_x(beta, gamma, tau, theta_deg)
    chord_AS, f2_c, _ = _apply_short_chord(chord_AS_raw, F2, beta, gamma, alpha)
    chord_AC_base = eqn13_chord_crown_x(beta, gamma, tau, theta_deg)
    chord_MIP = eqn8_chord_crown_ipb(beta, gamma, tau, theta_deg)   # reused, Table B-2's own note
    chord_MOP_raw = eqn16_chord_saddle_opb_x(beta, gamma, tau, theta_deg)
    chord_MOP, f3_c, _ = _apply_short_chord(chord_MOP_raw, F3, beta, gamma, alpha)

    brace_AS_raw = eqn14_brace_saddle_x(beta, gamma, tau, theta_deg)
    brace_AS, f2_b, _ = _apply_short_chord(brace_AS_raw, F2, beta, gamma, alpha)
    brace_AC_base = eqn15_brace_crown_x(beta, gamma, tau)
    brace_MIP = eqn9_brace_crown_ipb(beta, gamma, tau, theta_deg)
    brace_MOP_raw = eqn17_brace_saddle_opb_x(beta, gamma, tau, theta_deg)
    brace_MOP, f3_b, _ = _apply_short_chord(brace_MOP_raw, F3, beta, gamma, alpha)

    return dict(
        chord=_side_result(chord_AC_base, 0.0, chord_AS, chord_MIP, chord_MOP,
                            "13", "12(+F2 if a<12)", "8", "16(+F3 if a<12)", f2_c, f3_c),
        brace=_side_result(brace_AC_base, 0.0, brace_AS, brace_MIP, brace_MOP,
                            "15", "14(+F2 if a<12)", "9", "17(+F3 if a<12)", f2_b, f3_b),
    )


def scf_K_pair(self_row, other_row):
    """
    Table B-3 axial saddle (K-specific, eqn 20/21) + eqn 6b/7b crown (base +
    attachment coefficient, same as TY -- a K joint's chord DOES see a real
    bending moment from the brace load, same physical reasoning as TY) +
    Table B-3 OPB (eqn 23/24). self_row/other_row are two connection dicts
    sharing one K
    plane's sub_joint_id and chord_t_scenario (same gamma/zeta). Returns the
    SCF set for self_row's brace only -- call twice, swapping self/other, to
    get both braces of the plane.
    """
    beta_s, tau_s, theta_s, alpha_s = self_row["beta"], self_row["tau"], self_row["theta_deg"], self_row["alpha"]
    gamma = self_row["gamma"]
    beta_o, tau_o, theta_o = other_row["beta"], other_row["tau"], other_row["theta_deg"]
    zeta = self_row["zeta"]
    beta_max, beta_min = max(beta_s, beta_o), min(beta_s, beta_o)
    theta_max, theta_min = max(theta_s, theta_o), min(theta_s, theta_o)
    C_overlap = 0.0   # gap joint -- see eqn21's docstring

    chord_AS = eqn20_chord_saddle_k(beta_s, gamma, tau_s, theta_s, beta_max, beta_min,
                                     theta_max, theta_min, zeta)                        # Table B-3: "None"
    brace_AS = eqn21_brace_saddle_k(beta_s, gamma, tau_s, theta_s, beta_max, beta_min,
                                     theta_max, theta_min, zeta, C_overlap)              # Table B-3: "None"
    chord_AC_base = eqn6b_chord_crown_base(beta_s, gamma, tau_s, theta_s)
    brace_AC_base = eqn7b_brace_crown_base(beta_s, gamma, tau_s)
    chord_MIP = eqn8_chord_crown_ipb(beta_s, gamma, tau_s, theta_s)
    brace_MIP = eqn9_brace_crown_ipb(beta_s, gamma, tau_s, theta_s)                     # gap joint, no overlap factor

    eqn23_raw = eqn23_chord_saddle_opb_k(beta_s, gamma, tau_s, theta_s, beta_o, tau_o, theta_o, beta_max, zeta)
    eqn24_raw = eqn24_brace_saddle_opb_k(beta_s, gamma, tau_s, eqn23_raw)
    chord_MOP, f4_c, _ = _apply_short_chord(eqn23_raw, F4, beta_s, gamma, alpha_s)
    brace_MOP, f4_b, _ = _apply_short_chord(eqn24_raw, F4, beta_s, gamma, alpha_s)

    return dict(
        chord=_side_result(chord_AC_base, SCF_ATT, chord_AS, chord_MIP, chord_MOP,
                            "6b", "20", "8", "23(+F4 if a<12)", None, f4_c),
        brace=_side_result(brace_AC_base, SCF_ATT_BRACE_FACTOR * SCF_ATT, brace_AS, brace_MIP, brace_MOP,
                            "7b", "21", "9", "24(+F4 if a<12)", None, f4_b),
    )
# endregion


# region --- G2 sign-off CSV ---
def _group_k_planes(connections):
    """{(sub_joint_id, chord_t_scenario): [row_a, row_b]} for K family only."""
    groups = {}
    for c in connections:
        if c["family"] != "K":
            continue
        key = (c["sub_joint_id"], c["chord_t_scenario"])
        groups.setdefault(key, []).append(c)
    return groups


def compute_all_scf(connections):
    """
    Returns a list of output rows, one per (connection, treatment, side).
    treatment: "TY" for TY-family connections, "X" for X-family (exactly
    ONE treatment each), or "K" AND "Y" for K-family connections (BOTH
    computed -- the K-vs-Y envelope is taken downstream at the damage
    stage, after rainflow, never here -- see module docstring). Row count:
    24 TY x1 x2 sides + 32 X x1 x2 sides + 64 K x2 treatments x2 sides =
    48 + 64 + 256 = 368.
    """
    k_groups = _group_k_planes(connections)
    rows = []
    for c in connections:
        if c["family"] == "TY":
            treatments = [("TY", scf_TY(c["beta"], c["gamma"], c["tau"], c["theta_deg"], c["alpha"]))]
        elif c["family"] == "X":
            treatments = [("X", scf_X(c["beta"], c["gamma"], c["tau"], c["theta_deg"], c["alpha"]))]
        elif c["family"] == "K":
            key = (c["sub_joint_id"], c["chord_t_scenario"])
            pair = k_groups[key]
            other_row = pair[0] if pair[1] is c else pair[1]
            treatments = [
                ("K", scf_K_pair(c, other_row)),
                ("Y", scf_TY(c["beta"], c["gamma"], c["tau"], c["theta_deg"], c["alpha"])),
            ]
        else:
            raise ValueError(f"unknown family {c['family']!r}")

        for treatment, result in treatments:
            for side in ("chord", "brace"):
                sr = result[side]
                # AC excluded from this max -- it's not a static number any
                # more (needs the chord-bending signal, Step D). The three
                # still-static positions' worst is still a useful QA number.
                static_scf_max = max(sr["AS"], sr["MIP"], sr["MOP"])
                rows.append(dict(
                    node=c["node"], sub_joint_id=c["sub_joint_id"], plane_id=c["plane_id"],
                    family=c["family"], treatment=treatment,
                    brace_member=c["brace_member"], brace_end=c["brace_end"],
                    chord_t_scenario=c["chord_t_scenario"], direction=c["direction"],
                    side=side, beta=c["beta"], gamma=c["gamma"], tau=c["tau"],
                    theta_deg=c["theta_deg"], alpha=c["alpha"],
                    zeta=c["zeta"] if c["zeta"] is not None else "",
                    SCF_AC_base=sr["AC_base"], SCF_AC_att=sr["AC_att"],
                    SCF_AS=sr["AS"], SCF_MIP=sr["MIP"], SCF_MOP=sr["MOP"],
                    eqn_AC=sr["eqn_AC"], eqn_AS=sr["eqn_AS"], eqn_MIP=sr["eqn_MIP"], eqn_MOP=sr["eqn_MOP"],
                    F_AS=sr["F_AS"] if sr["F_AS"] is not None else "",
                    F_MOP=sr["F_MOP"] if sr["F_MOP"] is not None else "",
                    static_scf_max=static_scf_max,
                ))
    return rows


def write_scf_check(connections, out_path):
    """G2 sign-off CSV -- 368 rows (see compute_all_scf). Hand-check two
    connections against DNVGL-RP-0005 App. B yourself before flipping
    SCF_EQUATIONS_VERIFIED."""
    import csv
    all_rows = compute_all_scf(connections)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(all_rows[0].keys())
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for r in all_rows:
            writer.writerow([r[col] for col in cols])
    return all_rows
# endregion


def _self_check():
    print(f"SCF_EQUATIONS_VERIFIED = {SCF_EQUATIONS_VERIFIED}")
    print(f"Chord-end fixity (saddle only): C={CHORD_FIXITY_C}  C1={C1}")
    print(f"SCF_ATT (attachment SCF, crown chord-bending term, TY/K only) = {SCF_ATT} "
          f"(chord side); brace side = SCF_ATT_BRACE_FACTOR*SCF_ATT = "
          f"{SCF_ATT_BRACE_FACTOR}*{SCF_ATT} = {SCF_ATT_BRACE_FACTOR*SCF_ATT:.4f}\n")

    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    print(f"Loaded {len(connections)} connections from joint_geometry.py")

    # --- alpha<12 census -- unaffected by the crown rework (only the
    # SADDLE F2/F3/F4 corrections depend on this threshold).
    under12 = [c for c in connections if c["alpha"] < ALPHA_SHORT_CHORD_THRESHOLD]
    fams = {}
    for c in under12:
        fams[c["family"]] = fams.get(c["family"], 0) + 1
    print(f"\nConnections with alpha < {ALPHA_SHORT_CHORD_THRESHOLD} (real short-chord "
          f"correction population): {len(under12)}/{len(connections)}, by family: {fams}")
    assert fams == {"TY": 8}, (
        "expected exactly the 8 lower-mudbrace TY connections (alpha=3.12, genuinely "
        "below even the alpha=4 validity floor) to need F-correction, and no K/X "
        "connection. NOTE: this was 16 (the 8 lower-mudbrace PLUS 8 top-level, "
        "alpha=9.82) under the OLD halved-alpha formula -- Step A's full-span alpha "
        "revert roughly doubled the top-level connections' alpha, pushing them back "
        "above 12, so only the lower-mudbrace 8 remain. If this count changes again, "
        "the F1-vs-F2 'moot on this jacket' claim in the module docstring needs "
        "revisiting."
    )

    # --- write G2 CSV ---
    out_path = RESULTS_DIR / "scf_CHECK.csv"
    rows = write_scf_check(connections, out_path)
    print(f"\nwrote {out_path} ({len(rows)} rows, expect 368 = 24 TY x1x2 + 32 X x1x2 "
          f"+ 64 K x2-treatments x2-sides)")
    assert len(rows) == 368

    # --- row-count-by-treatment sanity ---
    import collections
    treat_counts = collections.Counter((r["family"], r["treatment"]) for r in rows)
    print(f"rows by (family, treatment): {dict(treat_counts)}")
    assert treat_counts[("TY", "TY")] == 48
    assert treat_counts[("X", "X")] == 64
    assert treat_counts[("K", "K")] == 128
    assert treat_counts[("K", "Y")] == 128

    # --- AC_att consistency: 0.0 for X (no chord-bending term needed),
    # SCF_ATT on the CHORD side and SCF_ATT_BRACE_FACTOR*SCF_ATT on the
    # BRACE side for every TY and K/Y treatment -- the two sides are NOT
    # the same value, per eqn 7b's own explicit 0.4 factor (a real bug --
    # bare SCF_ATT applied to both sides -- caught and fixed 14.08.2026,
    # see module docstring). SCF_ATT itself is 1.00 (T-curve), not the
    # box's own illustrative 1.27 (F/E-curve example) -- see SCF_ATT's own
    # definition for the derivation.
    brace_att = SCF_ATT_BRACE_FACTOR * SCF_ATT
    att_by_treatment = {r["treatment"] for r in rows}
    bad_att = [r for r in rows if
               (r["treatment"] == "X" and r["SCF_AC_att"] != 0.0) or
               (r["treatment"] in ("TY", "K", "Y") and r["side"] == "chord" and r["SCF_AC_att"] != SCF_ATT) or
               (r["treatment"] in ("TY", "K", "Y") and r["side"] == "brace" and r["SCF_AC_att"] != brace_att)]
    print(f"\nAC_att consistency (0.0 for X; chord={SCF_ATT}, brace={brace_att:.4f} "
          f"for TY/K/Y -- NOT equal, per eqn 7b's own 0.4 factor): "
          f"{len(bad_att)} mismatches (expect 0), treatments seen: {sorted(att_by_treatment)}")
    assert len(bad_att) == 0

    # --- base_AC regression: chord base_AC for TY and K treatments must
    # equal eqn13_chord_crown_x called directly (the proven-identical
    # equation) -- catches the alias breaking silently if eqn13 ever
    # changes without eqn6b_chord_crown_base being updated to match.
    max_alias_err = 0.0
    for r in rows:
        if r["side"] != "chord" or r["treatment"] not in ("TY", "K"):
            continue
        expect = eqn13_chord_crown_x(r["beta"], r["gamma"], r["tau"], r["theta_deg"])
        max_alias_err = max(max_alias_err, abs(r["SCF_AC_base"] - expect))
    print(f"\nmax |chord base_AC - eqn13_chord_crown_x| over TY/K chord rows: "
          f"{max_alias_err:.2e} (expect ~0 -- proven algebraic identity)")
    assert max_alias_err < 1e-9

    # --- SCFX > SCFY > SCFK ordering (DNV's own stated ordering, a free
    # consistency check on the transcription), AS position -- unaffected by
    # the crown rework, evaluated on ONE fixed synthetic geometry common to
    # all three formula sets since the real jacket's K/TY/X connections
    # don't share geometry to compare directly.
    print("\nSCF_X > SCF_Y > SCF_K ordering check (DNV's own stated ordering), "
          "synthetic geometry beta=0.667 gamma=12.0 tau=0.5 theta=55 alpha=20:")
    beta, gamma, tau, theta, alpha = 0.667, 12.0, 0.5, 55.0, 20.0
    ty = scf_TY(beta, gamma, tau, theta, alpha)
    x = scf_X(beta, gamma, tau, theta, alpha)
    fake_row = dict(beta=beta, gamma=gamma, tau=tau, theta_deg=theta, alpha=alpha, zeta=0.3)
    k = scf_K_pair(fake_row, fake_row)
    for pos in ("AS",):
        for side in ("chord", "brace"):
            vx, vty, vk = x[side][pos], ty[side][pos], k[side][pos]
            print(f"  {side} {pos}: X={vx:.3f}  Y(TY)={vty:.3f}  K={vk:.3f}  "
                  f"(X>Y>K: {vx > vty > vk})")
            assert vx > vty > vk, f"SCF_X > SCF_Y > SCF_K failed for {side}/{pos}"

    # --- K vs Y genuinely differ on the real jacket (a real sanity check --
    # if these were ever numerically identical it would mean the K-specific
    # equations weren't actually being exercised).
    real_k = [r for r in rows if r["treatment"] == "K"]
    real_y = [r for r in rows if r["treatment"] == "Y"]
    key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                      r["chord_t_scenario"], r["side"])
    y_by_key = {key(r): r for r in real_y}
    as_diffs = [abs(r["SCF_AS"] - y_by_key[key(r)]["SCF_AS"]) for r in real_k]
    print(f"\nK vs Y-at-K-geometry SCF_AS: min diff {min(as_diffs):.3f}, "
          f"mean diff {np.mean(as_diffs):.3f} (expect clearly nonzero -- "
          f"real K-specific equations, not accidentally identical to Y)")
    assert min(as_diffs) > 0.01

    # --- F-correction sanity: F -> 1 as alpha grows large (long chord, no
    # correction needed), and F < 1 for a genuinely short chord (alpha=2).
    print("\nF-correction sanity (F->1 for long chord, F<1 for short):")
    for name, fn in (("F1", F1), ("F2", F2), ("F3", F3), ("F4", F4)):
        f_long = fn(0.667, 12.0, 30.0)
        f_short = fn(0.667, 12.0, 2.0)
        print(f"  {name}(alpha=30) = {f_long:.4f} (expect close to 1)  "
              f"{name}(alpha=2) = {f_short:.4f} (expect < 1)")
        assert abs(f_long - 1.0) < 0.05
        assert f_short < 1.0

    # --- Correction threshold is genuinely alpha<12, not alpha<4.
    _, _, applied_9 = _apply_short_chord(1.0, F2, 0.667, 12.0, 9.0)
    _, _, applied_15 = _apply_short_chord(1.0, F2, 0.667, 12.0, 15.0)
    print(f"\nShort-chord correction fires at alpha=9 (valid but <12): {applied_9} (expect True)")
    print(f"Short-chord correction fires at alpha=15: {applied_15} (expect False)")
    assert applied_9 is True and applied_15 is False

    # --- Positivity: every static SCF (AC_base, AS, MIP, MOP) must be a
    # positive, finite real number across all rows -- catches a domain
    # error (negative base to a fractional power, etc.) silently producing
    # NaN/complex rather than a loud crash.
    bad = [r for r in rows
           for k_ in ("SCF_AC_base", "SCF_AS", "SCF_MIP", "SCF_MOP")
           if not (np.isfinite(r[k_]) and r[k_] > 0)]
    print(f"\nNon-finite or non-positive static SCF values across all {len(rows)} rows: "
          f"{len(bad)} (expect 0)")
    assert len(bad) == 0

    print("\n" + "=" * 78)
    if SCF_EQUATIONS_VERIFIED:
        print("SCF_EQUATIONS_VERIFIED = True -- signed off. Step D (hotspot_joint) may use these SCFs.")
    else:
        print("SIGN-OFF REQUIRED: open scf_CHECK.csv, hand-calculate two connections")
        print("against DNVGL-RP-0005 App. B yourself, then flip SCF_EQUATIONS_VERIFIED = True.")
        print("Downstream stages must refuse to run while it is False.")
    print("=" * 78)


if __name__ == "__main__":
    _self_check()
