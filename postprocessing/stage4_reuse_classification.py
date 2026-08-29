"""
Reuse acceptance-criteria classifier: assigns every splash-track member (and,
via bay aggregation, every joint) a reuse category -- Structural, Component,
Downgraded, or Recycle -- per the ordered procedure agreed 19.08.2026 (see
docs/decisions.md). Orchestration only: every
physics check it uses already exists and is reused as-is, except where noted.

Step 1 (L0, members only -- joints skip this step, Flag A, 19.08.2026):
    Derive a corrosion allowance per member by root-finding the wall
    thickness at which a x2-inflated governing load fails the EXISTING,
    unmodified eurocode_capacity() check (member_static_life_check.py), at
    the module's own DNV-ST-0126 baseline GAMMA_M=1.10 -- NOT P427's
    reclaimed-steel factor, since this step is asking whether the as-built,
    as-corroded structure still clears its own original design terms, not
    yet verifying reclaimed steel for a new application (that starts at
    Step 3). Compare against the member's actual loss at DESIGN_YEAR.

Step 2 (L1, per bay): needs a member/joint -> bay mapping. CONFIRMED NOT TO
    EXIST anywhere in this codebase (grepped 19.08.2026) -- see
    BAY_GROUPING_CSV below. Runs and produces real Structural-reuse results
    the moment that file exists; until then every member is conservatively
    treated as if its bay failed Step 2, and falls through to Step 3 (never
    silently assumed to pass).

Step 3 (L2, per member): P427's flat 5% section loss (unmodified) and a new
    D_25 <= 5% fatigue-budget criterion. A static capacity check at P427
    Section 4.5's gamma_M1,mod = 1.15 x gamma_M1 is still COMPUTED and
    reported (l2_capacity_N_b_Rd_N / l2_capacity_N_c_Rd_N) but does
    NOT gate l2_pass -- decided 19.08.2026, see reuse_acceptance_criteria
    memory. Reasoning: checking against this member's OLD governing load
    (from its original jacket application) tests the wrong demand once the
    member is cut out for reuse in a new, as-yet-unspecified application;
    P427's own framing (Section 4.5) puts that check on the FUTURE designer
    who specifies the reclaimed member for their own new design, not on
    this disposal-side assessment. Empirically this check never bound under
    the old gating either -- 0/112 members failed it.

Step 4 (L3/L4): relaxed fatigue/corrosion thresholds -- D_25 <= 50% and
    loss_mm <= this member's OWN Step-1 allowance_mm (not a fixed mm value,
    since the allowance itself already varies per member/geometry -- see
    L3_FATIGUE_D25_LIMIT / L3_CORROSION_ALLOWANCE_MULTIPLIER below) -- then
    CEV/coating screens (not yet automated, flagged per-row), falling
    through to Recycle (L4) if none of that passes. (No static check here
    either -- dropped from L2/L3 gating 19.08.2026, same reasoning as Step
    3 above.) **Proposed 19.08.2026, not independently reviewed -- see docs/decisions.md. Checked against
    real data: these two conditions are non-binding for all 32 current
    Step-4 candidates (max D_25 = 0.115; all already satisfy loss <=
    allowance since they passed Step 1) -- currently sends 32/32 to
    Downgraded reuse, 0 to Recycle. Not a bug -- a real property of this
    dataset under these numbers.**

Rate note (important, resolved 19.08.2026): this script uses the UpWind
Design Basis's GENERAL rate -- 0.30 mm/yr/surface brace (external only),
0.60 mm/yr leg (both surfaces) -- confirmed by the author as sourced directly
from the Design Basis, for every STATIC/section-loss check below (Steps 1,
3's section-loss, 4's corrosion). This is deliberately NOT the same as
sd_geometry.corroded_section()'s 0.15 mm/yr/surface, which the Design Basis
permits specifically for FATIGUE design and which the existing, already-
approved fatigue/D_25 pipeline (and member_static_life_check.py's own
trajectory) already uses throughout -- that pipeline is untouched here.
D_25 values below are READ from the existing approved CSVs, not recomputed,
so they still reflect the 0.15 rate; only this script's own static/
section-loss math uses the general 0.30/0.60 rate. See
docs/decisions.md for the full reasoning either way -- this
is a real, deliberate split, not an inconsistency.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_geometry as sg
from member_static_life_check import eurocode_capacity, GAMMA_M as BASE_GAMMA_M

# region --- confirmed parameters (19.08.2026) ---
DESIGN_YEAR = 25.0

GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE = {
    "leg": dict(ext=0.30, int=0.30),
    "brace": dict(ext=0.30, int=0.0),
}

STEP1_LOAD_FACTOR = 2.0          # replaces GAMMA_F for Step 1's allowance derivation only
STEP1_GAMMA_M = None             # None -> eurocode_capacity()'s own DNV-ST-0126 baseline (1.10), unmodified
RECLAIMED_GAMMA_M_MULTIPLIER = 1.15   # P427 Section 4.5, applied on top of GAMMA_M from Step 3 onward

L2_SECTION_LOSS_LIMIT = 0.05     # P427 Section 5.3, fraction of nominal thickness
L2_FATIGUE_D25_LIMIT = 0.05      # new criterion -- borrows P427's own conservatism level by analogy

# PROPOSED 19.08.2026, not independently reviewed 20.08.2026 -- see
# docs/decisions.md for the reasoning and the real-data check
# against these specific numbers. Change here and re-run; nothing else
# needs touching.
MARGIN_L1_JOINT_D25 = 0.25             # bay's worst joint D_25 must be <= this to pass Step 2
L3_FATIGUE_D25_LIMIT = 0.50            # "remaining fatigue life > 50%"
# L3 corrosion is NOT a fixed mm number -- Step 1's allowance already varies
# per member (geometry/load-dependent), so L3's corrosion pass condition is
# loss_mm <= L3_CORROSION_ALLOWANCE_MULTIPLIER * that member's OWN
# l0_allowance_mm column. 1.0 = "hasn't eaten into its own allowance at all"
# (the author's exact framing, 19.08.2026). Bump this up/down for a looser/
# tighter L3 corrosion bar without touching run_step4().
L3_CORROSION_ALLOWANCE_MULTIPLIER = 1.0

# CONFIRMED SCOPE (19.08.2026, revised same day after checking real scenario
# meanings): K family ONLY -- not all 10 scenario/family combinations in
# final_results_joint.csv. Reasoning: the other 8 (Y-plane) either fail
# too badly to be worth discussing individually, or carry too many extra
# assumptions.
#
# Real scenario meanings (verified against final_results_joint.csv's actual
# columns AND the raw per-load-case matrix files it's built from, NOT assumed
# from naming): S1=baseline no retrofit no corrosion (all 40 joints),
# S2=Retrofit A no corrosion (all 40), S3=Retrofit A + corrosion (year-graded,
# D_S3-K_5.._25 -- but ONLY the 8 flagged splash-zone joints 22/26/30/34/
# 45/46/47/48, confirmed at the raw matrix file level: corrosion was never
# modelled for any other joint, not a downstream filtering artifact),
# S4=Retrofit B no corrosion (all 40), S5=Retrofit B + corrosion (year-graded,
# same 8-joint-only coverage as S3).
#
# COMPOSITE construction (the author's own, 20.08.2026 -- better than an earlier
# same-day attempt that ANDed raw S3/S4/S5 columns with a NaN-pass-through):
# since S2/S4 cover all 40 joints and S3/S5 cover exactly the same 8 splash
# joints, build_composite_scenarios() below folds each retrofit level into
# ONE always-defined column per joint -- the real corroded value where
# corrosion was modelled, and the SAME-retrofit no-corrosion value everywhere
# else (not an approximation: those 32 joints genuinely don't corrode in this
# project's scope, so their no-corrosion value under that retrofit IS the
# correct number, not a fallback). D_A_K = Retrofit A composite (S3 else S2).
# D_B_K = Retrofit B composite (S5 else S4, the "account for corrosion, more
# optimistic" case). The earlier raw-column AND approach had a real flaw this
# fixes: with S3 always NaN for non-splash joints, the "Retrofit A, corroded"
# branch was never actually testing non-splash bays against Retrofit A at
# all -- it silently fell through to nothing there, leaving Retrofit B's
# value (S4) as the only real check regardless of which branch was supposedly
# being evaluated. The composite columns close that gap.
#
# TWO INDEPENDENT VERDICTS, not one ANDed pass/fail (corrected 20.08.2026,
# the author's own catch): Retrofit A and Retrofit B are two different real-world
# choices, not two conditions that must both hold at once. ANDing them into a
# single structural_reuse_pass would hide cases where a bay clears the margin
# under ONE retrofit but not the other -- e.g. at a looser margin than
# today's, Bay 1's D_B_K (0.979) can pass while D_A_K (1.836) still fails; an
# AND would report "fail" and bury that Retrofit B alone is sufficient. So
# run_step2() computes structural_reuse_pass_A and structural_reuse_pass_B
# separately, and main() carries the classification through twice (once per
# retrofit choice) rather than once. At TODAY's margin (0.25) every bay fails
# under BOTH individually anyway, so the two verdicts happen to agree in the
# current numbers -- that's a property of today's specific margin, not a
# reason the pipeline should only compute one.
RETROFIT_SCENARIOS = {"A": "D_A_K", "B": "D_B_K"}

# Member-track note: final_results_member.csv (read for D_25 in Steps 3/4) has NO
# scenario/resize columns at all -- checked 19.08.2026, it's a single canonical
# result, not a matrix. This makes sense structurally, not just conveniently: the
# Resize A/B scenarios are joint-CAN-thickness retrofits (per
# docs/decisions.md) -- they change joint capacity, not member
# section properties, so they have no mechanism to affect a member's own D_25. The
# member side of this script needs no scenario scoping; only Step 2's joint read
# does.

PROJECT = Path(__file__).resolve().parents[1]   # repo root
MEMBER_CSV = PROJECT / "results" / "final_results_member.csv"
JOINT_CSV = PROJECT / "results" / "final_results_joint.csv"
FORCE_CSV = PROJECT / "results" / "member_force_extremes.csv"
BAY_GROUPING_CSV = PROJECT / "results" / "bay_grouping.csv"
OUT_CSV = PROJECT / "results" / "reuse_classification.csv"
# endregion


# region --- Step 1: corrosion allowance (general rate, member-only) ---
def corroded_section_general_rate(model, mid, ext_loss_mm):
    """
    Same (D, t) geometric relationship as sd_geometry.corroded_section(),
    but driven directly by an external-surface loss (mm) rather than a
    year, and at the general (not fatigue-halved) rate's ext/int SPLIT for
    this member's type -- i.e. legs lose thickness from both faces, braces
    only the outer one, exactly as in the shared function's own docstring.
    Kept separate from sd_geometry.py so that module's existing, approved
    behaviour (used by every other script) is untouched by this one.
    """
    D0, t0, _pid = sg.member_section(model, mid)
    cls = sg.member_class(model, mid)
    rate = GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE[cls]
    # int_loss scales with ext_loss in the same fixed ratio as the rate dict
    # (1:1 for legs, 0:1 for braces) so a single scalar still parameterises
    # the whole corrosion state, exactly as year does in the shared function.
    int_loss_mm = ext_loss_mm * (rate["int"] / rate["ext"]) if rate["ext"] else 0.0
    D = D0 - 2 * (ext_loss_mm / 1000.0)
    t = t0 - ((ext_loss_mm + int_loss_mm) / 1000.0)
    return D, t


def general_rate_ext_loss_mm(model, mid, year):
    """External-surface loss (mm) at the general rate, at a given year."""
    cls = sg.member_class(model, mid)
    return GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE[cls]["ext"] * year


def general_rate_actual_loss_mm(model, mid, year):
    """Actual thickness lost (mm) by `year`, at the general rate -- this is
    the number Step 1 and Step 3's section-loss check both compare against
    a limit."""
    D0, t0, _pid = sg.member_section(model, mid)
    _D, t = corroded_section_general_rate(model, mid, general_rate_ext_loss_mm(model, mid, year))
    return (t0 - t) * 1000.0


def step1_allowance_mm(model, mid, L_m, F_tensile_N, F_compressive_N):
    """
    Linear sweep (mirrors find_static_life()'s own year-stepping style,
    just parameterised by external-surface loss instead of year, since the
    allowance itself is a pure geometry/capacity question, not a rate
    question -- see module docstring) to find the external-surface loss at
    which STEP1_LOAD_FACTOR x the raw governing force first fails the
    static check at STEP1_GAMMA_M. Returns allowance_mm (nominal t minus
    critical t), or None if the member survives the sweep entirely (flag
    for review, don't silently treat as an infinite allowance).
    """
    D0, t0, _pid = sg.member_section(model, mid)
    N_Ed_t = STEP1_LOAD_FACTOR * abs(F_tensile_N)
    N_Ed_c = STEP1_LOAD_FACTOR * abs(F_compressive_N)

    ext_loss_mm = 0.0
    step_mm = 0.01
    max_ext_loss_mm = t0 * 1000.0 * 0.99   # stop just short of zero thickness
    while ext_loss_mm <= max_ext_loss_mm:
        D_m, t_m = corroded_section_general_rate(model, mid, ext_loss_mm)
        N_b_Rd, N_c_Rd, _cls = eurocode_capacity(D_m, t_m, L_m, gamma_m=STEP1_GAMMA_M)
        if N_Ed_c > N_b_Rd or N_Ed_t > N_c_Rd:
            _D, t_crit = corroded_section_general_rate(model, mid, ext_loss_mm)
            return (t0 - t_crit) * 1000.0
        ext_loss_mm += step_mm

    return None
# endregion


# region --- Step 1 driver ---
def run_step1(model, members_df, forces_df):
    rows = []
    for mid in members_df["member_id"]:
        m = model["members"][mid]
        p1 = np.array(model["joints"][m["j1"]])
        p2 = np.array(model["joints"][m["j2"]])
        L_m = float(np.linalg.norm(p2 - p1))

        zone = members_df.set_index("member_id").loc[mid, "zone"]
        F_t = forces_df.loc[mid, "max_tensile_N"]
        F_c = forces_df.loc[mid, "max_compressive_N"]

        allowance_mm = step1_allowance_mm(model, mid, L_m, F_t, F_c)
        loss_mm = general_rate_actual_loss_mm(model, mid, DESIGN_YEAR) if zone == "splash" else 0.0

        passed = (allowance_mm is not None) and (loss_mm <= allowance_mm)
        rows.append(dict(
            member_id=mid, zone=zone, member_class=sg.member_class(model, mid),
            l0_loss_mm=loss_mm, l0_allowance_mm=allowance_mm,
            l0_pass=passed,
        ))
    return pd.DataFrame(rows)
# endregion


# region --- Step 3: component reuse (per member) ---
def run_step3(model, mid_list, members_df, forces_df, step1_df):
    """
    section_ok and fatigue_ok are the only gating criteria for l2_pass.
    The static capacity check (gamma_M1,mod = 1.15 x gamma_M1, P427 Section
    4.5) is still computed and reported as l2_capacity_N_b_Rd_N /
    l2_capacity_N_c_Rd_N -- informational only, for a future designer
    who specifies this member in a new application to use with their own
    actual demand. It does NOT gate l2_pass here: this member's OLD
    governing load (from its original jacket application) is not a
    meaningful demand to test against once the member is cut out for reuse
    elsewhere. See module docstring / docs/decisions.md,
    decided 19.08.2026.
    """
    m_idx = members_df.set_index("member_id")
    s1_idx = step1_df.set_index("member_id")
    rows = []
    for mid in mid_list:
        m = model["members"][mid]
        p1 = np.array(model["joints"][m["j1"]])
        p2 = np.array(model["joints"][m["j2"]])
        L_m = float(np.linalg.norm(p2 - p1))

        D0, t0, _pid = sg.member_section(model, mid)
        loss_mm = s1_idx.loc[mid, "l0_loss_mm"]
        section_loss_frac = loss_mm / (t0 * 1000.0)
        section_ok = section_loss_frac <= L2_SECTION_LOSS_LIMIT

        D_25 = m_idx.loc[mid, "D_25"]
        fatigue_ok = D_25 <= L2_FATIGUE_D25_LIMIT

        # Informational only from here down -- not part of l2_pass (see docstring).
        D_m, t_m = corroded_section_general_rate(
            model, mid, general_rate_ext_loss_mm(model, mid, DESIGN_YEAR)
        )
        gamma_m = BASE_GAMMA_M * RECLAIMED_GAMMA_M_MULTIPLIER
        N_b_Rd, N_c_Rd, _cls = eurocode_capacity(D_m, t_m, L_m, gamma_m=gamma_m)

        passed = section_ok and fatigue_ok
        rows.append(dict(
            member_id=mid, l2_section_loss_frac=section_loss_frac,
            l2_section_ok=section_ok, l2_D25=D_25, l2_fatigue_ok=fatigue_ok,
            l2_pass=passed,
            l2_capacity_N_b_Rd_N=N_b_Rd, l2_capacity_N_c_Rd_N=N_c_Rd,
        ))
    return pd.DataFrame(rows)
# endregion


# region --- Step 4: downgraded reuse / recycle (per member) ---
def run_step4(model, mid_list, members_df, forces_df, step1_df):
    """
    fatigue_ok: D_25 <= L3_FATIGUE_D25_LIMIT.
    corrosion_ok: loss_mm <= L3_CORROSION_ALLOWANCE_MULTIPLIER * THIS
        MEMBER'S OWN l0_allowance_mm -- not a fixed mm number, since the
        allowance already varies per member (geometry/load-dependent, see
        step1_allowance_mm()). A member with a None allowance (survived
        Step 1's sweep entirely -- flagged for review, not currently seen
        in real data) fails corrosion_ok rather than being silently passed.
    Both proposed 19.08.2026, not independently reviewed -- see module
    docstring / docs/decisions.md for the real-data check
    against these exact numbers (currently non-binding for all 32 real
    candidates).
    """
    if L3_FATIGUE_D25_LIMIT is None or L3_CORROSION_ALLOWANCE_MULTIPLIER is None:
        # Loud, not silent -- if these get unset again, every member
        # reaching Step 4 is reported as "NOT CONFIGURED" rather than
        # guessed into Downgraded or Recycle.
        return pd.DataFrame([
            dict(member_id=mid, l3_pass=None, l3_note="L3 thresholds not yet set")
            for mid in mid_list
        ])

    m_idx = members_df.set_index("member_id")
    s1_idx = step1_df.set_index("member_id")
    rows = []
    for mid in mid_list:
        D_25 = m_idx.loc[mid, "D_25"]
        loss_mm = s1_idx.loc[mid, "l0_loss_mm"]
        allowance_mm = s1_idx.loc[mid, "l0_allowance_mm"]
        fatigue_ok = D_25 <= L3_FATIGUE_D25_LIMIT
        corrosion_limit_mm = (
            L3_CORROSION_ALLOWANCE_MULTIPLIER * allowance_mm if pd.notna(allowance_mm) else None
        )
        corrosion_ok = (corrosion_limit_mm is not None) and (loss_mm <= corrosion_limit_mm)
        # CEV/coating screens: not yet automated, flag per-row once wired
        # up. No static check here -- dropped from L2/L3 gating 19.08.2026,
        # same reasoning as Step 3 (see that function's docstring).
        passed = fatigue_ok and corrosion_ok
        rows.append(dict(
            member_id=mid, l3_D25=D_25, l3_fatigue_ok=fatigue_ok,
            l3_loss_mm=loss_mm, l3_corrosion_limit_mm=corrosion_limit_mm,
            l3_corrosion_ok=corrosion_ok, l3_pass=passed, l3_note="",
        ))
    return pd.DataFrame(rows)
# endregion


def build_composite_scenarios(joint_df):
    """
    D_A_K = D_S3-K_25 where real (the 8 flagged splash joints), else D_S2-K
    (same Retrofit A, but the correct no-corrosion value for the other 32
    joints, which genuinely don't corrode in this project's scope).
    D_B_K = same pattern for Retrofit B (D_S5-K_25 else D_S4-K). Always
    defined for all 40 joints -- S2/S4 have full coverage, confirmed
    20.08.2026. See SCENARIOS_TO_RUN comment above for the full reasoning.
    """
    joint_df = joint_df.copy()
    joint_df["D_A_K"] = joint_df["D_S3-K_25"].fillna(joint_df["D_S2-K"])
    joint_df["D_B_K"] = joint_df["D_S5-K_25"].fillna(joint_df["D_S4-K"])
    return joint_df


# region --- Step 2: bay aggregation ---
def run_step2(step1_df, joint_df):
    """
    Uses BAY_GROUPING_CSV (build_bay_grouping.py's output, derived directly
    from SubDyn geometry -- see that script's docstring, 19.08.2026) to
    aggregate Step 1 member results and joint D_25 by bay, for EACH retrofit
    scenario in RETROFIT_SCENARIOS independently (the composite D_A_K/D_B_K
    columns built by build_composite_scenarios(), always defined for all 40
    joints).

    Produces a SEPARATE verdict per retrofit choice -- structural_reuse_
    pass_A and structural_reuse_pass_B -- not one ANDed pass/fail. See
    RETROFIT_SCENARIOS comment above for why: Retrofit A and B are different
    real-world choices, and a bay can legitimately clear the margin under one
    but not the other.

    A bay's final verdicts still can't be produced -- MARGIN_L1_JOINT_D25 is
    not yet set (docs/decisions.md) -- but the
    underlying numbers (did every member pass Step 1, what is the worst-joint
    D_25 in the bay per retrofit scenario) are computed and returned so they
    can inform that threshold choice, the same way Step 4 reports "not
    configured" rather than guessing.

    Returns a DataFrame: one row per bay, with member_ids/joint_ids,
    members_all_pass_l0, worst-joint D_25 per retrofit scenario, and
    structural_reuse_pass_{A,B} (None until MARGIN_L1_JOINT_D25 is set).
    """
    if not BAY_GROUPING_CSV.exists():
        return None
    bay_df = pd.read_csv(BAY_GROUPING_CSV)
    step1_idx = step1_df.set_index("member_id")
    joint_df = build_composite_scenarios(joint_df)
    joint_idx = joint_df.set_index("node")

    rows = []
    for bay_id, g in bay_df.groupby("bay_id"):
        member_ids = g[g["entity_type"] == "member"]["entity_id"].tolist()
        joint_ids = g[g["entity_type"] == "joint"]["entity_id"].tolist()

        members_all_pass = bool(step1_idx.loc[member_ids, "l0_pass"].all())

        row = dict(bay_id=bay_id, member_ids=member_ids, joint_ids=joint_ids,
                   n_members=len(member_ids), n_joints=len(joint_ids),
                   members_all_pass_l0=members_all_pass)
        for label, col in RETROFIT_SCENARIOS.items():
            vals = joint_idx.loc[joint_ids, col]
            row[f"worst_joint_{col}"] = vals.max()
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("bay_id")

    if MARGIN_L1_JOINT_D25 is None:
        for label in RETROFIT_SCENARIOS:
            out[f"structural_reuse_pass_{label}"] = None
        out["note"] = "margin_L1_joint_D25 not yet set"
    else:
        def bay_pass_under(row, col):
            # Composite columns (D_A_K/D_B_K) are always defined for all 40
            # joints (confirmed 20.08.2026), so this NaN guard should never
            # actually trigger now -- kept defensively in case
            # RETROFIT_SCENARIOS is ever pointed back at a raw, partially-NaN
            # scenario column: pd.notna() ensures a NaN is never silently
            # treated as a fail via `<=`.
            val = row[f"worst_joint_{col}"]
            joint_ok = pd.isna(val) or val <= MARGIN_L1_JOINT_D25
            return bool(row["members_all_pass_l0"] and joint_ok)

        def bay_unconfirmed(row, col):
            # Should always be False under the composite scenarios -- see
            # bay_pass_under() docstring. Kept as a live check, not removed,
            # so a future change that reintroduces a partially-NaN scenario
            # column is caught rather than silently unflagged.
            return pd.isna(row[f"worst_joint_{col}"])

        any_unconfirmed = pd.Series(False, index=out.index)
        for label, col in RETROFIT_SCENARIOS.items():
            out[f"structural_reuse_pass_{label}"] = out.apply(lambda r, c=col: bay_pass_under(r, c), axis=1)
            any_unconfirmed = any_unconfirmed | out.apply(lambda r, c=col: bay_unconfirmed(r, c), axis=1)
        out["uses_unconfirmed_nonretrofit_baseline"] = any_unconfirmed
        out["note"] = out["uses_unconfirmed_nonretrofit_baseline"].map(
            lambda x: (
                "some joints in this bay have no retrofit-scenario data -- their "
                "baseline (non-retrofit) D_25 is not directly confirmed, only "
                "inferred from not being flagged as needing retrofit -- see "
                "docs/decisions.md section 9 item 3"
            ) if x else ""
        )

    return out
# endregion


def main():
    model = sg.read_subdyn_model(sg.DEFAULT_SD_PATH)
    members_df = pd.read_csv(MEMBER_CSV)
    joint_df = pd.read_csv(JOINT_CSV)
    forces_df = pd.read_csv(FORCE_CSV).set_index("member_id")

    print(f"Step 1: {len(members_df)} members...")
    step1_df = run_step1(model, members_df, forces_df)
    n_step1_pass = step1_df["l0_pass"].sum()
    print(f"  {n_step1_pass}/{len(step1_df)} pass Step 1 (L0 admissibility)")

    print("Step 2: bay aggregation (per retrofit scenario)...")
    bay_result = run_step2(step1_df, joint_df)
    structural_reuse_members = {label: set() for label in RETROFIT_SCENARIOS}
    if bay_result is None:
        print(f"  SKIPPED -- {BAY_GROUPING_CSV} does not exist. "
              f"Every member falls through to Step 3 (not assumed to pass).")
    else:
        bay_result.to_csv(OUT_CSV.parent / "reuse_classification_bays.csv", index=False)
        for label in RETROFIT_SCENARIOS:
            col = f"structural_reuse_pass_{label}"
            n_pass = (bay_result[col] == True).sum()  # noqa: E712 (None != True)
            n_pending = bay_result[col].isna().sum()
            print(f"  Retrofit {label}: {len(bay_result)} bays, {n_pass} pass, "
                  f"{len(bay_result) - n_pass - n_pending} fail, {n_pending} pending threshold")
            for _, r in bay_result[bay_result[col] == True].iterrows():  # noqa: E712
                structural_reuse_members[label].update(r["member_ids"])

    # L2/L3/L4 (Steps 3/4) are retrofit-INDEPENDENT (Resize A/B are joint-can
    # retrofits, they don't touch member section properties -- see module
    # docstring), so they're computed ONCE for all 112 members regardless of
    # which retrofit's structural-reuse bay set is being asked about. Each
    # retrofit scenario's final label then just picks between "already
    # covered by ITS OWN structural-reuse bay" and this same shared L2/L3/L4
    # result underneath -- no need to recompute L2/L3/L4 twice.
    print(f"Step 3: {len(step1_df)} members (all -- L2 doesn't depend on retrofit choice)...")
    step3_df = run_step3(model, step1_df[step1_df["l0_pass"]]["member_id"].tolist(),
                          members_df, forces_df, step1_df)
    n_step3_pass = step3_df["l2_pass"].sum() if len(step3_df) else 0
    print(f"  {n_step3_pass}/{len(step3_df)} pass Step 3 (L2 component reuse)")

    step1_fail = step1_df[~step1_df["l0_pass"]]["member_id"].tolist()
    step3_fail = step3_df[~step3_df["l2_pass"]]["member_id"].tolist() if len(step3_df) else []
    step4_candidates = sorted(set(step1_fail) | set(step3_fail))
    print(f"Step 4: {len(step4_candidates)} members...")
    step4_df = run_step4(model, step4_candidates, members_df, forces_df, step1_df)
    if len(step4_df) and step4_df["l3_pass"].isna().all():
        print("  NOT CONFIGURED -- L3 thresholds not yet set, see module header.")

    # Assemble the shared (retrofit-independent) L0/L2/L3 result per member
    out = members_df[["member_id", "zone"]].merge(step1_df.drop(columns=["zone"]), on="member_id")
    out = out.merge(step3_df, on="member_id", how="left")
    out = out.merge(step4_df, on="member_id", how="left")

    # Every member (all 112) is sorted here, ONCE PER RETROFIT SCENARIO --
    # reuse_level_{A,B} is the flowchart-matching L1-L4 tag, reuse_category_
    # {A,B} adds the plain-English name. A member can land in L1 under one
    # retrofit and fall through to L2/L3/L4 under the other -- that's the
    # whole point of keeping the two verdicts separate (see RETROFIT_
    # SCENARIOS comment above), not an edge case to collapse away.
    def classify(row, label):
        if row["member_id"] in structural_reuse_members[label]:
            return "L1", "Structural reuse"
        if row.get("l2_pass") is True:
            return "L2", "Component reuse"
        if row.get("l3_pass") is True:
            return "L3", "Downgraded reuse"
        if row.get("l3_pass") is False:
            return "L4", "Recycle"
        return "Pending", "Pending (L3 thresholds not set)"

    for label in RETROFIT_SCENARIOS:
        out[f"in_structural_reuse_bay_{label}"] = out["member_id"].isin(structural_reuse_members[label])
        levels_names = out.apply(lambda r, lbl=label: classify(r, lbl), axis=1, result_type="expand")
        out[f"reuse_level_{label}"] = levels_names[0]
        out[f"reuse_category_{label}"] = levels_names[0] + " - " + levels_names[1]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(out)} rows to {OUT_CSV} (all {len(members_df)} members)")
    for label in RETROFIT_SCENARIOS:
        print(f"\nUnder Retrofit {label}:")
        print(out[f"reuse_category_{label}"].value_counts().to_string())


if __name__ == "__main__":
    main()
