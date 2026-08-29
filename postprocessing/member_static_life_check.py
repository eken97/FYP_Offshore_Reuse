"""
Static (tension/compression/buckling) life check per splash-zone member, per
EN 1993-1-1 with DNV-ST-0126 factors -- bounds the fatigue-extrapolation
window at the point a corroded member would fail a static check first.

Formulas replicate the author's own verified spreadsheet
(the author's own verified spreadsheet, not distributed here), not re-derived independently:
  - Class (CHS, S355): D/t <= 33 -> 1, <= 46 -> 2, <= 59 -> 3, else 4
  - A_eff = A * sqrt((90/(D/t)) * (235/fy))   [Class 4 only, else A_eff = A]
  - i_c = sqrt(D^2 + d^2) / 4                  (d = inner diameter)
  - lambda = Lcr / i_c ;  lambda_1 = pi*sqrt(E/fy) ;  lambda_bar = lambda/lambda_1
  - buckling curve "d" (alpha=0.76), the author's deliberate choice for the
    splash-zone/reuse context (extra conservatism on top of, not instead of,
    the corrosion-driven area loss -- see docs/decisions.md)
  - psi = 0.5*(1+alpha*(lambda_bar-0.2)+lambda_bar^2)
  - chi = 1/(psi+sqrt(psi^2-lambda_bar^2))
  - N_b,Rd = chi * A_eff * fy / gamma_M1        (compression, governs over
    N_c,Rd since chi<=1 always)
  - N_c,Rd = A_eff * fy / gamma_M1              (tension AND compression
    cross-section -- the author confirmed A_eff used for both, matching their sheet)
  - gamma_M0 = gamma_M1 = 1.10 (DNV-ST-0126 Table 4-8)
  - gamma_f = 1.35 (DNV-ST-0437, "normal" ULS condition) applied to the
    DEMAND side only: N_Ed = gamma_f * F_demand -- NOT the sheet's separate
    "SF=1.5" cell, which is superseded by this verified factor (the author
    confirmed 18-19.08.2026 session)

D and t both corrode over time (sd_geometry.corroded_section) -- D shrinks
at the external rate only, t at the combined external+internal rate; for
braces (external-only corrosion) these differ, so both must be recomputed
every step, not just t.

CORROSION RATE: this is a static/ULS capacity check, NOT a fatigue check,
so it uses the Design Basis GENERAL rate of 0.30 mm/yr/surface
(sd_geometry.GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE), passed
explicitly as rates= to corroded_section(). The halved 0.15 mm/yr rate is
a fatigue-design-only allowance and must not bound a static check --
using it previously roughly doubled every reported static life (fixed
28.08.2026; see docs/decisions.md).

Output: results/real_campaign/member_static_life.csv, one row per
splash-zone member: governing failure mode (tension/compression), the life
in years before that check first fails, and the section thickness at that
point.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

import sd_geometry as sg

# region --- constants (all sourced, see module docstring) ---
E_PA = 210e9
E_MPA = 210000.0
FY_MPA = 355.0
GAMMA_M = 1.10       # DNV-ST-0126 Table 4-8, gamma_M0 == gamma_M1
GAMMA_F = 1.35        # DNV-ST-0437, normal ULS condition
BUCKLING_CURVE_ALPHA = 0.76   # curve "d", the author's deliberate choice
K_EFFECTIVE_LENGTH = 1.0      # pinned-pinned default

YEAR_STEP = 0.1
MAX_YEAR = 200.0

PROJECT = Path(__file__).resolve().parents[1]   # repo root
FORCE_CSV = PROJECT / "results" / "member_force_extremes.csv"
MEMBER_CSV = PROJECT / "results" / "final_results_member.csv"
OUT_CSV = PROJECT / "results" / "member_static_life.csv"
# endregion


def eurocode_capacity(D_m, t_m, Lcr_m, gamma_m=None):
    """Return (N_b_Rd, N_c_Rd) in Newtons for the given corroded section,
    per the formulas in the module docstring.

    `gamma_m` overrides the module's GAMMA_M (DNV-ST-0126 baseline, 1.10)
    when given -- added 19.08.2026 for the reuse classifier's
    P427 Section 4.5 gamma_M1,mod = 1.15 x gamma_M1 check
    (stage4_reuse_classification.py). Defaults to None so every existing
    caller (this module's own main(), the already-approved static-life
    results) is completely unaffected."""
    gm = GAMMA_M if gamma_m is None else gamma_m
    D_mm, t_mm = D_m * 1000.0, t_m * 1000.0
    d_mm = D_mm - 2 * t_mm
    Dt = D_mm / t_mm

    A_mm2 = math.pi / 4 * (D_mm**2 - d_mm**2)
    cls = 1 if Dt <= 33 else 2 if Dt <= 46 else 3 if Dt <= 59 else 4
    if cls == 4:
        A_eff_mm2 = A_mm2 * math.sqrt((90.0 / Dt) * (235.0 / FY_MPA))
    else:
        A_eff_mm2 = A_mm2

    i_c_mm = math.sqrt(D_mm**2 + d_mm**2) / 4.0
    Lcr_mm = Lcr_m * 1000.0 * K_EFFECTIVE_LENGTH
    lam = Lcr_mm / i_c_mm
    lam_1 = math.pi * math.sqrt(E_MPA / FY_MPA)
    lam_bar = lam / lam_1

    psi = 0.5 * (1 + BUCKLING_CURVE_ALPHA * (lam_bar - 0.2) + lam_bar**2)
    chi = 1.0 / (psi + math.sqrt(psi**2 - lam_bar**2))

    N_b_Rd_N = chi * A_eff_mm2 * FY_MPA / gm
    N_c_Rd_N = A_eff_mm2 * FY_MPA / gm
    return N_b_Rd_N, N_c_Rd_N, cls


def find_static_life(model, mid, L_m, F_tensile_N, F_compressive_N, gamma_m=None):
    """Step forward in time until either the tension or compression check
    first fails. Returns (life_years, governing_mode, t_mm_at_failure,
    D_mm_at_failure), or (None, 'never', None, None) if it survives to
    MAX_YEAR.

    `gamma_m` passthrough added 19.08.2026, see eurocode_capacity() -- None
    preserves this module's exact existing behaviour."""
    N_Ed_t = GAMMA_F * abs(F_tensile_N)
    N_Ed_c = GAMMA_F * abs(F_compressive_N)

    year = 0.0
    while year <= MAX_YEAR:
        D_m, t_m = sg.corroded_section(
            model, mid, year,
            rates=sg.GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE,
        )
        N_b_Rd, N_c_Rd, _cls = eurocode_capacity(D_m, t_m, L_m, gamma_m=gamma_m)

        fails_compression = N_Ed_c > N_b_Rd
        fails_tension = N_Ed_t > N_c_Rd
        if fails_compression or fails_tension:
            mode = "compression" if fails_compression else "tension"
            if fails_compression and fails_tension:
                mode = "compression" if N_Ed_c / N_b_Rd >= N_Ed_t / N_c_Rd else "tension"
            return year, mode, t_m * 1000.0, D_m * 1000.0

        year += YEAR_STEP

    return None, "never", None, None


def main():
    model = sg.read_subdyn_model(sg.DEFAULT_SD_PATH)
    joints = model["joints"]

    members_df = pd.read_csv(MEMBER_CSV)
    splash_ids = members_df[members_df["zone"] == "splash"]["member_id"].tolist()

    forces_df = pd.read_csv(FORCE_CSV).set_index("member_id")

    rows = []
    for mid in splash_ids:
        m = model["members"][mid]
        p1 = np.array(joints[m["j1"]])
        p2 = np.array(joints[m["j2"]])
        L_m = float(np.linalg.norm(p2 - p1))

        F_tensile_N = forces_df.loc[mid, "max_tensile_N"]
        F_compressive_N = forces_df.loc[mid, "max_compressive_N"]

        life_years, mode, t_fail_mm, D_fail_mm = find_static_life(
            model, mid, L_m, F_tensile_N, F_compressive_N
        )

        D0_m, t0_m, _pid = sg.member_section(model, mid)
        rows.append(dict(
            member_id=mid,
            member_class=sg.member_class(model, mid),
            L_m=L_m,
            D0_mm=D0_m * 1000.0,
            t0_mm=t0_m * 1000.0,
            max_tensile_N=F_tensile_N,
            max_compressive_N=F_compressive_N,
            static_life_years=life_years,
            governing_mode=mode,
            t_at_failure_mm=t_fail_mm,
            D_at_failure_mm=D_fail_mm,
        ))

    out = pd.DataFrame(rows).sort_values("member_id")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} rows to {OUT_CSV}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
