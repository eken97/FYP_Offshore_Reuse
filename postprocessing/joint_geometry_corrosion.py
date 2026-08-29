"""
Joint-track corrosion, Step J1 -- per-connection geometry at a corrosion year.

NEW file, not an edit to joint_geometry.py. A long combined campaign run
(member-corrosion + joint-uncorroded) executes off exactly what is on disk,
and joint_geometry.py sits in that running pipeline's live import graph
(via stage2_joints.py) -- editing it risks a worker mid-run
re-importing a half-finished version (see docs/decisions.md). This module only IMPORTS joint_geometry/scf (read-only, zero risk
to a running process) and adds new, parallel functions; nothing in either
existing module changes.

WHAT CHANGES WITH CORROSION, WHAT DOESN'T -- read directly off
joint_geometry.compute_geometry_params's own formulas:
  - UNCHANGED (pure node/axis geometry -- corrosion thins wall/OD, it does
    not move the SubDyn beam centerline model): theta_deg, azimuth_deg,
    phi_deg, chord_a/b_phi_deg, chord_length, e_par/n/mop_axis.
  - CHANGED (built from D/T/t, which DO corrode): beta=d/D, gamma=D/2T,
    tau=t/T, alpha=2L/D (L unchanged, D corrodes), zeta=g/D (g depends on
    brace/chord D via joint_geometry._brace_near_toe_offset), and the chord
    leg-segments' own D/T/W (feeds stress.hotspot_joint's sigma_BendingChord
    via W).

Reuses sd_geometry.corroded_section() UNCHANGED (already corrosion-forward:
any member ID in, corroded (D,t) out -- works identically for a brace member
or a leg/chord-segment member, no brace-vs-chord distinction inside that
function) and joint_geometry's own private geometry helpers
(_zeta_and_validity) plus scf.py's own K-plane pairing (_group_k_planes)
rather than re-deriving either -- avoids a second, possibly-diverging
transcription of already G1/G2-signed-off maths. This module's own
self-check proves year=0 output matches joint_geometry.py's own stored
(uncorroded) values exactly, then extends to nonzero years.

Scope note: corrosion only matters for splash-zone connections (see
docs/decisions.md corrosion rule -- legs both-sides,
braces external-only, 0.15 mm/yr/surface). This module computes geometry
for whatever connections it's handed; the caller (stage2_joints_corrosion,
not yet built) is responsible for filtering to the splash-zone connections
before calling this repeatedly per year.
"""
from pathlib import Path

import sd_geometry as sdg
import joint_geometry as jg
import scf

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# endregion


def corroded_chord_segment(model, mid, year):
    """(D, T, W) for one chord leg-segment member at corrosion `year`.
    phi_deg is pure geometry (DCM-based) and unaffected by corrosion --
    callers reuse the connection's own year-0 chord_a/b_phi_deg unchanged."""
    D, T = sdg.corroded_section(model, mid, year)
    W = sdg.section_properties(D, T)["W"]
    return D, T, W


def corroded_connection_geometry(c, model, other_row, year):
    """
    Recomputes beta/gamma/tau/alpha/zeta (and the corroded brace_D/brace_t,
    chord_D/chord_T) for one connection `c` at corrosion `year`.

    other_row: the OTHER brace sharing c's K-plane (same sub_joint_id +
    chord_t_scenario) -- needed for zeta, same pairing role
    compute_geometry_params/scf.compute_all_scf already use. Pass None for
    TY/X (zeta is None for those families regardless, same as year-0).

    theta_deg and chord_length are geometric, carried through UNCHANGED from
    `c` (see module docstring) -- only D/T/t-derived quantities are
    recomputed.

    Returns a dict: brace_D, brace_t, chord_D, chord_T, beta, gamma, tau,
    alpha, zeta -- same field names/semantics as joint_geometry.py's own
    connection rows at year 0.
    """
    brace_D, brace_t = sdg.corroded_section(model, c["brace_member"], year)

    if c["family"] == "X":
        chord_mid = c["chord_members"][0][0]
        chord_D, chord_T = sdg.corroded_section(model, chord_mid, year)
    else:
        leg_a_mid = c["chord_members"][0][0]
        leg_b_mid = c["chord_members"][1][0]
        Da, Ta = sdg.corroded_section(model, leg_a_mid, year)
        Db, Tb = sdg.corroded_section(model, leg_b_mid, year)
        assert abs(Da - Db) < 1e-9, (
            f"node {c['node']}: corroded leg diameters diverge ({Da} vs {Db}) at year "
            f"{year} -- both leg segments corrode at the same rate (member_class='leg' "
            f"for both), so this should stay exact, matching joint_geometry.py's own "
            f"year-0 assertion"
        )
        chord_D = Da
        if c["chord_t_scenario"] == "thick":
            chord_T = max(Ta, Tb)
        elif c["chord_t_scenario"] == "thin":
            chord_T = min(Ta, Tb)
        else:
            chord_T = Ta   # "single" -- Ta == Tb by construction (year-0 chord_T_ambiguous=False)

    beta = brace_D / chord_D
    gamma = chord_D / (2.0 * chord_T)
    tau = brace_t / chord_T
    alpha = 2.0 * c["chord_length"] / chord_D

    zeta = None
    if c["family"] == "K":
        assert other_row is not None, "K-family connection needs its plane partner for zeta"
        other_brace_D, _other_brace_t = sdg.corroded_section(model, other_row["brace_member"], year)
        zeta = jg._zeta_and_validity(
            brace_D, c["theta_deg"], other_brace_D, other_row["theta_deg"], chord_D
        )

    return dict(brace_D=brace_D, brace_t=brace_t, chord_D=chord_D, chord_T=chord_T,
                beta=beta, gamma=gamma, tau=tau, alpha=alpha, zeta=zeta)


def _self_check():
    print(f"Parsing: {sdg.DEFAULT_SD_PATH}")
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    k_groups = scf._group_k_planes(connections)

    def other_row_for(c):
        if c["family"] != "K":
            return None
        pair = k_groups[(c["sub_joint_id"], c["chord_t_scenario"])]
        return pair[0] if pair[1] is c else pair[1]

    # --- year=0 must reproduce joint_geometry.py's own stored (uncorroded)
    # values exactly -- the decisive check that this module's independently
    # re-derived formulas match the already G1-signed-off originals.
    print("\n1. year=0 reproduces joint_geometry.py's own stored geometry exactly:")
    one_per_family = {}
    for c in connections:
        one_per_family.setdefault(c["family"], c)
    max_err = dict(beta=0.0, gamma=0.0, tau=0.0, alpha=0.0)
    zeta_checked = 0
    for c in connections:
        g0 = corroded_connection_geometry(c, model, other_row_for(c), year=0)
        for key in ("beta", "gamma", "tau", "alpha"):
            max_err[key] = max(max_err[key], abs(g0[key] - c[key]))
        assert abs(g0["brace_D"] - c["brace_D"]) < 1e-12
        assert abs(g0["chord_D"] - c["chord_D"]) < 1e-12
        assert abs(g0["chord_T"] - c["chord_T"]) < 1e-12
        if c["zeta"] is not None:
            assert g0["zeta"] is not None
            assert abs(g0["zeta"] - c["zeta"]) < 1e-9
            zeta_checked += 1
        else:
            assert g0["zeta"] is None
    print(f"   max |year0 - stored| over all {len(connections)} connections: {max_err}")
    print(f"   zeta matched on {zeta_checked} K-family connections")
    for key, err in max_err.items():
        assert err < 1e-9, f"{key}: year=0 diverges from joint_geometry.py's own value ({err:.3e})"

    # --- one representative connection per family, tracked across years,
    # monotonic thinning (D/T/beta/tau/gamma all move in the expected
    # direction as corrosion progresses).
    print("\n2. representative connection per family, year 0 vs 25:")

    def _fmt_zeta(z):
        return "None" if z is None else f"{z:.4f}"

    for fam, c in sorted(one_per_family.items()):
        g0 = corroded_connection_geometry(c, model, other_row_for(c), year=0)
        g25 = corroded_connection_geometry(c, model, other_row_for(c), year=25)
        print(f"   {fam}: brace_D {g0['brace_D']*1000:.2f}->{g25['brace_D']*1000:.2f}mm  "
              f"chord_D {g0['chord_D']*1000:.2f}->{g25['chord_D']*1000:.2f}mm  "
              f"beta {g0['beta']:.4f}->{g25['beta']:.4f}  "
              f"gamma {g0['gamma']:.3f}->{g25['gamma']:.3f}  "
              f"tau {g0['tau']:.4f}->{g25['tau']:.4f}  "
              f"zeta {_fmt_zeta(g0['zeta'])}->{_fmt_zeta(g25['zeta'])}")
        # brace is external-only (D loses 2x ext rate), chord/leg is
        # flooded both-sides (D loses 2x ext rate too -- ext rate is what
        # moves D regardless of int) -- both D's must strictly decrease.
        assert g25["brace_D"] < g0["brace_D"]
        assert g25["chord_D"] < g0["chord_D"]
        assert g25["chord_T"] < g0["chord_T"]

    # --- corroded_chord_segment: W must shrink with year (thinner section
    # -> smaller I, smaller R barely changes -> smaller W), spot-checked on
    # one real leg member.
    print("\n3. corroded_chord_segment W (section modulus) shrinks with year:")
    k_conn = next(c for c in connections if c["family"] == "K")
    leg_mid = k_conn["chord_members"][0][0]
    D0, T0, W0 = corroded_chord_segment(model, leg_mid, 0)
    D25, T25, W25 = corroded_chord_segment(model, leg_mid, 25)
    print(f"   member {leg_mid}: D {D0*1000:.2f}->{D25*1000:.2f}mm  T {T0*1000:.2f}->{T25*1000:.2f}mm  "
          f"W {W0:.6e}->{W25:.6e} m^3")
    assert W25 < W0

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
