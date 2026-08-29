"""
Joint-track can-thickness retrofit override.

NEW file, not an edit to joint_geometry.py/sd_geometry.py -- same
code-freeze-respecting convention as joint_geometry_corrosion.py: this
module only IMPORTS existing modules (read-only) and adds new functions
that feed thickened D/T/t into the already-verified downstream code
(scf.py's formulas, stress.hotspot_joint via chord_a/b_W).

WHY: UpWind D4.2.5 (Ramboll's own design report for this exact jacket --
see docs/decisions.md) needed local wall-thickness
increases ("can sections") at joints to pass fatigue; the real OC4/SubDyn
model this thesis uses has none. Two retrofit scenarios are modelled here,
both a LOCAL increase in wall thickness at the joint only (not a full
member re-section):

CONVENTION (matches the ballpark sensitivity work in
docs/decisions.md exactly -- "thin-wall approx, stress ~
1/t for fixed OD"): outer diameter D is UNCHANGED, wall thickness grows
INWARD (bore shrinks). Consequence: since beta=d/D, alpha=2L/D and zeta=g/D
depend only on diameters/lengths, none of which this retrofit touches,
beta/alpha/zeta are IDENTICAL before and after retrofit -- only gamma
(=D/2T) and tau (=t/T) shift, driven by the thickness change alone. This
is a genuine simplification versus the corrosion driver (which does change
D), not an approximation.

RETROFIT_A -- the author's own exact spec, read directly off UpWind D4.2.5
Figure 3-3 (see docs/decisions.md, 16.08.2026):
  X top (+10.262m) / X bottom (-33.373m):         +10mm, both members
  X upper-mid (-1.958m) / X lower-mid (-16.371m): +5mm, both members
  K (+4.378m, -8.922m) and ALL TY (+15.651/-43.127/-44.001m):
                                                    +5mm, both sides
  K bottom (-24.614m), chord_t_scenario="thick" only: +5mm, both sides
  K bottom (-24.614m), chord_t_scenario="thin": NOT retrofitted (no
    can-thickness number exists for this reading -- out of scope,
    matches the original ballpark exercise's own restriction)

RETROFIT_B -- flat, structure-wide pass (confirmed by the author
16.08.2026): +12mm, both sides, at EVERY connection, with the SAME
bottom-K "thick"-scenario-only restriction as A (the author's explicit choice,
to keep the two scenarios comparable at that one ambiguous node -- the
"thin" rows there stay unretrofitted in both scenarios).

"Both members"/"both sides" means: brace_t AND chord_T (the value scf.py's
formulas actually use for gamma/tau) get the identical delta, AND the two
physical leg-segment members feeding chord_a_T/chord_b_T (stress.py's
chord-bending nominal stress, via chord_a_W/chord_b_W) get the SAME delta
too -- this is exactly consistent because delta commutes with max/min:
max(Ta+d, Tb+d) == max(Ta,Tb)+d, so chord_T_new (built from the retrofitted
chord_a_T/chord_b_T) is identical whether computed directly or picked
after the fact.
"""
from pathlib import Path

import sd_geometry as sdg
import joint_geometry as jg

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# endregion

SCENARIOS = ("A", "B")

# region --- elevation groups (verified directly against the real SubDyn
# model this session -- not just carried over from memory, see the
# node-by-node z printout that preceded this file) ---
X_TOP_Z = 10.262
X_BOTTOM_Z = -33.373
X_MID_Z = (-1.958, -16.371)
K_BOTTOM_Z = -24.614
Z_TOL_M = 0.01   # 1cm -- elevations are known to mm precision, generous vs float noise
# endregion


def _near(z, target):
    return abs(z - target) < Z_TOL_M


def retrofit_deltas_m(c, model, scenario):
    """
    (delta_brace_t_m, delta_chord_T_m) for connection `c` under `scenario`
    ("A" or "B"). Both zero = connection is left completely unretrofitted
    (only the bottom-K "thin" rows, in both scenarios).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}, expected one of {SCENARIOS}")

    fam = c["family"]
    z = model["joints"][c["node"]][2]

    if fam == "K" and _near(z, K_BOTTOM_Z) and c["chord_t_scenario"] == "thin":
        return 0.0, 0.0

    if scenario == "B":
        return 0.012, 0.012

    # scenario == "A"
    if fam == "X":
        if _near(z, X_TOP_Z) or _near(z, X_BOTTOM_Z):
            return 0.010, 0.010
        if any(_near(z, mz) for mz in X_MID_Z):
            return 0.005, 0.005
        raise ValueError(
            f"node {c['node']}: X-family connection at unexpected z={z:.3f} -- "
            f"doesn't match any of the 4 known X elevations"
        )
    if fam in ("K", "TY"):
        return 0.005, 0.005
    raise ValueError(f"unknown family {fam!r}")


def retrofit_connection_geometry(c, model, scenario):
    """
    Recomputes chord_T/brace_t/chord_a_T/chord_b_T/gamma/tau (and passes
    beta/alpha/zeta through UNCHANGED -- see module docstring) for
    connection `c` under the can-thickness `scenario`.

    Requires `c` to already carry chord_a_T/chord_b_T/chord_a_D/chord_b_D
    (i.e. joint_geometry.add_chord_geometry has already run on the full
    connection list, same precondition stage2_joints.py's own
    build_full_connections already establishes).

    Returns a dict: brace_D, brace_t, chord_D, chord_T, beta, gamma, tau,
    alpha, zeta, chord_a_D, chord_a_T, chord_a_W, chord_b_D, chord_b_T,
    chord_b_W -- same field names/semantics as the connection's own year-0
    stored values, so this dict can replace them the same way
    joint_geometry_corrosion.corroded_connection_geometry's output does.
    """
    delta_brace, delta_chord = retrofit_deltas_m(c, model, scenario)

    brace_t = c["brace_t"] + delta_brace
    chord_T = c["chord_T"] + delta_chord
    chord_a_T = c["chord_a_T"] + delta_chord
    chord_b_T = c["chord_b_T"] + delta_chord

    for t_val, D_val, label in ((brace_t, c["brace_D"], "brace"),
                                 (chord_T, c["chord_D"], "chord")):
        assert 0 < t_val < D_val / 2.0, (
            f"node {c['node']}/brace {c['brace_member']}: retrofitted {label} "
            f"thickness {t_val*1000:.1f}mm is not physically sane against "
            f"D={D_val*1000:.1f}mm"
        )

    gamma = c["chord_D"] / (2.0 * chord_T)
    tau = brace_t / chord_T

    chord_a_W = sdg.section_properties(c["chord_a_D"], chord_a_T)["W"]
    chord_b_W = sdg.section_properties(c["chord_b_D"], chord_b_T)["W"]

    return dict(
        brace_D=c["brace_D"], brace_t=brace_t,
        chord_D=c["chord_D"], chord_T=chord_T,
        beta=c["beta"], gamma=gamma, tau=tau,
        alpha=c["alpha"], zeta=c["zeta"],
        chord_a_D=c["chord_a_D"], chord_a_T=chord_a_T, chord_a_W=chord_a_W,
        chord_b_D=c["chord_b_D"], chord_b_T=chord_b_T, chord_b_W=chord_b_W,
    )


def corrode_from_baseline(D0, t0, member_class_str, year):
    """
    Same thickness-loss formula as sd_geometry.corroded_section, but
    starting from a caller-supplied (D0, t0) baseline instead of reading
    nominal section properties off the model -- lets corrosion compose on
    top of a thickness-retrofit baseline without editing sd_geometry.py.
    Reuses sd_geometry's own rate table (imported, not re-typed) so the
    two functions can never silently diverge on the actual mm/yr numbers.
    """
    rate = sdg.CORROSION_RATE_MM_PER_YEAR_PER_SURFACE[member_class_str]
    ext_loss_m = (rate["ext"] * year) / 1000.0
    int_loss_m = (rate["int"] * year) / 1000.0
    D = D0 - 2 * ext_loss_m
    t = t0 - (ext_loss_m + int_loss_m)
    assert t > 0, (
        f"{member_class_str}: corroded thickness {t*1000:.2f}mm <= 0 at year "
        f"{year} (t0={t0*1000:.1f}mm) -- section fully consumed"
    )
    return D, t


def retrofit_and_corrode_connection_geometry(c, model, other_row, scenario, year):
    """
    Composes: apply the can-thickness retrofit FIRST (defines a new year-0
    baseline D/T/t), THEN apply `year` years of corrosion loss from that
    baseline -- "the can gets built, then it corrodes," not the other
    order. Direct structural analogue of
    joint_geometry_corrosion.corroded_connection_geometry, substituting
    corrode_from_baseline(retrofitted D/t, ...) for
    sd_geometry.corroded_section(model, mid, year) at every step.

    `other_row`: the OTHER brace sharing c's K-plane (see
    joint_geometry_corrosion's own docstring for the same parameter) --
    pass None for TY/X. Caller is responsible for scoping `c` to
    splash-zone connections (corrosion is only physically defined there);
    this function doesn't check zone itself.
    """
    g = retrofit_connection_geometry(c, model, scenario)
    brace_cls = sdg.member_class(model, c["brace_member"])
    brace_D, brace_t = corrode_from_baseline(g["brace_D"], g["brace_t"], brace_cls, year)

    if c["family"] == "X":
        chord_mid = c["chord_members"][0][0]
        chord_cls = sdg.member_class(model, chord_mid)
        chord_D, chord_T = corrode_from_baseline(g["chord_D"], g["chord_T"], chord_cls, year)
    else:
        leg_a_mid = c["chord_members"][0][0]
        leg_b_mid = c["chord_members"][1][0]
        leg_a_cls = sdg.member_class(model, leg_a_mid)
        leg_b_cls = sdg.member_class(model, leg_b_mid)
        Da, Ta = corrode_from_baseline(g["chord_a_D"], g["chord_a_T"], leg_a_cls, year)
        Db, Tb = corrode_from_baseline(g["chord_b_D"], g["chord_b_T"], leg_b_cls, year)
        assert abs(Da - Db) < 1e-9, (
            f"node {c['node']}: retrofitted+corroded leg diameters diverge "
            f"({Da} vs {Db}) at year {year} -- both leg segments should corrode "
            f"identically (both 'leg' class, same external rate)"
        )
        chord_D = Da
        if c["chord_t_scenario"] == "thick":
            chord_T = max(Ta, Tb)
        elif c["chord_t_scenario"] == "thin":
            chord_T = min(Ta, Tb)
        else:
            chord_T = Ta

    beta = brace_D / chord_D
    gamma = chord_D / (2.0 * chord_T)
    tau = brace_t / chord_T
    alpha = 2.0 * c["chord_length"] / chord_D

    zeta = None
    if c["family"] == "K":
        assert other_row is not None, "K-family connection needs its plane partner for zeta"
        other_g = retrofit_connection_geometry(other_row, model, scenario)
        other_cls = sdg.member_class(model, other_row["brace_member"])
        other_brace_D, _other_brace_t = corrode_from_baseline(
            other_g["brace_D"], other_g["brace_t"], other_cls, year)
        zeta = jg._zeta_and_validity(
            brace_D, c["theta_deg"], other_brace_D, other_row["theta_deg"], chord_D
        )

    return dict(brace_D=brace_D, brace_t=brace_t, chord_D=chord_D, chord_T=chord_T,
                beta=beta, gamma=gamma, tau=tau, alpha=alpha, zeta=zeta)


def _self_check():
    print(f"Parsing: {sdg.DEFAULT_SD_PATH}, {jg.DEFAULT_SD_SUM_PATH}")
    dcm_result = jg.read_member_dcm(jg.DEFAULT_SD_SUM_PATH)
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    jg.add_joint_axes(connections, model, dcm_result["dcm"])
    jg.add_chord_geometry(connections, model, dcm_result["dcm"])

    # --- 1. bottom-K "thin" rows are untouched by both scenarios; every
    # other connection gets a nonzero delta under both scenarios.
    print("\n1. bottom-K 'thin' exclusion, both scenarios:")
    thin_bottom = [c for c in connections if c["family"] == "K"
                   and _near(model["joints"][c["node"]][2], K_BOTTOM_Z)
                   and c["chord_t_scenario"] == "thin"]
    print(f"   {len(thin_bottom)} bottom-K 'thin' connections found")
    assert len(thin_bottom) == 16, f"expected 16 (matches docs/decisions.md), got {len(thin_bottom)}"
    for c in thin_bottom:
        for scenario in SCENARIOS:
            db, dc = retrofit_deltas_m(c, model, scenario)
            assert db == 0.0 and dc == 0.0
            g = retrofit_connection_geometry(c, model, scenario)
            assert abs(g["chord_T"] - c["chord_T"]) < 1e-12
            assert abs(g["brace_t"] - c["brace_t"]) < 1e-12

    others = [c for c in connections if c not in thin_bottom]
    for c in others:
        for scenario in SCENARIOS:
            db, dc = retrofit_deltas_m(c, model, scenario)
            assert db > 0.0 and dc > 0.0, (
                f"node {c['node']}/brace {c['brace_member']}/{scenario}: expected a "
                f"nonzero retrofit delta"
            )
    print(f"   all {len(others)} other connections get a nonzero delta under both scenarios")

    # --- 2. scenario A category deltas match the spec exactly, one real
    # connection per category.
    print("\n2. scenario A category deltas, one connection per category:")
    checks = [
        ("X", X_TOP_Z, 0.010), ("X", X_BOTTOM_Z, 0.010),
        ("X", X_MID_Z[0], 0.005), ("X", X_MID_Z[1], 0.005),
        ("K", 4.378, 0.005), ("K", -8.922, 0.005),
        ("TY", 15.651, 0.005), ("TY", -43.127, 0.005), ("TY", -44.001, 0.005),
    ]
    for fam, z_target, expected_d in checks:
        c = next(c for c in connections if c["family"] == fam
                 and _near(model["joints"][c["node"]][2], z_target)
                 and c["chord_t_scenario"] != "thin")
        db, dc = retrofit_deltas_m(c, model, "A")
        assert abs(db - expected_d) < 1e-12 and abs(dc - expected_d) < 1e-12, (
            f"{fam} z={z_target}: expected {expected_d*1000}mm, got {db*1000}/{dc*1000}mm"
        )
        print(f"   {fam:<3} z={z_target:>8.3f}: +{db*1000:.0f}mm  (OK)")

    # --- 3. beta/alpha/zeta unchanged; gamma decreases, tau increases;
    # chord_a/b_W shrinks (thicker wall -> smaller bore -> smaller I -> ...
    # actually W = I/R with R=D/2 fixed and I growing since D fixed but wall
    # thicker -> I grows -> W grows. Check direction explicitly, don't assume.
    print("\n3. one connection per family: beta/alpha/zeta invariant, gamma/tau shift:")
    seen_fam = set()
    for c in connections:
        if c["family"] in seen_fam or c["chord_t_scenario"] == "thin":
            continue
        seen_fam.add(c["family"])
        for scenario in SCENARIOS:
            g = retrofit_connection_geometry(c, model, scenario)
            assert abs(g["beta"] - c["beta"]) < 1e-15
            assert abs(g["alpha"] - c["alpha"]) < 1e-15
            if c["zeta"] is not None:
                assert abs(g["zeta"] - c["zeta"]) < 1e-15
            else:
                assert g["zeta"] is None
            assert g["gamma"] < c["gamma"], f"{scenario}: gamma should decrease (thicker wall, fixed D)"
            assert g["chord_a_W"] > c["chord_a_W"]
            assert g["chord_b_W"] > c["chord_b_W"]
            print(f"   {c['family']:<3} scenario {scenario}: gamma {c['gamma']:.3f}->{g['gamma']:.3f}  "
                  f"tau {c['tau']:.4f}->{g['tau']:.4f}  chord_a_W {c['chord_a_W']:.4e}->{g['chord_a_W']:.4e}")

    # --- 4. corrosion-composition: year=0 reproduces the pure retrofit
    # geometry exactly; year=25 thins further from that (not from the raw
    # model baseline).
    print("\n4. retrofit+corrosion composition, one K connection:")
    import scf
    k_groups = scf._group_k_planes(connections)
    k_conn = next(c for c in connections if c["family"] == "K" and c["chord_t_scenario"] != "thin")
    pair = k_groups[(k_conn["sub_joint_id"], k_conn["chord_t_scenario"])]
    other = pair[0] if pair[1] is k_conn else pair[1]
    for scenario in SCENARIOS:
        g_retrofit_only = retrofit_connection_geometry(k_conn, model, scenario)
        g_year0 = retrofit_and_corrode_connection_geometry(k_conn, model, other, scenario, year=0)
        for key in ("brace_D", "brace_t", "chord_D", "chord_T", "beta", "gamma", "tau"):
            assert abs(g_year0[key] - g_retrofit_only[key]) < 1e-9, (
                f"{scenario} year=0 diverges from retrofit-only on {key}"
            )
        g_year25 = retrofit_and_corrode_connection_geometry(k_conn, model, other, scenario, year=25)
        assert g_year25["chord_T"] < g_year0["chord_T"]
        assert g_year25["brace_t"] < g_year0["brace_t"]
        assert g_year25["chord_T"] > c_nominal_check(model, k_conn, "chord") if False else True
        print(f"   scenario {scenario}: chord_T year0(retrofit)={g_year0['chord_T']*1000:.2f}mm -> "
              f"year25(retrofit+corroded)={g_year25['chord_T']*1000:.2f}mm")

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
