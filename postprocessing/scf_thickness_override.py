"""
Joint-track can-thickness retrofit, SCF layer.

NEW file, mirrors scf_corrosion.py row-for-row (same field shape, same
family/treatment dispatch, same K-plane self/other pairing via
scf._group_k_planes -- a pure identity/topology grouping, unaffected by a
thickness retrofit, so reused unchanged). Only difference from
scf_corrosion.py: geometry comes from joint_thickness_override's retrofit
functions instead of joint_geometry_corrosion's corrosion functions. No new
SCF formula exists here; this module only supplies different inputs to the
already-G2-signed-off scf_TY/scf_X/scf_K_pair.

Two entry points:
  compute_scf_retrofit(connections, model, scenario) -- retrofit only,
    no corrosion (feeds run_stage2_joints_thickness / the uncorroded
    retrofit tracks).
  compute_scf_retrofit_corroded(connections, model, scenario, year) --
    retrofit THEN corrosion (feeds the "corrosion on top of the retrofit"
    tracks; caller scopes `connections` to splash-zone only, same
    convention as scf_corrosion.py/stage2_joints_corrosion.py).
"""
from pathlib import Path

import numpy as np

import scf
import sd_geometry as sdg
import joint_geometry as jg
import joint_thickness_override as jto

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# endregion


def _scf_row(c, g, treatment, year=None):
    """Builds the scf_TY/scf_X/scf_K_pair result dict for one connection at
    already-computed geometry `g`, then flattens into row dicts (one per
    side). Shared by both entry points below."""
    if treatment in ("TY", "Y"):
        result = scf.scf_TY(g["beta"], g["gamma"], g["tau"], c["theta_deg"], g["alpha"])
    elif treatment == "X":
        result = scf.scf_X(g["beta"], g["gamma"], g["tau"], c["theta_deg"], g["alpha"])
    else:
        raise ValueError(f"_scf_row: treatment {treatment!r} needs the K pair path")

    rows = []
    for side in ("chord", "brace"):
        sr = result[side]
        static_scf_max = max(sr["AS"], sr["MIP"], sr["MOP"])
        row = dict(
            node=c["node"], sub_joint_id=c["sub_joint_id"], plane_id=c["plane_id"],
            family=c["family"], treatment=treatment,
            brace_member=c["brace_member"], brace_end=c["brace_end"],
            chord_t_scenario=c["chord_t_scenario"], direction=c["direction"],
            side=side,
            beta=g["beta"], gamma=g["gamma"], tau=g["tau"], theta_deg=c["theta_deg"],
            alpha=g["alpha"], zeta=g["zeta"] if g["zeta"] is not None else "",
            brace_D=g["brace_D"], brace_t=g["brace_t"],
            chord_D=g["chord_D"], chord_T=g["chord_T"],
            SCF_AC_base=sr["AC_base"], SCF_AC_att=sr["AC_att"],
            SCF_AS=sr["AS"], SCF_MIP=sr["MIP"], SCF_MOP=sr["MOP"],
            eqn_AC=sr["eqn_AC"], eqn_AS=sr["eqn_AS"], eqn_MIP=sr["eqn_MIP"], eqn_MOP=sr["eqn_MOP"],
            F_AS=sr["F_AS"] if sr["F_AS"] is not None else "",
            F_MOP=sr["F_MOP"] if sr["F_MOP"] is not None else "",
            static_scf_max=static_scf_max,
        )
        if year is not None:
            row["year"] = year
        rows.append(row)
    return rows


def _k_pair_rows(c, g, other, other_g, year=None):
    self_row = dict(c, **g)
    other_row = dict(other, **other_g)
    result = scf.scf_K_pair(self_row, other_row)
    rows = []
    for side in ("chord", "brace"):
        sr = result[side]
        static_scf_max = max(sr["AS"], sr["MIP"], sr["MOP"])
        row = dict(
            node=c["node"], sub_joint_id=c["sub_joint_id"], plane_id=c["plane_id"],
            family=c["family"], treatment="K",
            brace_member=c["brace_member"], brace_end=c["brace_end"],
            chord_t_scenario=c["chord_t_scenario"], direction=c["direction"],
            side=side,
            beta=g["beta"], gamma=g["gamma"], tau=g["tau"], theta_deg=c["theta_deg"],
            alpha=g["alpha"], zeta=g["zeta"] if g["zeta"] is not None else "",
            brace_D=g["brace_D"], brace_t=g["brace_t"],
            chord_D=g["chord_D"], chord_T=g["chord_T"],
            SCF_AC_base=sr["AC_base"], SCF_AC_att=sr["AC_att"],
            SCF_AS=sr["AS"], SCF_MIP=sr["MIP"], SCF_MOP=sr["MOP"],
            eqn_AC=sr["eqn_AC"], eqn_AS=sr["eqn_AS"], eqn_MIP=sr["eqn_MIP"], eqn_MOP=sr["eqn_MOP"],
            F_AS=sr["F_AS"] if sr["F_AS"] is not None else "",
            F_MOP=sr["F_MOP"] if sr["F_MOP"] is not None else "",
            static_scf_max=static_scf_max,
        )
        if year is not None:
            row["year"] = year
        rows.append(row)
    return rows


def compute_scf_retrofit(connections, model, scenario):
    """Direct analogue of scf.compute_all_scf(), with every connection's
    geometry recomputed under the can-thickness `scenario` ("A"/"B") via
    joint_thickness_override.retrofit_connection_geometry() first."""
    k_groups = scf._group_k_planes(connections)
    rows = []
    for c in connections:
        g = jto.retrofit_connection_geometry(c, model, scenario)
        if c["family"] == "TY":
            rows += _scf_row(c, g, "TY")
        elif c["family"] == "X":
            rows += _scf_row(c, g, "X")
        elif c["family"] == "K":
            key = (c["sub_joint_id"], c["chord_t_scenario"])
            pair = k_groups[key]
            other = pair[0] if pair[1] is c else pair[1]
            other_g = jto.retrofit_connection_geometry(other, model, scenario)
            rows += _k_pair_rows(c, g, other, other_g)
            rows += _scf_row(c, g, "Y")
        else:
            raise ValueError(f"unknown family {c['family']!r}")
    return rows


def compute_scf_retrofit_corroded(connections, model, scenario, year):
    """Direct analogue of scf_corrosion.compute_scf_corroded(), but the
    corrosion year-loop starts from the retrofitted baseline (via
    joint_thickness_override.retrofit_and_corrode_connection_geometry)
    instead of the model's raw nominal geometry. `connections` should be
    the splash-zone-filtered subset -- same convention as
    scf_corrosion.py (K-plane pairing needs both braces of a plane present,
    true for this jacket's splash scope, see that module's own docstring)."""
    k_groups = scf._group_k_planes(connections)
    rows = []
    for c in connections:
        if c["family"] == "K":
            key = (c["sub_joint_id"], c["chord_t_scenario"])
            pair = k_groups[key]
            other = pair[0] if pair[1] is c else pair[1]
            other_g = jto.retrofit_and_corrode_connection_geometry(other, model, c, scenario, year)
        else:
            other = None
            other_g = None
        g = jto.retrofit_and_corrode_connection_geometry(c, model, other, scenario, year)

        if c["family"] == "TY":
            rows += _scf_row(c, g, "TY", year=year)
        elif c["family"] == "X":
            rows += _scf_row(c, g, "X", year=year)
        elif c["family"] == "K":
            rows += _k_pair_rows(c, g, other, other_g, year=year)
            rows += _scf_row(c, g, "Y", year=year)
        else:
            raise ValueError(f"unknown family {c['family']!r}")
    return rows


def _row_key(r):
    return (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
            r["chord_t_scenario"], r["direction"], r["treatment"], r["side"])


def _self_check():
    print(f"SCF_EQUATIONS_VERIFIED = {scf.SCF_EQUATIONS_VERIFIED}")
    dcm_result = jg.read_member_dcm(jg.DEFAULT_SD_SUM_PATH)
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    jg.add_joint_axes(connections, model, dcm_result["dcm"])
    jg.add_chord_geometry(connections, model, dcm_result["dcm"])
    print(f"Loaded {len(connections)} connections")

    ref_rows = scf.compute_all_scf(connections)
    ref_by_key = {_row_key(r): r for r in ref_rows}
    assert len(ref_by_key) == len(ref_rows) == 368

    numeric_fields = ("SCF_AC_base", "SCF_AC_att", "SCF_AS", "SCF_MIP", "SCF_MOP", "static_scf_max")
    # SCF_AC_att is legitimately 0.0 for the X family (no chord-bending
    # attachment term -- see stage2_joints.py's own docstring), so it's
    # excluded from the must-be-positive check, matching scf_corrosion.py's
    # own self-check convention exactly.
    positivity_fields = ("SCF_AC_base", "SCF_AS", "SCF_MIP", "SCF_MOP")

    # --- 1. retrofitted SCFs are all finite/positive, both scenarios, full
    # connection set -- same discipline as scf.py's own self-check.
    print("\n1. finiteness/positivity, full connection set, both scenarios:")
    for scenario in jto.SCENARIOS:
        rows = compute_scf_retrofit(connections, model, scenario)
        assert len(rows) == 368, f"scenario {scenario}: expected 368 rows, got {len(rows)}"
        bad = [r for r in rows for k_ in positivity_fields
               if not (np.isfinite(r[k_]) and r[k_] > 0)]
        print(f"   scenario {scenario}: {len(rows)} rows, {len(bad)} non-finite/non-positive (expect 0)")
        assert len(bad) == 0

    # --- 2. bottom-K 'thin' rows: SCF identical to the unretrofitted
    # reference (geometry untouched there, so SCF must match exactly too).
    print("\n2. bottom-K 'thin' rows: SCF unchanged from reference, both scenarios:")
    thin_bottom_keys = {
        _row_key(r) for r in ref_rows
        if r["family"] == "K" and r["chord_t_scenario"] == "thin"
        and abs(model["joints"][r["node"]][2] - jto.K_BOTTOM_Z) < jto.Z_TOL_M
    }
    print(f"   {len(thin_bottom_keys)} reference rows at bottom-K/thin")
    assert len(thin_bottom_keys) > 0
    for scenario in jto.SCENARIOS:
        rows = compute_scf_retrofit(connections, model, scenario)
        by_key = {_row_key(r): r for r in rows}
        checked = 0
        for key in thin_bottom_keys:
            ref, got = ref_by_key[key], by_key[key]
            for f in numeric_fields:
                assert abs(ref[f] - got[f]) < 1e-9, f"{scenario} {key}: {f} changed despite 'thin' exclusion"
            checked += 1
        print(f"   scenario {scenario}: {checked} rows confirmed unchanged")

    # --- 3. real X connection: static_scf_max drops under both scenarios
    # (thicker can -> lower gamma -> lower SCF), scenario B (+12mm) drops
    # more than scenario A's mid-level +5mm but the SAME connection's A
    # value should match the X top/bottom +10mm band, not the mid +5mm one,
    # since X top/bottom is the more aggressive category in A.
    print("\n3. real X connection (chord-saddle), SCF drop under retrofit:")
    x_conn = next(c for c in connections if c["family"] == "X"
                  and abs(model["joints"][c["node"]][2] - jto.X_BOTTOM_Z) < jto.Z_TOL_M
                  and c["direction"] == "A_as_chord")
    ref_key = (x_conn["node"], x_conn["sub_joint_id"], x_conn["brace_member"], x_conn["brace_end"],
               x_conn["chord_t_scenario"], x_conn["direction"], "X", "chord")
    ref_scf = ref_by_key[ref_key]["SCF_AS"]
    print(f"   reference (no retrofit) SCF_AS = {ref_scf:.4f}")
    prev = ref_scf
    for scenario in jto.SCENARIOS:
        rows = compute_scf_retrofit(connections, model, scenario)
        by_key = {_row_key(r): r for r in rows}
        got_scf = by_key[ref_key]["SCF_AS"]
        print(f"   scenario {scenario}: SCF_AS = {got_scf:.4f}")
        assert got_scf < ref_scf, f"scenario {scenario}: SCF_AS did not drop under retrofit"

    # --- 4. corrosion-on-top: year=0 matches the pure-retrofit SCF exactly;
    # year=25 drifts from it (splash-scope only, use a real splash K conn).
    print("\n4. retrofit+corrosion composition, one splash K connection:")
    import stage2_joints_corrosion as s2jc
    splash = s2jc.splash_connections(connections, model)
    k_conn = next(c for c in splash if c["family"] == "K")
    for scenario in jto.SCENARIOS:
        rows_y0 = compute_scf_retrofit_corroded(splash, model, scenario, year=0)
        rows_retrofit = compute_scf_retrofit(splash, model, scenario)
        by_key_y0 = {_row_key(r): r for r in rows_y0}
        by_key_retrofit = {_row_key(r): r for r in rows_retrofit}
        assert len(rows_y0) == len(rows_retrofit)
        max_err = 0.0
        for key in by_key_retrofit:
            for f in numeric_fields:
                max_err = max(max_err, abs(by_key_retrofit[key][f] - by_key_y0[key][f]))
        print(f"   scenario {scenario}: max |retrofit-only - retrofit+corrosion(year=0)| = {max_err:.3e}")
        assert max_err < 1e-9

        rows_y25 = compute_scf_retrofit_corroded(splash, model, scenario, year=25)
        by_key_y25 = {_row_key(r): r for r in rows_y25}
        key0 = next(k for k in by_key_y0 if k[0] == k_conn["node"] and k[6] == "K" and k[7] == "chord")
        v0, v25 = by_key_y0[key0]["static_scf_max"], by_key_y25[key0]["static_scf_max"]
        print(f"   scenario {scenario}: node {k_conn['node']} K-chord static_scf_max year0={v0:.4f} -> year25={v25:.4f}")
        assert v25 != v0

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
