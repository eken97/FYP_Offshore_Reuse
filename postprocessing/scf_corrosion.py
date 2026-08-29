"""
Joint-track corrosion, Step J2 -- SCF sets at a corrosion year.

NEW file, not an edit to scf.py (see joint_geometry_corrosion.py's own
docstring for the code-freeze reasoning -- same applies here: scf.py sits in
the running pipeline's live import graph via stage2_joints.py). This module only IMPORTS
scf.py (read-only) and calls its already-G2-signed-off pure functions
(scf_TY/scf_X/scf_K_pair) with corroded geometry instead of the year-0
values -- exactly the "corrosion-forward" design scf.py's own module
docstring says it was built for ("every function takes its geometry as
ARGUMENTS... so the same code is callable again per corrosion-year-step with
thinned D/T/t without redesign"). No new SCF formula exists here; this
module only supplies different inputs to the existing ones.

Design mirrors scf.compute_all_scf() row-for-row (same fields, same
family/treatment dispatch, same K-plane self/other pairing via
scf._group_k_planes -- a pure identity/topology grouping, unaffected by
corrosion, so reused unchanged), with one addition: every row also carries
`year` and the CORRODED beta/gamma/tau/alpha/zeta/brace_D/brace_t/chord_D/
chord_T actually used for that row, so a year-0 call is directly comparable
to (and, by the self-check below, numerically identical to) scf.py's own
compute_all_scf() output.
"""
from pathlib import Path

import numpy as np

import scf
import joint_geometry_corrosion as jgc
import sd_geometry as sdg
import joint_geometry as jg

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# endregion


def compute_scf_corroded(connections, model, year):
    """
    Direct analogue of scf.compute_all_scf(), with every connection's
    geometry parameters recomputed at corrosion `year` via
    joint_geometry_corrosion.corroded_connection_geometry() first. Returns
    the same row shape as compute_all_scf() (see that function's own
    docstring for the field list) plus `year`.

    `connections` may be the full 120-row list or an already-splash-filtered
    subset (the caller's choice, e.g. stage2_joints_corrosion.py -- this
    module has no zone knowledge of its own). K-plane pairing is looked up
    from `connections` itself via scf._group_k_planes(), so if a caller
    passes a splash-filtered subset it MUST include both braces of every K
    plane it wants K/Y rows for (true for the real jacket's splash scope --
    every K-family connection's plane-partner is at the same node, same
    zone, so filtering by zone never splits a K pair).
    """
    k_groups = scf._group_k_planes(connections)
    rows = []
    for c in connections:
        if c["family"] == "K":
            key = (c["sub_joint_id"], c["chord_t_scenario"])
            pair = k_groups[key]
            other = pair[0] if pair[1] is c else pair[1]
            other_g = jgc.corroded_connection_geometry(other, model, c, year)
            other_row = dict(other, **other_g)
        else:
            other = None
            other_row = None

        g = jgc.corroded_connection_geometry(c, model, other, year)
        self_row = dict(c, **g)

        if c["family"] == "TY":
            treatments = [("TY", scf.scf_TY(g["beta"], g["gamma"], g["tau"], c["theta_deg"], g["alpha"]))]
        elif c["family"] == "X":
            treatments = [("X", scf.scf_X(g["beta"], g["gamma"], g["tau"], c["theta_deg"], g["alpha"]))]
        elif c["family"] == "K":
            treatments = [
                ("K", scf.scf_K_pair(self_row, other_row)),
                ("Y", scf.scf_TY(g["beta"], g["gamma"], g["tau"], c["theta_deg"], g["alpha"])),
            ]
        else:
            raise ValueError(f"unknown family {c['family']!r}")

        for treatment, result in treatments:
            for side in ("chord", "brace"):
                sr = result[side]
                static_scf_max = max(sr["AS"], sr["MIP"], sr["MOP"])
                rows.append(dict(
                    year=year,
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
                ))
    return rows


def _row_key(r):
    """Same identity scf.compute_all_scf's rows are naturally keyed by --
    matches stage2_joints._connection_key + (treatment, side)."""
    return (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
            r["chord_t_scenario"], r["direction"], r["treatment"], r["side"])


def _self_check():
    print(f"SCF_EQUATIONS_VERIFIED = {scf.SCF_EQUATIONS_VERIFIED}")
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    print(f"Loaded {len(connections)} connections")

    # --- year=0 must reproduce scf.compute_all_scf()'s own output exactly
    # -- the decisive check that feeding corroded (=uncorroded at year 0)
    # geometry through this module's dispatch matches the already
    # G2-signed-off original row-for-row.
    print("\n1. year=0 reproduces scf.compute_all_scf() exactly:")
    ref_rows = scf.compute_all_scf(connections)
    corr_rows = compute_scf_corroded(connections, model, year=0)
    print(f"   reference: {len(ref_rows)} rows, corroded(year=0): {len(corr_rows)} rows")
    assert len(ref_rows) == len(corr_rows) == 368

    ref_by_key = {_row_key(r): r for r in ref_rows}
    assert len(ref_by_key) == len(ref_rows), "reference rows are not uniquely keyed"

    numeric_fields = ("beta", "gamma", "tau", "alpha", "SCF_AC_base", "SCF_AC_att",
                       "SCF_AS", "SCF_MIP", "SCF_MOP", "static_scf_max")
    max_err = {f: 0.0 for f in numeric_fields}
    zeta_checked = 0
    for r in corr_rows:
        ref = ref_by_key[_row_key(r)]
        for f in numeric_fields:
            max_err[f] = max(max_err[f], abs(r[f] - ref[f]))
        if r["zeta"] != "":
            assert ref["zeta"] != ""
            max_err.setdefault("zeta", 0.0)
            max_err["zeta"] = max(max_err["zeta"], abs(r["zeta"] - ref["zeta"]))
            zeta_checked += 1
        else:
            assert ref["zeta"] == ""
    print(f"   max |year0 - reference| per field: {max_err}")
    print(f"   zeta matched on {zeta_checked} rows")
    for f, err in max_err.items():
        assert err < 1e-9, f"{f}: year=0 diverges from scf.compute_all_scf() ({err:.3e})"

    # --- corrosion measurably changes the crown/saddle SCFs by year 25, on
    # a real K connection (both AC_base and AS should move -- gamma/tau/beta
    # all shift).
    print("\n2. one real K connection, chord side, K-treatment: SCF vs year:")
    k_conn = next(c for c in connections if c["family"] == "K")
    rows_by_year = {}
    for year in (0, 5, 10, 15, 20, 25):
        rows = compute_scf_corroded([c for c in connections
                                      if c["sub_joint_id"] == k_conn["sub_joint_id"]
                                      and c["chord_t_scenario"] == k_conn["chord_t_scenario"]],
                                     model, year)
        r = next(r for r in rows if r["brace_member"] == k_conn["brace_member"]
                 and r["brace_end"] == k_conn["brace_end"]
                 and r["treatment"] == "K" and r["side"] == "chord")
        rows_by_year[year] = r
        print(f"   year {year:>2}: AC_base={r['SCF_AC_base']:.4f}  AS={r['SCF_AS']:.4f}  "
              f"gamma={r['gamma']:.3f}  tau={r['tau']:.4f}  zeta={r['zeta']:.4f}")
    assert rows_by_year[25]["SCF_AC_base"] != rows_by_year[0]["SCF_AC_base"]
    assert rows_by_year[25]["SCF_AS"] != rows_by_year[0]["SCF_AS"]

    # --- positivity/finiteness, same discipline as scf.py's own self-check,
    # re-run at year=25 (a fresh geometry regime, not just year=0).
    print("\n3. positivity/finiteness at year=25, full connection set:")
    rows25 = compute_scf_corroded(connections, model, year=25)
    bad = [r for r in rows25
           for k_ in ("SCF_AC_base", "SCF_AS", "SCF_MIP", "SCF_MOP")
           if not (np.isfinite(r[k_]) and r[k_] > 0)]
    print(f"   non-finite or non-positive static SCF values: {len(bad)} (expect 0)")
    assert len(bad) == 0

    print("\n  all checks passed.")


if __name__ == "__main__":
    _self_check()
