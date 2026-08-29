"""
Joint track, Step 1 -- direction cosine matrix (DCM) reader.

See docs/methodology.md
and the docs/decisions.md for the full derivation. This module holds
everything GEOMETRIC for the joint track: DCM parsing (this step), member
connectivity + per-plane family classification (Step 2), DNV Appendix B
geometry parameters beta/gamma/tau/theta/zeta/alpha (Step 3), and the
IPB/OPB axis rotation (Step 4). Imports sd_geometry.py but does not edit
it -- that module is signed off and drives every already-produced member
number.

WHY THIS STEP EXISTS: DNV's hot-spot superposition (3.3.1) needs bending
split into in-plane (IPB, in the brace-chord plane) and out-of-plane (OPB,
perpendicular to it) -- a physically meaningful split that has nothing to
do with SubDyn's own MKxe/MKye member-local axes, which point wherever
SubDyn's internal axis-selection rule happens to put them. Reconstructing
that rule from documentation risks a silently swapped SCF_MIP/SCF_MOP (a
~2-3x vs ~6-8x mixup) with no downstream symptom -- a plausible-looking
wrong number, same failure class the whole pipeline's sign-off discipline
exists to catch.

Resolution: OpenFAST already computed and printed this rotation. Every run's
<case>.SD.sum.yaml (OutCOSM=False in SubDyn.dat, so this is NOT in the .outb
-- the summary file is the only source) contains, from a fixed heading,
"#Direction Cosine Matrices for all Members: GLOBAL-2-LOCAL", a 3x3 GLOBAL-
to-LOCAL rotation per member (all 112). Row 3 (DC(3,1..3)) is the member's
local z axis (the member axis) expressed in global coordinates; rows 1/2
are the exact local x/y that MKxe/MKye act about. So the IPB/OPB rotation
angle phi is MEASURED from OpenFAST's own output, not inferred.

The block is '#'-commented plain text laid out as one column-header line
then exactly NMembers data lines -- not valid YAML (comments aren't parsed
as data by any YAML loader), so this is a small dedicated text parser, not
a yaml.safe_load() call.
"""
import csv
import hashlib
import math
import re
from pathlib import Path

import numpy as np

import sd_geometry as sdg

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
DEV_FIXTURE_DIR = PROJECT / "data" / "example"
SD_SUM_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.SD.sum.yaml"
DEFAULT_SD_SUM_PATH = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001" / SD_SUM_NAME
# endregion


# region --- DCM parser ---
_DCM_HEADER_RE = re.compile(
    r"^#Direction Cosine Matrices for all Members:\s*GLOBAL-2-LOCAL\.\s*"
    r"No\.\s*of 3x3 matrices=\s*(\d+)"
)
# Data line: '#' then MemberID then 9 floats (row-major DC(1,1)..DC(3,3)).
# Fortran-style summary output, so accept both '1.0E+00' and '1.0D+00' just
# in case -- not seen in practice on this project's files, but cheap to
# allow rather than fail loud on a compiler-dependent exponent letter.
_DCM_ROW_RE = re.compile(
    r"^#\s*(\d+)((?:\s+[+-]?\d+\.\d+[EDed][+-]\d+){9})\s*$"
)


def read_member_dcm(sd_sum_path=DEFAULT_SD_SUM_PATH):
    """
    Parse the '#Direction Cosine Matrices for all Members: GLOBAL-2-LOCAL'
    block out of a SubDyn summary yaml.

    Returns dict:
        md5           -- of the summary file, for provenance stamping
                         (same role as sd_geometry's SubDyn.dat md5)
        n_members     -- count declared in the block's own header
        dcm           -- {member_id: 3x3 np.ndarray}, GLOBAL-to-LOCAL
                         rotation, row-major as printed: row 0 = local x,
                         row 1 = local y, row 2 = local z (member axis),
                         each expressed in global coordinates.
    """
    sd_sum_path = Path(sd_sum_path)
    raw_bytes = sd_sum_path.read_bytes()
    md5 = hashlib.md5(raw_bytes).hexdigest()
    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()

    header_idx = None
    n_members = None
    for i, line in enumerate(lines):
        m = _DCM_HEADER_RE.match(line)
        if m:
            header_idx = i
            n_members = int(m.group(1))
            break
    if header_idx is None:
        raise ValueError(
            f"{sd_sum_path}: 'Direction Cosine Matrices for all Members' "
            f"header not found"
        )

    # header_idx -> block title, header_idx+1 -> column names, data starts
    # at header_idx+2.
    dcm = {}
    for k in range(n_members):
        line = lines[header_idx + 2 + k]
        m = _DCM_ROW_RE.match(line)
        if not m:
            raise ValueError(
                f"{sd_sum_path}: line {header_idx + 2 + k + 1} did not match "
                f"the expected DCM row format: {line!r}"
            )
        mid = int(m.group(1))
        vals = [float(x.replace("D", "E").replace("d", "e"))
                for x in m.group(2).split()]
        assert len(vals) == 9, f"member {mid}: expected 9 DCM values, got {len(vals)}"
        dcm[mid] = np.array(vals, dtype=np.float64).reshape(3, 3)

    assert len(dcm) == n_members, (
        f"{sd_sum_path}: block header declared {n_members} matrices, parsed {len(dcm)}"
    )
    return dict(md5=md5, n_members=n_members, dcm=dcm)
# endregion


def _self_check_dcm():
    print(f"Parsing: {DEFAULT_SD_SUM_PATH}")
    result = read_member_dcm(DEFAULT_SD_SUM_PATH)
    dcm = result["dcm"]
    print(f"  md5 = {result['md5']}")
    print(f"  n_members declared in block header = {result['n_members']}")
    print(f"  parsed {len(dcm)} matrices")
    assert result["n_members"] == 112
    assert sorted(dcm) == list(range(1, 113))

    # --- Check 1: orthonormality. Every DCM must satisfy R . R^T = I to
    # near machine precision -- a real rotation matrix, not something
    # transcribed/parsed wrong.
    max_orthonormality_err = 0.0
    for mid, R in dcm.items():
        err = np.max(np.abs(R @ R.T - np.eye(3)))
        max_orthonormality_err = max(max_orthonormality_err, err)
    print(f"\n  max |R.R^T - I| over all 112 members: {max_orthonormality_err:.2e}")
    assert max_orthonormality_err < 1e-9, "a DCM is not orthonormal -- parse error"

    # --- Check 2: row 3 (member axis, local z) must be parallel to the
    # member's own J1->J2 direction, computed independently from the joint
    # coordinates in SubDyn.dat -- confirms row 3 really is the member axis
    # and not, say, row 1 (which would silently corrupt every downstream
    # IPB/OPB rotation).
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    max_axis_err = 0.0
    sign_flips = 0
    for mid, m in model["members"].items():
        j1 = np.array(model["joints"][m["j1"]])
        j2 = np.array(model["joints"][m["j2"]])
        axis_from_geometry = (j2 - j1) / np.linalg.norm(j2 - j1)
        axis_from_dcm = dcm[mid][2, :]   # row 3 (0-indexed row 2)
        # DCM row could point J1->J2 or J2->J1 depending on SubDyn's own
        # element-orientation convention -- check parallel, not identical,
        # and track which sign shows up (should be consistent, not mixed).
        dot = np.dot(axis_from_geometry, axis_from_dcm)
        if dot < 0:
            sign_flips += 1
        err = 1.0 - abs(dot)   # 0 if perfectly (anti)parallel
        max_axis_err = max(max_axis_err, err)
    print(f"  max axis-parallel error (1 - |cos(angle)|) over all 112 members: "
          f"{max_axis_err:.2e}")
    print(f"  members where DCM row3 points J2->J1 (opposite geometric J1->J2): "
          f"{sign_flips} / 112")
    assert max_axis_err < 1e-9, (
        "DCM row 3 is not parallel to the member's own geometric axis -- "
        "wrong row being read as the member axis, or a parsing error"
    )
    assert sign_flips in (0, 112), (
        f"sign convention is MIXED across members ({sign_flips}/112 flipped) -- "
        f"expected either all-same or all-opposite, a mix suggests a real bug, "
        f"not just a global sign convention"
    )

    m1 = model["members"][1]
    j1_xyz, j2_xyz = model["joints"][m1["j1"]], model["joints"][m1["j2"]]
    print(f"\n  member 1 DCM (leg segment {j1_xyz} -> {j2_xyz}, vertical -- "
          f"identity is a sane convention for a member along global Z):")
    print(dcm[1])
    print(f"\n  member 2 DCM (a real diagonal brace):")
    print(dcm[2])

    print("\n  all checks passed.")
# endregion


# region --- Step 2: connectivity + plane split + family classification ---
# A "welded node" is any SubDyn joint where members actually weld together --
# NOT every node in the model. sd_geometry's 64 nodes/112 members include 24
# degree-1/2 nodes that are just pile/TP interface stubs or straight-through
# node splits (no bracing), which carry no joint fatigue check of their own
# (the member track already covers the member length they sit on -- see
# docs/decisions.md). Welded nodes are degree 4 or 6 ONLY, split
# by member diameter into two structurally different situations:
#   - a LEG present (D >= D_LEG_MIN): the leg continues straight through the
#     node, braces frame into it from one or more azimuths -- K (2 braces
#     sharing an azimuth/"plane") or T/Y (1 brace in its own plane).
#   - NO leg (every member is brace-diameter): an X-brace crossing, where two
#     continuous braces cross -- assessed in BOTH directions (each as the
#     other's "chord"), per the 12.08.2026 decision.
# This threshold-based test (not a hard-coded node list) is what makes the
# classification a MEASUREMENT of the model rather than an assumption --
# see docs/decisions.md for why that mattered (it already caught one
# wrong assumption this session: the top/mudbrace nodes were assumed K,
# turned out to be Y/T).
D_LEG_MIN = 1.0   # m -- clean gap between the D=0.8 brace propset and every
                  # leg propset (D=1.2/2.082); see sd_geometry's propset table.

# Two braces are "in the same plane" (same jacket face) if their azimuths,
# measured perpendicular to the chord axis, agree within this tolerance.
#
# DNV-RP-C203, Appendix B (joint classification): "Joint classification is
# the process whereby the axial force in a given brace is subdivided into K,
# X and Y components of actions corresponding to the three joint types for
# which stress concentration equations exist. Such subdivision normally
# considers all of the members in one plane at a joint. For purposes of this
# provision, brace planes within +-15 deg of each other may be considered as
# being in a common plane." -- independently re-checked primary-source quote.
#
# Note the DIFFERENT purpose: DNV's own use of this +-15 deg test is for the
# K/X/Y axial-force LOAD-PATH classification (splitting one brace's force
# into K/X/Y fractions for a weighted SCF) -- the algorithm
# deliberately not built here, in favour of the X/real-type
# bounding pair (see docs/decisions.md). This module reuses the same
# "common plane" geometric definition for a different step: deciding how
# many uniplanar sub-joints exist at a multi-brace node, so each plane gets
# its own Appendix B calculation (the already-agreed multiplanar
# split-and-superpose method). Same geometric test, different downstream
# use -- not a reintroduction of the load-path classification.
#
# Confirms the earlier finding on this jacket (within-plane pairs agree to
# <0.1 deg, between-plane pairs differ by ~90 deg) sits nowhere near this
# boundary either way.
PLANE_AZIMUTH_TOL_DEG = 15.0

# Purely a REPORTING label (T vs Y), not a different equation set -- DNV's
# Table B-1 covers T/Y with ONE equation family, theta entering continuously
# (see docs/decisions.md). Every SCF formula, every validity check, and
# every damage number is IDENTICAL either way -- this constant only decides
# which word appears in one output column.
#
# NOT a DNV value -- DNV never draws this line because it never needs to.
# "T" (brace perpendicular to chord) vs "Y" (brace oblique) is informal
# engineering shorthand, and 75 deg is an arbitrary split I chose with no
# source behind it, picked because it happens to fall cleanly between this
# jacket's two real theta clusters (88.1 deg at the lower mudbrace level vs
# 29.5-38.6 deg everywhere else) -- ANY threshold from about 50 to 85 deg
# would label this jacket identically. Do not treat 75.0 as verified; if a
# future jacket has theta values near this boundary, revisit it deliberately
# rather than trusting the current default.
THETA_T_LABEL_MIN_DEG = 75.0


def _node_incidence(model):
    """{node_id: [(member_id, end), ...]} -- end 1 if the member's J1 is this
    node, end 2 if its J2 is. Every member contributes exactly 2 entries
    (one per end) across the whole dict."""
    inc = {}
    for mid, m in model["members"].items():
        inc.setdefault(m["j1"], []).append((mid, 1))
        inc.setdefault(m["j2"], []).append((mid, 2))
    return inc


def _member_axis_away_from_node(model, mid, end):
    """Unit vector of member `mid`, pointing AWAY from the node at `end`
    (end 1 -> J1->J2 direction; end 2 -> J2->J1 direction) -- i.e. the
    direction the member actually extends from this joint into the rest of
    the structure. Computed from SubDyn.dat joint coordinates directly
    (equivalent to DCM row 3 per Step 1's own regression, but self-contained
    here so Step 2 doesn't require the DCM to already be loaded)."""
    m = model["members"][mid]
    j1 = np.array(model["joints"][m["j1"]])
    j2 = np.array(model["joints"][m["j2"]])
    v = (j2 - j1) if end == 1 else (j1 - j2)
    return v / np.linalg.norm(v)


def _cluster_by_azimuth(items, tol_deg=PLANE_AZIMUTH_TOL_DEG):
    """
    items: [(key, azimuth_deg), ...]. Groups keys whose azimuths agree within
    tol_deg (circular, wraps at 360) via union-find -- robust to whatever
    order items arrive in, unlike a sort-and-walk approach. Returns a list of
    key-lists, one per cluster ("plane").
    """
    n = len(items)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    def angdiff(a, b):
        d = abs(a - b) % 360.0
        return min(d, 360.0 - d)

    for i in range(n):
        for j in range(i + 1, n):
            if angdiff(items[i][1], items[j][1]) < tol_deg:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i][0])
    return list(groups.values())


def _theta_and_azimuth(model, brace_mid, brace_end, chord_axis):
    """
    theta_deg: acute angle (0-90) between the brace (pointing away from the
    node) and the chord axis -- abs() on the dot product folds any obtuse
    result into the same 0-90 range DNV's own validity table expects, so
    this never needs a separate "which side" branch.
    azimuth_deg: angle of the brace's component PERPENDICULAR to the chord
    axis, about that axis -- the value _cluster_by_azimuth groups on. Uses
    global X/Y as the reference in-plane basis; consistent for every brace
    at a given node since they all share the same chord_axis, so relative
    groupings are meaningful even though the absolute angle has no special
    physical meaning on its own.
    """
    d = _member_axis_away_from_node(model, brace_mid, brace_end)
    chord_axis = chord_axis / np.linalg.norm(chord_axis)
    theta_deg = float(np.degrees(np.arccos(np.clip(abs(np.dot(d, chord_axis)), -1.0, 1.0))))

    # Perpendicular component, then its angle about chord_axis using two
    # basis vectors spanning the plane normal to it.
    perp = d - np.dot(d, chord_axis) * chord_axis
    ref = np.array([1.0, 0.0, 0.0]) if abs(chord_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(chord_axis, ref)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(chord_axis, e1)
    azimuth_deg = float(np.degrees(np.arctan2(np.dot(perp, e2), np.dot(perp, e1))))
    return theta_deg, azimuth_deg


def build_connections(model):
    """
    Every brace-to-chord connection needing its own DNV SCF check -- 120 of
    them (104 physically distinct brace-to-chord relationships, +16 extra
    rows at the 4 chord_T-ambiguous nodes -- see below). One row per
    connection:

        node, sub_joint_id, plane_id, family ("K"/"TY"/"X"), type_label
        (reporting only: "K"/"T"/"Y"/"X"), n_braces_in_plane,
        chord_members (list of (member_id, end)), chord_D, chord_T,
        chord_T_ambiguous (bool -- see below), chord_t_scenario
        ("single"/"thick"/"thin" -- see below), chord_T_upper, chord_T_lower,
        brace_member, brace_end, brace_D, brace_t,
        theta_deg, azimuth_deg, direction (X-crossings only: "A_as_chord" /
        "B_as_chord")

    sub_joint_id is (node, plane_index) for K/T/Y (one per plane -- the
    per-plane split IS the sub-joint, per the adopted multiplanar method),
    or just (node,) for X (one physical crossing, checked from both
    directions but a single sub-joint).

    chord_t_scenario: at the 4 nodes where the two leg segments meeting at
    a node have genuinely different wall thickness (the z=-24.61 KK level),
    there is no single correct chord T -- every brace at that node gets TWO
    rows instead of one, "thick" (chord_T = the thicker leg segment) and
    "thin" (the thinner), so downstream stages can report the resulting
    damage/life spread rather than silently picking one. Everywhere else
    chord_t_scenario is "single" and chord_T is unambiguous.
    """
    inc = _node_incidence(model)
    connections = []

    for node, ends in inc.items():
        if len(ends) not in (4, 6):
            continue   # not a welded joint -- degree-1/2 stub or split node

        legs = [(mid, e) for mid, e in ends if _member_diameter(model, mid) >= D_LEG_MIN]
        braces = [(mid, e) for mid, e in ends if _member_diameter(model, mid) < D_LEG_MIN]

        if len(legs) == 2 and len(braces) in (2, 4):
            connections.extend(_leg_node_connections(model, node, legs, braces))
        elif len(legs) == 0 and len(braces) == 4:
            connections.extend(_x_crossing_connections(model, node, braces))
        else:
            raise ValueError(
                f"node {node}: unexpected welded-node topology -- "
                f"{len(legs)} leg-diameter member(s), {len(braces)} brace-diameter "
                f"member(s) (degree {len(ends)}). Expected either (2 legs, 2 or 4 "
                f"braces) or (0 legs, 4 braces) -- new topology needs a human look, "
                f"not a silent default."
            )

    return connections


def _member_diameter(model, mid):
    D, _t, _pid = sdg.member_section(model, mid)
    return D


def _leg_node_connections(model, node, legs, braces):
    (leg_a_mid, leg_a_end), (leg_b_mid, leg_b_end) = legs
    Da, Ta, _ = sdg.member_section(model, leg_a_mid)
    Db, Tb, _ = sdg.member_section(model, leg_b_mid)
    assert abs(Da - Db) < 1e-9, (
        f"node {node}: the two leg members have different diameters "
        f"({Da} vs {Db}) -- chord D is assumed uniform through a leg node"
    )
    chord_D = Da
    chord_T_ambiguous = abs(Ta - Tb) > 1e-9
    # When the two leg segments meeting at this node have different wall
    # thickness (4 nodes on this jacket -- the z=-24.61 KK level, where the
    # leg steps 50mm -> 35mm), there is no single correct "chord T": report
    # BOTH scenarios -- thick-everywhere and thin-everywhere at this node --
    # as a spread, same treatment as the X/real classification bounding pair
    # and the multiplanar single/summed spread, rather than silently picking
    # one (supersedes an earlier "always use the thinner" interim rule).
    # Applied uniformly to every brace at the node, not per-brace -- there is
    # no structural basis for assigning individual braces to one leg segment
    # or the other; the ambiguity belongs to the joint can as a whole.
    if chord_T_ambiguous:
        chord_t_scenarios = [("thick", max(Ta, Tb)), ("thin", min(Ta, Tb))]
    else:
        chord_t_scenarios = [("single", Ta)]

    chord_axis = _member_axis_away_from_node(model, leg_a_mid, leg_a_end)

    brace_info = []
    for mid, end in braces:
        theta_deg, az_deg = _theta_and_azimuth(model, mid, end, chord_axis)
        brace_info.append((mid, end, theta_deg, az_deg))

    clusters = _cluster_by_azimuth([(bi[:2], bi[3]) for bi in brace_info])
    # deterministic plane ordering: by mean azimuth, so plane_id is stable
    # across reruns rather than depending on dict/set iteration order.
    az_by_key = {(mid, end): az for mid, end, _th, az in brace_info}
    clusters = sorted(clusters, key=lambda ks: np.mean([az_by_key[k] for k in ks]))

    rows = []
    for plane_idx, keys in enumerate(clusters):
        n_in_plane = len(keys)
        sub_joint_id = (node, plane_idx)
        if n_in_plane == 2:
            family, type_label = "K", "K"
        elif n_in_plane == 1:
            th = next(th for (mid, end, th, az) in brace_info if (mid, end) == keys[0])
            family = "TY"
            type_label = "T" if th >= THETA_T_LABEL_MIN_DEG else "Y"
        else:
            raise ValueError(
                f"node {node} plane {plane_idx}: {n_in_plane} braces clustered "
                f"into one plane -- expected 1 (T/Y) or 2 (K); a real 3+ brace "
                f"plane needs a human look, not a silent default."
            )

        for (mid, end) in keys:
            th = next(th for (m2, e2, th, az) in brace_info if (m2, e2) == (mid, end))
            az = next(az for (m2, e2, th2, az) in brace_info if (m2, e2) == (mid, end))
            brace_D, brace_t, _ = sdg.member_section(model, mid)
            for scenario_label, scenario_T in chord_t_scenarios:
                rows.append(dict(
                    node=node, sub_joint_id=sub_joint_id, plane_id=plane_idx,
                    family=family, type_label=type_label, n_braces_in_plane=n_in_plane,
                    chord_members=[(leg_a_mid, leg_a_end), (leg_b_mid, leg_b_end)],
                    chord_D=chord_D, chord_T=scenario_T, chord_T_ambiguous=chord_T_ambiguous,
                    chord_t_scenario=scenario_label,
                    chord_T_upper=Ta, chord_T_lower=Tb,
                    brace_member=mid, brace_end=end, brace_D=brace_D, brace_t=brace_t,
                    theta_deg=th, azimuth_deg=az, direction=None,
                ))
    return rows


def _x_crossing_connections(model, node, braces):
    # Identify the two collinear (continuous-brace) pairs: member-ends whose
    # away-from-node directions are ~180 deg apart, i.e. one physical brace
    # running straight through this node. Verified on the real model: exact
    # pairs, angle 180.00 deg (see docs/decisions.md).
    axes = {(mid, end): _member_axis_away_from_node(model, mid, end) for mid, end in braces}
    keys = list(axes)
    pairs = []
    used = set()
    for i in range(len(keys)):
        if keys[i] in used:
            continue
        for j in range(i + 1, len(keys)):
            if keys[j] in used:
                continue
            cos_ang = np.dot(axes[keys[i]], axes[keys[j]])
            if cos_ang < -0.999:   # ~180 deg, generous vs the observed 0.00 deg error
                pairs.append((keys[i], keys[j]))
                used.add(keys[i])
                used.add(keys[j])
                break
    assert len(pairs) == 2 and len(used) == 4, (
        f"node {node}: expected exactly 2 collinear brace pairs (2 continuous "
        f"braces crossing), found {len(pairs)} pair(s) from {len(braces)} "
        f"member-ends -- X-crossing topology assumption doesn't hold here"
    )

    sub_joint_id = (node,)
    rows = []
    # Both directions: pair A as chord / pair B's near end as the brace being
    # checked, then vice versa -- per the 12.08.2026 "assess both directions,
    # report the governing one" decision.
    for chord_pair, brace_pair, direction in (
        (pairs[0], pairs[1], "A_as_chord"),
        (pairs[1], pairs[0], "B_as_chord"),
    ):
        chord_mid, chord_end = chord_pair[0]
        chord_D, chord_T, _ = sdg.member_section(model, chord_mid)
        chord_axis = axes[chord_pair[0]]

        # The "brace" is whichever end of the crossing pair actually meets
        # this node -- both member-ends of brace_pair sit at this same node
        # (it's a crossing), so either works; use the first for a definite,
        # reproducible choice.
        brace_mid, brace_end = brace_pair[0]
        brace_D, brace_t, _ = sdg.member_section(model, brace_mid)
        theta_deg, az_deg = _theta_and_azimuth(model, brace_mid, brace_end, chord_axis)

        rows.append(dict(
            node=node, sub_joint_id=sub_joint_id, plane_id=0,
            family="X", type_label="X", n_braces_in_plane=4,
            chord_members=[chord_pair[0], (chord_pair[1][0], chord_pair[1][1])],
            chord_D=chord_D, chord_T=chord_T, chord_T_ambiguous=False,
            chord_t_scenario="single",
            chord_T_upper=chord_T, chord_T_lower=chord_T,
            brace_member=brace_mid, brace_end=brace_end, brace_D=brace_D, brace_t=brace_t,
            theta_deg=theta_deg, azimuth_deg=az_deg, direction=direction,
        ))
    return rows
# endregion


# region --- Step 3: DNV Appendix B geometry parameters + validity + G1 CSV ---
# Figure B-2 definitions (see docs/decisions.md, primary-source read
# 12.08.2026): lowercase = brace, uppercase = chord.
#   beta = d/D          brace diameter / chord diameter
#   gamma = D/(2T)       DNV's own /2 is built in -- NOT D/T
#   tau = t/T            brace thickness / chord thickness
#   alpha = 2*L/D        L = chord length between adjacent joints
#   zeta = g/D           g = gap between adjacent braces in the same plane
#                         (K joints only -- Y/T have one brace, nothing to
#                         gap against; X's own equations need no gap term)
#
# Validity ranges (p.111, applies to Tables B-1..B-5), primary-source read:
BETA_RANGE = (0.2, 1.0)
TAU_RANGE = (0.2, 1.0)
GAMMA_RANGE = (8.0, 32.0)
ALPHA_RANGE = (4.0, 40.0)
THETA_RANGE_DEG = (20.0, 90.0)
# zeta's lower bound depends on beta/theta -- not a fixed pair, see
# _zeta_validity below.
ZETA_UPPER = 1.0


def _member_length(model, mid):
    m = model["members"][mid]
    j1 = np.array(model["joints"][m["j1"]])
    j2 = np.array(model["joints"][m["j2"]])
    return float(np.linalg.norm(j2 - j1))


def _chord_length_leg_node(model, leg_a_mid, leg_b_mid):
    """
    'Chord length between the joints' for a leg node: per Lotsberg (2011),
    alpha = 2L/D with L = the distance between SUPPORTING joints for the
    chord -- the FULL span of both adjoining leg segments (out to the next
    node in each direction), not half of it. alpha is a local boundary-
    condition descriptor for this joint's own chord flexibility, not a
    quantity budgeted/shared across the structure, so two adjacent joints
    legitimately share the same full span (REVERTED 13.08.2026 -- the
    12.08.2026 halving was motivated by a double-counting worry that does
    not apply; see docs/decisions.md "ALPHA -- REVERTED to full span").
    Independent of chord_t_scenario (a thickness label, not a geometry
    one), so computed once per node.
    """
    return _member_length(model, leg_a_mid) + _member_length(model, leg_b_mid)


def _chord_length_x_crossing(model, chord_pair):
    """Same idea for an X-crossing: 'chord' is whichever collinear brace
    pair is playing the chord role for this direction; L = the FULL length
    of the two segments either side of the crossing node that make up that
    continuous brace (see _chord_length_leg_node docstring -- full span,
    not halved)."""
    return _member_length(model, chord_pair[0][0]) + _member_length(model, chord_pair[1][0])


def _brace_near_toe_offset(chord_D, brace_d, theta_deg):
    """
    How far along the chord axis a SINGLE brace's near-side (gap-facing) toe
    sits from the point directly 'above' the shared model node, at the
    chord's OUTER surface -- DNV Fig B-2's 2-brace K-joint construction,
    derived (not copied) via the line/plane intersection of the brace's
    offset tube-surface line with the chord's outer-surface plane. The author-
    verified this session via a decisive check (see below) after two prior
    attempts (pure sin, pure tan on the combined (D-d)/2 term) both failed it.

    SubDyn's beam model puts the shared K-plane node on the CHORD'S
    CENTERLINE (radius 0), not its outer surface (radius D/2) -- so getting
    from 'the model node' to 'where the physical weld toe actually sits'
    needs TWO separate corrections, each with its OWN trig function, NOT one
    combined (D-d)/2 term over a single sin or tan:
        - reaching the chord's own outer radius D/2, travelling along the
          brace direction: needs 1/tan(theta) (a pure centerline-crossing
          calc -- how far along the chord axis before the brace's centerline,
          extended, reaches height D/2).
        - the brace's OWN radius d/2, offset perpendicular to the brace axis
          toward the gap: needs 1/sin(theta) (a projection of that
          perpendicular offset onto the chord axis).
    These do not combine into a single fraction because they answer two
    different geometric questions.

    offset = (D/2)/tan(theta) - (d/2)/sin(theta)

    DECISIVE CHECK (why this form, and not sin-only or tan-only): at
    theta=90 deg (a brace exactly perpendicular to the chord), the brace is
    a vertical tube -- its footprint on the chord surface is just its own
    radius d/2, completely independent of the chord's diameter D. Only this
    combined form gives that (D/2)/tan(90) -> 0, leaving exactly -d/2); a
    single (D-d)/2 divided by one trig function cannot reproduce this,
    because it can never fully cancel D's contribution. Verified against
    this check in _self_check_geometry().
    """
    th = math.radians(theta_deg)
    D_term = (chord_D / 2.0) / math.tan(th)
    d_term = (brace_d / 2.0) / math.sin(th)
    return D_term - d_term


def _zeta_and_validity(brace_D_a, theta_a_deg, brace_D_b, theta_b_deg, chord_D):
    """
    Single physical gap g for a 2-brace K-plane (DNV Fig B-2's middle
    diagram -- "Brace A / Brace B", ONE zeta = g/D; the two-gap g_AB/g_BC
    split only applies to Fig B-2's bottom diagram, a genuine 3-brace K-T
    joint with a real middle brace, which we don't have -- confirmed
    against the primary source this session, see docs/decisions.md).

    g = the sum of each brace's own near-toe offset from the shared node's
    surface projection (Fig B-2's chevron K: braces diverge in OPPOSITE
    along-chord directions from the shared node, so their two offsets add
    to give the total physical separation between their near toes).
    """
    offset_a = _brace_near_toe_offset(chord_D, brace_D_a, theta_a_deg)
    offset_b = _brace_near_toe_offset(chord_D, brace_D_b, theta_b_deg)
    gap = offset_a + offset_b
    zeta = gap / chord_D
    return zeta


def _param_validity(beta, gamma, tau, alpha, theta_deg, zeta):
    """Per-parameter in/out-of-DNV-range flags + one combined notes string."""
    notes = []
    beta_ok = BETA_RANGE[0] <= beta <= BETA_RANGE[1]
    if not beta_ok:
        notes.append(f"beta={beta:.3f} outside [{BETA_RANGE[0]},{BETA_RANGE[1]}]")
    gamma_ok = GAMMA_RANGE[0] <= gamma <= GAMMA_RANGE[1]
    if not gamma_ok:
        notes.append(f"gamma={gamma:.2f} outside [{GAMMA_RANGE[0]},{GAMMA_RANGE[1]}]")
    tau_ok = TAU_RANGE[0] <= tau <= TAU_RANGE[1]
    if not tau_ok:
        notes.append(f"tau={tau:.3f} outside [{TAU_RANGE[0]},{TAU_RANGE[1]}]")
    alpha_ok = ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]
    if not alpha_ok:
        if alpha > ALPHA_RANGE[1]:
            notes.append(
                f"alpha={alpha:.1f} above [{ALPHA_RANGE[0]},{ALPHA_RANGE[1]}] "
                f"(NOT benign -- a genuine validity breach, not just a "
                f"short-chord-correction non-issue; the short-chord F1-F4 "
                f"trigger is a SEPARATE threshold, alpha<12, not this ceiling)"
            )
        else:
            notes.append(
                f"alpha={alpha:.1f} BELOW [{ALPHA_RANGE[0]},{ALPHA_RANGE[1]}] "
                f"(NOT benign -- genuinely short chord, F1-F4 may deviate "
                f"from 1 and need real evaluation, not the F->1 shortcut)"
            )
    theta_ok = THETA_RANGE_DEG[0] <= theta_deg <= THETA_RANGE_DEG[1]
    if not theta_ok:
        notes.append(f"theta={theta_deg:.1f} outside [{THETA_RANGE_DEG[0]},{THETA_RANGE_DEG[1]}]deg")

    if zeta is None:
        zeta_ok = None
    else:
        zeta_lower = -0.6 * beta / math.sin(math.radians(theta_deg))
        zeta_ok = zeta_lower <= zeta <= ZETA_UPPER
        if not zeta_ok:
            notes.append(
                f"zeta={zeta:.3f} outside [{zeta_lower:.3f},{ZETA_UPPER}] "
                f"(open item -- see docs/decisions.md)"
            )

    return beta_ok, gamma_ok, tau_ok, alpha_ok, theta_ok, zeta_ok, "; ".join(notes)


def compute_geometry_params(connections, model):
    """
    Adds beta/gamma/tau/alpha/zeta (theta already present from Step 2) and
    per-parameter validity flags to every connection row, in place, and
    returns the same list for chaining. zeta is computed once per K
    sub_joint_id (needs BOTH braces in the plane) and copied onto both of
    that plane's connection rows; None for TY/X (no gap parameter in either
    family's equations).
    """
    # Chord length is a per-(node) or per-(node, direction) property, not
    # per-brace -- compute once and reuse across every connection that
    # shares it, rather than recomputing per row.
    chord_length_by_node = {}
    for c in connections:
        if c["family"] == "X":
            continue
        node = c["node"]
        if node not in chord_length_by_node:
            leg_a, leg_b = c["chord_members"]
            chord_length_by_node[node] = _chord_length_leg_node(model, leg_a[0], leg_b[0])

    # X-crossing chord length depends on direction (which collinear pair is
    # playing chord), keyed by (node, direction).
    chord_length_by_x = {}
    for c in connections:
        if c["family"] != "X":
            continue
        key = (c["node"], c["direction"])
        if key not in chord_length_by_x:
            chord_length_by_x[key] = _chord_length_x_crossing(model, c["chord_members"])

    # zeta needs both braces of a K plane together -- group first.
    zeta_by_sub_joint = {}
    by_sub_joint = {}
    for c in connections:
        by_sub_joint.setdefault(c["sub_joint_id"], []).append(c)
    for sub_joint_id, rows in by_sub_joint.items():
        if rows[0]["family"] != "K":
            continue
        # A K plane has exactly 2 braces; chord_t_scenario duplicates them,
        # so restrict to one scenario's pair before computing gap geometry
        # (the two braces' D/theta don't depend on chord_t_scenario at all).
        pair = [r for r in rows if r["chord_t_scenario"] == rows[0]["chord_t_scenario"]][:2]
        assert len(pair) == 2, f"sub_joint {sub_joint_id}: expected 2 K braces, got {len(pair)}"
        (a, b) = pair
        zeta_by_sub_joint[sub_joint_id] = _zeta_and_validity(
            a["brace_D"], a["theta_deg"], b["brace_D"], b["theta_deg"], a["chord_D"]
        )

    for c in connections:
        beta = c["brace_D"] / c["chord_D"]
        gamma = c["chord_D"] / (2.0 * c["chord_T"])
        tau = c["brace_t"] / c["chord_T"]
        if c["family"] == "X":
            chord_length = chord_length_by_x[(c["node"], c["direction"])]
        else:
            chord_length = chord_length_by_node[c["node"]]
        alpha = 2.0 * chord_length / c["chord_D"]
        zeta = zeta_by_sub_joint.get(c["sub_joint_id"])   # None for TY/X

        beta_ok, gamma_ok, tau_ok, alpha_ok, theta_ok, zeta_ok, notes = _param_validity(
            beta, gamma, tau, alpha, c["theta_deg"], zeta
        )

        c.update(
            beta=beta, gamma=gamma, tau=tau, alpha=alpha, chord_length=chord_length,
            zeta=zeta,
            beta_valid=beta_ok, gamma_valid=gamma_ok, tau_valid=tau_ok,
            alpha_valid=alpha_ok, theta_valid=theta_ok, zeta_valid=zeta_ok,
            validity_notes=notes,
        )
    return connections


def write_joint_geometry_check(connections, out_path):
    """G1 sign-off CSV -- one row per connection (120 rows), every DNV
    Appendix B geometry parameter plus its validity flag, traceable back to
    the member/node identity already carried on each row."""
    cols = [
        "node", "sub_joint_id", "plane_id", "family", "type_label",
        "n_braces_in_plane", "chord_D", "chord_T", "chord_T_ambiguous",
        "chord_t_scenario", "chord_length", "brace_member", "brace_end",
        "brace_D", "brace_t", "theta_deg", "azimuth_deg", "direction",
        "beta", "gamma", "tau", "alpha", "zeta",
        "beta_valid", "gamma_valid", "tau_valid", "alpha_valid",
        "theta_valid", "zeta_valid", "validity_notes",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for c in connections:
            writer.writerow([c[col] for col in cols])
    return connections
# endregion


def _self_check_geometry():
    print(f"Parsing: {sdg.DEFAULT_SD_PATH}")
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = build_connections(model)
    compute_geometry_params(connections, model)

    out_path = sdg.RESULTS_DIR / "joint_geometry_CHECK.csv"
    write_joint_geometry_check(connections, out_path)
    print(f"  wrote {out_path} ({len(connections)} rows)")

    # --- spot-check against the values already hand-derived (docs/decisions.md): beta=0.667, tau=0.400/0.571, gamma=12.0/17.14.
    node5 = [c for c in connections if c["node"] == 5]
    betas = sorted({round(c["beta"], 3) for c in node5})
    taus = sorted({round(c["tau"], 3) for c in node5})
    gammas = sorted({round(c["gamma"], 3) for c in node5})
    print(f"\n  node 5 (KK, ambiguous chord_T) beta values: {betas} (expect [0.667])")
    print(f"  node 5 tau values: {taus} (expect [0.4, 0.571] approx -- t=50/35mm legs)")
    print(f"  node 5 gamma values: {gammas} (expect [12.0, 17.14] approx)")
    assert betas == [0.667]
    assert taus == [0.4, 0.571]
    assert gammas == [12.0, 17.143]

    # alpha: previously hand-estimated 41-57 (different, rougher theta
    # sample) -- check it lands in a sane jacket-panel-height ballpark
    # rather than forcing an exact match to that earlier estimate.
    alphas = [c["alpha"] for c in connections if c["family"] != "X"]
    print(f"\n  alpha range over all non-X connections: "
          f"[{min(alphas):.1f}, {max(alphas):.1f}]")
    alpha_high = sum(1 for c in connections if not c["alpha_valid"] and c["alpha"] > ALPHA_RANGE[1])
    alpha_low = sum(1 for c in connections if not c["alpha_valid"] and c["alpha"] < ALPHA_RANGE[0])
    print(f"  connections with alpha ABOVE DNV range (genuine breach, NOT benign): {alpha_high}/{len(connections)}")
    print(f"  connections with alpha BELOW DNV range (NOT benign -- real short-chord "
          f"correction needed): {alpha_low}/{len(connections)} -- expect these to be the "
          f"lower-mudbrace nodes (short pile-transition leg stub)")
    low_nodes = sorted({c["node"] for c in connections
                         if not c["alpha_valid"] and c["alpha"] < ALPHA_RANGE[0]})
    print(f"  nodes with alpha below range: {low_nodes}")

    # X-crossings: beta = tau = 1.0 exactly (both braces same diameter),
    # sitting exactly on the upper validity bound -- must be flagged True
    # (in-range, right at the edge), not spuriously failed by float noise.
    x_rows = [c for c in connections if c["family"] == "X"]
    x_betas = sorted({round(c["beta"], 6) for c in x_rows})
    x_taus = sorted({round(c["tau"], 6) for c in x_rows})
    print(f"\n  X-crossing beta values: {x_betas} (expect [1.0])")
    print(f"  X-crossing tau values: {x_taus} (expect [1.0])")
    assert x_betas == [1.0] and x_taus == [1.0]
    assert all(c["beta_valid"] and c["tau_valid"] for c in x_rows), (
        "beta=tau=1.0 sits exactly on the DNV upper bound (<=1.0) and must "
        "still validate True, not fail on a float boundary"
    )
    assert all(c["zeta"] is None and c["zeta_valid"] is None for c in x_rows), (
        "X joints take no gap parameter -- zeta must be unset, not silently 0"
    )

    # TY joints: also no zeta (single brace, nothing to gap against).
    ty_rows = [c for c in connections if c["family"] == "TY"]
    assert all(c["zeta"] is None and c["zeta_valid"] is None for c in ty_rows), (
        "T/Y joints take no gap parameter -- zeta must be unset"
    )

    # Decisive check for _brace_near_toe_offset (see its docstring): a brace
    # exactly perpendicular to the chord (theta=90) is a vertical tube, so
    # its footprint on the chord surface is just its own radius d/2,
    # completely independent of the chord's own diameter D. This is what
    # separates the correct combined formula from two plausible-looking but
    # wrong single-trig-function attempts (both ruled out this session by
    # this exact check).
    off_90 = _brace_near_toe_offset(chord_D=1.2, brace_d=0.8, theta_deg=90.0)
    print(f"\n  theta=90 decisive check: near-toe offset = {off_90:.6f} "
          f"(expect exactly -d/2 = -0.4, independent of chord_D)")
    assert abs(off_90 - (-0.4)) < 1e-9, (
        "at theta=90 the near-toe offset must equal -d/2 exactly and not "
        "depend on chord_D at all -- formula regression"
    )

    # K joints: zeta IS set, and (per this session's corrected DNV Fig B-2
    # 2-brace construction) should now be POSITIVE and in-range -- a real
    # gap, not the large negative overlap the earlier (wrong) formula gave.
    k_rows = [c for c in connections if c["family"] == "K"]
    assert all(c["zeta"] is not None for c in k_rows), "every K connection needs a zeta"
    zeta_values = sorted({round(c["zeta"], 3) for c in k_rows})
    n_zeta_invalid = sum(1 for c in k_rows if not c["zeta_valid"])
    print(f"\n  K-joint zeta values (distinct): {zeta_values[:5]}{'...' if len(zeta_values) > 5 else ''}")
    print(f"  K connections with zeta outside DNV's validity floor: "
          f"{n_zeta_invalid}/{len(k_rows)} (expect 0 -- corrected formula gives "
          f"a real, in-range gap; see docs/decisions.md)")
    assert all(z > 0 for z in zeta_values), (
        "expected every K-plane zeta to be positive (a real gap) under the "
        "corrected Fig B-2 formula -- got a negative value, formula regression"
    )
    assert n_zeta_invalid == 0, (
        "expected every K-plane zeta to now be within DNV's validity range "
        "under the corrected formula"
    )

    # Same zeta value must appear on both connection rows of a K plane
    # (shared between the two braces sharing that plane, not per-brace).
    for sub_joint_id, rows in by_sub_joint_for_check(k_rows):
        zetas_in_plane = {round(r["zeta"], 9) for r in rows if r["chord_t_scenario"] == rows[0]["chord_t_scenario"]}
        assert len(zetas_in_plane) == 1, (
            f"sub_joint {sub_joint_id}: zeta disagrees between the two braces sharing "
            f"the plane -- should be one shared gap value, not per-brace"
        )

    print("\n  all checks passed.")


def by_sub_joint_for_check(rows):
    grouped = {}
    for r in rows:
        grouped.setdefault(r["sub_joint_id"], []).append(r)
    return grouped.items()


def _self_check_connections():
    print(f"Parsing: {sdg.DEFAULT_SD_PATH}")
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = build_connections(model)
    print(f"  {len(connections)} connections built (expect 120 = 104 physically "
          f"distinct + 16 extra rows from the 4 chord_T-ambiguous nodes)")
    assert len(connections) == 120

    sub_joints = sorted({c["sub_joint_id"] for c in connections})
    print(f"  {len(sub_joints)} distinct sub-joints (expect 64 -- chord_t_scenario "
          f"is a separate dimension from physical sub-joint identity)")
    assert len(sub_joints) == 64

    welded_nodes = sorted({c["node"] for c in connections})
    print(f"  {len(welded_nodes)} distinct welded nodes (expect 40)")
    assert len(welded_nodes) == 40

    import collections
    fam_conn = collections.Counter(c["family"] for c in connections)
    fam_sub = collections.Counter()
    seen_sub = set()
    for c in connections:
        if c["sub_joint_id"] not in seen_sub:
            seen_sub.add(c["sub_joint_id"])
            fam_sub[c["family"]] += 1
    print(f"\n  connections by family: {dict(fam_conn)} (expect K=64, TY=24, X=32)")
    print(f"  sub-joints by family:  {dict(fam_sub)} (expect K=24, TY=24, X=16)")
    assert dict(fam_conn) == {"K": 64, "TY": 24, "X": 32}
    assert dict(fam_sub) == {"K": 24, "TY": 24, "X": 16}

    scenario_conn = collections.Counter(c["chord_t_scenario"] for c in connections)
    print(f"  connections by chord_t_scenario: {dict(scenario_conn)} "
          f"(expect single=88, thick=16, thin=16)")
    assert dict(scenario_conn) == {"single": 88, "thick": 16, "thin": 16}

    type_label_conn = collections.Counter(c["type_label"] for c in connections)
    print(f"  connections by type_label (reporting only): {dict(type_label_conn)}")

    # Per-level theta -- cross-check against the values hand-derived earlier
    # this session (33.20/31.01 at the -24.61 KK level, 38.55 at top, 29.49/
    # 88.11 at the two mudbrace levels, 62.65 at the crossings).
    by_node_theta = collections.defaultdict(list)
    for c in connections:
        by_node_theta[c["node"]].append(round(c["theta_deg"], 2))
    print(f"\n  spot-check node 5 (KK, z=-24.61) theta values: {sorted(by_node_theta[5])} "
          f"(expect ~[31.01, 31.01, 33.20, 33.20])")
    print(f"  spot-check node 23 (top Y) theta values: {sorted(by_node_theta[23])} "
          f"(expect ~[38.55, 38.55])")
    print(f"  spot-check node 3 (mud T) theta values: {sorted(by_node_theta[3])} "
          f"(expect ~[88.11, 88.11])")
    print(f"  spot-check node 4 (mud Y) theta values: {sorted(by_node_theta[4])} "
          f"(expect ~[29.49, 29.49])")
    print(f"  spot-check node 37 (X-crossing) theta values: {sorted(by_node_theta[37])} "
          f"(expect ~[62.65, 62.65])")

    # K-plane discriminator: within a K plane, the two braces must share an
    # azimuth (same face) but point in OPPOSITE along-chord directions (one
    # up, one down -- the chevron/K shape) -- verify this is what the
    # clustering actually found, not merely that it produced pairs of 2.
    # Node 5 is chord_T-ambiguous (4 rows/plane, 2 braces x thick/thin), so
    # filter to one scenario first -- the along-chord shape is a property of
    # the braces themselves, independent of which chord_T assumption is used.
    k_planes_checked = 0
    node5_legs = [(mid, e) for mid, e in _node_incidence(model)[5]
                  if _member_diameter(model, mid) >= D_LEG_MIN]
    chord_axis_5 = _member_axis_away_from_node(model, *node5_legs[0])
    node5_braces = [c for c in connections if c["node"] == 5 and c["chord_t_scenario"] == "thick"]
    for plane_id in sorted({c["plane_id"] for c in node5_braces}):
        pair = [c for c in node5_braces if c["plane_id"] == plane_id]
        assert len(pair) == 2
        along = [np.dot(_member_axis_away_from_node(model, c["brace_member"], c["brace_end"]),
                         chord_axis_5) for c in pair]
        print(f"  node 5 plane {plane_id}: along-chord components {[round(a, 3) for a in along]} "
              f"(expect opposite signs -- chevron/K shape)")
        assert along[0] * along[1] < 0, "K-plane braces should point opposite ways along the chord"
        k_planes_checked += 1
    assert k_planes_checked == 2

    # Chord-T ambiguity: confirmed 4 KK nodes (the z=-24.61 level, where the
    # leg wall thickness steps 50mm->35mm) have genuinely different T on
    # their two leg segments.
    ambiguous_nodes = sorted({c["node"] for c in connections if c["chord_T_ambiguous"]})
    print(f"\n  nodes with chord_T_ambiguous=True: {ambiguous_nodes} (expect [5, 10, 15, 20])")
    assert ambiguous_nodes == [5, 10, 15, 20]

    # Every brace at an ambiguous node must appear exactly twice (thick +
    # thin), with chord_T equal to the max/min of the node's two real leg
    # thicknesses -- not some other value (e.g. an accidental average).
    for node in ambiguous_nodes:
        node_rows = [c for c in connections if c["node"] == node]
        by_brace = collections.defaultdict(list)
        for c in node_rows:
            by_brace[(c["plane_id"], c["brace_member"], c["brace_end"])].append(c)
        for key, rows2 in by_brace.items():
            scenarios = sorted(r["chord_t_scenario"] for r in rows2)
            assert scenarios == ["thick", "thin"], (
                f"node {node} brace {key}: expected exactly [thick, thin], got {scenarios}"
            )
            Ts = {r["chord_t_scenario"]: r["chord_T"] for r in rows2}
            expect_thick, expect_thin = max(rows2[0]["chord_T_upper"], rows2[0]["chord_T_lower"]), \
                                          min(rows2[0]["chord_T_upper"], rows2[0]["chord_T_lower"])
            assert abs(Ts["thick"] - expect_thick) < 1e-12
            assert abs(Ts["thin"] - expect_thin) < 1e-12
    print(f"  every brace at the 4 ambiguous nodes carries exactly one 'thick' "
          f"and one 'thin' row, with chord_T = max/min of the node's real leg "
          f"thicknesses (not an average or a silent pick)")

    # Every OTHER connection must be chord_t_scenario="single".
    non_ambiguous = [c for c in connections if c["node"] not in ambiguous_nodes]
    assert all(c["chord_t_scenario"] == "single" for c in non_ambiguous)
    assert len(non_ambiguous) == 120 - 32   # 16 ambiguous braces x 2 scenarios

    # X-crossing symmetry: both directions at a crossing must see the SAME
    # theta (crossing angle is symmetric regardless of which brace is called
    # "chord") -- a real check on the both-directions logic, not a tautology.
    x_rows = [c for c in connections if c["family"] == "X" and c["node"] == 37]
    assert len(x_rows) == 2
    thetas = [round(r["theta_deg"], 3) for r in x_rows]
    print(f"\n  node 37 (X-crossing) both-direction thetas: {thetas} (expect equal)")
    assert abs(thetas[0] - thetas[1]) < 1e-6

    print("\n  all checks passed.")


# region --- Step 4: IPB/OPB rotation ---
# DNV's (3.3.1) hot-spot superposition needs bending split into IN-PLANE
# (IPB -- bending whose DEFLECTION stays inside the brace-chord plane, so the
# moment VECTOR points perpendicular to that plane) and OUT-OF-PLANE (OPB --
# deflection leaves the plane, so the moment vector lies WITHIN the plane,
# transverse to the brace axis). This is a standard beam-bending fact (a
# moment vector is always perpendicular to the plane the bending curve lies
# in), not a joint-specific assumption.
#
# SubDyn's own MKxe/MKye (-> stress.nominal_components' sig_ipb/sig_opb, a
# pre-existing member-track NAME COLLISION -- those are just "bending stress
# about local x" / "about local y", nothing joint-specific yet) are computed
# about the member's OWN local x/y (DCM rows 0/1), which point wherever
# SubDyn's internal element-orientation rule happens to put them -- not
# aligned with the brace-chord plane in general. rotate_to_joint_axes() is
# the fix: a plain 2D rotation of the (sig_ipb, sig_opb) vector by the angle
# phi between local-x and the joint's own in-plane direction.
#
# Because it is a rotation (not a re-derivation), sqrt(mip^2 + mop^2) ==
# sqrt(ipb^2 + opb^2) is an ALGEBRAIC IDENTITY of the formula below, true for
# any phi -- so it is not, by itself, proof phi was computed correctly (a
# wrong phi still preserves the magnitude). It is still worth checking as a
# regression test (catches a non-orthogonal coding slip, e.g. two cosines),
# and phi itself is checked independently and geometrically in
# _self_check_rotation via the DCM-rotated-axes-vs-geometric-axes comparison,
# which is the check that actually catches a transposed DCM or a sign error.


def joint_plane_axes(model, c):
    """
    Per-connection geometric frame, purely from SubDyn.dat joint coordinates
    (independent of the DCM -- this is the ground truth the DCM-derived phi
    gets checked against in _self_check_rotation).

    brace_axis, chord_axis: unit vectors, both pointing AWAY from the shared
        node (same convention as Steps 2/3's own _member_axis_away_from_node
        / _theta_and_azimuth -- chord_axis recomputed the identical way Step
        2 did internally, from chord_members[0], rather than threading a new
        field through the signed-off Step 2/3 row-building code).
    n: the brace-chord PLANE NORMAL, unit vector -- the axis a moment vector
        points along for IN-PLANE bending (deflection stays in-plane ->
        moment perpendicular to it).
    e_par: unit vector, IN the brace-chord plane, perpendicular to the brace
        axis -- the projection of chord_axis onto the plane normal to the
        brace. This is the axis a moment vector points along for OUT-OF-PLANE
        bending, and (by the same "crown sits where the brace visibly meets
        the chord, i.e. along the plane" logic already used for theta/azimuth
        in Step 2) it is also theta=0's physical direction for the joint's
        own crown/saddle circumferential convention: crown at 0/180 deg
        (along e_par, in-plane, toe/heel), saddle at 90/270 deg (along n,
        out-of-plane).
    """
    brace_axis = _member_axis_away_from_node(model, c["brace_member"], c["brace_end"])
    chord_mid, chord_end = c["chord_members"][0]
    chord_axis = _member_axis_away_from_node(model, chord_mid, chord_end)

    n = np.cross(brace_axis, chord_axis)
    n_norm = np.linalg.norm(n)
    assert n_norm > 1e-9, (
        f"connection (node={c['node']}, brace={c['brace_member']}): brace and "
        f"chord axes are parallel -- no well-defined brace-chord plane. Should "
        f"be geometrically impossible on this jacket (theta is always well "
        f"inside (0,90] deg); a hit here means the wrong members were paired."
    )
    n = n / n_norm

    e_par_unnorm = chord_axis - np.dot(chord_axis, brace_axis) * brace_axis
    e_par = e_par_unnorm / np.linalg.norm(e_par_unnorm)

    return dict(brace_axis=brace_axis, chord_axis=chord_axis, n=n, e_par=e_par)


def compute_member_phi_deg(dcm, mid, axes):
    """
    Generalised form of compute_phi_deg (below): angle (deg) from ANY
    member's own DCM local-x (row 0) to the joint's shared e_par, measured
    about that member's own axis in the local-x -> local-y sense (matches
    hotspot_member's own theta convention: cos multiplies the "x-like" term,
    -sin the "y-like" term -- see the module-level rotate_to_joint_axes
    derivation). e_par lies in the local x-y plane by construction for the
    BRACE (it is perpendicular to brace_axis, and DCM row 2 IS brace_axis up
    to Step 1's own confirmed sign convention) -- same holds for a CHORD
    segment's own local x-y plane, since e_par is perpendicular to
    chord_axis too (see joint_plane_axes), and DCM row 2 IS a member's own
    axis regardless of which member. So this one formula is correct for
    both brace and chord segments (Step B reuses it unchanged), and is a
    clean 2D atan2 either way, not an approximation.
    """
    R = dcm[mid]
    ex, ey = R[0, :], R[1, :]
    e_par = axes["e_par"]
    return float(np.degrees(np.arctan2(np.dot(e_par, ey), np.dot(e_par, ex))))


def compute_phi_deg(dcm, c, axes):
    """phi for the connection's own brace member -- see compute_member_phi_deg
    for the general form (Step B reuses it for the chord segments too)."""
    return compute_member_phi_deg(dcm, c["brace_member"], axes)


def rotate_to_joint_axes(sig_ipb, sig_opb, phi_deg):
    """
    Rotate SubDyn member-local bending stresses (about local x / local y --
    stress.nominal_components' sig_ipb/sig_opb) into the joint's TRUE
    in-plane / out-of-plane bending stresses (sig_mip, sig_mop), by the angle
    phi_deg from compute_phi_deg.

    Derivation: hotspot_member's own formula is
        sigma(theta) = ax + ipb*cos(theta) - opb*sin(theta)
    with theta measured from local-x. Substituting theta = theta' + phi (theta'
    now measured from e_par, the joint's own crown/saddle reference) and
    collecting cos(theta')/sin(theta') terms gives:
        mip = ipb*cos(phi) - opb*sin(phi)
        mop = ipb*sin(phi) + opb*cos(phi)
    -- an ordinary 2D vector rotation by -phi (equivalently: the (ipb, opb)
    vector expressed in a frame rotated by +phi). sig_ipb/sig_opb may be
    arrays (a time signal); phi_deg is a single connection-level scalar (a
    fixed geometric property, not time-varying).
    """
    phi = math.radians(phi_deg)
    sig_ipb = np.asarray(sig_ipb, dtype=np.float64)
    sig_opb = np.asarray(sig_opb, dtype=np.float64)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    sig_mip = sig_ipb * cos_p - sig_opb * sin_p
    sig_mop = sig_ipb * sin_p + sig_opb * cos_p
    return sig_mip, sig_mop


def add_joint_axes(connections, model, dcm):
    """
    Adds, in place, to every connection row: n (plane_normal), e_par (both
    as plain 3-tuples, CSV/JSON friendly), phi_deg, mop_axis. Returns the
    same list.

    plane_normal vs mop_axis -- two related but distinct vectors, both
    perpendicular to the brace-chord plane, kept as SEPARATE fields rather
    than merged, because they answer different questions and only one of
    them is safe to treat as end-independent:
      - plane_normal: from brace_axis x chord_axis (both "away from node"),
        a purely geometric description of the plane. Its sign happens to
        depend on brace_end (an accident of which end SubDyn.dat calls J1
        for this member, not a physical asymmetry) -- fine for a geometry
        column, NOT safe to assume matches any other module's frame.
      - mop_axis: rotate(DCM local-y, phi) -- the axis the rotation formula
        (rotate_to_joint_axes) ACTUALLY treats as its "90 degrees from
        e_par" reference, i.e. the axis mop is physically a bending stress
        about. Verified in _self_check_rotation to be exactly +-plane_normal
        (perpendicular to the plane to machine precision), sign tracking
        brace_end. This is the field the Grasshopper JSON export must use
        to place the 8 crown/saddle points, since it is the one guaranteed
        consistent with the (mip, mop) values rotate_to_joint_axes produces.
    """
    for c in connections:
        axes = joint_plane_axes(model, c)
        phi_deg = compute_phi_deg(dcm, c, axes)
        R = dcm[c["brace_member"]]
        ex, ey = R[0, :], R[1, :]
        phi = math.radians(phi_deg)
        mop_axis = -ex * math.sin(phi) + ey * math.cos(phi)
        c.update(
            plane_normal=tuple(axes["n"]),
            e_par=tuple(axes["e_par"]),
            brace_axis=tuple(axes["brace_axis"]),
            chord_axis=tuple(axes["chord_axis"]),
            phi_deg=phi_deg,
            mop_axis=tuple(mop_axis),
        )
    return connections
# endregion


# region --- Step B: chord-side geometry (phi + section modulus, both segments) ---
# DNV eqn 6b/7b's crown term needs the CHORD's own bending stress (M/W) at
# the crown position, added on top of the brace's axial-driven base term
# (see docs/decisions.md, "EQN 6b/7b ADOPTED for the crown"). That
# needs the same two things Step 4 built for the brace, but for the chord
# instead: (1) a phi angle rotating the chord's OWN local x/y (its own DCM,
# a different member from the brace) into the shared joint e_par/n frame,
# and (2) the chord's own section modulus W = I/R.
#
# Both chord LEG SEGMENTS meeting at a node are carried forward separately,
# never combined into one signal (see docs/decisions.md, "CHORD-
# SEGMENT HANDLING SUPERSEDED" -- rainflowing max(|M_a|,|M_b|) risks
# manufacturing spurious turning points at the crossover that don't
# correspond to a real physical reversal in either segment; the two
# segments are instead evaluated as two independent candidate signals all
# the way through rainflow, worst damage taken after). chord_members[0]/[1]
# (already on every connection row since Step 2) are segment A and segment
# B -- this step adds each segment's own phi/section alongside its existing
# member/end identity. X-crossing connections have the identical
# two-member chord_members structure (the two collinear members forming
# whichever brace pair is playing "chord" for that direction), so no
# family-specific branching is needed here.
#
# D/T for each segment come from sd_geometry.member_section (the segment's
# REAL physical section), deliberately NOT the chord_t_scenario thick/thin
# pick Step 3 uses for beta/gamma/tau. That pick exists because the DNV
# Appendix B equations need ONE chord-T assumption per joint and the two
# real leg segments genuinely differ (the reason chord_T_ambiguous is True
# at 4 nodes) -- but here the two segments are real, physically distinct
# beams, each with its own real bending stiffness, so each gets its own
# real T (and hence its own real W), not an assumed scenario value.
#
# TARGET AXIS -- NOT e_par. e_par (Step 4) lies in the BRACE's own local
# x-y plane (perpendicular to brace_axis) by construction, but is NOT in
# general perpendicular to chord_axis, so it does not lie in the CHORD's
# own local x-y plane -- rotating a chord segment's DCM ex/ey (confined to
# a plane perpendicular to that segment's own axis) can never reach it.
# The correct chord-side target is the MIRROR construction: project
# brace_axis onto the plane perpendicular to the CHORD SEGMENT'S OWN axis
# (instead of the other way around):
#
#     chord_e_par = normalize(brace_axis - dot(brace_axis, axis)*axis)
#
# This is perpendicular to `axis` by construction. Computed SEPARATELY per
# segment (using each segment's own away-from-node axis), NOT once from
# segment A and reused for B: the two leg segments meeting at a node are
# close to collinear but not exactly (measured on this model: ~0.003 deg
# kink at a spot-checked node -- a real, physically negligible panel-point
# effect, not a bug) -- reusing one segment's chord_e_par for the other
# leaves phi unable to land on it exactly (checked directly: ~1e-3 residual
# for segment B when segment A's chord_e_par was reused, well above the
# 1e-9 machine-precision bar every other check in this module holds to).
# Two independent chord_e_par values, each exactly perpendicular to its own
# segment's axis, avoids the issue entirely -- correct rather than merely
# "close enough". Physically: chord_e_par is the direction, in a chord
# segment's own cross-section, pointing toward the brace -- the chord-side
# view of the same physical crown/toe location e_par already describes from
# the brace side. Computed here from c["brace_axis"] (a Step 4 field, read
# only) and each segment's own axis (recomputed via the Step 2 helper), not
# added into joint_plane_axes/add_joint_axes themselves, so Step 4's own
# signed-off code and outputs stay untouched.


def chord_segment_geometry(model, dcm, mid, axes):
    """(D, T, W, phi_deg) for one chord segment member. See the region
    docstring for why D/T are the segment's real physical section rather
    than the chord_t_scenario thick/thin pick used elsewhere."""
    D, T, _pid = sdg.member_section(model, mid)
    W = sdg.section_properties(D, T)["W"]
    phi_deg = compute_member_phi_deg(dcm, mid, axes)
    return D, T, W, phi_deg


def chord_e_par_axis(c, model, mid, end):
    """The chord-side mirror of Step 4's e_par, for ONE specific chord
    segment member -- see the region docstring for why this must be
    computed per-segment rather than shared between segments A and B."""
    brace_axis = np.array(c["brace_axis"])
    axis = _member_axis_away_from_node(model, mid, end)
    unnorm = brace_axis - np.dot(brace_axis, axis) * axis
    return unnorm / np.linalg.norm(unnorm)


def add_chord_geometry(connections, model, dcm):
    """
    Adds, in place, to every connection row: chord_a_e_par, chord_a_member,
    chord_a_end, chord_a_D, chord_a_T, chord_a_W, chord_a_phi_deg, and the
    same seven fields suffixed _b for the second chord segment
    (chord_members[1]) -- chord_a_e_par/chord_b_e_par are each segment's OWN
    target axis (see chord_e_par_axis), not a shared one. Requires
    add_joint_axes to have already run (reads c["brace_axis"]). Returns the
    same list.
    """
    for c in connections:
        (a_mid, a_end), (b_mid, b_end) = c["chord_members"]
        e_par_a = chord_e_par_axis(c, model, a_mid, a_end)
        e_par_b = chord_e_par_axis(c, model, b_mid, b_end)
        Da, Ta, Wa, phi_a = chord_segment_geometry(model, dcm, a_mid, dict(e_par=e_par_a))
        Db, Tb, Wb, phi_b = chord_segment_geometry(model, dcm, b_mid, dict(e_par=e_par_b))
        c.update(
            chord_a_e_par=tuple(e_par_a),
            chord_a_member=a_mid, chord_a_end=a_end,
            chord_a_D=Da, chord_a_T=Ta, chord_a_W=Wa, chord_a_phi_deg=phi_a,
            chord_b_e_par=tuple(e_par_b),
            chord_b_member=b_mid, chord_b_end=b_end,
            chord_b_D=Db, chord_b_T=Tb, chord_b_W=Wb, chord_b_phi_deg=phi_b,
        )
    return connections


def write_chord_geometry_check(connections, out_path):
    """Step B sign-off CSV -- one row per connection (120 rows), both chord
    segments' own real section + phi, for hand-checking before scf.py
    (Step C) consumes them. Kept as its own file rather than added to the
    already-G1-signed-off joint_geometry_CHECK.csv, to avoid changing the
    schema of a closed sign-off artifact."""
    cols = [
        "node", "sub_joint_id", "family", "type_label", "brace_member",
        "brace_end", "chord_a_member", "chord_a_end", "chord_a_D",
        "chord_a_T", "chord_a_W", "chord_a_phi_deg",
        "chord_b_member", "chord_b_end", "chord_b_D", "chord_b_T",
        "chord_b_W", "chord_b_phi_deg",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for c in connections:
            writer.writerow([c[col] for col in cols])
    return connections
# endregion


def _self_check_chord_geometry():
    print(f"Parsing: {DEFAULT_SD_SUM_PATH}, {sdg.DEFAULT_SD_PATH}")
    dcm_result = read_member_dcm(DEFAULT_SD_SUM_PATH)
    dcm = dcm_result["dcm"]
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = build_connections(model)
    compute_geometry_params(connections, model)
    add_joint_axes(connections, model, dcm)
    add_chord_geometry(connections, model, dcm)

    out_path = sdg.RESULTS_DIR / "joint_chord_geometry_CHECK.csv"
    write_chord_geometry_check(connections, out_path)
    print(f"  wrote {out_path} ({len(connections)} rows)")

    # --- Check 1: chord segment identity matches chord_members exactly,
    # order preserved (a=[0], b=[1]) -- trivial but real bookkeeping check.
    for c in connections:
        (a_mid, a_end), (b_mid, b_end) = c["chord_members"]
        assert c["chord_a_member"] == a_mid and c["chord_a_end"] == a_end
        assert c["chord_b_member"] == b_mid and c["chord_b_end"] == b_end
    print("\n  chord_a/chord_b identity matches chord_members[0]/[1] on all "
          f"{len(connections)} connections: passed")

    # --- Check 2: real physical D/T match sd_geometry directly (NOT the
    # chord_t_scenario thick/thin pick) -- spot-checked at node 5, the
    # chord_T-ambiguous KK node (leg steps 50mm -> 35mm).
    node5 = [c for c in connections if c["node"] == 5][0]
    Da_expect, Ta_expect, _ = sdg.member_section(model, node5["chord_a_member"])
    Db_expect, Tb_expect, _ = sdg.member_section(model, node5["chord_b_member"])
    print(f"\n  node 5 chord_a: D={node5['chord_a_D']:.3f} T={node5['chord_a_T']:.4f} "
          f"(expect D={Da_expect:.3f} T={Ta_expect:.4f} -- real member section)")
    print(f"  node 5 chord_b: D={node5['chord_b_D']:.3f} T={node5['chord_b_T']:.4f} "
          f"(expect D={Db_expect:.3f} T={Tb_expect:.4f})")
    assert abs(node5["chord_a_D"] - Da_expect) < 1e-9
    assert abs(node5["chord_a_T"] - Ta_expect) < 1e-9
    assert abs(node5["chord_b_D"] - Db_expect) < 1e-9
    assert abs(node5["chord_b_T"] - Tb_expect) < 1e-9
    # This node is chord_T_ambiguous precisely because its two real segments
    # differ -- confirm chord geometry actually sees that (unlike the single
    # scenario-picked chord_T field used for beta/gamma/tau).
    seg_Ts = {round(node5["chord_a_T"], 3), round(node5["chord_b_T"], 3)}
    print(f"  node 5 real segment thicknesses differ: {seg_Ts} (expect {{0.035, 0.05}})")
    assert seg_Ts == {0.035, 0.05}

    # --- Check 3: section modulus is the standard I/R formula, independently
    # recomputed here rather than trusting section_properties uncritically a
    # second time.
    for c in connections[:5] + connections[-5:]:
        for seg in ("a", "b"):
            D, T, W = c[f"chord_{seg}_D"], c[f"chord_{seg}_T"], c[f"chord_{seg}_W"]
            d_inner = D - 2 * T
            I_expect = math.pi * (D ** 4 - d_inner ** 4) / 64
            W_expect = I_expect / (D / 2)
            assert abs(W - W_expect) / W_expect < 1e-9, (
                f"chord_{seg}_W does not match the standard I/R formula"
            )
    print("\n  section modulus formula spot-checked on 10 connections "
          "(first+last 5): passed")

    # --- Check 4 (the real phi check, mirroring Step 4's own decisive check
    # for the brace): rotating EACH chord segment's own DCM local-x by its
    # own chord_phi_deg must land exactly on that SAME segment's own
    # chord_a_e_par/chord_b_e_par (the CHORD-side target axis -- NOT the
    # brace-side e_par, and NOT shared between segments; see the region
    # docstring for why both of those would be wrong) -- this is what would
    # catch a transposed chord DCM or a sign error.
    max_err_a = 0.0
    max_err_b = 0.0
    for c in connections:
        Ra = dcm[c["chord_a_member"]]
        exa, eya = Ra[0, :], Ra[1, :]
        phia = math.radians(c["chord_a_phi_deg"])
        exa_rot = exa * math.cos(phia) + eya * math.sin(phia)
        max_err_a = max(max_err_a, np.max(np.abs(exa_rot - np.array(c["chord_a_e_par"]))))

        Rb = dcm[c["chord_b_member"]]
        exb, eyb = Rb[0, :], Rb[1, :]
        phib = math.radians(c["chord_b_phi_deg"])
        exb_rot = exb * math.cos(phib) + eyb * math.sin(phib)
        max_err_b = max(max_err_b, np.max(np.abs(exb_rot - np.array(c["chord_b_e_par"]))))
    print(f"\n  max |rotate(chord_a ex, phi) - chord_a_e_par|: {max_err_a:.2e} (expect ~0)")
    print(f"  max |rotate(chord_b ex, phi) - chord_b_e_par|: {max_err_b:.2e} (expect ~0)")
    assert max_err_a < 1e-9 and max_err_b < 1e-9, (
        "rotating a chord segment's own DCM local-x by its chord_phi_deg "
        "does not reproduce that segment's own chord_e_par -- phi "
        "formula/sign error, same failure class as Step 4's own check 2"
    )

    # --- Check 5: each segment's own target axis IS perpendicular to that
    # SAME segment's own axis (by construction, but verified rather than
    # assumed), and the two segments' target axes are close to each other
    # (not identical -- real small panel-point kinks, see the region
    # docstring -- but should agree to within a fraction of a degree, not
    # be wildly different vectors, which would indicate a real topology
    # problem rather than a benign kink).
    max_perp_a, max_perp_b = 0.0, 0.0
    max_ab_angle_deg = 0.0
    for c in connections:
        e_par_a = np.array(c["chord_a_e_par"])
        e_par_b = np.array(c["chord_b_e_par"])
        axis_a = _member_axis_away_from_node(model, c["chord_a_member"], c["chord_a_end"])
        axis_b = _member_axis_away_from_node(model, c["chord_b_member"], c["chord_b_end"])
        max_perp_a = max(max_perp_a, abs(np.dot(e_par_a, axis_a)))
        max_perp_b = max(max_perp_b, abs(np.dot(e_par_b, axis_b)))
        ang = math.degrees(math.acos(np.clip(np.dot(e_par_a, e_par_b), -1.0, 1.0)))
        max_ab_angle_deg = max(max_ab_angle_deg, ang)
    print(f"\n  max |chord_a_e_par . chord_a_axis|: {max_perp_a:.2e} (expect ~0)")
    print(f"  max |chord_b_e_par . chord_b_axis|: {max_perp_b:.2e} (expect ~0)")
    print(f"  max angle between chord_a_e_par and chord_b_e_par at the same "
          f"connection: {max_ab_angle_deg:.3f} deg (expect small -- real "
          f"panel-point kinks, not a topology error)")
    assert max_perp_a < 1e-9 and max_perp_b < 1e-9, (
        "a segment's own target axis is not perpendicular to its own axis "
        "-- formula regression"
    )
    assert max_ab_angle_deg < 1.0, (
        "the two chord segments' target axes disagree by more than 1 deg -- "
        "too large to be a benign panel-point kink, worth a human look"
    )

    # --- Spot values, one per family, for eyeballing.
    print("\n  spot chord geometry by family (one connection each):")
    seen = set()
    for c in connections:
        if c["family"] in seen:
            continue
        seen.add(c["family"])
        print(f"    family={c['family']:>2} node={c['node']:>3} "
              f"chord_a=(mid={c['chord_a_member']:>3}, "
              f"phi={c['chord_a_phi_deg']:7.2f}, W={c['chord_a_W']:.4e})  "
              f"chord_b=(mid={c['chord_b_member']:>3}, "
              f"phi={c['chord_b_phi_deg']:7.2f}, W={c['chord_b_W']:.4e})")
        if len(seen) == 3:
            break

    print("\n  all checks passed.")
    return connections


def _self_check_rotation():
    print(f"Parsing: {DEFAULT_SD_SUM_PATH}, {sdg.DEFAULT_SD_PATH}")
    dcm_result = read_member_dcm(DEFAULT_SD_SUM_PATH)
    dcm = dcm_result["dcm"]
    model = sdg.read_subdyn_model(sdg.DEFAULT_SD_PATH)
    connections = build_connections(model)
    compute_geometry_params(connections, model)
    add_joint_axes(connections, model, dcm)

    # --- Check 1: n perpendicular to the brace-chord plane, i.e. perpendicular
    # to BOTH brace_axis and e_par, to machine precision -- and e_par itself
    # perpendicular to brace_axis (a genuine in-plane transverse direction,
    # not just "close enough").
    max_n_dot_brace = max(abs(np.dot(c["plane_normal"], c["brace_axis"])) for c in connections)
    max_n_dot_epar = max(abs(np.dot(c["plane_normal"], c["e_par"])) for c in connections)
    max_epar_dot_brace = max(abs(np.dot(c["e_par"], c["brace_axis"])) for c in connections)
    print(f"\n  max |n . brace_axis|: {max_n_dot_brace:.2e} (expect ~0 -- n is the plane normal)")
    print(f"  max |n . e_par|:      {max_n_dot_epar:.2e} (expect ~0 -- n perp e_par)")
    print(f"  max |e_par . brace_axis|: {max_epar_dot_brace:.2e} (expect ~0 -- e_par is transverse)")
    assert max_n_dot_brace < 1e-9 and max_n_dot_epar < 1e-9 and max_epar_dot_brace < 1e-9, (
        "the OPB axis (n) is not perpendicular to the brace-chord plane -- "
        "geometry regression"
    )

    # --- Check 2 (the real phi check, not the trivial magnitude one): rotate
    # the DCM's own local x/y by phi. rotate(ex,phi) must land EXACTLY on
    # e_par (e_par is direction-independent of brace_end -- see docstring
    # note below), catching a transposed DCM or a sign error in phi.
    #
    # rotate(ey,phi) ("mop_axis", the axis MOP is actually a bending stress
    # about) is a DIFFERENT story: DCM row 2 (ez) is fixed to the member's
    # own J1->J2 direction, but this module's `n` (plane_normal) is built
    # from brace_axis, the AWAY-FROM-NODE direction -- which equals ez for
    # an end-1 brace but -ez for an end-2 brace (row 2's own direction never
    # flips; "away from node" does). So mop_axis is only guaranteed to be
    # +-n, with the sign tracking brace_end, not identically n. That sign is
    # a harmless labelling artifact (which physical saddle point gets called
    # "positive"), not a physics error -- both signs are genuine points
    # perpendicular to the plane, and the 8-position hotspot sweep covers
    # both anyway. The real check is |mop_axis . n| == 1 (mop_axis truly IS
    # the plane normal, to machine precision) with the predicted sign.
    max_ex_rot_err = 0.0
    max_mop_axis_perp_err = 0.0
    sign_mismatches = 0
    for c in connections:
        R = dcm[c["brace_member"]]
        ex, ey = R[0, :], R[1, :]
        phi = math.radians(c["phi_deg"])
        ex_rot = ex * math.cos(phi) + ey * math.sin(phi)
        mop_axis = np.array(c["mop_axis"])
        max_ex_rot_err = max(max_ex_rot_err, np.max(np.abs(ex_rot - np.array(c["e_par"]))))
        dot = np.dot(mop_axis, c["plane_normal"])
        max_mop_axis_perp_err = max(max_mop_axis_perp_err, abs(abs(dot) - 1.0))
        expected_sign = 1.0 if c["brace_end"] == 1 else -1.0
        if (dot > 0) != (expected_sign > 0):
            sign_mismatches += 1
    print(f"\n  max |rotate(ex, phi) - e_par|: {max_ex_rot_err:.2e} (expect ~0)")
    print(f"  max ||mop_axis . plane_normal| - 1|: {max_mop_axis_perp_err:.2e} "
          f"(expect ~0 -- mop_axis IS the plane normal, up to a per-end sign)")
    print(f"  connections where mop_axis's sign vs plane_normal didn't match the "
          f"predicted brace_end rule: {sign_mismatches}/{len(connections)} (expect 0)")
    assert max_ex_rot_err < 1e-9, (
        "rotating the DCM's own local x by phi does not reproduce e_par -- "
        "phi's sign/formula is wrong, this is the check that would catch a "
        "transposed DCM"
    )
    assert max_mop_axis_perp_err < 1e-9, (
        "rotate(ey, phi) (mop_axis) is not perpendicular to the brace-chord "
        "plane -- this is the plan's required check and it failed"
    )
    assert sign_mismatches == 0, (
        "mop_axis's sign relative to plane_normal doesn't follow the predicted "
        "brace_end rule -- the end-dependent sign flip isn't understood/handled "
        "correctly, this is a real bug, not just a labelling choice"
    )

    # --- Check 3: resultant bending magnitude is invariant under the
    # rotation (algebraic identity of the formula, see module docstring --
    # a genuine regression test, not independent proof of phi's correctness,
    # but still worth catching a non-orthogonal coding slip).
    rng = np.random.default_rng(0)
    ipb_synth = rng.normal(size=2000)
    opb_synth = rng.normal(size=2000)
    max_mag_err = 0.0
    for phi_deg in (0.0, 30.0, 62.65, 90.0, 145.0, -73.2, 180.0, 271.4):
        mip, mop = rotate_to_joint_axes(ipb_synth, opb_synth, phi_deg)
        mag_before = np.hypot(ipb_synth, opb_synth)
        mag_after = np.hypot(mip, mop)
        max_mag_err = max(max_mag_err, np.max(np.abs(mag_after - mag_before)))
    print(f"\n  max resultant-magnitude error over synthetic signals, 8 phi values: "
          f"{max_mag_err:.2e} (expect ~0)")
    assert max_mag_err < 1e-9, "rotation is not magnitude-preserving -- formula regression"

    # --- Check 4: phi=0 must be a no-op (mip=ipb, mop=opb exactly) -- a
    # trivial but real sanity check on the formula's sign convention.
    mip0, mop0 = rotate_to_joint_axes(ipb_synth, opb_synth, 0.0)
    assert np.max(np.abs(mip0 - ipb_synth)) < 1e-12
    assert np.max(np.abs(mop0 - opb_synth)) < 1e-12
    print("  phi=0 no-op check: passed")

    # --- Spot values for one of each family, for eyeballing alongside the
    # Grasshopper JSON export.
    print("\n  spot phi_deg by family (one connection each):")
    seen = set()
    for c in connections:
        if c["family"] in seen:
            continue
        seen.add(c["family"])
        print(f"    family={c['family']:>2} node={c['node']:>3} "
              f"brace={c['brace_member']:>3} theta={c['theta_deg']:6.2f} "
              f"phi_deg={c['phi_deg']:7.2f}")
        if len(seen) == 3:
            break

    print("\n  all checks passed.")
    return connections


if __name__ == "__main__":
    _self_check_dcm()
    _self_check_connections()
    _self_check_geometry()
    _self_check_rotation()
    _self_check_chord_geometry()
