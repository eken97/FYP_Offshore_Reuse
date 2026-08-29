"""
Step 2 -- SubDyn geometry + section properties.

Parses NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat directly (never imports
Validation/plot_channel_map.py -- that module has import-time side effects:
it parses a hard-coded file and prints at import, and pulls in the 46 kB
run_validation_lc module). This is a fresh, self-contained parser built on
one generic rule that holds for every table in the file:

    a banner line of dashes (sometimes absent), then a line whose FIRST
    token is an integer count and whose remaining tokens are the table's
    variable name (e.g. "64   NJoints     - Number of joints (-)"), then
    a column-name line, then a units line, then exactly `count` data rows.

Zero-count tables (e.g. NPropSetsRec=0) still have both header lines --
this generic reader handles that correctly since it only ever reads
`count` rows after the header, never scans until a non-numeric line.

THIS IS ONE OF THE TWO SIGN-OFF STEPS (see the plan's "Values you verify,
not me"). Section properties (D, t, and the derived A/I/R/W) decide every
stress number the whole pipeline produces, and a wrong value here is
invisible downstream -- the damage table looks equally plausible either
way. This script emits section_properties_CHECK.csv with every value
traced back to its exact line in SubDyn.dat, and building continues past
this step only once you've checked it against the file yourself.

Steps:
    1. Read NDiv, OutAll, OutCOSM (scalar fields), and the JOINTS,
       MEMBERS, CIRCULAR BEAM CROSS-SECTION PROPERTIES, base-reaction, and
       interface-joint tables, using the generic table rule above.
    2. Compute A, I, R, W from each propset's (D, t) via plain thin/thick
       tube formulas -- no approximation, exact circular-annulus geometry.
    3. Join members to their section properties, asserting the model is
       prismatic (MPropSetID1 == MPropSetID2) per member, since a taper
       would need a different stress formula than the one this pipeline
       uses.
    4. Emit fatigue_results/section_properties_CHECK.csv, one row per
       member, with the exact SubDyn.dat line number each value came from.
"""
import re
import math
import hashlib
import os
from pathlib import Path

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# Dev/test fixture data stays at its original location -- see
# See docs/decisions.md, 10.08.2026 folder-reorg session.
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

SD_NAME = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"
# Section properties are identical across every case (shared jacket
# geometry) -- any one run's SubDyn.dat works as the default source.


# Resolution order for "some SubDyn.dat describing this jacket":
#   1. OC4_SUBDYN_DAT, if you want to point at a specific run's copy
#   2. a campaign run folder under data/example/, if one is present
#   3. the pristine NREL deck staged by scripts/fetch_openfast_inputs.py
#
# All three describe the SAME structure. A campaign copy differs from the
# pristine r-test deck by exactly one line -- OutAll False -> True, written
# by simulation/of_inputs.py so SubDyn reports every member's end forces.
# That flag does not affect any geometry or section property parsed here.
def _resolve_default_sd() -> Path:
    override = os.environ.get("OC4_SUBDYN_DAT")
    if override:
        return Path(override)
    campaign_copy = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001" / SD_NAME
    if campaign_copy.exists():
        return campaign_copy
    return PROJECT / "inputs" / "r-test-baseline" / SD_NAME


DEFAULT_SD_PATH = _resolve_default_sd()

# Independently reconfirmed 05.08.2026 (this session, via `hashlib.md5`
# directly on the file) -- every case in the real campaign and every test
# run checked so far shares this exact SubDyn.dat. Used only as an
# informational match/mismatch report, never to silently substitute a
# cached model.
KNOWN_SD_MD5 = {
    "58258b464559f579f7b711007749cb08": "campaign deck (OutAll=True)",
    "dfd72b375b9d8047403541ba52c9da56": "pristine NREL r-test deck (OutAll=False)",
}
KNOWN_CAMPAIGN_MD5 = "58258b464559f579f7b711007749cb08"
# endregion


# region --- generic table reader ---
_COUNT_LINE_RE_CACHE = {}


def _count_line_re(varname):
    if varname not in _COUNT_LINE_RE_CACHE:
        _COUNT_LINE_RE_CACHE[varname] = re.compile(
            r"^\s*(\d+)\s+" + re.escape(varname) + r"\b"
        )
    return _COUNT_LINE_RE_CACHE[varname]


def _find_count_line(lines, varname):
    """Return (line_index, count) for the line '<count>   <varname> ...'."""
    pat = _count_line_re(varname)
    for i, line in enumerate(lines):
        m = pat.match(line)
        if m:
            return i, int(m.group(1))
    raise ValueError(f"{varname!r} count line not found")


def _read_rows(lines, count_line_idx, count):
    """
    Read `count` data rows starting 2 lines after the count line (skipping
    the column-name line and the units line). Returns a list of (row_no,
    file_line_no, parts) -- file_line_no is 1-based, matching a text
    editor's line numbers, for traceability into section_properties_CHECK.csv.
    """
    start = count_line_idx + 3
    rows = []
    for k in range(count):
        file_line_no = start + k + 1  # 1-based
        raw = lines[start + k].split("!")[0]  # strip inline "! M1" comments
        parts = raw.split()
        rows.append((k, file_line_no, parts))
    return rows


def _scalar(lines, varname):
    """Read a single scalar value field: '<value>   <varname> ...'."""
    pat = re.compile(r"^\s*(\S+)\s+" + re.escape(varname) + r"\b")
    for line in lines:
        m = pat.match(line)
        if m:
            return m.group(1)
    raise ValueError(f"{varname!r} scalar line not found")
# endregion


# region --- section property formulas ---
def section_properties(D, t):
    """
    Circular hollow section, exact (not thin-wall-approximated) formulas.

        d = D - 2t                          inner diameter
        A = pi*(D^2 - d^2) / 4               cross-sectional area
        I = pi*(D^4 - d^4) / 64              second moment of area
        R = D / 2                            outer radius (weld toe location)
        W = I / R                            section modulus

    Returns a dict: D, t, d, A, I, R, W.
    """
    d = D - 2 * t
    A = math.pi * (D ** 2 - d ** 2) / 4
    I = math.pi * (D ** 4 - d ** 4) / 64
    R = D / 2
    W = I / R
    return dict(D=D, t=t, d=d, A=A, I=I, R=R, W=W)
# endregion


# region --- SubDyn.dat parser ---
def read_subdyn_model(path):
    """
    Parse a SubDyn.dat into a dict:
        md5, ndiv, out_all, out_cosm,
        joints        {jid: (x, y, z)}
        joint_line    {jid: file line number}
        members       {mid: dict(j1, j2, propset1, propset2, mtype)}
        member_line   {mid: file line number}
        circ_props    {propset_id: dict(E, G, rho, D, t)}
        propset_line  {propset_id: file line number}
        reaction_joints   [jid, ...]
        interface_joints  [jid, ...]
    """
    path = Path(path)
    raw_bytes = path.read_bytes()
    md5 = hashlib.md5(raw_bytes).hexdigest()
    lines = raw_bytes.decode("utf-8", errors="replace").splitlines()

    ndiv = int(_scalar(lines, "NDiv"))
    out_all = _scalar(lines, "OutAll").strip().lower() == "true"
    out_cosm = _scalar(lines, "OutCOSM").strip().lower() == "true"

    # JOINTS: JointID JointXss JointYss JointZss JointType JointDirX JointDirY JointDirZ JointStiff
    i, n = _find_count_line(lines, "NJoints")
    joints, joint_line = {}, {}
    for _, ln, parts in _read_rows(lines, i, n):
        jid = int(parts[0])
        joints[jid] = (float(parts[1]), float(parts[2]), float(parts[3]))
        joint_line[jid] = ln

    # MEMBERS: MemberID MJointID1 MJointID2 MPropSetID1 MPropSetID2 MType MSpin/COSMID
    i, n = _find_count_line(lines, "NMembers")
    members, member_line = {}, {}
    for _, ln, parts in _read_rows(lines, i, n):
        mid = int(parts[0])
        members[mid] = dict(
            j1=int(parts[1]), j2=int(parts[2]),
            propset1=int(parts[3]), propset2=int(parts[4]),
            mtype=parts[5],  # STRING, e.g. "1c" -- not an int
        )
        member_line[mid] = ln

    # CIRCULAR BEAM CROSS-SECTION PROPERTIES:
    # PropSetID YoungE ShearG MatDens XsecD XsecT
    i, n = _find_count_line(lines, "NPropSetsCyl")
    circ_props, propset_line = {}, {}
    for _, ln, parts in _read_rows(lines, i, n):
        pid = int(parts[0])
        circ_props[pid] = dict(
            E=float(parts[1]), G=float(parts[2]), rho=float(parts[3]),
            D=float(parts[4]), t=float(parts[5]),
        )
        propset_line[pid] = ln

    # BASE REACTION JOINTS: RJointID ...
    i, n = _find_count_line(lines, "NReact")
    reaction_joints = [int(parts[0]) for _, _, parts in _read_rows(lines, i, n)]

    # INTERFACE JOINTS: IJointID ...
    i, n = _find_count_line(lines, "NInterf")
    interface_joints = [int(parts[0]) for _, _, parts in _read_rows(lines, i, n)]

    return dict(
        md5=md5, ndiv=ndiv, out_all=out_all, out_cosm=out_cosm,
        joints=joints, joint_line=joint_line,
        members=members, member_line=member_line,
        circ_props=circ_props, propset_line=propset_line,
        reaction_joints=reaction_joints, interface_joints=interface_joints,
    )


def member_section(model, mid):
    """
    Return (D, t, propset_id) for a member, asserting it is prismatic
    (MPropSetID1 == MPropSetID2). A tapered member would need a different
    stress formula than the one stress.py (Step 4) uses -- fail loudly
    rather than silently pick one end's property.
    """
    m = model["members"][mid]
    p1, p2 = m["propset1"], m["propset2"]
    assert p1 == p2, (
        f"member {mid}: not prismatic (MPropSetID1={p1} != MPropSetID2={p2}) "
        f"-- stress.py assumes a constant section along the member length"
    )
    props = model["circ_props"][p1]
    return props["D"], props["t"], p1


def member_end_z(model, mid, end):
    """z-coordinate (global) of a member's end. end: 1 (J1/start) or 2 (J2/end)."""
    m = model["members"][mid]
    jid = m["j1"] if end == 1 else m["j2"]
    return model["joints"][jid][2]


# Same threshold joint_geometry.py's own D_LEG_MIN already uses to split
# leg-diameter from brace-diameter members at a node (a clean gap: braces
# are all D=0.8, every leg/pile propset is D>=1.2 -- see that module's own
# comment). Promoted here so the member track's corrosion step (which needs
# a whole-MEMBER leg/brace classification, not a per-node one) reuses the
# identical, already-established number rather than inventing a second one.
LEG_DIAMETER_MIN_M = 1.0


def member_class(model, mid):
    """
    'leg' or 'brace' for a whole member, by its (prismatic) section diameter
    against LEG_DIAMETER_MIN_M. Includes the pile/leg-transition members
    (D=2.082, z<mudline) as 'leg' -- structurally a continuation of the leg
    below the mudline -- though they never matter for corrosion since they
    are always 'buried' (see member_zone), never splash-zone.
    """
    D, _t, _pid = member_section(model, mid)
    return "leg" if D >= LEG_DIAMETER_MIN_M else "brace"
# endregion


# region --- section-loss corrosion: thickness-loss function ---
# Final rule, confirmed against the UpWind Design Basis
# directly, 14.08.2026 (SUPERSEDES the earlier "both-sides everywhere at the
# full 0.3mm/yr rate" version -- see docs/decisions.md for the full
# reasoning): the Design Basis's standard rate is 0.3 mm/yr per surface, but
# it separately states the corrosion allowance MAY BE HALVED for fatigue
# design specifically (as opposed to extreme/ULS design) -- so the rate
# actually applied per surface is 0.15 mm/yr, not 0.3. Legs are flooded
# (both internal and external surfaces corrode); braces are not flooded
# (external surface only).
#
# Splash-zone members only -- caller's responsibility to gate on zone via
# member_zone()/environment_zone(); this function does not check zone
# itself since it doesn't know which run's mudline_z to use.
CORROSION_RATE_MM_PER_YEAR_PER_SURFACE = {
    "leg": dict(ext=0.15, int=0.15),    # flooded: both surfaces
    "brace": dict(ext=0.15, int=0.0),   # not flooded: external only
}

# The Design Basis's GENERAL corrosion rate -- 0.30 mm/yr per surface, i.e.
# the full (un-halved) allowance. The halved 0.15 rate above is permitted
# for FATIGUE design specifically; every non-fatigue check (static/ULS
# capacity, section-loss / L0-L3 classification, mass balance) must use this
# general rate instead. Pass it as `rates=` to corroded_section(). Same
# ext/int split as the fatigue dict: legs flooded (both surfaces), braces
# external only. Mirrors stage4_reuse_classification.py's own
# GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE (kept in sync).
GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE = {
    "leg": dict(ext=0.30, int=0.30),
    "brace": dict(ext=0.30, int=0.0),
}


def corroded_section(model, mid, year, rates=None):
    """
    (D_corroded, t_corroded) in metres for member `mid` after `year` years
    of exposure.

    `rates` selects the per-surface mm/yr rate table. Default (None) uses
    the fatigue-design halved rate CORROSION_RATE_MM_PER_YEAR_PER_SURFACE
    -- correct for the S-N/Miner fatigue pipeline, which is what every
    existing caller wants, so their behaviour is unchanged. Non-fatigue
    checks pass rates=GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE for
    the full 0.30 mm/yr/surface Design Basis rate.

    D and t do NOT lose the same amount in general -- only D depends on the
    outer radius receding (external corrosion), while t depends on BOTH
    surfaces. For legs (both surfaces corrode at the same 0.15 mm/yr rate)
    this happens to make D-loss == t-loss numerically. For braces (external
    only) it does NOT: D loses 2x the external rate (diameter = 2x outer
    radius, and only the outer radius is moving), while t loses only the
    external rate itself (the inner radius is fixed, nothing corroding it).
    Getting this wrong would corrupt R=D/2 (the actual weld-toe / bending-
    stress location in section_properties()), not just the wall thickness --
    this is why the two surfaces are tracked separately rather than as one
    lumped "wall loss" scalar.

        ext_loss = rate['ext'] * year   (mm)
        int_loss = rate['int'] * year   (mm)
        D' = D0 - 2*ext_loss/1000        (only the outer radius moves)
        t' = t0 - (ext_loss+int_loss)/1000   (both radii's movement, summed)
    """
    D0, t0, _pid = member_section(model, mid)
    cls = member_class(model, mid)
    rate_table = CORROSION_RATE_MM_PER_YEAR_PER_SURFACE if rates is None else rates
    rate = rate_table[cls]
    ext_loss_m = (rate["ext"] * year) / 1000.0
    int_loss_m = (rate["int"] * year) / 1000.0
    D = D0 - 2 * ext_loss_m
    t = t0 - (ext_loss_m + int_loss_m)
    assert t > 0, (
        f"member {mid} ({cls}): corroded thickness {t*1000:.2f}mm <= 0 at "
        f"year {year} (t0={t0*1000:.1f}mm) -- section fully consumed, "
        f"check the year range being requested"
    )
    return D, t
# endregion


# region --- environment zone ---
# Splash zone extent for the K13 site, resolved 05.08.2026. Source: "UpWind
# Design Basis - K13 Deep Water Site", Annex C.4.3 "Water levels" (the same
# document OC4_K13_scatter_data.xlsx's own ReadMe cites as the source of
# the V-Hs-Tp scatter table, Annex C, pages 122-128 -- same Annex, water
# levels precede the scatter diagrams within it). Formula per that
# document's own citation to DNV ("According to DNV [8]"):
#
#   SZ_upper = HAT + 0.6 * (1/3) * Hmax(100yr)
#   SZ_lower = LAT - 0.4 * (1/3) * Hmax(100yr)
#
# Site inputs, read directly off the document's Table 45 and Section C.4.3
# text (measured water levels, 22-year record; extreme value from the same
# analysis): HAT = +1.16 m MSL, LAT = -1.06 m MSL, Hmax(100yr) = 18.41 m.
# Computed here from those three inputs, not hardcoded as SZ_upper/SZ_lower
# literals, so the derivation is visible and self-checking -- verified
# independently (05.08.2026) to reproduce the document's own stated result
# (+4.84 m / -3.51 m) to the precision printed there.
#
# SubDyn's z=0 is MSL (OC4 model description, Vorpahl et al. 2013: "The
# origin lies at MSL in the centerline of the tower"), so these apply
# directly with no offset.
K13_HAT_M = 1.16
K13_LAT_M = -1.06
K13_HMAX_100YR_M = 18.41

ZONE_SPLASH_ZMIN = K13_LAT_M - 0.4 * (1 / 3) * K13_HMAX_100YR_M
ZONE_SPLASH_ZMAX = K13_HAT_M + 0.6 * (1 / 3) * K13_HMAX_100YR_M

# Worst-to-best severity for S-N curve selection (see fatigue-postpro-design
# memory): splash (free corrosion, no S-N knee) is worst; seawater-with-CP
# (submerged) is next; atmospheric (in-air) and buried (no electrolyte
# cycling) are both treated as the best tier. Higher number = worse.
ZONE_SEVERITY = {"splash": 3, "submerged": 2, "atmospheric": 1, "buried": 1}


def environment_zone(z, mudline_z, splash_zmin=ZONE_SPLASH_ZMIN, splash_zmax=ZONE_SPLASH_ZMAX):
    """
    Classify a single elevation z (global, MSL=0) into one of:
    'buried', 'submerged', 'splash', 'atmospheric'.

    A single-point classifier -- used for reporting each member's two
    endpoint zones, NOT for deciding which S-N curve a member gets (see
    member_zone below for why a member needs more than its endpoints).
    """
    if z <= mudline_z:
        return "buried"
    if z < splash_zmin:
        return "submerged"
    if z <= splash_zmax:
        return "splash"
    return "atmospheric"


def member_zone(model, mid, mudline_z, splash_zmin=ZONE_SPLASH_ZMIN, splash_zmax=ZONE_SPLASH_ZMAX):
    """
    Classify a MEMBER (not a point) by the worst-severity zone touched
    ANYWHERE along its length, per the author's decision (05.08.2026): if
    any part of the member sits in a worse zone, the whole member is
    treated as that zone.

    This is NOT the same as max(environment_zone(z1), environment_zone(z2))
    on the two endpoints alone. The zone bands are ordered by elevation as
    buried < submerged < splash < atmospheric, but SEVERITY is not
    monotonic in elevation -- splash sits physically BETWEEN submerged and
    atmospheric while being the worst of the three. A member can have one
    end in 'submerged' and the other in 'atmospheric' (neither endpoint
    classified as 'splash') while its length still passes straight through
    the entire splash band in between -- endpoint-only logic would miss
    that silently. This function checks the member's full elevation
    interval against every zone band's range, not just its two endpoints.

    Returns (worst_zone, sorted list of all zones touched).
    """
    z1 = member_end_z(model, mid, 1)
    z2 = member_end_z(model, mid, 2)
    z_lo, z_hi = min(z1, z2), max(z1, z2)

    bands = [
        ("buried", -math.inf, mudline_z),
        ("submerged", mudline_z, splash_zmin),
        ("splash", splash_zmin, splash_zmax),
        ("atmospheric", splash_zmax, math.inf),
    ]
    touched = [name for name, b_lo, b_hi in bands if z_hi >= b_lo and z_lo <= b_hi]
    assert touched, f"member {mid}: z-range [{z_lo}, {z_hi}] touched no zone band"
    worst = max(touched, key=lambda name: ZONE_SEVERITY[name])
    return worst, touched
# endregion


# region --- CSV sign-off report ---
def write_section_properties_check(model, out_path):
    """
    One row per member: propset_id, both traceable line numbers, D, t,
    and the derived A/I/R/W -- the sign-off artifact for this step.
    """
    mudline_z = min(z for _, _, z in model["joints"].values())

    rows = []
    for mid in sorted(model["members"]):
        D, t, pid = member_section(model, mid)
        props = section_properties(D, t)
        m = model["members"][mid]
        z1 = member_end_z(model, mid, 1)
        z2 = member_end_z(model, mid, 2)
        worst_zone, touched = member_zone(model, mid, mudline_z)
        rows.append(dict(
            member_id=mid,
            j1=m["j1"], j2=m["j2"],
            mtype=m["mtype"],
            propset_id=pid,
            member_line_no=model["member_line"][mid],
            propset_line_no=model["propset_line"][pid],
            D=props["D"], t=props["t"], d_inner=props["d"],
            A=props["A"], I=props["I"], R=props["R"], W=props["W"],
            z1=z1, z2=z2,
            zone_end1=environment_zone(z1, mudline_z),
            zone_end2=environment_zone(z2, mudline_z),
            zone_WORST=worst_zone,
            zones_touched="+".join(sorted(touched)),
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
    print(f"Parsing: {DEFAULT_SD_PATH}")
    model = read_subdyn_model(DEFAULT_SD_PATH)

    print(f"  md5 = {model['md5']}")
    print(f"  known deck: {KNOWN_SD_MD5.get(model['md5'], 'UNRECOGNISED -- geometry may differ')}")
    print(f"  NDiv={model['ndiv']}  OutAll={model['out_all']}  "
          f"OutCOSM={model['out_cosm']}")
    print(f"  {len(model['joints'])} joints, {len(model['members'])} members, "
          f"{len(model['circ_props'])} circular propsets")
    print(f"  {len(model['reaction_joints'])} reaction joints: "
          f"{sorted(model['reaction_joints'])}")
    print(f"  {len(model['interface_joints'])} interface joints: "
          f"{sorted(model['interface_joints'])}")

    member_ids_sorted = sorted(model["members"])
    print(f"  MemberIDs sequential 1..{len(member_ids_sorted)}: "
          f"{member_ids_sorted == list(range(1, len(member_ids_sorted) + 1))}")

    prismatic = [mid for mid, m in model["members"].items()
                 if m["propset1"] == m["propset2"]]
    print(f"  prismatic members (PropSetID1 == PropSetID2): "
          f"{len(prismatic)}/{len(model['members'])}")

    mtypes = {m["mtype"] for m in model["members"].values()}
    print(f"  MType values present: {mtypes}")

    print("\n  Circular propsets:")
    for pid in sorted(model["circ_props"]):
        props = model["circ_props"][pid]
        sp = section_properties(props["D"], props["t"])
        members_using = sorted(mid for mid, m in model["members"].items()
                                if m["propset1"] == pid)
        span = f"{members_using[0]}-{members_using[-1]}" if members_using else "-"
        print(f"    PropSet {pid}  D={props['D']:.3f} t={props['t']:.3f}  "
              f"members {span} (n={len(members_using)})  "
              f"A={sp['A']:.6f} m2  I={sp['I']:.8f} m4  R={sp['R']:.3f}  "
              f"W={sp['W']:.8f} m3")

    # Hand-checkable example: PropSet 1, D=0.8, t=0.02. No "expected" value
    # is printed here on purpose -- work it out on a calculator yourself
    # and compare, rather than checking this output against a number that
    # itself came from this same codebase (an earlier draft of the build
    # plan had a wrong value here -- 0.00354453 for I instead of the
    # correct 0.00372957 -- precisely because it wasn't independently
    # recomputed at the time).
    sp = section_properties(0.8, 0.02)
    print(f"\n  Calculator check (PropSet 1, D=0.8, t=0.02) -- work these out")
    print(f"  yourself and compare, don't trust this printout alone:")
    print(f"    d = D - 2t = 0.8 - 2*0.02 = {0.8 - 2*0.02}")
    print(f"    A = pi*(D^2-d^2)/4 = pi*(0.8^2-0.76^2)/4 = {sp['A']:.8f} m2")
    print(f"    I = pi*(D^4-d^4)/64 = pi*(0.8^4-0.76^4)/64 = {sp['I']:.8f} m4")
    print(f"    R = D/2 = 0.8/2 = {sp['R']} m")
    print(f"    W = I/R = {sp['I']:.8f}/{sp['R']} = {sp['W']:.8f} m3")

    z_all = [z for _, _, z in model["joints"].values()]
    print(f"\n  z range: {min(z_all):.3f} .. {max(z_all):.3f}")

    out_path = RESULTS_DIR / "section_properties_CHECK.csv"
    rows = write_section_properties_check(model, out_path)
    print(f"\n  wrote {out_path}  ({len(rows)} rows)")

    zone_counts = {}
    for r in rows:
        zone_counts[r["zone_WORST"]] = zone_counts.get(r["zone_WORST"], 0) + 1
    print(f"  per-member worst-zone census (112 members, K13 site-specific thresholds (UpWind Design Basis Annex C.4.3)): "
          f"{zone_counts}")

    class_counts = {}
    for mid in model["members"]:
        cls = member_class(model, mid)
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"  per-member leg/brace census (LEG_DIAMETER_MIN_M={LEG_DIAMETER_MIN_M}): {class_counts}")
    assert class_counts == {"brace": 68, "leg": 44}, (
        f"leg/brace census {class_counts} != expected {{'brace': 68, 'leg': 44}} -- "
        f"either the SubDyn.dat propset table changed or LEG_DIAMETER_MIN_M no longer "
        f"cleanly separates the two propset families, check section_properties_CHECK.csv"
    )

    splash_legs = sum(1 for r in rows
                       if r["zone_WORST"] == "splash"
                       and member_class(model, r["member_id"]) == "leg")
    splash_braces = sum(1 for r in rows
                         if r["zone_WORST"] == "splash"
                         and member_class(model, r["member_id"]) == "brace")
    print(f"  splash-zone members by class: {splash_legs} leg(s), {splash_braces} brace(s) "
          f"(expect 32 total, corrosion track's actual scope)")
    assert splash_legs + splash_braces == 32

    # Does the "worst zone touched anywhere along the member" logic ever
    # actually differ from a naive max-of-endpoints check, for THESE
    # placeholder thresholds and THIS jacket's geometry? Report it either
    # way -- a "no" here is informative (confirms no member currently spans
    # clean across the whole splash band) but does not make the general
    # interval-overlap logic optional, since real thresholds may differ.
    naive_disagreements = [
        r for r in rows
        if r["zone_WORST"] not in (r["zone_end1"], r["zone_end2"])
    ]
    print(f"  members where the worst zone touched is NEITHER endpoint's zone "
          f"(the case naive endpoint-only logic would miss): {len(naive_disagreements)}")
    for r in naive_disagreements:
        print(f"    member {r['member_id']}: ends={r['zone_end1']}/"
              f"{r['zone_end2']}  worst={r['zone_WORST']}  "
              f"touched={r['zones_touched']}")

    print("\n" + "=" * 78)
    print("SIGN-OFF REQUIRED: open section_properties_CHECK.csv and verify D, t,")
    print("A, I, R, W for each PropSet against SubDyn.dat yourself before Step 3.")
    print("=" * 78)


if __name__ == "__main__":
    _self_check()
