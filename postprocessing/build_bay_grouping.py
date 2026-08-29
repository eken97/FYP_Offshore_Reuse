"""
Derives the jacket's bay structure directly from SubDyn geometry -- for the
reuse classifier's Step 2 (L1, sub-assembly reuse), which needs to know
which members and joints belong to which physically liftable panel.

Method (no assumed level count or boundaries -- read off the real model):
1. Every joint level (4 joints per level, one per leg -- verified 19.08.2026,
   16 distinct Z values, 4 joints each) is a candidate boundary.
2. A level is a real BRACING boundary only if a diagonal brace (z1 != z2,
   member_class == "brace") starts or ends there -- separates the 8
   individual X/K panels from the mudline pile-transition cluster and the
   top interface-transition cluster (neither has diagonals, both excluded).
3. A level is a real LEG-SEGMENT boundary only if a leg member starts or
   ends there. Legs in this model are NOT segmented at every diagonal-brace
   level -- they run straight through the X-family joints without a break
   (confirmed 19.08.2026), so leg boundaries are a smaller set than bracing
   boundaries.
4. A bay boundary is a level that is BOTH -- the intersection of 2 and 3.
   This gives 4 bays (not 8): each one is a complete X+K panel pair with NO
   member ever needing to be split across two bays -- every diagonal and
   every leg segment fits inside exactly one bay. Simpler and more
   physically natural than the finer 8-single-panel version tried first
   (each bay there needed an artificial rule to handle legs spanning two
   panels) -- dropped in favour of this after review, 19.08.2026.

Cross-checked 19.08.2026 against docs/decisions.md known
splash-zone joint list (K-top 22/26/30/34 at z=4.378, X-upper-mid 45-48 at
z=-1.958) -- both sit inside the same bay (the jacket's middle third),
consistent with the splash zone being roughly mid-height at this water
depth.

Output: results/bay_grouping.csv, columns (entity_type, entity_id, bay_id).
"""
from pathlib import Path

import pandas as pd

import sd_geometry as sg

PROJECT = Path(__file__).resolve().parents[1]   # repo root
OUT_CSV = PROJECT / "results" / "bay_grouping.csv"


def main():
    model = sg.read_subdyn_model(sg.DEFAULT_SD_PATH)
    joints = model["joints"]
    members = model["members"]

    def z_of_joint(jid):
        return round(joints[jid][2], 3)

    bracing_levels, leg_levels = set(), set()
    for mid, m in members.items():
        z1, z2 = z_of_joint(m["j1"]), z_of_joint(m["j2"])
        cls = sg.member_class(model, mid)
        if cls == "brace" and z1 != z2:
            bracing_levels.update((z1, z2))
        elif cls == "leg":
            leg_levels.update((z1, z2))

    bay_bounds = sorted(bracing_levels & leg_levels)
    print(f"Bay boundary levels ({len(bay_bounds)}, bracing AND leg-segment boundary "
          f"both): {bay_bounds}")

    all_levels = sorted(set(round(v[2], 3) for v in joints.values()))
    excluded_levels = [z for z in all_levels if z not in bay_bounds]
    print(f"Excluded (not a bay boundary -- foundation/interface, or an X-level the "
          f"legs pass straight through): {excluded_levels}")

    bays = list(zip(bay_bounds[:-1], bay_bounds[1:]))
    print(f"\n{len(bays)} bays derived:")
    for i, (z_lo, z_hi) in enumerate(bays, start=1):
        print(f"  Bay {i}: z=[{z_lo}, {z_hi}]  (height {z_hi - z_lo:.2f} m)")

    rows = []
    n_member_unassigned = 0
    for mid, m in members.items():
        z1, z2 = z_of_joint(m["j1"]), z_of_joint(m["j2"])
        zlo, zhi = min(z1, z2), max(z1, z2)
        assigned = False
        for i, (blo, bhi) in enumerate(bays, start=1):
            if zlo >= blo - 1e-2 and zhi <= bhi + 1e-2:
                rows.append(dict(entity_type="member", entity_id=mid, bay_id=i))
                assigned = True
                break
        if not assigned:
            n_member_unassigned += 1  # foundation/interface members -- expected, not an error

    # joints: assign to every bay it bounds (top of one, bottom of the next --
    # a joint's own condition doesn't change depending which neighbour bay is
    # asking, and the middle-level joints of the old 8-panel version are now
    # simply INSIDE one bay, no sharing involved at all)
    for jid in joints:
        z = z_of_joint(jid)
        for i, (blo, bhi) in enumerate(bays, start=1):
            if blo - 1e-2 <= z <= bhi + 1e-2:
                rows.append(dict(entity_type="joint", entity_id=jid, bay_id=i))

    out = pd.DataFrame(rows).drop_duplicates().sort_values(["bay_id", "entity_type", "entity_id"])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\n{n_member_unassigned} members excluded (foundation/interface, expected -- "
          f"see module docstring), {len(out)} rows written to {OUT_CSV}")
    print(out.groupby(["bay_id", "entity_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
