"""
plan_assignment.py — computes the per-machine case split and writes
assignment.json.

Throughput-weighted: each machine's share of the 69 bins is proportional to
(workers / hours-per-case), i.e. how many cases it can churn through per
hour. Describe each machine with --machine NAME:CORES:HOURS_PER_CASE; with
no --machine given, everything is assigned to this machine alone.

Split at the BIN level, sorted by wind speed — the faster machine takes the
low-V end working up, the other takes the high-V end working down, meeting
in the middle. NOT split at the individual-case level: an earlier
interleaved (deficit round-robin) version split some bins' 6 seeds across
BOTH machines (e.g. 2 seeds on one, 4 on the other for the same wind/wave
condition) — harmless
correctness-wise (merge.py's per-case owner check still works fine either
way) but a real practical headache, since completing any post-processing
for that one bin would mean pulling data from both machines instead of one.
Splitting whole bins avoids that entirely: every bin's full 6-seed set lands
on exactly one machine.

Default is a dry run (prints the computed split, writes nothing). Pass --apply
to write assignment.json. If a file already exists, diffs old vs new per
case_id — any case that would change owner needs --force (protects against
silently reassigning a case whose machine already meant something, e.g. mid-
campaign); new/removed case ids (the case list itself changed) are reported
but don't block.

Run it directly:
  python plan_assignment.py                             (dry run, 1 machine)
  python plan_assignment.py --machine fast:8:3.2 --machine slow:4:3.2
  python plan_assignment.py --machine fast:8:3.2 --machine slow:4:3.2 --apply
  python plan_assignment.py ... --apply --force   (only needed if an existing
                                                   case's machine would change)
"""

import argparse
import json

import config


# Parses repeatable --machine NAME:CORES:HOURS_PER_CASE specs into
# {name: (cores, hours_per_case)}. With none given, this machine alone.
def parse_machines(specs: list[str] | None) -> dict[str, tuple[int, float]]:
    if not specs:
        return {config.WORKER: (config.CORES, config.EST_HOURS_PER_CASE)}
    machines: dict[str, tuple[int, float]] = {}
    for spec in specs:
        try:
            name, cores, hours = spec.split(":")
            machines[name] = (int(cores), float(hours))
        except ValueError:
            raise SystemExit(
                f"Bad --machine spec {spec!r}. Expected NAME:CORES:HOURS_PER_CASE, "
                f"e.g. --machine fast:8:3.2"
            )
    return machines


# Groups CASES into per-bin (v_hub, hs, tp) case_id lists, sorted by v_hub
# ascending — the ordering the low-to-high / high-to-low split walks.
def group_bins() -> list[tuple[float, list[str]]]:
    bins: dict[tuple, list[str]] = {}
    for c in config.CASES:
        bins.setdefault((c.v_hub, c.hs, c.tp), []).append(c.case_id)
    return sorted(bins.items(), key=lambda item: item[0][0])  # sort by v_hub


# The faster machine gets the low-wind-speed bins working up, the other
# gets the high-wind-speed bins working down, meeting in the middle — bin
# counts sized by throughput share so wall-clock time still comes out
# balanced. Every bin's full set of case_ids goes entirely to one machine or
# the other, never split.
def compute_assignment(case_ids: list[str],
                       machines: dict[str, tuple[int, float]]) -> dict[str, list[str]]:
    bins = group_bins()  # [(v_hub, [case_ids]), ...] ascending by v_hub

    if len(machines) == 1:
        only = next(iter(machines))
        return {only: [cid for _, ids in bins for cid in ids]}
    if len(machines) != 2:
        raise SystemExit(
            f"compute_assignment handles 1 or 2 machines, got {len(machines)}. The "
            f"\"meet in the middle\" split would need rethinking past 2."
        )

    throughput = {name: cores / hours for name, (cores, hours) in machines.items()}
    total = sum(throughput.values())
    low_machine, high_machine = sorted(throughput, key=lambda m: -throughput[m])
    low_share = throughput[low_machine] / total
    n_low_bins = round(low_share * len(bins))

    # low_laptop's list is already ascending V (starts at the low end, ends at
    # the boundary). high_laptop's list is reversed (starts at the HIGH end,
    # ends at the boundary too) — both machines converge toward the boundary
    # as they work through their list, so campaign.py's processing order
    # naturally means the LEAST-touched bins on any machine are always the
    # boundary-adjacent ones. That matters for rebalance.py: it moves cases
    # from the END of a machine's list (the ones least likely started) — with
    # both lists converging on the boundary, any rebalance just nudges the
    # boundary a little rather than handing off a disconnected bin from the
    # far end, so the contiguous-by-wind-speed split stays contiguous.
    assignment: dict[str, list[str]] = {name: [] for name in throughput}
    for _, ids in bins[:n_low_bins]:
        assignment[low_machine].extend(ids)
    for _, ids in reversed(bins[n_low_bins:]):
        assignment[high_machine].extend(ids)
    return assignment


# Compares an old assignment (or {} if none existed) against a newly computed
# one. Returns (changed, added, removed) where `changed` is
# {case_id: (old_machine, new_machine)} for any case that switched machines —
# the thing --force exists to gate.
def diff_assignment(old: dict, new: dict) -> tuple[dict, set, set]:
    old_owner = {cid: lp for lp, ids in old.items() for cid in ids}
    new_owner = {cid: lp for lp, ids in new.items() for cid in ids}

    changed = {
        cid: (old_owner[cid], new_owner[cid])
        for cid in (set(old_owner) & set(new_owner))
        if old_owner[cid] != new_owner[cid]
    }
    added = set(new_owner) - set(old_owner)
    removed = set(old_owner) - set(new_owner)
    return changed, added, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write assignment.json; default is dry-run")
    parser.add_argument("--force", action="store_true",
                         help="allow already-assigned cases to change machine")
    parser.add_argument("--machine", action="append", metavar="NAME:CORES:HOURS",
                        help="a machine and its throughput; repeatable. "
                             "Default: this machine alone.")
    args = parser.parse_args()

    machines = parse_machines(args.machine)
    case_ids = [c.case_id for c in config.CASES]
    new_assignment = compute_assignment(case_ids, machines)
    config.check_assignment(new_assignment)  # validate before printing/writing anything

    by_id = {c.case_id: c for c in config.CASES}

    print(f"{'APPLY' if args.apply else 'DRY RUN'}: computed split for {len(case_ids)} cases\n")
    for name, ids in new_assignment.items():
        cores, hours_per_case = machines[name]
        wall_h = len(ids) * hours_per_case / cores
        v_range = sorted({by_id[cid].v_hub for cid in ids})
        print(f"  {name}: {len(ids)} cases ({len(ids) / len(config.CAMPAIGN_SEEDS):.0f} bins, "
              f"V={v_range[0]:g}-{v_range[-1]:g} m/s)  "
              f"~{wall_h:.0f}h wall-clock at {cores} workers / {hours_per_case}h per case")

    old_assignment = config._load_assignment()
    changed, added, removed = diff_assignment(old_assignment, new_assignment)

    if old_assignment:
        print(f"\nCompared to existing assignment.json:")
        print(f"  {len(changed)} case(s) would change machine")
        print(f"  {len(added)} new case(s) (not in the old file)")
        print(f"  {len(removed)} case(s) removed (in the old file, not in the new case list)")
        if changed and not args.force:
            print("\n  Sample of changed cases:")
            for cid, (old_m, new_m) in list(changed.items())[:10]:
                print(f"    {cid}: {old_m} -> {new_m}")

    if not args.apply:
        print("\n(dry run — nothing written. Re-run with --apply to write assignment.json.)")
        return

    if changed and not args.force:
        raise SystemExit(
            f"\n{len(changed)} case(s) would change machine — pass --force to confirm this "
            f"is intentional (e.g. after a --machine spec changed), or fix the "
            f"inputs if it wasn't expected."
        )

    config.ASSIGNMENT_FILE.write_text(json.dumps(new_assignment, indent=2), encoding="utf-8")
    print(f"\nWrote {config.ASSIGNMENT_FILE}")


if __name__ == "__main__":
    main()
