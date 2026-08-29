"""
rebalance.py — moves not-yet-started cases from one machine's assignment to
another, mid-campaign, when one machine is running behind pace.

Only ever touches assignment.json — never touches run data directly. A case
is only eligible to move if it hasn't been built yet on its current owner
(build.is_built()) and hasn't finished (run.is_finished()). Requires the
external drive to be attached before doing anything: eligibility needs to see
BOTH machines' real progress, and a machine's own _staging/ fallback only
ever holds its own cases, not the other machine's — so this can only be run
reliably from whichever machine currently has the drive.

Takes the least-likely-to-be-in-progress cases first: from the END of the
source machine's assignment list, since campaign.py processes its list in
order — cases near the end are the ones least likely to have been touched.

Default is a dry run. Pass --apply to actually write the change.

Run it directly:
  python rebalance.py --from machine-a --to machine-b --count 20   (dry run)
  python rebalance.py --from machine-a --to machine-b --count 20 --apply
"""

import argparse
import json

import build
import config
import run as run_mod


# Filters case_ids down to ones that haven't been built or finished yet on
# whichever machine currently owns them — safe to hand to a different machine.
def eligible_to_move(case_ids: list[str], by_id: dict) -> list[str]:
    eligible = []
    for cid in case_ids:
        case_dir = config.case_dir(by_id[cid])
        if build.is_built(case_dir) or run_mod.is_finished(case_dir):
            continue
        eligible.append(cid)
    return eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_laptop", required=True)
    parser.add_argument("--to", dest="to_laptop", required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--apply", action="store_true", help="write assignment.json; default is dry-run")
    args = parser.parse_args()

    if config.find_drive() is None:
        raise SystemExit(
            "External drive not attached — rebalance needs to see both machines' actual "
            "progress (case.json/.outb on the drive), which local staging can't provide "
            "for the other machine's cases. Attach the drive first."
        )

    assignment = config._load_assignment()
    if not assignment:
        raise SystemExit("No assignment.json found — run plan_assignment.py first.")
    if args.from_laptop not in assignment or args.to_laptop not in assignment:
        raise SystemExit(
            f"Unknown machine(s) — assignment.json only has: {list(assignment)}"
        )
    if args.from_laptop == args.to_laptop:
        raise SystemExit("--from and --to are the same machine — nothing to do.")

    by_id = {c.case_id: c for c in config.CASES}
    from_ids = assignment[args.from_laptop]
    eligible = eligible_to_move(from_ids, by_id)

    if len(eligible) < args.count:
        print(f"Only {len(eligible)} not-yet-started case(s) available on "
              f"{args.from_laptop} (out of {len(from_ids)} assigned), "
              f"requested {args.count} — moving all {len(eligible)}.")
    to_move = eligible[-args.count:]

    print(f"{'APPLY' if args.apply else 'DRY RUN'}: moving {len(to_move)} case(s) "
          f"from {args.from_laptop} to {args.to_laptop}\n")
    for cid in to_move:
        print(f"  {cid}")

    if not args.apply:
        print("\n(dry run — nothing written. Re-run with --apply to write assignment.json.)")
        return

    if not to_move:
        print("\nNothing to move.")
        return

    new_assignment = dict(assignment)
    new_assignment[args.from_laptop] = [c for c in from_ids if c not in to_move]
    new_assignment[args.to_laptop] = assignment[args.to_laptop] + to_move

    config.check_assignment(new_assignment)  # validate before writing
    config.ASSIGNMENT_FILE.write_text(json.dumps(new_assignment, indent=2), encoding="utf-8")
    print(f"\nWrote {config.ASSIGNMENT_FILE}")


if __name__ == "__main__":
    main()
