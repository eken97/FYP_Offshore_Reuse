"""
merge.py — copies a local staging folder onto the external drive.

Needed when cases were built/run while the drive wasn't attached (fell back to
_staging/) and the drive has since been reattached. Walks every condition/seed
folder actually present under staging_root, NOT just config.my_cases() —
_staging/ lives inside a cloud-synced project tree, so another machine's
staged data can genuinely show up here too (observed directly: one machine's
campaign.py --smoke staging check fell back to its own local staging, which
then synced onto the other machine's mirrored _staging/ folder). Safety doesn't depend on
scope — every case is still owner.json-checked before touching the drive, so
widening from "my cases" to "every case present" doesn't weaken anything.

Default is a dry run (shows what would happen, copies nothing). Pass --apply
to actually copy.

Run it directly:
  python merge.py             (dry run)
  python merge.py --apply
"""

import argparse
import json
import shutil
from pathlib import Path

import config


# Reads owner.json from a case folder, or None if it doesn't exist.
def read_owner(case_dir: Path) -> dict | None:
    owner_path = case_dir / "owner.json"
    if not owner_path.exists():
        return None
    return json.loads(owner_path.read_text(encoding="utf-8"))


# After copytree, confirms every file made it across with the right size —
# a cheap correctness check on a USB-drive copy, not a full checksum.
def verify_copy(src_dir: Path, dst_dir: Path) -> None:
    src_files = sorted(p.relative_to(src_dir) for p in src_dir.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(dst_dir) for p in dst_dir.rglob("*") if p.is_file())
    if src_files != dst_files:
        raise RuntimeError(f"File list mismatch after copying {src_dir} -> {dst_dir}")
    for rel in src_files:
        src_size = (src_dir / rel).stat().st_size
        dst_size = (dst_dir / rel).stat().st_size
        if src_size != dst_size:
            raise RuntimeError(f"Size mismatch for {rel}: staged={src_size}B, drive={dst_size}B")


# Finds every <condition>/<seed>/ folder actually present under staging_root
# (two levels deep, seed folders identified by owner.json — anything else at
# that depth is ignored rather than assumed to be a case). Returns (case_id,
# relative_path) pairs; case_id comes from owner.json, not from config.CASES,
# so this works even for a case this machine's own config doesn't know about.
def discover_staged_cases(staging_root: Path) -> list[tuple[str, Path]]:
    found = []
    for condition_dir in sorted(staging_root.iterdir()):
        if not condition_dir.is_dir():
            continue
        for seed_dir in sorted(condition_dir.iterdir()):
            if not seed_dir.is_dir() or not (seed_dir / "owner.json").exists():
                continue
            owner = read_owner(seed_dir)
            rel = seed_dir.relative_to(staging_root)
            found.append((owner.get("case_id", str(rel)), rel))
    return found


# Merges one staged case folder (given as a path relative to staging_root)
# onto the drive. Returns a short status string rather than raising, so one
# bad case doesn't stop the others from merging — main() collects these and
# refuses to finish quietly if any case reports an ABORT.
def merge_case(rel_path: Path, staging_root: Path, external_root: Path, apply: bool) -> str:
    src_dir = staging_root / rel_path
    dst_dir = external_root / rel_path

    if not src_dir.exists():
        return "no staged data — nothing to merge"

    if dst_dir.exists():
        src_owner = read_owner(src_dir)
        dst_owner = read_owner(dst_dir)
        if src_owner is None or dst_owner is None:
            return "ABORT: missing owner.json — refusing to touch"
        if src_owner["machine"] != dst_owner["machine"]:
            return (f"ABORT: owner mismatch (staged={src_owner['machine']}, "
                    f"drive={dst_owner['machine']}) — refusing to merge")
        return "already present, same owner — skipping"

    if not apply:
        return "would copy"

    shutil.copytree(src_dir, dst_dir)
    verify_copy(src_dir, dst_dir)
    return "copied"


# Merges one manifest_<machine>.jsonl from staging onto the drive. Any
# manifest_*.jsonl present in staging is merged, not just this machine's own —
# same reasoning as discover_staged_cases() above, a synced _staging/ folder
# can carry another machine's manifest too. If a manifest already exists on
# the drive (e.g. from an earlier direct-to-drive run), appends only the
# lines not already present rather than overwriting — nothing from either
# side is lost.
def merge_manifest(src: Path, external_root: Path, apply: bool) -> str:
    dst = external_root / src.name
    src_lines = src.read_text(encoding="utf-8").splitlines()
    dst_lines = dst.read_text(encoding="utf-8").splitlines() if dst.exists() else []
    new_lines = [line for line in src_lines if line not in dst_lines]

    if not new_lines:
        return "manifest already up to date"
    if not apply:
        return f"would add {len(new_lines)} manifest line(s)"

    combined = dst_lines + new_lines
    dst.write_text("\n".join(combined) + "\n", encoding="utf-8")
    return f"added {len(new_lines)} manifest line(s)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually copy; default is dry-run")
    args = parser.parse_args()
    apply = args.apply

    staging_root = config.PROJECT_DIR / "_staging" / config.CAMPAIGN_NAME
    drive = config.find_drive()
    if drive is None:
        raise SystemExit("External drive not attached (no .oc4_campaign_drive marker found) — nothing to merge onto.")
    external_root = drive / config.CAMPAIGN_NAME

    if not staging_root.exists():
        print(f"No local staging data at {staging_root} — nothing to merge.")
        return

    print(f"{'APPLY' if apply else 'DRY RUN'}: {staging_root}  ->  {external_root}\n")

    staged_cases = discover_staged_cases(staging_root)
    any_abort = False
    for case_id, rel_path in staged_cases:
        status = merge_case(rel_path, staging_root, external_root, apply)
        print(f"  {case_id} ({rel_path}): {status}")
        if status.startswith("ABORT"):
            any_abort = True
    if not staged_cases:
        print("  (no case folders found in staging)")

    manifest_paths = sorted(staging_root.glob("manifest_*.jsonl"))
    if not manifest_paths:
        print("\nManifest: no staged manifest")
    for manifest_path in manifest_paths:
        status = merge_manifest(manifest_path, external_root, apply)
        print(f"\nManifest ({manifest_path.name}): {status}")

    if not apply:
        print("\n(dry run — nothing was copied. Re-run with --apply to actually merge.)")

    if any_abort:
        raise SystemExit("\nOne or more cases had an owner mismatch — resolve manually before re-running.")


if __name__ == "__main__":
    main()
