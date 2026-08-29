"""
fetch_openfast_inputs.py -- stage the NREL OpenFAST input decks into inputs/.

This project does NOT redistribute NREL's models. The OC4 jacket case and the
shared NREL 5MW baseline files it depends on belong to NREL, are published as
part of the OpenFAST regression-test suite (r-test), and are licensed
Apache-2.0. This script copies them from a local r-test checkout into the
layout the campaign code expects, and verifies every file against the SHA-256
manifest in scripts/rtest_manifest.json.

Verification is the point. The manifest was generated from the exact r-test
commit the published results were produced with, so a mismatch means your
model differs from the one behind the numbers in results/ -- which is
something you want to be told loudly, not discover later.

Nothing this script writes is tracked by git: inputs/ is ignored wholesale.

Usage
-----
  # if you already have an r-test checkout
  python scripts/fetch_openfast_inputs.py --rtest C:/OpenFAST/r-test

  # or let it clone the pinned commit for you (needs git on PATH)
  python scripts/fetch_openfast_inputs.py --clone

  # check an existing inputs/ tree without copying anything
  python scripts/fetch_openfast_inputs.py --verify-only

OC4_RTEST_ROOT is used when --rtest is not given.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().parent / "rtest_manifest.json"
INPUTS = REPO / "inputs"
DEFAULT_CLONE_DIR = REPO / ".rtest"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Manifest not found at {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def clone_rtest(manifest, dest):
    """Shallow-fetch exactly the pinned commit, nothing else."""
    if (dest / ".git").exists():
        print(f"reusing existing checkout at {dest}")
    else:
        dest.mkdir(parents=True, exist_ok=True)
        print(f"cloning {manifest['upstream']} @ {manifest['commit'][:12]} into {dest}")
        subprocess.run(["git", "init", "-q", str(dest)], check=True)
        subprocess.run(["git", "-C", str(dest), "remote", "add", "origin",
                        manifest["upstream"]], check=True)
    subprocess.run(["git", "-C", str(dest), "fetch", "--depth", "1", "-q",
                    "origin", manifest["commit"]], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", manifest["commit"]],
                   check=True)
    return dest


def verify(manifest, quiet=False):
    """Check every staged file against the manifest. Returns a list of problems."""
    problems = []
    for rel, entry in manifest["files"].items():
        target = INPUTS / rel
        if not target.exists():
            problems.append(f"MISSING  {rel}")
            continue
        actual = sha256(target)
        if actual != entry["sha256"]:
            problems.append(f"MISMATCH {rel}\n"
                            f"           expected {entry['sha256']}\n"
                            f"           got      {actual}")
    if not quiet:
        n = len(manifest["files"])
        if problems:
            print(f"\n{len(problems)} of {n} file(s) failed verification:")
            for p in problems:
                print("  " + p)
        else:
            print(f"\nverified {n}/{n} files against the manifest -- "
                  f"r-test commit {manifest['commit'][:12]}")
    return problems


def stage(manifest, rtest_root):
    copied = skipped = 0
    for rel, entry in manifest["files"].items():
        src = rtest_root / entry["src"]
        dst = INPUTS / rel
        if not src.exists():
            raise SystemExit(
                f"Not found in the r-test checkout: {entry['src']}\n"
                f"  looked under: {rtest_root}\n"
                f"  Is this really an r-test checkout, and is it at commit "
                f"{manifest['commit'][:12]}?"
            )
        if dst.exists() and sha256(dst) == entry["sha256"]:
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    print(f"staged into {INPUTS.relative_to(REPO)}/: {copied} copied, "
          f"{skipped} already present and matching")


def main():
    parser = argparse.ArgumentParser(
        description="Stage NREL OpenFAST input decks into inputs/ (not redistributed here).")
    parser.add_argument("--rtest", metavar="PATH",
                        help="path to an existing OpenFAST r-test checkout "
                             "(default: $OC4_RTEST_ROOT)")
    parser.add_argument("--clone", action="store_true",
                        help=f"git-clone the pinned r-test commit into {DEFAULT_CLONE_DIR.name}/")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify the existing inputs/ tree and exit")
    args = parser.parse_args()

    manifest = load_manifest()

    if args.verify_only:
        sys.exit(1 if verify(manifest) else 0)

    if args.clone:
        rtest_root = clone_rtest(manifest, DEFAULT_CLONE_DIR)
    else:
        raw = args.rtest or os.environ.get("OC4_RTEST_ROOT")
        if not raw:
            raise SystemExit(
                "No r-test checkout given.\n"
                "  Either pass --rtest PATH, set OC4_RTEST_ROOT, or pass --clone\n"
                "  to fetch the pinned commit automatically. See docs/setup.md."
            )
        rtest_root = Path(raw)
        if not rtest_root.exists():
            raise SystemExit(f"r-test path does not exist: {rtest_root}")

    stage(manifest, rtest_root)
    problems = verify(manifest)
    if problems:
        print("\nThe staged decks do NOT match the commit these results were "
              "produced with. Re-check out the r-test at "
              f"{manifest['commit']} before running the campaign.")
        sys.exit(1)

    print("\nNext: set OC4_OPENFAST_EXE and OC4_TURBSIM_EXE, then run\n"
          "  python simulation/config.py\nto print the readiness report.")


if __name__ == "__main__":
    main()
