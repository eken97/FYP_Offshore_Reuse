"""
run_example.py -- end-to-end smoke test on the worked example shipped in data/.

Runs the member-track Stage 3 aggregation (Miner summation, seed averaging,
probability weighting, 25-year scaling) over the single Stage-2 histogram
cache in data/example/, then compares the result against the checked-in
expected output. No OpenFAST installation and no campaign data are needed.

WHAT THIS IS NOT
----------------
The shipped cache holds ONE of the campaign's 69 metocean bins, at ONE of
its 6 seeds. The damage numbers it produces are therefore NOT the thesis
results -- they are the arithmetic of a single condition carried through the
same code path. Treat this as a check that the pipeline runs and stays
numerically stable, not as a reproduction of any published figure. The real
campaign results are in results/, produced from all 414 runs.

Usage:
  python scripts/run_example.py            compare against the expected output
  python scripts/run_example.py --write    regenerate the expected output
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLE_STAGE2 = REPO / "data" / "example" / "stage2"
EXPECTED = REPO / "data" / "example" / "expected_stage3_member.csv"

# Columns that are floating-point and compared with a tolerance.
FLOAT_COLS = {"D", "t", "z", "D_life", "D_life_seed_std", "D_life_min",
              "D_life_max", "life_years", "worst_theta_deg"}
RTOL = 1e-9


def load_rows():
    os.environ.setdefault("OC4_STAGE2_DIR", str(EXAMPLE_STAGE2))
    sys.path.insert(0, str(REPO / "postprocessing"))
    import stage3_damage as s3

    if not EXAMPLE_STAGE2.exists():
        raise SystemExit(f"Worked-example cache not found at {EXAMPLE_STAGE2}")

    rows, meta = s3.compute_stage3(stage2_root=EXAMPLE_STAGE2, category="B1")
    return rows, meta


def write_expected(rows):
    EXPECTED.parent.mkdir(parents=True, exist_ok=True)
    with EXPECTED.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {EXPECTED.relative_to(REPO)} ({len(rows)} rows)")


def compare(rows):
    if not EXPECTED.exists():
        raise SystemExit(f"No expected output at {EXPECTED}. Run with --write first.")

    with EXPECTED.open(encoding="utf-8") as fh:
        expected = list(csv.DictReader(fh))

    if len(rows) != len(expected):
        raise SystemExit(f"FAIL: got {len(rows)} rows, expected {len(expected)}")

    problems = []
    for i, (got, want) in enumerate(zip(rows, expected)):
        for col, want_raw in want.items():
            got_val = got.get(col)
            if col in FLOAT_COLS:
                try:
                    a, b = float(got_val), float(want_raw)
                except (TypeError, ValueError):
                    if str(got_val) != want_raw:
                        problems.append(f"row {i} col {col}: {got_val!r} != {want_raw!r}")
                    continue
                if math.isnan(a) or math.isnan(b):
                    ok = math.isnan(a) and math.isnan(b)
                elif math.isinf(a) or math.isinf(b):
                    # Zero-damage points give life_years = inf; that is a real
                    # result, not a failure, so compare it exactly.
                    ok = a == b
                elif b == 0:
                    ok = a == 0
                else:
                    ok = abs(a - b) <= RTOL * abs(b)
                if not ok:
                    problems.append(f"row {i} col {col}: {a!r} != {b!r}")
            elif str(got_val) != want_raw:
                problems.append(f"row {i} col {col}: {got_val!r} != {want_raw!r}")

    if problems:
        print(f"FAIL: {len(problems)} mismatch(es); first 10:")
        for p in problems[:10]:
            print("  " + p)
        raise SystemExit(1)

    print(f"PASS: {len(rows)} rows match {EXPECTED.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="regenerate the expected output instead of comparing")
    args = parser.parse_args()

    rows, meta = load_rows()
    print(f"bins available in the example: {meta['n_bins_available']} "
          f"of {meta['n_bins_total_campaign']} -- these are NOT thesis numbers")

    if args.write:
        write_expected(rows)
    else:
        compare(rows)


if __name__ == "__main__":
    main()
