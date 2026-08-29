"""
Bin-range QA scan (supporting tool, not one of the 11 numbered pipeline
steps -- tied to Step 6's frozen bin range assumption).

Scans real runs and records the minimum and maximum rainflow cycle range
actually observed, across every SCREENED member end and theta, so
fatigue_config.BIN_LO_MPA/BIN_HI_MPA can eventually be set from real data
rather than a guessed decade range. Members in
fatigue_config.SCREENING_EXCLUDED_MEMBER_IDS are skipped entirely here --
see that constant's docstring for why this does NOT affect the real
Stage 2/3 pipeline, which still processes all 112.

THIS IS A FIRST-PASS CHECK ONLY: the 414-run real campaign is still in
progress. Only what's on disk right now can be scanned -- one complete
real bin (LC_V20_H3p5_T8, 6 seeds) plus 18 older smoke-test runs across 7
other conditions. Re-run this script once the full campaign finishes for
the real answer; treat today's numbers as provisional.

Performance note: all 672 fatigue channels for a run (112 members x 2
ends x 3 components) are read in ONE memmap read, not per member-end --
Step 4's compute_member_stress rereads channels per call, fine for one-off
use but would reopen the file up to 224 times per run here.

Reports TWO minimums per run:
    min_range_mpa_all        -- literal global minimum, every member
    min_range_mpa_screened   -- minimum over screened (non-excluded)
                                 members only -- the meaningful number,
                                 since the excluded members' "cycles" are
                                 near-zero numerical noise on a near-rigid
                                 signal, not real small stress cycles.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/

import numpy as np
import rainflow

import fatigue_config as cfg
import outb_reader as obr
import sd_geometry as sdg
import stress

PROJECT = Path(__file__).resolve().parents[1]   # repo rootPOSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
# Dev/test fixture data stays at its original location -- see
# See docs/decisions.md, 10.08.2026 folder-reorg session.
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

REAL_BIN_ROOT = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8"
OLD_TEST_ROOT = DEV_FIXTURE_DIR / "_staging" / "TestScenario"

OUTB_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb"
SD_NAME = "NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat"

ALL_MEMBER_IDS = list(range(1, 113))
SCREENED_MEMBER_IDS = [m for m in ALL_MEMBER_IDS if m not in cfg.SCREENING_EXCLUDED_MEMBER_IDS]


def _discover_cases(root):
    return sorted(p.parent for p in Path(root).rglob("owner.json"))


def scan_run(case_dir):
    """
    One run -> dict of case metadata + observed min/max cycle range (MPa).
    Returns None if the run didn't complete normally (skip, don't crash
    the sweep -- e.g. the known-corrupt LC_V23_H4p0_T10/S654321 fixture).
    """
    case_dir = Path(case_dir)
    outb_path = case_dir / OUTB_NAME
    sd_path = case_dir / SD_NAME
    case_json_path = case_dir / "case.json"

    try:
        header = obr.read_outb_header(outb_path)  # raises on truncated/corrupt file
    except AssertionError:
        print(f"  SKIP (crashed/corrupt): {case_dir.parent.name}/{case_dir.name}")
        return None

    duration_s = (header["n_t"] - 1) * header["t_incr"]
    if duration_s <= stress.TRANSIENT_CUTOFF_S:
        print(f"  SKIP (duration {duration_s:.1f}s <= TRANSIENT_CUTOFF_S="
              f"{stress.TRANSIENT_CUTOFF_S}s, nothing left after trim): "
              f"{case_dir.parent.name}/{case_dir.name}")
        return None

    case_json = json.loads(case_json_path.read_text()) if case_json_path.exists() else {}
    model = sdg.read_subdyn_model(sd_path)

    # ONE bulk read for every fatigue channel of every member -- see module
    # docstring on why this matters for performance.
    names = obr.member_end_channels(ALL_MEMBER_IDS, obr.FATIGUE_COMPONENTS)
    t_full, arr = obr.read_channels(outb_path, header, names)
    n_comp = len(obr.FATIGUE_COMPONENTS)

    max_all = -np.inf
    max_loc = None
    min_all = np.inf
    min_all_loc = None
    min_screened = np.inf
    min_screened_loc = None

    for i, mid in enumerate(ALL_MEMBER_IDS):
        D, wall_t, pid = sdg.member_section(model, mid)
        is_screened = mid in cfg.SCREENING_EXCLUDED_MEMBER_IDS
        if is_screened:
            continue  # skip degenerate/not-of-interest members entirely
        for e_idx, end in enumerate((1, 2)):
            base = (i * 2 + e_idx) * n_comp
            N, Mkx, Mky = arr[:, base], arr[:, base + 1], arr[:, base + 2]
            t_trim, sigma = stress.member_end_stress_history(t_full, N, Mkx, Mky, D, wall_t)
            for k_theta in range(sigma.shape[1]):
                for rng, mean, count, i_s, i_e in rainflow.extract_cycles(sigma[:, k_theta]):
                    if rng <= 0:
                        continue
                    loc = (mid, end, k_theta)
                    if rng > max_all:
                        max_all, max_loc = rng, loc
                    if rng < min_all:
                        min_all, min_all_loc = rng, loc
                    if rng < min_screened:
                        min_screened, min_screened_loc = rng, loc

    return dict(
        case_dir=f"{case_dir.parent.name}/{case_dir.name}",
        case_id=case_json.get("case_id", ""),
        mode=case_json.get("mode", ""),
        v_hub=case_json.get("v_hub", float("nan")),
        hs=case_json.get("hs", float("nan")),
        tp=case_json.get("tp", float("nan")),
        max_range_mpa=max_all,
        max_loc_member=max_loc[0] if max_loc else None,
        max_loc_end=max_loc[1] if max_loc else None,
        max_loc_theta_idx=max_loc[2] if max_loc else None,
        min_range_mpa_screened=min_screened,
        min_loc_member=min_screened_loc[0] if min_screened_loc else None,
        min_loc_end=min_screened_loc[1] if min_screened_loc else None,
        min_loc_theta_idx=min_screened_loc[2] if min_screened_loc else None,
    )


def sweep(roots):
    rows = []
    for root in roots:
        for case_dir in _discover_cases(root):
            t0 = time.time()
            row = scan_run(case_dir)  # prints its own reason if skipped
            dt = time.time() - t0
            if row is None:
                continue
            row["source"] = "real_campaign" if root == REAL_BIN_ROOT else "old_smoke_test"
            row["scan_seconds"] = dt
            rows.append(row)
            print(f"  {row['case_dir']}: max={row['max_range_mpa']:.2f} MPa  "
                  f"min_screened={row['min_range_mpa_screened']:.2e} MPa  ({dt:.1f}s)")
    return rows


def write_csv(rows, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def plot_range_check(rows, out_path):
    """
    Point-cloud plot, NOT one labelled x-position per run -- meant to scale
    to all 414 once the campaign finishes, and 414 case-name x-tick labels
    would be unreadable regardless of sorting. Every run gets ONE shared
    jittered x-position, used for BOTH its max_range_mpa and its
    min_range_mpa_screened point -- so a given run's two points sit
    directly above/below each other in the same column, and the two
    quantities read as two separate horizontal bands (upper = max, lower =
    min) rather than two unrelated clouds. Log-y, with
    BIN_LO_MPA/BIN_HI_MPA drawn as horizontal reference lines -- the two
    numbers this whole scan exists to sanity-check.

    No real-vs-old-smoke-test marker split: the old smoke-test fixtures
    (30-60s) are always too short for TRANSIENT_CUTOFF_S=100s and get
    skipped before reaching this function (see scan_run), so that
    distinction never actually has anything to plot -- dropped rather than
    carried as dead legend entries.

    Jitter uses a fixed seed so re-plotting the same data doesn't shuffle
    point positions between runs of this script.
    """
    import matplotlib.pyplot as plt

    max_vals = np.array([r["max_range_mpa"] for r in rows])
    min_vals = np.array([r["min_range_mpa_screened"] for r in rows])

    rng = np.random.default_rng(0)
    x = rng.uniform(-0.15, 0.15, size=len(rows))

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.axhline(cfg.BIN_HI_MPA, color="firebrick", ls="--", lw=1.2, zorder=1)
    ax.axhline(cfg.BIN_LO_MPA, color="steelblue", ls="--", lw=1.2, zorder=1)
    ax.text(0.65, cfg.BIN_HI_MPA, f"BIN_HI_MPA = {cfg.BIN_HI_MPA:g}",
            color="firebrick", va="bottom", ha="left", fontsize=9)
    ax.text(0.65, cfg.BIN_LO_MPA, f"BIN_LO_MPA = {cfg.BIN_LO_MPA:g}",
            color="steelblue", va="top", ha="left", fontsize=9)

    ax.scatter(x, max_vals, marker="o", color="darkorange", alpha=0.6,
               label="max range per run", zorder=3)
    ax.scatter(x, min_vals, marker="o", color="teal", alpha=0.6,
               label="min range per run (screened members)", zorder=3)

    ax.set_yscale("log")
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([])
    ax.set_ylabel("rainflow cycle range (MPa)")
    ax.set_title(f"Observed cycle-range extremes vs. bin cutoffs\n"
                 f"({len(rows)} runs, {len(SCREENED_MEMBER_IDS)}/{len(ALL_MEMBER_IDS)} members "
                 f"screened -- provisional)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=1, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _self_check():
    outb_path = (DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001" /
                 "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb")

    print(f"screening {len(SCREENED_MEMBER_IDS)}/{len(ALL_MEMBER_IDS)} members "
          f"(excluded: {cfg.SCREENING_EXCLUDED_MEMBER_IDS})\n")

    print(f"timing one real 700s run ({outb_path.parent.name})...")
    t0 = time.time()
    row = scan_run(outb_path.parent)
    dt = time.time() - t0
    print(f"  took {dt:.1f} s")
    print(f"  max_range_mpa={row['max_range_mpa']:.3f}  "
          f"(member {row['max_loc_member']} end {row['max_loc_end']} "
          f"theta_idx {row['max_loc_theta_idx']})")
    print(f"  min_range_mpa_screened={row['min_range_mpa_screened']:.3e}  "
          f"(member {row['min_loc_member']} end {row['min_loc_end']} "
          f"theta_idx {row['min_loc_theta_idx']})")

    n_real_runs_available = len(_discover_cases(REAL_BIN_ROOT))
    n_old_runs_available = len(_discover_cases(OLD_TEST_ROOT))
    print(f"\ncurrently on disk: {n_real_runs_available} real 700s runs, "
          f"{n_old_runs_available} old 30-60s smoke runs")
    print(f"extrapolated time for currently-available data (real runs at "
          f"{dt:.1f}s each, old runs much shorter): "
          f"~{dt * n_real_runs_available / 60:.1f} min for the real runs alone")
    print(f"extrapolated time for the full 414-run campaign (all ~700s): "
          f"~{dt * 414 / 3600:.2f} h single-core")

    print(f"\nfull sweep over everything currently on disk:")
    rows = sweep([REAL_BIN_ROOT, OLD_TEST_ROOT])
    print(f"\n{len(rows)} runs scanned")

    out_csv = RESULTS_DIR / "bin_range_check.csv"
    write_csv(rows, out_csv)
    print(f"wrote {out_csv}")

    global_max = max(r["max_range_mpa"] for r in rows)
    global_min = min(r["min_range_mpa_screened"] for r in rows)
    print(f"\nGLOBAL max_range_mpa across all scanned runs: {global_max:.3f} MPa "
          f"(BIN_HI_MPA={cfg.BIN_HI_MPA:g}, headroom factor {cfg.BIN_HI_MPA/global_max:.1f}x)")
    print(f"GLOBAL min_range_mpa_screened across all scanned runs: {global_min:.3e} MPa "
          f"(BIN_LO_MPA={cfg.BIN_LO_MPA:g})")

    out_png = RESULTS_DIR / "bin_range_check.png"
    plot_range_check(rows, out_png)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    _self_check()
