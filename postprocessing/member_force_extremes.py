"""
Per-member static-check extremes: max tensile, max compressive, and
sign-separated mean axial force (FKze), across the full 414-run campaign.

Feeds a real strength/buckling bound on the fatigue-life extrapolation
(see docs/decisions.md) -- fatigue extrapolation predicts a
member survives some number of years assuming section loss alone, with no
lower bound on remaining thickness; this gives the static axial-force
envelope needed to check that assumption against an actual capacity
limit instead of extrapolating the maths forever.

Sign convention (verified empirically 18.08.2026 against the leg members'
known compressive self-weight load): FKze negative = compression,
positive = tension.

Per member, per timestep: the two end nodes (J1/J2) are collapsed to
whichever has the larger |FKze| at that instant (confirmed by the author
18.08.2026) -- the worst instantaneous load the member sees, wherever
along it it acts, not two independently-tracked end series.

max_tensile / max_compressive: the single worst value across all 414
runs, unweighted -- a static/buckling check cares about the worst case
that can occur, not how often it occurs. Sign-filtered the same way the
means are (a value only counts toward max_tensile if it is actually
positive at that instant, toward max_compressive if actually negative):
a member that never sees one sign across the whole campaign reports 0.0
for that extreme, never a same-signed-as-the-opposite-column value --
e.g. an always-compressive member must NOT show a negative max_tensile.

mean_tensile / mean_compressive: sign-separated (a signed mean would let
tension and compression cancel and understate both) and reported two
ways:
  *_unweighted -- pooled across all timesteps of all runs equally
  *_weighted   -- per-bin mean (pooled across that bin's 6 seeds),
                  combined by OC4_Final_Bins.xlsx's own raw Probability
                  column -- same weighting convention as
                  stage3_damage.py's fatigue-damage weighting.

Transient trim: first 100s of each 700s run dropped (TRANSIENT_CUTOFF_S,
same value as fatigue_config.py), leaving the 600s (10-min) usable block
per seed -- 6 seeds x 10 min per load case, as simulated.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import outb_reader as ob

# region --- paths / constants ---
CAMPAIGN_ROOT = Path(r"D:\OC4_CAMPAIGN\Simulation")
OUTB_NAME = "5MW_OC4Jckt_DLL_WTurb_WavesIrr_MGrowth.outb"

PROJECT = Path(__file__).resolve().parents[1]   # repo root
BINS_CSV = PROJECT / "data" / "oc4_k13_bins.csv"
RESULTS_DIR = PROJECT / "results"

TRANSIENT_CUTOFF_S = 100.0
N_MEMBERS = 112
CASE_ID_RE = re.compile(r"^LC(\d+)_S\d+$")
# endregion


# region --- bin probabilities ---
def load_bin_probabilities(table_path=BINS_CSV):
    """{bin_number (1-based, matching case_id's LC<nn>): p_bin}, the sheet's
    own raw Probability column, used as-is (same convention as
    stage3_damage.load_bin_probabilities)."""
    df = pd.read_csv(table_path)
    return {idx: float(p) for idx, p in enumerate(df["Probability"], start=1)}
# endregion


# region --- run discovery ---
def discover_runs(root=CAMPAIGN_ROOT):
    """[(outb_path, case_json_path), ...] for every LC_*/S* run folder."""
    runs = []
    for lc_dir in sorted(root.glob("LC_*")):
        if not lc_dir.is_dir():
            continue
        for seed_dir in sorted(lc_dir.glob("S*")):
            outb = seed_dir / OUTB_NAME
            case_json = seed_dir / "case.json"
            if outb.exists() and case_json.exists():
                runs.append((outb, case_json))
    return runs


def bin_number_of(case_json_path):
    case = json.loads(case_json_path.read_text())
    m = CASE_ID_RE.match(case["case_id"])
    assert m, f"{case_json_path}: unexpected case_id {case['case_id']!r}"
    return int(m.group(1))
# endregion


# region --- per-run read ---
def member_merged_series(outb_path):
    """(n_t_trimmed, 112) array: per member, per trimmed timestep, the
    FKze of whichever end (J1/J2) has the larger |FKze| at that instant."""
    header = ob.read_outb_header(outb_path)
    wanted = ob.member_end_channels(range(1, N_MEMBERS + 1), components=("FKze",))
    t, arr = ob.read_channels(outb_path, header, wanted)

    arr = arr[t >= TRANSIENT_CUTOFF_S]
    arr = arr.reshape(arr.shape[0], N_MEMBERS, 2)  # (n_t, member, end)
    end_choice = np.argmax(np.abs(arr), axis=2)  # (n_t, 112)
    merged = np.take_along_axis(arr, end_choice[:, :, None], axis=2)[:, :, 0]
    return merged
# endregion


# region --- accumulation over the whole campaign ---
def accumulate(runs):
    """One pass over every run. Returns per-member global (pooled,
    unweighted) accumulators plus per-bin accumulators (pooled across a
    bin's 6 seeds) needed for probability weighting."""
    acc = dict(
        global_max=np.full(N_MEMBERS, -np.inf),
        global_min=np.full(N_MEMBERS, np.inf),
        global_sum_pos=np.zeros(N_MEMBERS),
        global_n_pos=np.zeros(N_MEMBERS, dtype=np.int64),
        global_sum_neg=np.zeros(N_MEMBERS),
        global_n_neg=np.zeros(N_MEMBERS, dtype=np.int64),
        bin_sum_pos={}, bin_n_pos={}, bin_sum_neg={}, bin_n_neg={},
    )

    for i, (outb_path, case_json_path) in enumerate(runs, 1):
        bin_number = bin_number_of(case_json_path)
        merged = member_merged_series(outb_path)

        # Sign-filtered extrema: a value only counts toward the tensile
        # max if it's actually positive, toward the compressive min if
        # actually negative -- run_max_pos/run_min_neg are -inf/+inf for
        # a member with no sample of that sign in THIS run, which leaves
        # the running global extreme unchanged (see module docstring).
        run_max_pos = np.where(merged > 0, merged, -np.inf).max(axis=0)
        run_min_neg = np.where(merged < 0, merged, np.inf).min(axis=0)
        acc["global_max"] = np.maximum(acc["global_max"], run_max_pos)
        acc["global_min"] = np.minimum(acc["global_min"], run_min_neg)

        pos = np.where(merged > 0, merged, 0.0)
        neg = np.where(merged < 0, merged, 0.0)
        n_pos = (merged > 0).sum(axis=0)
        n_neg = (merged < 0).sum(axis=0)

        acc["global_sum_pos"] += pos.sum(axis=0)
        acc["global_n_pos"] += n_pos
        acc["global_sum_neg"] += neg.sum(axis=0)
        acc["global_n_neg"] += n_neg

        for key, val, n in (("pos", pos.sum(axis=0), n_pos), ("neg", neg.sum(axis=0), n_neg)):
            sums = acc[f"bin_sum_{key}"]
            counts = acc[f"bin_n_{key}"]
            if bin_number not in sums:
                sums[bin_number] = np.zeros(N_MEMBERS)
                counts[bin_number] = np.zeros(N_MEMBERS, dtype=np.int64)
            sums[bin_number] += val
            counts[bin_number] += n

        print(f"  [{i}/{len(runs)}] {outb_path.parent.parent.name}/{outb_path.parent.name} (bin {bin_number})")

    return acc
# endregion


# region --- final table ---
def _safe_mean(sums, counts):
    return np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts > 0)


def build_table(acc, p_bin):
    # A member with zero positive (or negative) samples across the whole
    # campaign never had its running extreme touched -- still -inf/+inf
    # here. Report 0.0 for that extreme (consistent with mean_tensile/
    # mean_compressive already being 0.0 for the same member), not an
    # opposite-signed value.
    global_max = np.where(np.isfinite(acc["global_max"]), acc["global_max"], 0.0)
    global_min = np.where(np.isfinite(acc["global_min"]), acc["global_min"], 0.0)
    acc = {**acc, "global_max": global_max, "global_min": global_min}

    mean_tensile_uw = _safe_mean(acc["global_sum_pos"], acc["global_n_pos"])
    mean_compressive_uw = _safe_mean(acc["global_sum_neg"], acc["global_n_neg"])

    weighted_tensile = np.zeros(N_MEMBERS)
    weighted_compressive = np.zeros(N_MEMBERS)
    weight_used = 0.0
    for bin_number, pb in p_bin.items():
        if bin_number not in acc["bin_sum_pos"]:
            continue  # bin not present in this campaign snapshot
        bin_mean_pos = _safe_mean(acc["bin_sum_pos"][bin_number], acc["bin_n_pos"][bin_number])
        bin_mean_neg = _safe_mean(acc["bin_sum_neg"][bin_number], acc["bin_n_neg"][bin_number])
        weighted_tensile += pb * bin_mean_pos
        weighted_compressive += pb * bin_mean_neg
        weight_used += pb

    rows = []
    for mid in range(1, N_MEMBERS + 1):
        k = mid - 1
        rows.append(dict(
            member_id=mid,
            max_tensile_N=acc["global_max"][k],
            max_compressive_N=acc["global_min"][k],
            mean_tensile_unweighted_N=mean_tensile_uw[k],
            mean_compressive_unweighted_N=mean_compressive_uw[k],
            mean_tensile_weighted_N=weighted_tensile[k],
            mean_compressive_weighted_N=weighted_compressive[k],
        ))
    df = pd.DataFrame(rows).sort_values("member_id").reset_index(drop=True)
    return df, weight_used
# endregion


def run(campaign_root=CAMPAIGN_ROOT, out_path=RESULTS_DIR / "member_force_extremes.csv"):
    runs = discover_runs(campaign_root)
    print(f"found {len(runs)} runs under {campaign_root}")
    if not runs:
        raise SystemExit(
            f"No runs found under {campaign_root} -- refusing to overwrite "
            f"{out_path} with an all-zero table. Point --root (or "
            f"OC4_RUN_ROOT) at a folder that actually holds .outb output, "
            f"or leave the shipped results/member_force_extremes.csv alone "
            f"if you don't have your own campaign data."
        )
    p_bin = load_bin_probabilities()
    acc = accumulate(runs)
    df, weight_used = build_table(acc, p_bin)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"p_bin weight used = {weight_used:.5f} (raw sheet sums to ~1.00046, not renormalized)")
    print(f"wrote {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    run()
