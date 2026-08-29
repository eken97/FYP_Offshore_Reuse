"""
Joint-track can-thickness retrofit, Stage 3 -- Miner damage -> life-years,
full population, no corrosion. Direct analogue of stage3_joint_damage.py
(same grouping rule, same "K/Y treatment never merged" discipline, same
184-row-per-run shape), reusing run_connection_damage/aggregate_condition's
UNDERLYING physics via stage3_damage.point_theta_damage() -- but this
module supplies its own scf_lookup (built from the RETROFITTED SCF set,
not the baseline) and reads stage2 files from stage2_joints_thickness.py's
loader instead of stage2_joints.py's.

WHY a separate scf_lookup matters: the S-N thickness-correction exponent k
depends on the connection's own static_scf_max (see
stage3_joint_damage.py's own docstring, "REAL WRINKLE"). Under the retrofit
the SCF drops meaningfully (e.g. the X-bottom worst point: 6.33 -> 4.22 at
+10mm) -- using the baseline (unretrofitted) SCF here would be internally
inconsistent, even though in practice this dataset never crosses the
k-branch threshold (SCF=10) either way.

t_mm for the S-N thickness correction is read from the retrofitted
brace_t/chord_T already stored in stage2_joints_thickness.py's own point
table (built from the retrofitted geometry, see that module's docstring) --
no separate lookup needed for that part.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sd_geometry as sdg
import sn_curves as sn
import stage2_joints as s2j
import stage2_joints_thickness as s2jt
import stage3_damage as s3
import stage3_joint_damage as s3j
import joint_thickness_override as jto
import scf_thickness_override as sto

# region --- paths ---
PROJECT = Path(__file__).resolve().parents[1]   # repo root
POSTPRO_DIR = PROJECT / "postprocessing"
RESULTS_DIR = PROJECT / "results"
DEV_FIXTURE_DIR = PROJECT / "data" / "example"

BINS_CSV = s3.BINS_CSV
OUTB_NAME = s2j.OUTB_NAME
SD_NAME = s2j.SD_NAME
SD_SUM_NAME = s2j.SD_SUM_NAME
# endregion

DESIGN_LIFE_YEARS = s3.DESIGN_LIFE_YEARS
BLOCKS_PER_LIFE = s3.BLOCKS_PER_LIFE
CATEGORY = s3j.CATEGORY   # "T" -- the joint track always assesses against the T-curve


def build_scf_lookup_retrofit(sd_path, sd_sum_path, scenario):
    """Same shape as stage3_joint_damage.build_scf_lookup(), but sourced
    from scf_thickness_override.compute_scf_retrofit() under `scenario`
    instead of the unretrofitted baseline."""
    import joint_geometry as jg
    dcm_result = jg.read_member_dcm(sd_sum_path)
    model = sdg.read_subdyn_model(sd_path)
    connections = jg.build_connections(model)
    jg.compute_geometry_params(connections, model)
    jg.add_joint_axes(connections, model, dcm_result["dcm"])
    jg.add_chord_geometry(connections, model, dcm_result["dcm"])

    rows = sto.compute_scf_retrofit(connections, model, scenario)
    lookup = {}
    for r in rows:
        lookup_key = (r["node"], str(r["sub_joint_id"]), r["brace_member"], r["brace_end"],
                      r["chord_t_scenario"], r["direction"] or "", r["treatment"], r["side"])
        lookup[lookup_key] = r["static_scf_max"]
    return lookup


def aggregate_condition_retrofit(cond_dir, scf_lookup, mudline_z):
    """Direct analogue of stage3_joint_damage.aggregate_condition(), reading
    stage2_joints_thickness's own .npz shape instead of stage2_joints's."""
    npz_paths = sorted(Path(cond_dir).glob("*.npz"))
    assert npz_paths, f"no seeds found in {cond_dir}"

    per_seed_damage = []
    per_seed_detail = []
    bin_numbers = set()
    meta_by_group = {}
    n_excluded = 0

    for npz_path in npz_paths:
        stage2 = s2jt.load_stage2_joints_thickness(npz_path)
        stamp = stage2["stamp"]
        if not s3._seed_usable(stamp, stamp["transient_cutoff_s"]):
            n_excluded += 1
            print(f"    excluding {npz_path.name}: duration too short after transient trim")
            continue
        case_id = stamp["case_json"]["case_id"]
        bin_numbers.add(s3.bin_number_from_case_id(case_id))
        rows = s3j.run_connection_damage(stage2, scf_lookup, mudline_z)
        group_key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                                r["chord_t_scenario"], r["direction"], r["treatment"])
        per_seed_damage.append({group_key(r): r["damage_block"] for r in rows})
        per_seed_detail.append({group_key(r): (r["worst_side"], r["worst_position"], r["worst_segment"])
                                 for r in rows})
        for r in rows:
            meta_by_group.setdefault(group_key(r), r)

    assert per_seed_damage, f"{cond_dir}: every seed excluded, nothing usable"
    assert len(bin_numbers) == 1, f"{cond_dir}: seeds disagree on bin number: {bin_numbers}"
    bin_number = bin_numbers.pop()
    n_seeds = len(per_seed_damage)

    out_rows = []
    for key, meta in sorted(meta_by_group.items()):
        vals = np.array([seed[key] for seed in per_seed_damage])
        i_dominant = int(np.argmax(vals))
        side_rep, pos_rep, seg_rep = per_seed_detail[i_dominant][key]
        r = {k: v for k, v in meta.items()
             if k not in ("damage_block", "worst_side", "worst_position", "worst_segment")}
        r.update(
            bin_number=bin_number,
            damage_block_mean=float(vals.mean()),
            damage_block_std=float(vals.std(ddof=0)),
            damage_block_min=float(vals.min()),
            damage_block_max=float(vals.max()),
            n_seeds_used=n_seeds,
            n_seeds_excluded=n_excluded,
            worst_side=side_rep, worst_position=pos_rep, worst_segment=seg_rep,
        )
        out_rows.append(r)
    return bin_number, out_rows


def build_per_bin_retrofit(stage2_root, scf_lookup, mudline_z):
    cond_dirs = sorted(p for p in Path(stage2_root).iterdir() if p.is_dir())
    assert cond_dirs, f"no condition folders found under {stage2_root}"

    per_bin = {}
    for i, cond_dir in enumerate(cond_dirs, 1):
        print(f"  [{i}/{len(cond_dirs)}] aggregating {cond_dir.name} ...")
        bin_number, rows = aggregate_condition_retrofit(cond_dir, scf_lookup, mudline_z)
        key = lambda r: (r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                          r["chord_t_scenario"], r["direction"], r["treatment"])
        assert bin_number not in per_bin, f"bin {bin_number} appears in more than one condition folder"
        per_bin[bin_number] = {key(r): r for r in rows}
    return per_bin


def compute_stage3_retrofit(stage2_root, sd_path, sd_sum_path, scenario, bins_table=BINS_CSV):
    """Discovers every condition folder under stage2_root (expected:
    stage2_joints_thickness.STAGE2_JOINTS_THICKNESS_DIR / scenario),
    aggregates seeds per bin, combines by probability, scales to 25yr --
    direct analogue of stage3_joint_damage.compute_stage3()."""
    p_bin, raw_total = s3.load_bin_probabilities(bins_table)
    mudline_z = s3j.mudline_z_from_sd(sd_path)
    scf_lookup = build_scf_lookup_retrofit(sd_path, sd_sum_path, scenario)
    per_bin = build_per_bin_retrofit(stage2_root, scf_lookup, mudline_z)
    return s3j.stage3_from_per_bin(per_bin, p_bin, raw_total)


def write_stage3_joint_damage_thickness(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(
        ["node", "brace_member", "brace_end", "treatment", "chord_t_scenario"]
    ).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df


def _self_check():
    import stress
    import scf as scf_mod
    print(f"HOTSPOT_JOINT_VERIFIED = {stress.HOTSPOT_JOINT_VERIFIED}")
    print(f"SCF_EQUATIONS_VERIFIED = {scf_mod.SCF_EQUATIONS_VERIFIED}")

    case_dir = DEV_FIXTURE_DIR / "LC_V20_H3p5_T8" / "S100001"
    sd_path = case_dir / SD_NAME
    sd_sum_path = list(case_dir.glob("*.SD.sum.yaml"))[0]
    mudline_z = s3j.mudline_z_from_sd(sd_path)

    # --- baseline (unretrofitted) reference, for comparison throughout ---
    ref_cond_dir = s2j.STAGE2_JOINTS_DIR / "LC_V20_H3p5_T8"
    if not ref_cond_dir.exists():
        ref_cond_dir = RESULTS_DIR / "_stage2_joints_selfcheck" / "LC_V20_H3p5_T8"
    assert ref_cond_dir.exists(), f"no unretrofitted joint Stage 2 reference found (checked {ref_cond_dir})"
    ref_scf_lookup = s3j.build_scf_lookup(sd_path, sd_sum_path)
    ref_bin_number, ref_rows = s3j.aggregate_condition(ref_cond_dir, ref_scf_lookup, mudline_z)
    ref_by_key = {(r["node"], r["brace_member"], r["brace_end"], r["chord_t_scenario"],
                   r["direction"], r["treatment"]): r for r in ref_rows}

    for scenario in jto.SCENARIOS:
        print(f"\n=== scenario {scenario} ===")
        cond_dir = RESULTS_DIR / "_stage2_joints_thickness_selfcheck" / scenario / "LC_V20_H3p5_T8"
        assert cond_dir.exists(), f"missing Stage 2 retrofit fixture: {cond_dir}"

        print("1. build_scf_lookup_retrofit + aggregate_condition_retrofit:")
        scf_lookup = build_scf_lookup_retrofit(sd_path, sd_sum_path, scenario)
        assert len(scf_lookup) == 368
        bin_number, rows = aggregate_condition_retrofit(cond_dir, scf_lookup, mudline_z)
        print(f"   bin_number={bin_number} (expect 65), {len(rows)} rows (expect 184)")
        assert bin_number == 65
        assert len(rows) == 184
        assert all(r["n_seeds_used"] == 6 for r in rows), "expected all 6 real seeds usable"
        assert all(np.isfinite(r["damage_block_mean"]) for r in rows)

        by_key = {(r["node"], r["brace_member"], r["brace_end"], r["chord_t_scenario"],
                   r["direction"], r["treatment"]): r for r in rows}

        # --- 2. bottom-K 'thin' connections: damage must match the
        # unretrofitted reference EXACTLY (geometry untouched there).
        print("2. bottom-K 'thin' rows: damage unchanged from reference:")
        thin_keys = [k for k in by_key if k[3] == "thin"
                     and abs(model_z_for(sd_path, by_key[k]["node"]) - jto.K_BOTTOM_Z) < jto.Z_TOL_M]
        assert len(thin_keys) == 32, f"expected 32 (16 connections x K+Y treatments), got {len(thin_keys)}"
        max_rel_diff = 0.0
        for k in thin_keys:
            d_ref = ref_by_key[k]["damage_block_mean"]
            d_got = by_key[k]["damage_block_mean"]
            rel_diff = abs(d_ref - d_got) / d_ref if d_ref > 0 else abs(d_ref - d_got)
            max_rel_diff = max(max_rel_diff, rel_diff)
        print(f"   {len(thin_keys)} rows, max rel.diff vs reference = {max_rel_diff:.3e}")
        assert max_rel_diff < 1e-9

        # --- 3. real X-bottom worst point (node 39/brace 47): damage drops
        # relative to the unretrofitted reference, and by a similar order to
        # the earlier hand-ballpark (life 0.34yr -> a few years, not >10x
        # over-optimistic vs. that estimate).
        print("3. node 39/brace 47 (X-bottom worst point): damage vs reference:")
        key_x = next(k for k in by_key if k[0] == 39 and k[1] == 47)
        d_ref = ref_by_key[key_x]["damage_block_mean"]
        d_got = by_key[key_x]["damage_block_mean"]
        print(f"   reference D_block={d_ref:.6e}  scenario {scenario} D_block={d_got:.6e}  "
              f"ratio={d_ref/d_got:.2f}x less damage")
        assert d_got < d_ref, "retrofit should reduce damage at this point, not increase it"

        # --- 4. compute_stage3_retrofit on the dev fixture (single bin) ---
        print("4. compute_stage3_retrofit (single-bin dev fixture):")
        p_bin, raw_total = s3.load_bin_probabilities()
        per_bin = {bin_number: {(r["node"], r["sub_joint_id"], r["brace_member"], r["brace_end"],
                                  r["chord_t_scenario"], r["direction"], r["treatment"]): r
                                 for r in rows}}
        stage3_rows, meta = s3j.stage3_from_per_bin(per_bin, p_bin, raw_total)
        print(f"   {len(stage3_rows)} rows (expect 184)")
        assert len(stage3_rows) == 184
        assert all(np.isfinite(r["D_life"]) and r["D_life"] >= 0 for r in stage3_rows)

        out_path = write_stage3_joint_damage_thickness(
            stage3_rows, RESULTS_DIR / f"_stage3_joint_damage_thickness_selfcheck_{scenario}.csv")
        print(f"   wrote {out_path}")

    print("\n  all checks passed.")


_z_cache = {}


def model_z_for(sd_path, node):
    if sd_path not in _z_cache:
        model = sdg.read_subdyn_model(sd_path)
        _z_cache[sd_path] = model["joints"]
    return _z_cache[sd_path][node][2]


if __name__ == "__main__":
    _self_check()
