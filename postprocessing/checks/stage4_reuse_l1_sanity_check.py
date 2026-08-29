"""
Synthetic validation test, NOT part of the real thesis pipeline: does the
classifier actually promote a bay to L1 (Structural reuse) when the numbers
genuinely support it? With real data every bay currently fails Step 2 (see
docs/decisions.md), so the L1 pathway has never been
exercised end-to-end -- this is worth confirming isn't dead code before
relying on it.

Scenario, exactly as specified 20.08.2026: base everything on Retrofit A
(S2/S3) only, take Bay 1 (the lowest/deepest bay by elevation, z=[-43.127,
-24.614] -- also happens to have the lowest real worst-joint D_A_K of the
four bays, 1.836, making it the "easiest" real candidate to push under the
margin), and artificially set D_25 = 0.15 (15%) for every one of its 12
joints. By the coded logic (MARGIN_L1_JOINT_D25 = 0.25, members_all_pass_l0
already True for every bay with real data), this SHOULD flip Bay 1 to
structural_reuse_pass_A = True and promote its 20 members to L1.

Real members/joints, real L0/L2/L3 results -- only the joint D_25 INPUT for
Bay 1 is synthetic. Nothing here overwrites the real reuse_classification*
CSVs; outputs are written to clearly separate _SANITY_CHECK files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # postprocessing/

import pandas as pd

import sd_geometry as sg
import stage4_reuse_classification as s4

TEST_BAY_ID = 1
SYNTHETIC_D25 = 0.15
PROJECT = Path(__file__).resolve().parents[2]   # repo root
OUT_MEMBERS = PROJECT / "results" / "reuse_classification_SANITY_CHECK.csv"
OUT_BAYS = PROJECT / "results" / "reuse_classification_bays_SANITY_CHECK.csv"


def main():
    model = sg.read_subdyn_model(sg.DEFAULT_SD_PATH)
    members_df = pd.read_csv(s4.MEMBER_CSV)
    joint_df = pd.read_csv(s4.JOINT_CSV)
    forces_df = pd.read_csv(s4.FORCE_CSV).set_index("member_id")
    bay_df = pd.read_csv(s4.BAY_GROUPING_CSV)

    bay1_joint_ids = bay_df[
        (bay_df["bay_id"] == TEST_BAY_ID) & (bay_df["entity_type"] == "joint")
    ]["entity_id"].tolist()
    bay1_member_ids = bay_df[
        (bay_df["bay_id"] == TEST_BAY_ID) & (bay_df["entity_type"] == "member")
    ]["entity_id"].tolist()
    print(f"Bay {TEST_BAY_ID}: {len(bay1_joint_ids)} joints {bay1_joint_ids}, "
          f"{len(bay1_member_ids)} members")

    # Real numbers for comparison, before the override.
    real_worst_A = joint_df.set_index("node").loc[bay1_joint_ids, "D_S3-K_25"]
    print(f"\nReal D_S3-K_25 for Bay {TEST_BAY_ID}'s joints (mostly NaN -- "
          f"none of these are among the 8 flagged splash joints):")
    print(real_worst_A.to_string())

    # The synthetic override: force every one of Bay 1's joints to D_25=0.15
    # under Retrofit A (S3 column -- composite D_A_K reads this directly,
    # not via the fillna(S2) path, since we're setting S3 itself here).
    joint_df_test = joint_df.copy()
    joint_df_test.loc[joint_df_test["node"].isin(bay1_joint_ids), "D_S3-K_25"] = SYNTHETIC_D25
    print(f"\nOverrode D_S3-K_25 = {SYNTHETIC_D25} for all {len(bay1_joint_ids)} of "
          f"Bay {TEST_BAY_ID}'s joints. Retrofit B (S4/S5) columns untouched -- "
          f"this test is Scenario A (S2/S3) only, as specified.")

    # Steps 1/3/4 are unaffected by a joint-only change -- reuse the REAL
    # results (L0/L2/L3 never touch joint data).
    step1_df = s4.run_step1(model, members_df, forces_df)
    step3_df = s4.run_step3(
        model, step1_df[step1_df["l0_pass"]]["member_id"].tolist(),
        members_df, forces_df, step1_df,
    )
    step4_fail_candidates = sorted(
        set(step1_df[~step1_df["l0_pass"]]["member_id"])
        | set(step3_df[~step3_df["l2_pass"]]["member_id"])
    )
    step4_df = s4.run_step4(model, step4_fail_candidates, members_df, forces_df, step1_df)

    # Step 2 re-run against the MODIFIED joint data -- this is the actual
    # code path under test, not a hand-injected verdict.
    bay_result = s4.run_step2(step1_df, joint_df_test)
    bay_result.to_csv(OUT_BAYS, index=False)

    print(f"\n--- Bay results (Retrofit A only shown) ---")
    print(bay_result[["bay_id", "worst_joint_D_A_K", "worst_joint_D_B_K",
                       "structural_reuse_pass_A", "structural_reuse_pass_B"]].to_string(index=False))

    bay1_row = bay_result[bay_result["bay_id"] == TEST_BAY_ID].iloc[0]
    passed = bool(bay1_row["structural_reuse_pass_A"])
    print(f"\n=== SANITY CHECK: Bay {TEST_BAY_ID} structural_reuse_pass_A "
          f"under the synthetic D_25={SYNTHETIC_D25} override: {passed} "
          f"({'PASS -- L1 pathway confirmed working' if passed else 'FAIL -- something is wrong, see code'}) ===\n")

    structural_reuse_members_A = set()
    for _, r in bay_result[bay_result["structural_reuse_pass_A"] == True].iterrows():  # noqa: E712
        structural_reuse_members_A.update(r["member_ids"])

    out = members_df[["member_id", "zone"]].merge(step1_df.drop(columns=["zone"]), on="member_id")
    out = out.merge(step3_df, on="member_id", how="left")
    out = out.merge(step4_df, on="member_id", how="left")

    def classify(row):
        if row["member_id"] in structural_reuse_members_A:
            return "L1", "Structural reuse"
        if row.get("l2_pass") is True:
            return "L2", "Component reuse"
        if row.get("l3_pass") is True:
            return "L3", "Downgraded reuse"
        if row.get("l3_pass") is False:
            return "L4", "Recycle"
        return "Pending", "Pending (L3 thresholds not set)"

    levels_names = out.apply(classify, axis=1, result_type="expand")
    out["reuse_level_TEST_A"] = levels_names[0]
    out["reuse_category_TEST_A"] = levels_names[0] + " - " + levels_names[1]
    out["in_test_bay"] = out["member_id"].isin(bay1_member_ids)

    OUT_MEMBERS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_MEMBERS, index=False)
    print(f"Wrote {OUT_MEMBERS} and {OUT_BAYS}\n")
    print("Bay 1 members' labels under the synthetic scenario:")
    print(out[out["in_test_bay"]][["member_id", "reuse_level_TEST_A", "reuse_category_TEST_A"]]
          .to_string(index=False))
    print("\nFull count:")
    print(out["reuse_category_TEST_A"].value_counts().to_string())


if __name__ == "__main__":
    main()
