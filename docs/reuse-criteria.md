# Reuse criteria (L0–L4)

The acceptance ladder that turns damage numbers into a reuse verdict. Every
threshold below lives in
[`postprocessing/stage4_reuse_classification.py`](../postprocessing/stage4_reuse_classification.py)
as a named constant — change it there and re-run; nothing else needs touching.

**These thresholds are proposed, not standardised.** They were not
independently reviewed. Treat them as one defensible construction, not as an
authority. See [limitations.md](limitations.md).

---

## The ladder

| Level | Meaning |
|---|---|
| **L0** | Admissibility — is this member a candidate for reuse at all? |
| **L1** | Structural reuse — the member stays in a structural role |
| **L2** | Component reuse |
| **L3** | Downgraded reuse — a less demanding role |
| **L4** | Recycle |

A member falls through the ladder: it is tested at each level and drops to the
next if it fails. L4 is the terminal outcome, not a judgement of failure —
recycling is still a real outcome with real avoided emissions.

> **Naming caveat.** L3 is referred to by more than one name across the thesis
> text, this code, and the result tables. The definition is consistent; the
> label is not. If you are comparing sources, match on the level number.

---

## Thresholds

### Corrosion allowance (Step 1)

Each member's own allowance is derived per member, from its geometry and
loads, rather than assumed uniform. The derivation uses a load factor of 2.0
in place of γ_F, and the DNV-ST-0126 baseline γ_M of 1.10 unmodified.

From Step 3 onward, a further **×1.15** is applied to γ_M for reclaimed steel,
per SCI P427 §4.5.

### L1 — structural reuse

A bay's worst joint must satisfy:

```
MARGIN_L1_JOINT_D25 = 0.25
```

i.e. the worst joint in the bay must have accumulated no more than 25% of its
fatigue capacity at 25 years. This is a **bay-level** gate: a member sits in a
bay, and the bay's worst joint governs whether structural reuse is credible
for anything in it.

### L2 — component reuse

```
L2_SECTION_LOSS_LIMIT  = 0.05   # SCI P427 §5.3, fraction of nominal thickness
L2_FATIGUE_D25_LIMIT   = 0.05   # fatigue analogue of the same conservatism
```

The section-loss limit is P427's. The fatigue limit is **new** — P427 does not
give one, because the reuse guidance it encodes was not written with fatigue
in mind. It borrows P427's own level of conservatism by analogy rather than
inventing a number from nothing.

This is the substantive methodological addition, and the honest framing of it
is that existing reuse guidance has a fatigue-shaped hole.

### L3 — downgraded reuse

```
L3_FATIGUE_D25_LIMIT               = 0.50   # "more than 50% of fatigue life remaining"
L3_CORROSION_ALLOWANCE_MULTIPLIER  = 1.0
```

Note that L3's corrosion condition is deliberately **not a fixed millimetre
figure**. Step 1 already derives a per-member allowance that varies with
geometry and load, so L3 asks whether measured loss has eaten into *that
member's own* allowance:

```
loss_mm  <=  L3_CORROSION_ALLOWANCE_MULTIPLIER × member's own l0_allowance_mm
```

At the default multiplier of 1.0 this reads as "has not eaten into its own
allowance at all". Raise or lower the multiplier for a looser or tighter bar
without touching the classification logic.

### Screens not automated

CEV (carbon equivalent value) and coating condition are part of the criteria
but are **not automated** — they are flagged per row for manual assessment.
They need inspection data this analysis does not have.

---

## Scope

### Only the K family is carried through

`final_results_joint.csv` contains 10 scenario/family combinations. The reuse
classification uses the **K family only**.

The eight Y-plane combinations either fail so comprehensively that discussing
them individually adds nothing, or carry enough extra assumptions that
including them would import uncertainty into the verdict without improving it.

### Retrofit A and Retrofit B are independent verdicts

The two can-thickness retrofit scenarios produce two separate classifications,
`reuse_level_A` and `reuse_level_B`. They describe **two different real-world
interventions**, so do not AND them together and do not treat one as a
refinement of the other.

- **Retrofit A** — a reading of UpWind D4.2.5 Fig 3-3: X joints +10 mm top and
  bottom, +5 mm mid; K and Y +5 mm both sides; bottom-K restricted to the
  thick scenario only.
- **Retrofit B** — a flat +12 mm both sides everywhere, with the same
  bottom-K restriction.

### Two member scopes appear in the results

Some figures cover **80 members** (the bay members in the reuse scope) and
others **104** (all non-pile members). Both are deliberate. Check which scope
a figure uses before comparing totals across figures — the difference is not
an inconsistency.

---

## Reading the outputs

| File | Contents |
|---|---|
| `results/reuse_classification.csv` | per-member level and category, for both retrofit scenarios |
| `results/reuse_classification_bays.csv` | per-bay rollup |
| `results/final_results_member.csv` | damage and life per member at each 5-year step |
| `results/final_results_joint.csv` | per joint node, all 10 scenario/family combinations |
| `results/member_remaining_life.csv` | min(fatigue, static) governing life |

`stage4_reuse_l1_sanity_check.py` exercises the L1 promotion path against
synthetic data — it exists because in the real campaign no member ever reaches
L1, so without it the L1 branch would be untested dead code.
