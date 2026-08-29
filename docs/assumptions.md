# Assumptions

Everything this analysis takes as given, and how confident you should be in
each. The authoritative machine-readable version is
[`postprocessing/fatigue_config.py`](../postprocessing/fatigue_config.py) —
open it alone and you can see every assumed number in one place, with its
status, rather than hunting through the pipeline for scattered constants.

This page explains the *discipline*; that file carries the *values*.

---

## Two categories, and why the distinction is enforced

**VERIFIED** — checked against a primary source, or derived from the model
itself. Section properties are the clearest example: they are recomputed from
the SubDyn deck per run and signed off against it, rather than stored as fixed
constants anywhere.

**ASSUMED** — a modelling choice or a literature value that has not been
independently checked for *this* structure. Marked UNVERIFIED until someone
actually checks it.

Three module-level flags gate the pipeline on this:

| Flag | Module | Meaning |
|---|---|---|
| `SN_CONSTANTS_VERIFIED` | `sn_curves.py` | S-N constants checked against DNV-RP-C203. **Hard gate** — Stage 3 refuses to run while False, because a wrong S-N constant produces a wrong-but-plausible damage number. |
| `SCF_EQUATIONS_VERIFIED` | `scf.py` | SCF equations checked against DNVGL-RP-0005 Appendix B |
| `HOTSPOT_JOINT_VERIFIED` | `stress.py` | hot-spot superposition signed off. **Soft gate** — prints a loud banner and continues, because the formula is fully built and self-consistent; it is a review gate, not a data-correctness gate. |

The hard/soft distinction is deliberate. A gate that blocks everything gets
switched off; a gate that blocks only what would be silently wrong gets
respected.

---

## Cheap mistakes and expensive mistakes

This is the single most useful thing to understand before changing a constant.

Some values are **read at the point of use**. Get one wrong, fix it, re-run
Stage 3 — minutes. `log_a`, the S-N knee, the thickness exponent are all like
this: Stage 3 reads them fresh against a Stage-2 cache that is still correct.

Other values are **baked into stored data**. The Wöhler exponents, the bin
edges, and the number of circumferential points are all frozen into every
Stage-2 `.npz` at the moment it is written. Getting one wrong means
recomputing 3.6 GB of cache from 119 GB of raw output.

`fatigue_config.PIPELINE_VERSION` exists for exactly this. Consumers stamp it
and refuse any file whose stamp does not match, so stale data is detected
rather than silently reused. **Bump it whenever you change something on the
expensive side of that line.**

It has been bumped once, from 1 to 2, when the Wöhler exponent set was
corrected — see below.

---

## The correction worth reading about

`WOHLER_EXPONENTS` was originally `(3, 5)`, on the assumption that the
exponents were uniform across S-N curves. They are not.

DNV-RP-C203's B1 air and seawater-with-cathodic-protection curves use
**m₁ = 4.0** above the knee, not 3. Those two curves cover 80 of the 112
members — every atmospheric and submerged member. Only B1's free-corrosion
curve and all of the T curves use m₁ = 3.

So the exponent set actually needed is `(3, 4, 5)`. With `(3, 5)`, every
atmospheric and submerged member's high-stress-branch damage would have been
computed from a stored power sum for the wrong exponent. The output would have
looked entirely plausible and been wrong.

Two things about how this was caught matter more than the fix:

1. It was found by **cross-checking the config against `sn_curves.py`'s actual
   (category, environment) table** before Stage 3 could consume it — not by
   reading the code and thinking it looked right. Inspection would not have
   found it.
2. It was on the **expensive** side of the line, so it forced a
   `PIPELINE_VERSION` bump and a full recompute. Had it been caught later it
   would have cost days.

---

## Members that get a number you should not trust

Twelve of the 112 members are flagged `not_assessable`. **Nothing is dropped**
— stress recovery, rainflow counting and a damage number are computed for
them exactly like every other member. The flag only attaches a reason string
so a reader knows not to read that number as a fatigue life:

| Members | Reason | Why |
|---|---|---|
| 101–104 | `interface_degenerate` | transition-piece interface stubs; elastic force ~1.3×10⁻⁸ N, i.e. numerical noise |
| 105–108 | `grouted_equivalent` | grouted equivalent tube, density 3339 kg/m³ — not steel |
| 109–112 | `buried_pile` | real members, but outside this pipeline's splash/corrosion zoning |

Computing and flagging, rather than excluding, means the output is complete
and the exclusion is visible and auditable rather than an unexplained gap.

A separate list, `SCREENING_EXCLUDED_MEMBER_IDS`, controls what ad-hoc QA
scripts bother scanning, purely for speed. It is derived from the same table
so the two cannot drift apart, and it must never be imported by Stage 2 or
Stage 3.

---

## Known-soft assumptions

Recorded honestly rather than buried:

- **Rainflow bin upper bound.** 1000 MPa was confirmed to be un-exceeded on a
  single mid-severity condition, not across the full severity envelope of all
  414 runs. Confirm `n_over == 0` on your most severe bins before relying on
  it.
- **Transient cutoff.** 100 s is discarded, applied by time value rather than
  sample index. Chosen once and applied consistently; not tuned per case.
- **Joint SCFs.** Parametric equations from DNVGL-RP-0005 Appendix B, with
  validity ranges checked per connection. Where a connection falls outside a
  formula's stated validity range, that is recorded rather than silently
  extrapolated.
- **The joint track as a whole** was not independently reviewed. See
  [limitations.md](limitations.md).
