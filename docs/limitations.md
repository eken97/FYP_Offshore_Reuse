# Limitations

What this work does **not** establish. Read this before quoting a number from
`results/`.

---

## It is a single structure, in simulation

Everything here concerns the OC4 reference jacket carrying the NREL 5MW
reference turbine, at one site's metocean conditions. No physical structure
was inspected, measured or tested. There is no as-built geometry, no
inspection record, no measured thickness, no coating survey, no weld quality
data.

So the outputs describe **what a design-basis structure would have
accumulated under simulated loading**, not the condition of any jacket
standing in the sea. A real reuse assessment needs inspection data this
analysis does not have and cannot substitute for.

## The joint track is indicative, not quantitative

This is the most important caveat and it is deliberate framing, not modesty.

The member track is presented as a quantitative damage map. The joint track
is a **first-order estimate**. It depends on:

- parametric SCF equations, which carry their own scatter
- a hot-spot stress superposition that was signed off by the author but
  **not independently reviewed** (`HOTSPOT_JOINT_VERIFIED` is a soft gate)
- a joint family classification, evaluated as a bounding pair rather than a
  load-path decomposition
- an assumption of rigid joints inherited from SubDyn's formulation

Two of the hot-spot formulation's open judgement calls — the crown-heel sign
convention and whether intermediate-point attachment SCF is included — were
resolved by primary-source re-checks and by the structure of the DNV equation
itself. They were not silently guessed. They were also never put to an
independent reviewer as a batch, which is the honest status.

Do not present member-track and joint-track numbers at equal confidence.

## It stops at the damage number

This repository computes fatigue damage per member and per joint — pristine,
on the corrosion trajectory to 25 years, and (for joints) under two
can-thickness retrofit scenarios. What the dissertation then did with those
numbers — a proposed L0–L4 reuse acceptance classification, a static/ULS
governing-life check, and an embodied-carbon comparison — is **not** in this
repository. Any statement about whether the structure can be reused, about
which check governs remaining life, or about carbon savings belongs to that
downstream work and cannot be reconstructed from what is published here.

## Scope boundaries baked into the model

- **Rigid joints.** SubDyn's formulation assumes rigid connections. Local
  joint flexibility is not represented, which affects load distribution near
  the joints — exactly where the joint track is trying to say something.
- **No marine growth evolution.** Marine growth is as configured in the
  HydroDyn deck, not evolved over the 25-year horizon.
- **No inspection or repair history.** The structure is assumed to have
  accumulated damage monotonically with no intervention.
- **Corrosion is uniform.** A constant rate per surface, by zone. Pitting,
  crevice corrosion and coating breakdown are not modelled.
- **Twelve members carry numbers that are not fatigue lives** — interface
  stubs, grouted equivalents and buried piles. They are flagged, not dropped.
  See [assumptions.md](assumptions.md#members-that-get-a-number-you-should-not-trust).

## The worked example is not a reproduction

`data/example/` holds **1 of 69 bins at 1 of 6 seeds**. It demonstrates that
the pipeline runs and stays numerically stable. Its damage numbers are not
comparable to anything in `results/` and should never be quoted.

Full reproduction from raw inputs requires re-running 414 OpenFAST cases —
roughly two machine-weeks and 119 GB of output. This repository gives you the
code, the decisions, the intermediate contract and the final tables; it does
not give you a one-command rebuild, and it would be dishonest to imply
otherwise.

## Verification status is uneven

`fatigue_config.py` records, per constant, whether it was independently
verified or assumed. That register is accurate and worth reading before
relying on any particular number. Not everything in it is verified, and the
file says so.

## What it does support

Given all of the above, the analysis does support:

- a **relative** damage map across the structure — which members and joints
  are worst, and by how much
- the finding that member fatigue damage stays low over the 25-year horizon,
  while **splash-zone joints** carry damage ratios more than an order of
  magnitude higher
- how that joint damage responds to corrosion and to can-thickness
  reinforcement, as a first-order estimate

Those are conclusions about method and relative severity. They are not a
certification of any structure.
