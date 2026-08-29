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

## The thresholds are proposed, not standardised

The L0–L4 ladder in [reuse-criteria.md](reuse-criteria.md) is a construction.
The section-loss limit comes from SCI P427; the **fatigue** limits do not,
because P427 has no fatigue criterion to borrow. They were set by analogy to
P427's own conservatism.

That is a defensible choice and it is also the weakest link between the
damage numbers and the reuse verdict. Someone else applying different
thresholds to the same damage data would reach different levels.

## Emission factors are literature values

The embodied-carbon work uses published emission factors (IStructE *How to
Calculate Embodied Carbon*, 3rd ed.) rather than project-specific EPDs. No
supply chain was modelled, no transport or reprocessing energy for a real
reuse route was costed, and the carbon accounting boundary is a stated
assumption rather than a derived one.

There is also a genuine gap in the literature here: offshore-specific reuse
emission factors are scarce, so the reuse figure inherits uncertainty that
the recycling and primary-steel figures do not.

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
- the finding that the **static check governs** remaining life for much of
  the structure, rather than fatigue
- the observation that **splash-zone joints** are the binding constraint on
  reuse, and that they fail every scenario examined
- a framework in which reuse acceptance is **condition-gated** rather than
  assumed, with the carbon consequence following from the condition verdict

Those are conclusions about method and relative severity. They are not a
certification of any structure.
