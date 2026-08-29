# Decisions

Why the pipeline is shaped the way it is. Choices, alternatives that were
considered and dropped, and what would have to change to revisit them.

This deliberately does not re-derive standard theory — read DNV-RP-C203 for
that. It records the judgement calls, which is the part you cannot look up.

---

## Campaign design

### 20-minute runs, not 60

IEC 61400 asks for six 10-minute realisations per fatigue bin. The obvious
worry is whether a longer run would find damage a shorter one misses — low
frequency content, slow drift, rare large cycles.

This was tested rather than assumed: a load case was run at both durations
and compared channel by channel. 12 of 14 channels agreed. The two that did
not showed a **duration-independent offset**, not a duration-dependent trend
— which is a different phenomenon and does not argue for longer runs.

Each case therefore runs 700 s: 600 s usable plus a 100 s discarded startup
transient. Six seeds per bin. At 414 runs and roughly 3.2 hours each, the
campaign is already about two machine-weeks; 1-hour runs would have tripled
that for no demonstrated gain.

### 6 seeds, and the same 6 everywhere

Six realisations per condition, per the IEC fatigue rule. The seed values
(`100001, 200002, …, 600006`) are reused across every bin — a seed only needs
to be distinct *within* a condition, not globally.

They were picked as obviously-distinct round numbers that do not collide with
any seed used by the earlier test cases, specifically so that leftover test
data on disk can never be mistaken for real campaign data. Wind and wave
seeds differ from each other (`wave_seed = wind_seed + 1`) so that each of the
6 realisations is a genuinely independent draw rather than co-varying.

### 69 bins from the K13 scatter table

The metocean bins come from binning the OC4 K13 site scatter data into 69
wind/wave conditions, each with an occurrence probability. The full table is
`data/oc4_k13_bins.csv`, including a `NAME` column.

`config.py` recomputes each row's name from its own wind/wave values and
asserts it matches the `NAME` column. 414 cases is too many to eyeball, and
this makes a parsing or formatting mistake fail loudly at import time instead
of quietly producing a wrong folder name.

### Coarse Tp bins are fine; this was checked

Peak wave period was binned coarsely. That is a real approximation, so it was
tested: two sensitivity studies found mean DEL changes of roughly 3% and 1.4%
from Tp bin resolution. The jacket's first natural frequency is 0.311 Hz,
comfortably clear of the wave energy, which is why the structure is not
especially sensitive to where exactly within a band Tp falls.

Given that, spending campaign budget on finer Tp resolution instead of more
wind bins would have been the wrong trade.

### The V ≥ 25 m/s bin is idling, not operating

Above cut-out the turbine is not producing power, and modelling it as if it
were would badly misrepresent the loads. That bin runs in an idle
configuration with feathered blades. Getting this genuinely right needs more
than blade pitch — see [traps.md](traps.md#drtrdof-when-idling).

### OpenFAST version

Results were produced with **OpenFAST v5.0.0** (double-precision release
build) and the NREL model decks at **r-test commit `dd5feaaa`** (2026-03-12),
pinned in `scripts/rtest_manifest.json`.

Pinning the model matters more than pinning the binary: a changed deck changes
the structure, and then the numbers in `results/` describe something else.

---

## Post-processing architecture

### Four stages, with exactly one durable intermediate

Stage 2's `.npz` histogram cache is the only thing written to disk between
raw `.outb` and final tables. Recovered stress time series are never
persisted — they are large, trivially recomputable, and keeping them would
have multiplied storage for no benefit.

This makes the expensive boundary explicit: everything upstream of Stage 2 is
recomputed from 119 GB of `.outb`; everything downstream is recomputed from
3.6 GB of cache. Which side of that boundary a mistake falls on determines
whether fixing it costs minutes or days.

### The Stage-2 cache stores power sums, not just counts

Each histogram bin stores the cycle count **and** the exact power sums
Σ(count · range^m) for every Wöhler exponent in use.

The alternative — storing counts alone and applying exponents later — is
simpler but throws away precision, since a binned count reconstructs the
power sum only approximately. Storing the sums means a later change to an S-N
constant re-runs Stage 3 in minutes, without touching the cache.

The exponents themselves were originally hardcoded in `rainflow_hist.py` as
`sum_r3`/`sum_r5`, on the assumption they were uniform across curves. They
are not, for the categories actually used here. Pulling them into
`fatigue_config.WOHLER_EXPONENTS` made that correction a one-line edit plus a
recompute rather than a code change.

### 16 circumferential points, not 8

Stress is sampled at 16 angles around each member end. For the member track
the *choice* of origin angle is provably irrelevant, but the *count* is not:
at n = 8 the worst-case sampling error undercounts damage by
cos(22.5°)³ = 0.7886 — a 21% underestimate. `test_invariants.py` checks this
against synthetic signals.

Changing `N_THETA` changes the stored array shape, so it requires a
`PIPELINE_VERSION` bump and a full recompute.

### 256 log-spaced bins from 0.01 to 1000 MPa

Log spacing because stress ranges span orders of magnitude and damage lives
in the upper tail, where log spacing buys resolution cheaply. The lower bound
is far below anything that damages steel; the upper bound has headroom above
observed ranges.

The bins are frozen: identical edges across every run, seed and member, since
Stage 3 sums histograms across seeds and bins and that only works if the edges
match. Note the caveat recorded in `fatigue_config.py` — the upper bound was
confirmed against a mid-severity condition, not the full severity envelope.

### Two-tier framing: quantitative members, indicative joints

The member track is presented as a quantitative damage map. The joint track
is presented as an **indicative, first-order** estimate.

This is not modesty, it is scope. The joint track depends on parametric SCF
equations, a hot-spot superposition, and a family classification, each
carrying its own uncertainty — and unlike the member track it was not
independently reviewed. Presenting both at the same confidence would have
overstated the joints. See [limitations.md](limitations.md).

---

## Fatigue methodology

### B1 for members, T for joints

Members use DNV-RP-C203 category B1; joints use the T curve, which is the
tubular-joint hot-spot curve. Three environments per category — air, seawater
with cathodic protection, free corrosion — assigned by member zone.

### No design fatigue factor

A DFF was considered and rejected. The question this work asks is what the
structure's *actual* accumulated fatigue damage is. Folding in a design
safety factor conflates "what has happened to this steel" with "what margin a
new design would require", and leaves the damage number uninterpretable.

### Corrosion rate

The design basis gives 0.30 mm/yr per surface generally, and permits halving
the allowance for fatigue design specifically. This repository's fatigue
tracks therefore use **0.15 mm/yr**, applied to both surfaces of the
(flooded) legs and the external surface only of the braces. The un-halved
0.30 mm/yr rate is defined in `sd_geometry.py` for completeness but is not
exercised by any code here — the non-fatigue checks that used it are not
published in this repository.

An earlier version applied "both sides everywhere at the full 0.3 mm/yr" and
was superseded.

### 25-year horizon

Damage is evaluated at 0, 5, 10, 15, 20 and 25 years, with corrosion thinning
composed step by step rather than applied as a single end-state. Extending to
50 years is a one-line change, and was deliberately left as one.

### Joint families: a bounding pair, not a load-path decomposition

DNV's ±15° coplanarity test is used in the standard to split a brace's axial
force into K/X/Y fractions for a weighted SCF. That decomposition was
**deliberately not built** here. Instead each joint is evaluated as a bounding
pair — the X treatment and its real-type treatment — which brackets the
answer without requiring a load-path split that would add assumptions the
data cannot support.

`joint_geometry.py` reuses the same coplanarity geometry for a different
purpose: deciding how connections group, not how force divides.

---

## Campaign logistics

### The work split is by bin, not by case

`plan_assignment.py` divides whole bins between machines, weighted by measured
throughput (cores ÷ hours-per-case).

An earlier version used a deficit round-robin at the individual-case level.
That is correct — `merge.py`'s per-case ownership check works either way — but
it split some bins' 6 seeds across both machines. Completing post-processing
for one bin then meant pulling data from two machines instead of one, which is
a genuine practical headache for no benefit.

Whole bins, sorted by wind speed: the faster machine works up from the low-V
end, the other works down from the high-V end, and they meet in the middle.
Because both lists converge on the boundary, the least-touched cases on either
machine are always the boundary-adjacent ones — which is what lets
`rebalance.py` move work mid-campaign by taking from the end of a list.

### Find the drive by marker file, never by letter

Run data lives on an external drive that enumerates as a different letter
depending on which machine it is plugged into. An early version hardcoded
`D:`. Drive letters are scanned for a marker file instead.

### Code freeze while a campaign runs

Long campaign runs execute off whatever is on disk at the moment each worker
imports a module. Editing a module inside a running pipeline's import graph
risks a worker importing a half-written file.

This is why the corrosion variants are separate modules —
`joint_geometry_corrosion.py`, `scf_corrosion.py` — that only *import* the
frozen originals and add parallel functions, rather than edits to the
originals. It is more files, and it is the right trade while a multi-day run
is in flight.
