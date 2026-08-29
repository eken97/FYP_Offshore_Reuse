# Traps

Things that cost real time on this project, or that silently produce
plausible-looking wrong answers. Most were found by checking a number that
looked fine.

If you read one file in `docs/`, read this one.

---

## Rainflow counting

### Rainflow is not monotonic, so enveloping is never conservative

The instinct — "take the worst case, analyse that, it will bound the rest" —
is wrong for fatigue, and wrong in the unsafe direction.

Rainflow counting is a **path-dependent** operation on the whole time series.
A signal that is pointwise larger than another can produce *less* damage,
because what matters is the sequence of reversals, not the envelope. Combining
or enveloping load cases before counting can understate damage by up to a
factor of about 4 in cases checked here.

Count each realisation separately, then sum the damage. Never envelope first.

### Range versus amplitude is a factor-8 error

S-N curves in DNV-RP-C203 are defined in terms of stress **range**
(peak-to-trough). Half the literature, and most people's intuition, works in
**amplitude** (half of that).

Get it backwards with a Wöhler exponent of m = 3 and you are wrong by 2³ = 8.
The result still looks like a fatigue number. It is off by nearly an order of
magnitude.

Check which one your rainflow library returns. Check it again against a
hand-counted triangle wave. `test_invariants.py` does exactly this against
synthetic signals.

### Output sampling rate destroys damage you cannot get back

Damage lost to a coarse output timestep is **not recoverable** by any
post-processing. Small, fast cycles simply are not in the record.

This is a decision you make *before* the campaign runs, in the OpenFAST input
deck, and it is expensive to get wrong: re-running is the only fix. This
project used `DT_out = 0.05 s`. Sanity-check yours against the highest
structural frequency you care about before committing compute.

### Log-spaced bins need geometric midpoints

The 256 histogram bins are log-spaced from 0.01 to 1000 MPa, because stress
ranges span orders of magnitude and damage is dominated by the upper tail.

The representative stress for a bin is then `sqrt(lo × hi)`, **not**
`(lo + hi)/2`. The arithmetic midpoint of a log-spaced bin sits systematically
high, and with m = 3 that bias is cubed. Use the geometric mean.

---

## OpenFAST

### `set_waves()` does not set `WaveTMax`, and the sea silently repeats

The single most expensive gotcha here.

If `WaveTMax` is left at its default while the simulation runs longer than it,
HydroDyn/SeaState **loops the wave train**. In one case the sea repeated every
60 seconds for the entire run. The simulation completes, produces no warning,
and yields a beautifully periodic time series that is worthless for fatigue —
it contains one minute of sea state repeated over and over, so the rainflow
count is dominated by an artefact.

Set `WaveTMax > TMax` explicitly, per case. This project uses `TMAX = 700 s`
with `WAVE_TMAX = 750 s`.

Nothing catches this for you. Check it in your own input files.

### `CompAero = 1` versus `2`

Some OpenFAST versions behave differently, or fail, depending on which
aerodynamics module is selected. If a case that should run does not, this is
an early thing to check. See the version notes in
[decisions.md](decisions.md#openfast-version).

### `DrTrDOF` when idling

Setting up an idling or parked case (used here for the V ≥ 25 m/s bin) is not
just a matter of pitching the blades to feather. The drivetrain degree of
freedom needs attention too, or the rotor behaves in ways you did not intend.
See `idle` handling in `simulation/of_inputs.py`.

### `OutAll` is enough, and the channel names are `M{id}J{1|2}`

SubDyn's `OutAll` flag emits end forces and moments for every member. That is
sufficient for this analysis — an explicit mid-member output check found that
axial stress accounts for 79–91% of the dynamic stress and bending for
9–21%, so member ends govern.

The resulting channels follow the convention `M<member_id>J<1|2>` — member
id, then which end. This is not documented prominently anywhere obvious and
was established by probing real output against the older `NMOutputs` list.

### The SubDyn deck you use barely matters, except once

Every SubDyn deck in this project describes the same jacket. A campaign copy
differs from the pristine NREL r-test deck by **exactly one line**:
`OutAll: False → True`, written at run time by `of_inputs.py`.

So for reading geometry or section properties, any copy works —
`sd_geometry.py` falls back to the fetched pristine deck for exactly this
reason. But if you are running the campaign and `OutAll` is `False`, you get
no member end forces at all, and the whole pipeline has nothing to read.

---

## Validation and diagnostics

### A rigid support structure severs the wind-to-jacket load path

When a validation load case showed almost no load fluctuation in the jacket,
the cause was that the **support structure** was modelled as rigid — which
removes the path by which rotor thrust variation reaches the substructure.

Describe this precisely. It is a *rigid support structure*, not a "rigid
rotor"; the distinction changes what the result means and the wrong phrasing
sends the next reader down the wrong diagnostic path entirely.

### Damage-equivalent load needs its reference cycle count stated

A DEL is meaningless without saying what `N_ref` it is referenced to. Two
DELs computed with different reference cycle counts are not comparable, and
nothing in the number itself tells you. A normalisation error here was found
and fixed during validation.

### Power spectral density: resolution and units

Two separate errors, both found by cross-checking against reference data:

- comparing PSDs computed at different frequency resolutions — a 15×
  mismatch — which makes the curves disagree for purely numerical reasons
- labelling a PSD axis in dB when the quantity plotted was not in dB

Neither changes the physics; both make a comparison plot lie.

### The damage jump at V = 10 m/s is physical

A sharp increase in damage between the 8 and 10 m/s bins looks like a bug. It
is not: it is the **rated thrust peak** of the NREL 5MW turbine, which sits
just below rated wind speed.

This also corrects an easy over-claim. It is tempting to conclude "waves
dominate the fatigue"; the aerodynamic thrust peak is clearly visible in the
damage distribution and the honest statement is more nuanced.

---

## Corrosion

### Two rates, and they are not interchangeable

The design basis gives a general corrosion rate of **0.30 mm/yr per surface**,
and separately permits the allowance to be **halved for fatigue design
specifically** — as distinct from extreme/ULS design.

So this project uses:

- **0.15 mm/yr per surface** for fatigue life calculations
- **0.30 mm/yr per surface** for the static/ULS check

and applies it to:

- **legs** — both surfaces (they are flooded)
- **braces** — external surface only

Using the fatigue rate in a static check is a real bug that was made and
caught here. It makes the structure look considerably healthier than it is,
and because both numbers are defensible in isolation, nothing looks wrong.
Keep the two rates explicitly separate in code, as
`CORROSION_RATE_MM_PER_YEAR_PER_SURFACE` and
`GENERAL_CORROSION_RATE_MM_PER_YEAR_PER_SURFACE` do in `sd_geometry.py`.

---

## S-N curves and stress

### The mean-stress reduction factor does not apply everywhere

DNV's mean-stress reduction factor `f_m` is not automatically applicable to
every category and track. It does not apply to the B1 / member track as used
here. Check the clause conditions against your own case rather than applying
it because it is available.

### The knee moves between tables

In DNV-RP-C203, the S-N knee is not at a fixed cycle count across
environments: N = 1e7 for air, N = 1e6 for seawater with cathodic protection,
and free corrosion has no knee at all — it is single-slope m = 3 throughout.

Transcribing one table's knee across all three environments is an easy and
quiet error.

### `α = 2L/D`, and *L* is not what you first assume

In the SCF equations, the chord-length parameter α uses *L* = the distance
between **supporting joints** — not the brace length, not the member length
as it appears in the model. Getting this wrong shifts every SCF that depends
on α.

---

## Discipline

### Never assert an engineering constant from memory

Every constant in this pipeline that came from a standard was checked against
a primary source before being used, and `fatigue_config.py` records the
verification status of each one explicitly as VERIFIED or ASSUMED.

This is not pedantry. Doing it caught a genuinely wrong value that had been
recalled confidently and would have propagated into stored Stage-2 data,
where fixing it means recomputing gigabytes rather than editing a line.

Which brings us to the last one:

### Know which errors are cheap and which are expensive

A wrong S-N constant is cheap — it lives in Stage 3, which reads a cache that
is still correct. A wrong Wöhler exponent or bin edge is expensive — it is
baked into every Stage-2 `.npz`, and correcting it means recomputing all of
them from raw `.outb`.

`fatigue_config.py` carries `PIPELINE_VERSION` for exactly this reason:
consumers stamp it and refuse data whose stamp does not match. Before
changing anything in that file, work out which side of the line it falls on.
