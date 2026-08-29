# Fatigue-driven reuse assessment of an OC4 offshore wind jacket

Code, results and documentation from an MEng dissertation at University
College Cork asking a fairly narrow question: **when a jacket substructure is
decommissioned, how much of it could actually be reused — and what decides
that?**

The approach is a 414-run OpenFAST campaign over the OC4 reference jacket
carrying the NREL 5MW turbine, followed by a fatigue post-processing pipeline
that produces a per-member and per-joint damage map, a corrosion trajectory
to 25 years, a static/ULS check, and an L0–L4 reuse classification.

Existing reuse guidance for structural steel — SCI P427 and similar — has a
fatigue-shaped hole in it: it was written for buildings, where fatigue is
rarely the governing question. This work is an attempt to fill that gap for
offshore substructures.

---

## What it found

- **Members are not fatigue-critical.** The worst member reaches a damage
  ratio of about 0.115 at 25 years — a fatigue life of roughly 217 years.
- **The static check governs, not fatigue.** Of the splash-zone members whose
  remaining life is limited at all, all 32 are governed by the static
  compression or tension check (27 and 5 respectively), never by fatigue. The
  shortest governing life is 34.5 years.
- **Joints are the binding constraint.** Several splash-zone joints exceed a
  damage ratio of 1.0 by more than an order of magnitude, and the same
  splash-zone joints fail under every scenario examined.
- **No member reaches structural reuse.** Under both can-thickness retrofit
  scenarios, 80 of 112 members classify as L2 (component reuse) and 32 as L3
  (downgraded reuse). All four bays fail the L1 structural-reuse gate.

The practical reading is that reuse of this structure is credible at the
component level and not at the structural level, and that the constraint sits
at the joints — which is where existing reuse guidance says least.

Numbers come from `results/`. Read [docs/limitations.md](docs/limitations.md)
before quoting any of them.

---

## Layout

```
simulation/       OpenFAST campaign: config, deck editing, build, run, work split
postprocessing/   the fatigue pipeline: geometry, stress, rainflow, S-N, damage,
                  reuse classification, figure builders
scripts/          fetch the NREL decks; run the worked example
data/             metocean bin table + one worked-example Stage-2 cache
results/          the campaign's result tables (CSV/XLSX)
figures/          the report figures (PNG + SVG pairs)
docs/             why it is built this way, and what will bite you
```

## Start here

```bash
pip install -r requirements.txt
python scripts/run_example.py
```

That runs the member-track Stage 3 aggregation over the shipped example cache
and checks it against the expected output. No OpenFAST, no downloads, no
campaign data.

Then:

| Document | For |
|---|---|
| [docs/setup.md](docs/setup.md) | installing, environment variables, running the campaign |
| [docs/pipeline.md](docs/pipeline.md) | what runs in what order, and the file formats between stages |
| [docs/decisions.md](docs/decisions.md) | why it is built this way; alternatives considered and dropped |
| [docs/traps.md](docs/traps.md) | **the most useful file here** — things that silently produce wrong answers |
| [docs/assumptions.md](docs/assumptions.md) | every assumed value and its verification status |
| [docs/reuse-criteria.md](docs/reuse-criteria.md) | the L0–L4 ladder and its thresholds |
| [docs/limitations.md](docs/limitations.md) | what this does **not** establish |

---

## What is not here, and why

**The NREL model files.** The OC4 jacket decks, the shared 5MW baseline data
and the compiled `DISCON.dll` are NREL's work, published in the OpenFAST
r-test under Apache-2.0. They are not redistributed here. Every deck this
project consumes was verified byte-identical to its upstream copy — all of
this project's modifications are applied at run time, in code, by
`simulation/of_inputs.py`. Stage them with:

```bash
python scripts/fetch_openfast_inputs.py --clone
```

which fetches the pinned r-test commit and verifies all 72 files by SHA-256.
See [NOTICE](NOTICE).

**The raw campaign output.** 119 GB of `.outb` across 414 runs, plus a 3.6 GB
Stage-2 histogram cache. Neither belongs in git. `data/example/` ships one
condition at one seed so the pipeline is runnable, not reproducible — the
distinction matters and is spelled out in
[docs/pipeline.md](docs/pipeline.md#the-worked-example).

**The standards.** DNV-RP-C203, DNV-ST-0126, DNVGL-RP-0005, EN 1993-1-1 and
IEC 61400 are licensed documents and are not reproduced. Where a parameter
comes from one, the source clause or table is cited in the code so you can
check it against your own copy.

**The dissertation itself**, and third-party validation datasets.

---

## Reproducibility, honestly

You cannot rebuild this from scratch without re-running the campaign: roughly
two machine-weeks of compute and 119 GB of intermediate output. What you get
instead is the complete code path, the decisions behind it, the exact model
revision it ran against, the durable intermediate contract, and every result
table and figure.

`scripts/run_example.py` doubles as a regression test — it will tell you if a
change to the pipeline has altered its arithmetic.

---

## Licence

Everything in this repository — code, results, figures, data and docs — is
CC BY 4.0: use it for anything, including commercially, just give credit.
See [LICENSE](LICENSE) and [CITATION.cff](CITATION.cff).

Third-party material (the NREL model files, the standards referenced) is not
distributed here at all — see [NOTICE](NOTICE).
