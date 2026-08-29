# Fatigue damage assessment of an OC4 offshore wind jacket

Code, results and documentation from an MEng dissertation at University
College Cork. The wider dissertation asks how much of a decommissioned
jacket substructure could be reused; **this repository is the fatigue
engine underneath it** — everything up to and including the accumulated
fatigue damage number for each member and each joint.

The approach is a 414-run OpenFAST campaign over the OC4 reference jacket
carrying the NREL 5MW turbine, followed by a fatigue post-processing pipeline
that produces a per-member and per-joint damage map, a corrosion trajectory
to 25 years, and — for the joints — two can-thickness retrofit scenarios.

The static/ULS governing-life check and the embodied-carbon comparison that
the dissertation builds on top of these numbers are **not** published here.
The reuse-acceptance classification code is not here either, though the
interactive viewer below does show its per-member outcomes. See *What is not
here* below.

---

## Interactive viewer

**[Explore the results in 3D →](https://eken97.github.io/FYP_Offshore_Reuse/)**

A self-contained page showing the whole jacket with every member and joint
colour-coded by fatigue damage, across the corrosion trajectory to 25 years
and both can-thickness retrofit scenarios. A second tab visualises the
dissertation's L0–L4 reuse-classification outcomes. Built from the tables in
`results/`; the page source is [`index.html`](index.html).

---

## What it found

- **Members are not fatigue-critical.** The worst member reaches a damage
  ratio of about 0.115 at 25 years — a fatigue life of roughly 217 years.
- **Joints are where the fatigue is.** Several splash-zone joints exceed a
  damage ratio of 1.0 by more than an order of magnitude, and the same
  splash-zone joints stay the worst under every scenario examined —
  pristine, corroded, and after either can-thickness retrofit.

Numbers come from `results/`. Read [docs/limitations.md](docs/limitations.md)
before quoting any of them.

---

## Layout

```
simulation/       OpenFAST campaign: config, deck editing, build, run, work split
postprocessing/   the fatigue pipeline: geometry, stress, rainflow, S-N,
                  damage (pristine / corrosion / can-retrofit), figure builders
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

**The downstream assessment code.** The dissertation takes the damage numbers
this repository produces and adds a proposed L0–L4 reuse-acceptance
classification, a static/ULS governing-life check, and an embodied-carbon
comparison. The code and result tables for those steps are not published
here — this repository ends at the fatigue damage number per member and per
joint. The interactive viewer does display the reuse-classification
*outcomes* (retained / downgraded / recycle per member) next to the fatigue
results as a visual summary; the classification method itself is defined in
the dissertation.

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
