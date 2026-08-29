# Setup

There are two quite different things you might want to do with this
repository, and they need very different amounts of setup.

| I want to… | What I need |
|---|---|
| Read the results, regenerate the figures, run the worked example | Python and `pip install -r requirements.txt`. Nothing else. |
| Re-run the fatigue post-processing on my own OpenFAST output | The above, plus your own `.outb` files |
| Re-run the whole 414-case campaign | The above, plus OpenFAST, TurbSim, the NREL decks, and roughly a fortnight of wall-clock time |

Start at the top. Only go further down if you actually need to.

---

## 1. Python

Python 3.10 or newer (the code uses `X | None` annotations).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Check it worked by running the worked example:

```bash
python scripts/run_example.py
```

You should see `PASS: 224 rows match expected_stage3_member.csv`. That runs
the member-track Stage 3 aggregation over the small histogram cache shipped
in `data/example/` and compares it against the checked-in expected output.
No OpenFAST, no campaign data, no downloads.

Read [pipeline.md](pipeline.md) for what that example actually computes, and
why its numbers are not the thesis numbers.

---

## 2. Regenerating the figures

The figure builders read the tables in `results/` and write PNG + SVG pairs
into `figures/`:

```bash
cd postprocessing
python report_figures_members.py        # 11 member-track figures
python report_figures_static_check.py   # 2 static-check figures
python report_figures_joints.py         # 11 joint-track figures
```

The joint and member builders need the jacket geometry, which they read from
a SubDyn input deck. If you have not fetched the NREL decks (step 4 below),
put any copy of `NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat` somewhere and set:

```bash
export OC4_SUBDYN_DAT=/path/to/NRELOffshrBsline5MW_OC4Jacket_SubDyn.dat
```

Every copy of that deck used in this project describes the same structure —
see [traps.md](traps.md#the-subdyn-deck-you-use-barely-matters-except-once).

---

## 3. Running the post-processing on your own OpenFAST output

Point the pipeline at your run tree and at somewhere to keep the Stage-2
histogram cache:

```bash
export OC4_RUN_ROOT=/path/to/your/runs         # folders holding the .outb files
export OC4_STAGE2_DIR=/path/to/stage2-cache    # ~9 MB per run/seed
cd postprocessing
python run_pipeline.py
```

`run_pipeline.py` has a single `SELECT` dict near the bottom that decides
which stages and which tracks execute. It is deliberately an explicit dict
rather than a command-line switchboard, so that any given run of the pipeline
is auditable after the fact. Edit it, do not add flags.

Budget the disk: the Stage-2 cache for the full 414-run campaign is about
3.6 GB, and the raw OpenFAST output it is derived from is about 119 GB.

---

## 4. Fetching the NREL OpenFAST model

**This repository does not contain any NREL model files.** The OC4 jacket
decks, the shared 5MW baseline data and the compiled `DISCON.dll` controller
belong to NREL and are published in the OpenFAST regression-test suite. See
[NOTICE](../NOTICE).

To stage them locally:

```bash
# let the script fetch the pinned commit for you (needs git)
python scripts/fetch_openfast_inputs.py --clone

# or point it at an r-test checkout you already have
python scripts/fetch_openfast_inputs.py --rtest /path/to/r-test
```

This copies 72 files into `inputs/` and verifies every one against the
SHA-256 manifest in `scripts/rtest_manifest.json`. `inputs/` is gitignored
in full, so nothing NREL-owned can be committed back by accident.

The manifest pins r-test commit `dd5feaaa` (2026-03-12) — the exact revision
the published results were produced from. If verification fails, your model
differs from the one behind the numbers in `results/`. That is worth knowing
before you spend a fortnight of compute on it.

To re-check an existing tree without copying:

```bash
python scripts/fetch_openfast_inputs.py --verify-only
```

---

## 5. Running the campaign

You need OpenFAST and TurbSim binaries. Build or download them from
<https://github.com/OpenFAST/openfast>. The published results used the
double-precision OpenFAST release build.

Environment variables — all optional except the two executables:

| Variable | Meaning | Default |
|---|---|---|
| `OC4_OPENFAST_EXE` | OpenFAST binary | `openfast` on `PATH` |
| `OC4_TURBSIM_EXE` | TurbSim binary | `turbsim` on `PATH` |
| `OC4_RTEST_5MW_BASELINE` | shared NREL baseline folder | `inputs/base_files/5MW_Baseline` |
| `OC4_PROJECT_ROOT` | root the campaign writes under | the repo |
| `OC4_RUN_ROOT` | explicit run-data location | auto-detected, see below |
| `OC4_WORKER` | this machine's name in `assignment.json` | the hostname |
| `OC4_CORES` | concurrent OpenFAST processes | `cpu_count() - 1` |
| `OC4_EST_HOURS_PER_CASE` | mean hours per case, for deadline arithmetic | `3.2` |
| `OC4_STAGE2_DIR` | Stage-2 histogram cache | auto-detected, see below |
| `OC4_SUBDYN_DAT` | specific SubDyn deck to read geometry from | auto-detected |

Then print the readiness report, which checks every path before you commit
compute to anything:

```bash
python simulation/config.py
```

It reports the resolved paths, whether the executables and decks exist, how
much free space the run root has, and expands the 69 bins into 414 cases with
a per-bin sanity check of rotor speed, pitch and turbulence intensity.

To run:

```bash
cd simulation
python plan_assignment.py --apply      # decide which machine runs which cases
python campaign.py                     # build + run this machine's share
```

### Splitting the work across machines

The campaign was run across two machines at once. `plan_assignment.py`
divides the 69 bins by measured throughput and writes `assignment.json`:

```bash
python plan_assignment.py --machine fast:8:3.2 --machine slow:4:3.23 --apply
```

The spec is `NAME:CORES:HOURS_PER_CASE`. With no `--machine` at all,
everything is assigned to the current machine. Machines are matched by
`OC4_WORKER` (defaulting to hostname), so set that to one of the names you
used. `rebalance.py` moves not-yet-started cases between machines mid-campaign.

Why bins and not individual cases: see
[decisions.md](decisions.md#the-work-split-is-by-bin-not-by-case).

### Where run data goes

`run_root()` resolves in this order:

1. `OC4_RUN_ROOT`, if set
2. an external drive — drive letters `D:` to `Z:` are scanned for a folder
   named `OC4_CAMPAIGN` containing a marker file `.oc4_campaign_drive`
3. `simulation/_staging/`, as a fallback

The marker-file scan exists because the same physical drive enumerates as a
different letter on different machines. Hardcoding `D:` was tried first and
broke exactly as you would expect. Override the folder and marker names with
`OC4_DRIVE_SUBFOLDER` and `OC4_DRIVE_MARKER`.
