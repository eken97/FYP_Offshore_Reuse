"""
CANONICAL script for the reuse-track results figures -- produces ONLY the
figures actually approved for the report, nothing else. Trimmed from the
broader iteration script `reuse_results_final.py` (which stays the place
for candidate work still under discussion -- see docs/decisions.md for why
this project keeps a two-script split per track).

Approved: C1 (L0->L1->L2->L3->L4 classification cascade), C3 (jacket reuse
isometric), C4 (mass breakdown), C5 (carbon savings by baseline). C2 (the
bay-level L1 margin dumbbell) was not promoted -- its code stays in
reuse_results_final.py for anyone who wants it, but it is not part of the
approved report set.

Run directly to regenerate every approved report figure from scratch:
    python report_figures_reuse.py

Output: figures/reuse_track/final/ (PNG + SVG each).
"""
from pathlib import Path

import reuse_results_final as rrf

PROJECT = Path(__file__).resolve().parents[1]   # repo root
OUT_DIR = PROJECT / "figures" / "reuse_track" / "final"

if __name__ == "__main__":
    rrf.OUT_DIR = OUT_DIR  # redirect the shared module's save target to final/

    members, bays = rrf.load()
    rrf.fig_C1_cascade(members, bays)
    rrf.fig_C3_jacket_reuse_isometric(members)
    rrf.fig_C4_mass_breakdown()
    rrf.fig_C5_carbon_savings_by_baseline()

    print(f"\n4 approved report figures written to {OUT_DIR}")
    print("C1_reuse_classification_cascade, C3_jacket_reuse_isometric,")
    print("C4_mass_breakdown, C5_carbon_savings_by_baseline")
    print("C2 (bay-level L1 margin dumbbell) was not promoted -- see")
    print("reuse_results_final.py if you want it.")
