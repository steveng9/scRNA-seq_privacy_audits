#!/usr/bin/env python3
"""
Representative HLCA sweep — scDesign2 + scVI, donors 10/20/35, 3 trials each,
enhanced (Class B) scMAMA-MIA, BB quad (+aux/-aux) and (sd2) WB quad.

Answers reviewer JvKY's "cross-model generalization rests on too few datasets"
by adding a 4th, distinct-tissue dataset (Human Lung Cell Atlas core) and
showing the attack transfers across a copula generator (scDesign2) and a
neural generator (scVI).

Thin wrapper over experiments/sdg_comparison/run_full_sweep.py — overrides only
the sweep tables and N_TRIALS, so all the resource-gating / OOM-retry / detach
machinery is reused unchanged. Accepts the same CLI flags (--status, --dry-run,
--sdg, --nd, --skip-wb, --skip-bb, ...).

Splits must already exist (build_hlca_splits.py); they are fully disjoint aux.

Run:
    python experiments/hlca_onboard/run_hlca_sweep.py --dry-run
    nohup python experiments/hlca_onboard/run_hlca_sweep.py \
        > experiments/hlca_onboard/_sweep_logs/hlca_sweep.log 2>&1 &
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "sdg_comparison"))

import run_full_sweep as fs  # noqa: E402

# --- HLCA-only overrides (module globals are read at call time) ---
HLCA_ND = [10, 20, 35]
fs.SD2_SWEEP = [("hlca", "scdesign2/no_dp", HLCA_ND)]
fs.OTHER_SWEEP = [("hlca", "scvi/no_dp")]
fs.OTHER_SWEEP_ND = HLCA_ND
fs.N_TRIALS = 3


if __name__ == "__main__":
    if "--log-dir" not in sys.argv:
        sys.argv += ["--log-dir", os.path.join(_HERE, "_sweep_logs")]
    fs.main()
