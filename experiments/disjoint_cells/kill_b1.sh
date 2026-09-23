#!/bin/bash
# Kill the B1 sweep schedulers (scDesign2 + scVI) and all trial workers.
# NOTE: patterns are limited to the script filenames on purpose. A broader
# pattern like "b1_disjoint" (the data path) also matches any launching/
# monitoring shell whose command line contains that string -> accidental
# self-kill. Always run this script ALONE, not inside a compound command whose
# own line mentions these script names.
pkill -f run_b1_scvi_sweep.py && echo "  scVI scheduler killed"
pkill -f run_b1_sweep.py       && echo "  scDesign2 scheduler killed"
pkill -f run_b1_trial.py       && echo "  trial drivers killed"
sleep 3
pkill -9 -f run_b1_trial.py 2>/dev/null
# scVI trainer subprocesses spawned by a trial driver:
pkill -f run_scvi_standalone.py 2>/dev/null
echo "B1 (scDesign2 + scVI) schedulers + trial workers signalled."
