#!/usr/bin/env bash
# Kill the B2 (marginal-noise) sweep scheduler + all its trial workers.
# NOTE: patterns are limited to the two script filenames on purpose — a broader
# pattern like "b2_marginal" also matches any launching/monitoring shell whose
# command line contains that path, causing an accidental self-kill.
pkill -f run_b2_sweep.py
pkill -f run_b2_trial.py
echo "B2 scheduler + trial workers signalled."
