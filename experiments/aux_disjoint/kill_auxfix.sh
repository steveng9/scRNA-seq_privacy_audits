#!/bin/bash
# Kill the aux-disjoint sweep scheduler and all trial workers.
# NOTE: patterns match the script FILENAMES only (never the data path
# "aux_disjoint", which would also match this shell's own command line and
# cause an accidental self-kill). Always run this script ALONE.
pkill -f run_auxfix_sweep.py && echo "  scheduler killed"
pkill -f run_auxfix_trial.py && echo "  trial workers killed"
sleep 3
pkill -9 -f run_auxfix_trial.py 2>/dev/null
echo "aux-disjoint scheduler + trial workers signalled."
