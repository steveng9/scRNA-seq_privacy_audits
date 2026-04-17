#!/bin/bash
# Re-runs quality evals for AIDA 2d, 5d, 10d (all 5 trials) and 20d (trials 1-4)
# using updated code that produces mmd, lisi, ari_real_vs_syn

SRC=/home/golobs/scRNA-seq_privacy_audits/src
PYTHON=python3
LOG=/tmp/aida_small_quality_evals_log.txt
CFG_DIR=/home/golobs/data/aida/exp_cfgs

echo "===== AIDA small-setting quality evals =====" | tee -a $LOG
echo "Started at: $(date)" | tee -a $LOG

for SETTING in 2d 5d 10d 20d; do
    echo "" | tee -a $LOG
    echo "--- AIDA $SETTING ---" | tee -a $LOG
    # run_quality_eval picks the next incomplete trial automatically; loop 5x
    for i in 1 2 3 4 5; do
        $PYTHON -u $SRC/run_quality_eval.py T $CFG_DIR/${SETTING}_000.yaml 2>&1 | tee -a $LOG
        echo "Python exit code: ${PIPESTATUS[0]}" | tee -a $LOG
    done
done

echo "" | tee -a $LOG
echo "===== Done =====" | tee -a $LOG
echo "Finished at: $(date)" | tee -a $LOG
