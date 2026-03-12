#!/bin/bash
# Run quality evaluations for HFRA (cg) trials that are missing quality results.
# Trials 4 and 5 are missing for all donor sizes that have 5 completed MIA trials.
# Run each config twice (once per missing trial); run_quality_eval.py picks the next
# incomplete trial automatically via the tracking.csv.

SRC=/home/golobs/scRNA-seq_privacy_audits/src
PYTHON=python3
LOG=/tmp/cg_quality_eval_log.txt

echo "===== HFRA (cg) quality evaluations =====" | tee -a $LOG
echo "Started at: $(date)" | tee -a $LOG

for DONORS in 2d 5d 10d 20d; do
    CFG=/home/golobs/data/cg/exp_cfgs/${DONORS}_000.yaml
    echo "" | tee -a $LOG
    echo "--- cg ${DONORS} trial 4 ---" | tee -a $LOG
    $PYTHON -u $SRC/run_quality_eval.py T $CFG 2>&1 | tee -a $LOG
    echo "Exit code: ${PIPESTATUS[0]}" | tee -a $LOG

    echo "" | tee -a $LOG
    echo "--- cg ${DONORS} trial 5 ---" | tee -a $LOG
    $PYTHON -u $SRC/run_quality_eval.py T $CFG 2>&1 | tee -a $LOG
    echo "Exit code: ${PIPESTATUS[0]}" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "===== cg quality evaluations complete =====" | tee -a $LOG
echo "Finished at: $(date)" | tee -a $LOG
