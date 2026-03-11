#!/bin/bash
# Runs the 5 missing AIDA 50d quality evaluations sequentially.
# Each call to run_quality_eval.py auto-picks the next quality=0 trial.

SRC=/home/golobs/scRNA-seq_privacy_audits/src
PYTHON=python3
LOG=/tmp/aida_50d_quality_eval_log.txt

echo "===== AIDA 50d quality evaluations =====" | tee -a $LOG
echo "Started at: $(date)" | tee -a $LOG

for i in 1 2 3 4 5; do
    echo "" | tee -a $LOG
    echo "--- AIDA 50d trial $i ---" | tee -a $LOG
    $PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/50d_000.yaml 2>&1 | tee -a $LOG
    echo "Exit code: $?" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "===== AIDA 50d quality evaluations complete =====" | tee -a $LOG
echo "Finished at: $(date)" | tee -a $LOG
