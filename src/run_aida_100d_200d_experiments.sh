#!/bin/bash
# Runs the AIDA 100d and 200d MIA experiments (BB-aux threat model, tm:000).
# This generates synthetic data AND runs scMAMA-MIA for each setting.
# After this completes, run_quality_eval.py can be run for those settings.

SRC=/home/golobs/scRNA-seq_privacy_audits/src
PYTHON=python3
LOG=/tmp/aida_100d_200d_experiment_log.txt

echo "===== AIDA 100d/200d MIA experiments =====" | tee -a $LOG
echo "Started at: $(date)" | tee -a $LOG

echo "" | tee -a $LOG
echo "--- AIDA 100d trial 1 (tm:000, BB-aux) ---" | tee -a $LOG
$PYTHON -u $SRC/run_experiment.py T /home/golobs/data/aida/exp_cfgs/100d_000.yaml 2>&1 | tee -a $LOG
echo "Python exit code: ${PIPESTATUS[0]}" | tee -a $LOG

echo "" | tee -a $LOG
echo "--- AIDA 200d trial 1 (tm:000, BB-aux) ---" | tee -a $LOG
$PYTHON -u $SRC/run_experiment.py T /home/golobs/data/aida/exp_cfgs/200d_000.yaml 2>&1 | tee -a $LOG
echo "Python exit code: ${PIPESTATUS[0]}" | tee -a $LOG

echo "" | tee -a $LOG
echo "===== AIDA 100d/200d experiments complete =====" | tee -a $LOG
echo "Finished at: $(date)" | tee -a $LOG
