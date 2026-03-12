#!/bin/bash
# Runs 5 trials of scMAMA-MIA (BB+aux, tm:100) for AIDA 200d.
# Trial 1 already has donor splits, h5ad splits, and trained model copulas
# from a prior run; it will redo generation+assembly (now fixed for OOM)
# then train the BB shadow copula and run the MIA attack.

SRC=/home/golobs/scRNA-seq_privacy_audits/src
PYTHON=python3
LOG=/tmp/aida_200d_bb_aux_log.txt

echo "===== AIDA 200d BB+aux (tm:100) MIA experiments =====" | tee -a $LOG
echo "Started at: $(date)" | tee -a $LOG

for TRIAL in 1 2 3 4 5; do
    echo "" | tee -a $LOG
    echo "--- AIDA 200d trial ${TRIAL} (tm:100, BB+aux) ---" | tee -a $LOG
    $PYTHON -u $SRC/run_experiment.py T /home/golobs/data/aida/exp_cfgs/200d_100.yaml 2>&1 | tee -a $LOG
    echo "Python exit code: ${PIPESTATUS[0]}" | tee -a $LOG
done

echo "" | tee -a $LOG
echo "===== AIDA 200d BB+aux experiments complete =====" | tee -a $LOG
echo "Finished at: $(date)" | tee -a $LOG
