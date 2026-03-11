#!/bin/bash
# Runs all missing quality evaluations (trials with quality=0 that have synthetic.h5ad).
# Each invocation of run_quality_eval.py picks the next quality=0 trial automatically.

SRC=/home/golobs/scRNA-seq_privacy_audits/src
PYTHON=python3

echo "===== Starting missing quality evaluations ====="
echo "Started at: $(date)"

# HFRA (dataset=ok): 2d trial 6 (trial 5 already completed)
echo ""
echo "--- HFRA 2d trial 6 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/ok/exp_cfgs/2d_000.yaml

# HFRA (dataset=ok): 20d trials 4-5
echo ""
echo "--- HFRA 20d trial 4 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/ok/exp_cfgs/20d_000.yaml
echo "--- HFRA 20d trial 5 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/ok/exp_cfgs/20d_000.yaml

# AIDA: 20d trial 5
echo ""
echo "--- AIDA 20d trial 5 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/20d_000.yaml

# AIDA: 50d trials 1-5
echo ""
echo "--- AIDA 50d trial 1 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/50d_000.yaml
echo "--- AIDA 50d trial 2 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/50d_000.yaml
echo "--- AIDA 50d trial 3 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/50d_000.yaml
echo "--- AIDA 50d trial 4 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/50d_000.yaml
echo "--- AIDA 50d trial 5 ---"
$PYTHON $SRC/run_quality_eval.py T /home/golobs/data/aida/exp_cfgs/50d_000.yaml

echo ""
echo "===== All quality evaluations complete ====="
echo "Finished at: $(date)"
