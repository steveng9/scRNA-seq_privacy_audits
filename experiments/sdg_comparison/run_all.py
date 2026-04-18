"""
Master orchestrator — synthetic data GENERATION ONLY.

Trains each SDG method and saves synthetic.h5ad for every (dataset, donor
count, trial) combination.  No MIA attacks are run at this stage.

Resource budget
---------------
  CPU pool (3 workers):  scDesign3 jobs — each spawns 15 R subprocesses
                         (3 × 15 = 45, safe on 48 cores)
  GPU-0 (1 worker):      scVI and scDiffusion jobs, round-robin
  GPU-1 (1 worker):      scVI and scDiffusion jobs, round-robin

Ordering
--------
  All OneK1K jobs are queued before AIDA jobs.  Within each dataset,
  CPU and GPU pools run concurrently.

Kill-all
--------
  Every subprocess process-group ID is appended to PID_FILE.
  Run:  bash experiments/sdg_comparison/kill_all.sh

Launch (detached from SSH session)
------------------------------------
  nohup conda run --no-capture-output -n tabddpm_ \\
      python experiments/sdg_comparison/run_all.py \\
      > /tmp/sdg_comparison.log 2>&1 &
  echo $! >> /tmp/sdg_comparison_pids.txt

  tail -f /tmp/sdg_comparison.log
  ls  /tmp/sdg_comparison_logs/     # per-job logs
  bash experiments/sdg_comparison/kill_all.sh
"""

import argparse
import os
import signal
import subprocess
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO     = "/home/golobs/scRNA-seq_privacy_audits"
DATA     = "/home/golobs/data/scMAMAMIA"
LOG_DIR  = "/tmp/sdg_comparison_logs"
PID_FILE = "/tmp/sdg_comparison_pids.txt"

CONDA_ENV      = "tabddpm_"
SCVI_CONDA_ENV = "scvi_"
SCDF_CONDA_ENV = "scdiff_"
NMF_CONDA_ENV  = "nmf_"

PYTHON       = f"conda run --no-capture-output -n {CONDA_ENV} python"
GEN_TRIAL_PY = f"{REPO}/experiments/sdg_comparison/generate_trial.py"
COMPUTE_HVG  = f"{REPO}/experiments/sdg_comparison/compute_hvgs.py"

N_TRIALS          = 5
N_CPU_WORKERS     = 1   # each scDesign3 job spawns up to 15 R workers; 1 concurrent avoids OOM
N_NMF_WORKERS     = 4   # NMF is CPU-only but fast and low-memory; allow more concurrent jobs
GPU_IDS           = [0, 1]

# Donor counts per method × dataset  (from the experiment plan)
# 490 = CAMDA-scale (OneK1K: 491 train / 490 holdout, aux = 200 holdout subsample)
OK_SD3G   = [2, 5, 10, 20, 50, 100, 200, 490]
OK_SD3V   = [10, 20, 50, 100, 490]
OK_SCVI   = [5, 10, 20, 50, 100, 490]
OK_SCDF   = [10, 20, 50, 490]
OK_NMF    = [10, 20, 50, 100, 490]

AIDA_SD3G = [10, 20, 50, 100]
AIDA_SD3V = [10, 20, 50]
AIDA_SCVI = [10, 20, 50]
AIDA_SCDF = [20, 50]
AIDA_NMF  = [10, 20, 50, 100]

# ---------------------------------------------------------------------------
# PID tracking
# ---------------------------------------------------------------------------
os.makedirs(LOG_DIR, exist_ok=True)
_pid_lock = threading.Lock()


def _register_pgid(pgid: int):
    with _pid_lock:
        with open(PID_FILE, "a") as f:
            f.write(f"{pgid}\n")


def _run_subprocess(cmd: str, log_path: str, env: dict = None) -> int:
    """Launch cmd in a new process group; stream to log.  Returns exit code."""
    merged = {**os.environ, **(env or {})}
    with open(log_path, "a") as log:
        log.write(f"\n{'='*60}\nCMD: {cmd}\n{'='*60}\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, shell=True,
            stdout=log, stderr=subprocess.STDOUT,
            env=merged,
            preexec_fn=os.setsid,   # new process group → clean kill
        )
        _register_pgid(os.getpgid(proc.pid))
        ret = proc.wait()
        log.write(f"\n[EXIT {ret}]\n")
    return ret


# ---------------------------------------------------------------------------
# GPU pool
# ---------------------------------------------------------------------------

class _GpuPool:
    """Round-robin GPU slot — max one active job per device."""
    def __init__(self, gpu_ids):
        self._available = list(gpu_ids)
        self._cond = threading.Condition()

    def acquire(self) -> int:
        with self._cond:
            while not self._available:
                self._cond.wait()
            return self._available.pop(0)

    def release(self, gpu_id: int):
        with self._cond:
            self._available.append(gpu_id)
            self._cond.notify()


# ---------------------------------------------------------------------------
# Per-trial job builders
# ---------------------------------------------------------------------------

def _sd3_jobs(src, copula, donor_counts):
    """Yield (label, cmd, log, env) for every scDesign3 trial.

    src    — base dataset key, e.g. "ok" or "aida"
    copula — "gaussian" or "vine"
    """
    full_h5ad  = f"{DATA}/{src}/full_dataset_cleaned.h5ad"
    hvg_path   = f"{DATA}/{src}/hvg_full.csv"
    out_prefix = f"{DATA}/{src}/scdesign3/{copula}"

    for nd in donor_counts:
        for trial in range(1, N_TRIALS + 1):
            splits_dir = f"{DATA}/{src}/scdesign2/no_dp/{nd}d/{trial}/datasets"
            out_dir    = f"{out_prefix}/{nd}d/{trial}"
            label      = f"{src}_sd3{copula[0]}_{nd}d_t{trial}"
            cmd = (
                f"{PYTHON} {GEN_TRIAL_PY} "
                f"--generator sd3_{copula} "
                f"--dataset {full_h5ad} "
                f"--splits-dir {splits_dir} "
                f"--out-dir {out_dir} "
                f"--hvg-path {hvg_path}"
            )
            log = os.path.join(LOG_DIR, f"{label}.log")
            yield label, cmd, log, {}


def _scvi_jobs(src, donor_counts):
    """Yield (label, cmd, log, env) for every scVI trial (env set later)."""
    full_h5ad = f"{DATA}/{src}/full_dataset_cleaned.h5ad"
    hvg_path  = f"{DATA}/{src}/hvg_full.csv"

    for nd in donor_counts:
        for trial in range(1, N_TRIALS + 1):
            splits_dir = f"{DATA}/{src}/scdesign2/no_dp/{nd}d/{trial}/datasets"
            out_dir    = f"{DATA}/{src}/scvi/no_dp/{nd}d/{trial}"
            label      = f"{src}_scvi_{nd}d_t{trial}"
            cmd = (
                f"{PYTHON} {GEN_TRIAL_PY} "
                f"--generator scvi "
                f"--dataset {full_h5ad} "
                f"--splits-dir {splits_dir} "
                f"--out-dir {out_dir} "
                f"--hvg-path {hvg_path} "
                f"--conda-env {SCVI_CONDA_ENV}"
            )
            log = os.path.join(LOG_DIR, f"{label}.log")
            yield label, cmd, log   # env added by caller


def _scdf_jobs(src, donor_counts):
    """Yield (label, cmd, log) for every scDiffusion trial."""
    full_h5ad = f"{DATA}/{src}/full_dataset_cleaned.h5ad"
    hvg_path  = f"{DATA}/{src}/hvg_full.csv"

    for nd in donor_counts:
        for trial in range(1, N_TRIALS + 1):
            splits_dir = f"{DATA}/{src}/scdesign2/no_dp/{nd}d/{trial}/datasets"
            out_dir    = f"{DATA}/{src}/scdiffusion/no_dp/{nd}d/{trial}"
            label      = f"{src}_scdf_{nd}d_t{trial}"
            cmd = (
                f"{PYTHON} {GEN_TRIAL_PY} "
                f"--generator scdiffusion "
                f"--dataset {full_h5ad} "
                f"--splits-dir {splits_dir} "
                f"--out-dir {out_dir} "
                f"--hvg-path {hvg_path} "
                f"--conda-env {SCDF_CONDA_ENV}"
            )
            log = os.path.join(LOG_DIR, f"{label}.log")
            yield label, cmd, log   # env added by caller


def _nmf_jobs(src, donor_counts, n_components=20, dp_mode="none"):
    """Yield (label, cmd, log, env) for every NMF trial."""
    full_h5ad = f"{DATA}/{src}/full_dataset_cleaned.h5ad"
    hvg_path  = f"{DATA}/{src}/hvg_full.csv"

    for nd in donor_counts:
        for trial in range(1, N_TRIALS + 1):
            splits_dir = f"{DATA}/{src}/scdesign2/no_dp/{nd}d/{trial}/datasets"
            out_dir    = f"{DATA}/{src}/nmf/no_dp/{nd}d/{trial}"
            label      = f"{src}_nmf_{nd}d_t{trial}"
            cmd = (
                f"{PYTHON} {GEN_TRIAL_PY} "
                f"--generator nmf "
                f"--dataset {full_h5ad} "
                f"--splits-dir {splits_dir} "
                f"--out-dir {out_dir} "
                f"--hvg-path {hvg_path} "
                f"--conda-env {NMF_CONDA_ENV} "
                f"--n-components {n_components} "
                f"--dp-mode {dp_mode}"
            )
            log = os.path.join(LOG_DIR, f"{label}.log")
            yield label, cmd, log, {}


# ---------------------------------------------------------------------------
# Job runners (run in threads)
# ---------------------------------------------------------------------------

_cpu_sem = threading.Semaphore(N_CPU_WORKERS)
_nmf_sem = threading.Semaphore(N_NMF_WORKERS)


def _run_cpu_job(label, cmd, log, env):
    _cpu_sem.acquire()
    try:
        print(f"[START cpu] {label}", flush=True)
        ret = _run_subprocess(cmd, log, env)
        status = "OK" if ret == 0 else f"ERR({ret})"
        print(f"[{status}]  {label}", flush=True)
        return label, ret
    finally:
        _cpu_sem.release()


def _run_nmf_job(label, cmd, log, env):
    _nmf_sem.acquire()
    try:
        print(f"[START nmf] {label}", flush=True)
        ret = _run_subprocess(cmd, log, env)
        status = "OK" if ret == 0 else f"ERR({ret})"
        print(f"[{status}]  {label}", flush=True)
        return label, ret
    finally:
        _nmf_sem.release()


def _run_gpu_job(label, cmd, log, gpu_pool: _GpuPool):
    gpu_id = gpu_pool.acquire()
    env = {"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    try:
        print(f"[START GPU-{gpu_id}] {label}", flush=True)
        ret = _run_subprocess(cmd, log, env)
        status = "OK" if ret == 0 else f"ERR({ret})"
        print(f"[{status}]  {label}", flush=True)
        return label, ret
    finally:
        gpu_pool.release(gpu_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic scRNA-seq data for all SDG methods."
    )
    parser.add_argument(
        "--generators", nargs="+",
        choices=["sd3", "scvi", "scdiff", "nmf"],
        default=None,
        help="Only run these generators (default: all). "
             "sd3=scDesign3, scvi=scVI, scdiff=scDiffusion, nmf=NMF",
    )
    parser.add_argument(
        "--skip-hvg", action="store_true",
        help="Skip HVG recomputation step (use existing hvg_full.csv files)",
    )
    args = parser.parse_args()

    run_sd3  = args.generators is None or "sd3"    in args.generators
    run_scvi = args.generators is None or "scvi"   in args.generators
    run_scdf = args.generators is None or "scdiff" in args.generators
    run_nmf  = args.generators is None or "nmf"    in args.generators

    # ------------------------------------------------------------------
    # Step 0: Recompute HVGs (blocks everything else — must finish first)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 0: Recomputing HVGs from full datasets")
    print("="*60, flush=True)
    if args.skip_hvg:
        print("  [SKIP] --skip-hvg passed; using existing hvg_full.csv files.", flush=True)
    else:
        ret = _run_subprocess(
            f"{PYTHON} {COMPUTE_HVG}",
            os.path.join(LOG_DIR, "compute_hvgs.log"),
        )
        if ret != 0:
            sys.exit(f"HVG computation failed (exit {ret}). Check {LOG_DIR}/compute_hvgs.log")

    # ------------------------------------------------------------------
    # Step 1: Build job lists — OneK1K first, then AIDA
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("STEP 1: Queuing generation jobs")
    print("="*60, flush=True)

    # scDesign3 CPU jobs
    sd3_jobs = (
        list(_sd3_jobs("ok",   "gaussian", OK_SD3G)) +
        list(_sd3_jobs("ok",   "vine",     OK_SD3V)) +
        list(_sd3_jobs("aida", "gaussian", AIDA_SD3G)) +
        list(_sd3_jobs("aida", "vine",     AIDA_SD3V))
    ) if run_sd3 else []

    # NMF CPU jobs (fast; separate semaphore allows N_NMF_WORKERS concurrent)
    nmf_jobs = (
        list(_nmf_jobs("ok",   OK_NMF)) +
        list(_nmf_jobs("aida", AIDA_NMF))
    ) if run_nmf else []

    # GPU jobs: scVI (OneK1K → AIDA) then scDiffusion (OneK1K → AIDA)
    gpu_scvi_jobs = (
        list(_scvi_jobs("ok",   OK_SCVI)) +
        list(_scvi_jobs("aida", AIDA_SCVI))
    ) if run_scvi else []
    gpu_scdf_jobs = (
        list(_scdf_jobs("ok",   OK_SCDF)) +
        list(_scdf_jobs("aida", AIDA_SCDF))
    ) if run_scdf else []
    all_gpu_jobs  = gpu_scvi_jobs + gpu_scdf_jobs

    n_cpu = len(sd3_jobs)
    n_nmf = len(nmf_jobs)
    n_gpu = len(all_gpu_jobs)
    print(f"  scDesign3 trials : {n_cpu}")
    print(f"  NMF trials       : {n_nmf}")
    print(f"  GPU trials       : {n_gpu}")
    print(f"  Total            : {n_cpu + n_nmf + n_gpu}", flush=True)

    # ------------------------------------------------------------------
    # Step 2: Launch with resource management
    # ------------------------------------------------------------------
    gpu_pool = _GpuPool(GPU_IDS)
    futures  = {}

    max_workers = len(sd3_jobs) + len(nmf_jobs) + len(all_gpu_jobs)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:

        # scDesign3 — CPU (semaphore enforces N_CPU_WORKERS at a time)
        for label, cmd, log, env in sd3_jobs:
            f = ex.submit(_run_cpu_job, label, cmd, log, env)
            futures[f] = label

        # NMF — CPU (separate semaphore allows N_NMF_WORKERS concurrently)
        for label, cmd, log, env in nmf_jobs:
            f = ex.submit(_run_nmf_job, label, cmd, log, env)
            futures[f] = label

        # GPU jobs — round-robin GPU assignment
        for item in all_gpu_jobs:
            label, cmd, log = item
            f = ex.submit(_run_gpu_job, label, cmd, log, gpu_pool)
            futures[f] = label

        # Collect
        failed = []
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                _, ret = fut.result()
                if ret != 0:
                    failed.append(name)
            except Exception as e:
                print(f"[ERROR] {name}: {e}", flush=True)
                failed.append(name)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total  = len(futures)
    n_fail = len(failed)
    print("\n" + "="*60)
    print(f"GENERATION COMPLETE:  {total - n_fail}/{total} succeeded")
    if failed:
        print(f"Failed jobs ({n_fail}):")
        for j in failed:
            print(f"  {j}")
    print(f"Logs: {LOG_DIR}/")
    print("="*60)


if __name__ == "__main__":
    # Register own PID so kill_all.sh can also stop the master process
    with open(PID_FILE, "a") as f:
        f.write(f"{os.getpid()}\n")
    main()
