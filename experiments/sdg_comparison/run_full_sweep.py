#!/usr/bin/env python3
"""
run_full_sweep.py — Complete scMAMA-MIA Class B sweep across all SDG methods and datasets.

Runs all scMAMA-MIA attacks with the Class B enhancement (optimal gamma=auto), covering:

  scDesign2 (BB quad + WB quad — standard + ClassB in one pass):
    ok   : 2, 5, 10, 20, 50, 100, 200, 490d
    aida : 2, 5, 10, 20, 50, 100, 200d
    cg   : 2, 5, 10, 11d

  Other SDG methods (BB quad only):
    ok + aida : 10, 20, 50d
    Methods   : scvi, scdiffusion, sd3g, sd3v, zinbwave, nmf, nmf+dp, sd2+dp

Generates scDesign2 datasets if they don't exist.
Skips non-SD2 entries where synthetic.h5ad hasn't been generated yet.

Completion is tracked by the presence of AUC values in mamamia_results_classb.csv:
  BB quad → tm:100 and tm:101 in classb file
  WB quad → tm:000 and tm:001 in classb file

Memory-aware parallel execution:
  - Before each launch, /proc/meminfo is checked
  - Max concurrent jobs scales with available RAM and donor-count tier
  - OOM kills (returncode -9) trigger a retry with halved internal workers
  - Other failures are logged to failed_jobs.json for easy manual restart

Usage
-----
  python run_full_sweep.py [options]

  --dry-run           Print all pending jobs without running them
  --status            Print completion table and exit
  --dataset DATASET   Only process entries whose dataset_name contains DATASET
  --sdg SDG           Only process entries whose sdg_key contains SDG
  --nd N              Only process this donor count
  --max-concurrent N  Override automatic concurrent-job limit
  --log-dir PATH      Where to write per-job log files (default: _sweep_logs/)
  --skip-wb           Skip white-box quad runs (do BB quad only)
  --skip-bb           Skip black-box quad runs (do WB quad only, SD2 only)
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
DATA_DIR  = "/home/golobs/data/scMAMAMIA"
RUNNER    = os.path.join(SRC_DIR, "run_experiment.py")
N_TRIALS  = 5
MIN_AUX_DONORS = 10

# ---------------------------------------------------------------------------
# scMAMA-MIA hyper-parameters (Class B optimal from ablation study)
# ---------------------------------------------------------------------------
MAMAMIA_PARAMS = {
    "IMPORTANCE_OF_CLASS_B_FPs": 0.17,
    "epsilon":                   0.0001,
    "mahalanobis":               True,
    "uniform_remapping_fn":      "zinb_cdf",
    "lin_alg_inverse_fn":        "pinv_gpu",
    "closeness_to_correlation_fn": "closeness_to_correlation_1",
    "class_b_gene_set":          "secondary",
    "class_b_scoring":           "llr",
    "class_b_gamma":             "auto",
    "class_b_gamma_noaux":       "auto",
}

# ---------------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------------

# scDesign2 — full donor-count range, both WB and BB quad
SD2_SWEEP = [
    # (base_dataset, sdg_subpath, donor_counts)
    ("ok",   "scdesign2/no_dp", [2, 5, 10, 20, 50, 100, 200, 490]),
    ("aida", "scdesign2/no_dp", [2, 5, 10, 20, 50, 100, 200]),
    ("cg",   "scdesign2/no_dp", [2, 5, 10, 11]),
]

# Non-SD2 — BB quad only; ok and aida; 10/20/50d only
# Entries where synthetic.h5ad doesn't exist are silently skipped.
OTHER_SWEEP = [
    # (base_dataset, sdg_subpath)
    ("ok",   "scvi/no_dp"),
    ("ok",   "scdiffusion/no_dp"),
    ("ok",   "scdesign3/gaussian"),
    ("ok",   "scdesign3/vine"),
    ("ok",   "zinbwave/no_dp"),
    ("ok",   "nmf/no_dp"),
    ("ok",   "nmf/eps_1"),
    ("ok",   "nmf/eps_2.8"),
    ("ok",   "nmf/eps_10"),
    ("ok",   "nmf/eps_100"),
    ("ok",   "nmf/eps_1000"),
    ("ok",   "nmf/eps_10000"),
    ("ok",   "nmf/eps_100000"),
    ("ok",   "nmf/eps_1000000"),
    ("ok",   "nmf/eps_10000000"),
    ("ok",   "nmf/eps_100000000"),
    ("ok",   "scdesign2/eps_1"),
    ("ok",   "scdesign2/eps_10"),
    ("ok",   "scdesign2/eps_100"),
    ("ok",   "scdesign2/eps_1000"),
    ("ok",   "scdesign2/eps_10000"),
    ("ok",   "scdesign2/eps_100000"),
    ("ok",   "scdesign2/eps_1000000"),
    ("ok",   "scdesign2/eps_10000000"),
    ("ok",   "scdesign2/eps_100000000"),
    ("ok",   "scdesign2/eps_1000000000"),
    ("aida", "scvi/no_dp"),
    ("aida", "scdiffusion/no_dp"),
    ("aida", "scdesign3/gaussian"),
    ("aida", "scdesign3/vine"),
    ("aida", "zinbwave/no_dp"),
    ("aida", "nmf/no_dp"),
    ("aida", "nmf/eps_1"),
    ("aida", "nmf/eps_2.8"),
    ("aida", "nmf/eps_10"),
    ("aida", "nmf/eps_100"),
    ("aida", "nmf/eps_1000"),
    ("aida", "nmf/eps_10000"),
    ("aida", "nmf/eps_100000"),
    ("aida", "scdesign2/eps_1"),
    ("aida", "scdesign2/eps_10"),
    ("aida", "scdesign2/eps_100"),
    ("aida", "scdesign2/eps_1000"),
    ("aida", "scdesign2/eps_10000"),
    ("aida", "scdesign2/eps_100000"),
    ("aida", "scdesign2/eps_1000000"),
    ("aida", "scdesign2/eps_10000000"),
    ("aida", "scdesign2/eps_100000000"),
    ("aida", "scdesign2/eps_1000000000"),
]
OTHER_SWEEP_ND = [10, 20, 50]  # donor counts to sweep for all non-SD2 entries

# ---------------------------------------------------------------------------
# Memory / concurrency thresholds
#   nd_tier → (min_free_gb_to_launch, parallel_workers_in_job)
# Jobs are launched only when MemAvailable > min_free_gb.
# parallel_workers is the number of cell-type workers inside run_experiment.py.
# ---------------------------------------------------------------------------
MEM_TIERS = [
    (490,  20, 2),   # nd ≥ 490  : need 20 GB free, 2 internal workers
    (200,  16, 2),   # nd ≥ 200
    (100,  12, 3),   # nd ≥ 100
    (50,    8, 4),   # nd ≥ 50
    (20,    6, 4),   # nd ≥ 20
    (0,     4, 4),   # nd < 20
]

# Global cap on concurrent jobs regardless of memory
MAX_CONCURRENT_HARD = 4

# OOM retry: how many times to retry with halved workers before giving up
MAX_OOM_RETRIES = 3


# ===========================================================================
# Completion checking
# ===========================================================================

def _classb_auc_present(results_classb_path, tm_code):
    """True if mamamia_results_classb.csv has a non-NaN AUC for the given tm_code."""
    if not os.path.exists(results_classb_path):
        return False
    try:
        df = pd.read_csv(results_classb_path)
        auc_row = df[df["metric"] == "auc"]
        if auc_row.empty:
            return False
        col = f"tm:{tm_code}"
        return (col in auc_row.columns) and pd.notna(auc_row[col].values[0])
    except Exception:
        return False


def count_done_quad(data_dir, nd, mode):
    """
    Count trials with complete quad results in mamamia_results_classb.csv.
      bb_quad : looks for tm:100 and tm:101
      wb_quad : looks for tm:000 and tm:001
    """
    tm_aux, tm_noaux = ("000", "001") if mode == "wb_quad" else ("100", "101")
    count = 0
    for t in range(1, N_TRIALS + 1):
        f = os.path.join(data_dir, f"{nd}d", str(t), "results", "mamamia_results_classb.csv")
        if _classb_auc_present(f, tm_aux) and _classb_auc_present(f, tm_noaux):
            count += 1
    return count


def synth_exists(data_dir, nd, trial):
    """True if synthetic.h5ad has been generated for the given trial."""
    return os.path.exists(
        os.path.join(data_dir, f"{nd}d", str(trial), "datasets", "synthetic.h5ad")
    )


def n_synth_available(data_dir, nd):
    """Count trials where synthetic.h5ad already exists."""
    return sum(1 for t in range(1, N_TRIALS + 1) if synth_exists(data_dir, nd, t))


# ===========================================================================
# Config generation
# ===========================================================================

def _parallel_workers_for(nd):
    """Return the configured parallel_workers for a job with this donor count."""
    for threshold, _, pw in MEM_TIERS:
        if nd >= threshold:
            return pw
    return 4


def write_config(dataset_name, base_dataset, nd, mode, cfg_dir, parallel_workers=None):
    """
    Write a run_experiment.py YAML config and return its path.

    mode : "bb_quad" | "wb_quad"
    """
    white_box = (mode == "wb_quad")
    pw = parallel_workers or _parallel_workers_for(nd)

    strategy_fn = "sample_donors_strategy_490" if nd >= 490 else "sample_donors_strategy_2"

    # Look for hvg.csv; fall back to hvg_full.csv
    hvg_path = os.path.join(DATA_DIR, base_dataset, "hvg.csv")
    if not os.path.exists(hvg_path):
        hvg_path = os.path.join(DATA_DIR, base_dataset, "hvg_full.csv")

    cfg = {
        "dir_list": {
            "local":  {"home": REPO_ROOT, "data": DATA_DIR},
            "server": {"home": REPO_ROOT, "data": DATA_DIR},
        },
        "dataset_name":    dataset_name,
        "hvg_path":        hvg_path,
        "generator_name":  "scdesign2",
        "plot_results":    False,
        "parallelize":     True,
        "parallel_workers": pw,
        "min_aux_donors":  MIN_AUX_DONORS,
        "mamamia_params":  dict(MAMAMIA_PARAMS),
        "mia_setting": {
            "sample_donors_strategy_fn": strategy_fn,
            "num_donors": nd,
            "white_box":  white_box,
            "use_wb_hvgs": True,
            "use_aux":    True,   # needed so tracking key is classb:000/classb:100
            "run_quad_bb": True,
        },
    }

    cfg_name = f"{nd}d_{mode}.yaml"
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, cfg_name)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


# ===========================================================================
# Memory monitoring
# ===========================================================================

def get_available_memory_gb():
    """Read MemAvailable from /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 16.0  # conservative fallback


def get_cpu_load_1min():
    """Read 1-minute load average from /proc/loadavg."""
    try:
        with open("/proc/loadavg") as fh:
            return float(fh.read().split()[0])
    except Exception:
        return 0.0


def min_free_gb_for(nd):
    """Minimum free GB required before launching a job with this donor count."""
    for threshold, min_gb, _ in MEM_TIERS:
        if nd >= threshold:
            return min_gb
    return 4


def can_launch(nd, n_currently_running, max_concurrent):
    """True if system resources allow launching one more job."""
    if n_currently_running >= max_concurrent:
        return False
    mem_gb = get_available_memory_gb()
    if mem_gb < min_free_gb_for(nd):
        return False
    # Don't launch if system is already heavily loaded (daniilf protection)
    cpu_count = os.cpu_count() or 8
    if get_cpu_load_1min() > cpu_count * 0.85:
        return False
    return True


# ===========================================================================
# Job data structure and queue builder
# ===========================================================================

class Job:
    def __init__(self, dataset_name, base_dataset, nd, mode, label, is_generative=False):
        self.dataset_name  = dataset_name
        self.base_dataset  = base_dataset
        self.nd            = nd
        self.mode          = mode            # "bb_quad" or "wb_quad"
        self.label         = label
        self.is_generative = is_generative   # True → run_experiment.py generates dataset
        self.cfg_path      = None
        self.retries       = 0
        self.parallel_workers = _parallel_workers_for(nd)

    def __repr__(self):
        return f"Job({self.label}, nd={self.nd}, {self.mode})"


def build_job_queue(args):
    """Build ordered list of pending Job objects from SD2_SWEEP and OTHER_SWEEP."""
    jobs = []
    cfg_root = os.path.join(REPO_ROOT, "experiments", "sdg_comparison", "_sweep_cfgs")

    # ----------------------------------------------------------------
    # scDesign2 jobs (bb_quad + wb_quad)
    # ----------------------------------------------------------------
    for base_dataset, sdg_subpath, donor_counts in SD2_SWEEP:
        dataset_name = f"{base_dataset}/{sdg_subpath}"
        data_dir = os.path.join(DATA_DIR, base_dataset, *sdg_subpath.split("/"))
        cfg_dir  = os.path.join(cfg_root, dataset_name.replace("/", "_"))

        # Apply filters
        if args.dataset and args.dataset not in dataset_name:
            continue
        if args.sdg and args.sdg not in dataset_name:
            continue

        for nd in donor_counts:
            if args.nd and args.nd != nd:
                continue

            modes = []
            if not args.skip_bb:
                modes.append("bb_quad")
            if not args.skip_wb and sdg_subpath.startswith("scdesign2"):
                modes.append("wb_quad")

            for mode in modes:
                n_done = count_done_quad(data_dir, nd, mode)
                # For scDesign2, we generate datasets if not present → use N_TRIALS
                n_needed = N_TRIALS - n_done
                if n_needed <= 0:
                    continue

                label = f"{dataset_name}  {nd}d  [{mode}]"
                # SD2 jobs are generative: run_experiment.py creates data if missing
                job = Job(dataset_name, base_dataset, nd, mode, label, is_generative=True)
                job.cfg_path = write_config(dataset_name, base_dataset, nd, mode, cfg_dir)
                jobs.append(job)

    # ----------------------------------------------------------------
    # Non-SD2 / other SDG jobs (bb_quad only)
    # ----------------------------------------------------------------
    if not args.skip_bb:
        for base_dataset, sdg_subpath in OTHER_SWEEP:
            dataset_name = f"{base_dataset}/{sdg_subpath}"
            data_dir = os.path.join(DATA_DIR, base_dataset, *sdg_subpath.split("/"))

            if args.dataset and args.dataset not in dataset_name:
                continue
            if args.sdg and args.sdg not in dataset_name:
                continue

            cfg_dir = os.path.join(cfg_root, dataset_name.replace("/", "_"))

            for nd in OTHER_SWEEP_ND:
                if args.nd and args.nd != nd:
                    continue

                # Skip if no synthetic data exists
                n_avail = n_synth_available(data_dir, nd)
                if n_avail == 0:
                    continue

                n_done   = count_done_quad(data_dir, nd, "bb_quad")
                n_needed = min(N_TRIALS, n_avail) - n_done
                if n_needed <= 0:
                    continue

                label = f"{dataset_name}  {nd}d  [bb_quad]"
                job = Job(dataset_name, base_dataset, nd, "bb_quad", label)
                job.cfg_path = write_config(dataset_name, base_dataset, nd, "bb_quad", cfg_dir)
                jobs.append(job)

    # Sort: smaller nd first (cheaper jobs first, enables more parallelism early)
    jobs.sort(key=lambda j: (j.nd, j.dataset_name, j.mode))
    return jobs


# ===========================================================================
# Status display
# ===========================================================================

def print_status():
    """Print a summary table of quad completion across all sweep entries."""
    print(f"\n{'=' * 80}")
    print(f"  scMAMA-MIA Quad Sweep Completion")
    print(f"{'=' * 80}\n")
    print(f"  {'Dataset':<40} {'nd':>6}  {'BB-quad':>9}  {'WB-quad':>9}")
    print(f"  {'-'*40} {'-'*6}  {'-'*9}  {'-'*9}")

    for base_dataset, sdg_subpath, donor_counts in SD2_SWEEP:
        dataset_name = f"{base_dataset}/{sdg_subpath}"
        data_dir = os.path.join(DATA_DIR, base_dataset, *sdg_subpath.split("/"))
        for nd in donor_counts:
            bb = count_done_quad(data_dir, nd, "bb_quad")
            wb = count_done_quad(data_dir, nd, "wb_quad")
            bb_s = "✓" if bb == N_TRIALS else (f"~{bb}" if bb > 0 else "·")
            wb_s = "✓" if wb == N_TRIALS else (f"~{wb}" if wb > 0 else "·")
            print(f"  {dataset_name:<40} {nd:>4}d  {bb_s:>9}  {wb_s:>9}")
        print()

    print(f"  {'Dataset':<40} {'nd':>6}  {'BB-quad':>9}")
    print(f"  {'-'*40} {'-'*6}  {'-'*9}")
    for base_dataset, sdg_subpath in OTHER_SWEEP:
        dataset_name = f"{base_dataset}/{sdg_subpath}"
        data_dir = os.path.join(DATA_DIR, base_dataset, *sdg_subpath.split("/"))
        for nd in OTHER_SWEEP_ND:
            avail = n_synth_available(data_dir, nd)
            if avail == 0:
                continue
            bb = count_done_quad(data_dir, nd, "bb_quad")
            bb_s = "✓" if bb == N_TRIALS else (f"~{bb}" if bb > 0 else "·")
            print(f"  {dataset_name:<40} {nd:>4}d  {bb_s:>9}")
        print()

    print(f"\n  ✓ = all {N_TRIALS} trials done   ~N = N done   · = none\n")


# ===========================================================================
# Job execution
# ===========================================================================

def launch_job(job, log_dir):
    """
    Launch run_experiment.py as a subprocess.
    Returns the Popen object.
    """
    os.makedirs(log_dir, exist_ok=True)
    safe_label = job.label.replace("/", "_").replace(" ", "_").replace("[", "").replace("]", "")
    log_path = os.path.join(log_dir, f"{safe_label}_r{job.retries}.log")

    cmd = [sys.executable, RUNNER, job.cfg_path]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )
    proc._log_path = log_path
    proc._log_fh   = log_fh
    return proc


def check_completed_after_job(job):
    """Re-count done trials; returns n_done."""
    data_dir = os.path.join(DATA_DIR, job.base_dataset, *job.dataset_name.split("/")[1:])
    return count_done_quad(data_dir, job.nd, job.mode)


def is_oom_kill(returncode):
    """Return True if returncode indicates the process was killed by the OOM killer."""
    return returncode in (-9, -signal.SIGKILL)


# ===========================================================================
# Main loop
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dry-run",       action="store_true",
                        help="Print all pending jobs without running them")
    parser.add_argument("--status",        action="store_true",
                        help="Print completion table and exit")
    parser.add_argument("--dataset",       default=None,
                        help="Filter: only entries whose dataset_name contains this")
    parser.add_argument("--sdg",           default=None,
                        help="Filter: only entries whose sdg_key contains this")
    parser.add_argument("--nd",            type=int, default=None,
                        help="Filter: only this donor count")
    parser.add_argument("--max-concurrent", type=int, default=0,
                        help="Override automatic concurrent-job limit (0 = auto)")
    parser.add_argument("--log-dir",       default=None,
                        help="Directory for per-job log files (default: _sweep_logs/)")
    parser.add_argument("--skip-wb",       action="store_true",
                        help="Skip white-box quad runs")
    parser.add_argument("--skip-bb",       action="store_true",
                        help="Skip black-box quad runs")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    log_dir = args.log_dir or os.path.join(
        REPO_ROOT, "experiments", "sdg_comparison", "_sweep_logs"
    )
    failed_jobs_path = os.path.join(log_dir, "failed_jobs.json")

    # ----------------------------------------------------------------
    # Build pending queue
    # ----------------------------------------------------------------
    print(f"\n[sweep] Building job queue…", flush=True)
    all_jobs = build_job_queue(args)

    if not all_jobs:
        print("[sweep] No pending jobs. All done or no data found.", flush=True)
        print_status()
        return

    print(f"[sweep] {len(all_jobs)} pending job configs (each runs until 5 trials complete).")

    if args.dry_run:
        print("\n[DRY-RUN] Planned jobs:")
        for j in all_jobs:
            data_dir = os.path.join(DATA_DIR, j.base_dataset, *j.dataset_name.split("/")[1:])
            n_done  = count_done_quad(data_dir, j.nd, j.mode)
            n_avail = n_synth_available(data_dir, j.nd)
            n_target = (N_TRIALS if j.is_generative else min(N_TRIALS, n_avail))
            avail_str = f"({n_avail} synth available)" if not j.is_generative else ""
            print(f"  {j.label:60s}  {n_done}/{n_target} done  {avail_str}")
        return

    max_concurrent = args.max_concurrent if args.max_concurrent > 0 else MAX_CONCURRENT_HARD

    # ----------------------------------------------------------------
    # Tracking state
    # ----------------------------------------------------------------
    running  = {}     # pid → {"proc": Popen, "job": Job, "started": float}
    failed   = []     # list of {"job": Job, "returncode": int, "log": str}
    oom_requeue = []  # jobs to retry with lower workers

    total_launched  = 0
    total_completed = 0
    total_failed    = 0

    # ----------------------------------------------------------------
    # Graceful shutdown on Ctrl+C
    # ----------------------------------------------------------------
    shutting_down = [False]

    def _sigint_handler(sig, frame):
        print("\n[sweep] Ctrl+C received — finishing running jobs, then stopping.", flush=True)
        shutting_down[0] = True

    signal.signal(signal.SIGINT,  _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    # ----------------------------------------------------------------
    # Main polling loop
    # ----------------------------------------------------------------
    pending = list(all_jobs)   # mutable queue; already sorted by nd

    print(f"\n[sweep] Starting.  max_concurrent={max_concurrent}  log_dir={log_dir}\n",
          flush=True)

    while pending or running or oom_requeue:
        # ---- Handle any OOM requeues from previous iteration ----
        if oom_requeue:
            pending = oom_requeue + pending
            oom_requeue.clear()

        # ---- Poll running jobs ----
        for pid in list(running.keys()):
            entry = running[pid]
            proc  = entry["proc"]
            job   = entry["job"]
            rc    = proc.poll()

            if rc is None:
                continue   # still running

            # Job finished
            proc._log_fh.close()
            del running[pid]

            elapsed = time.time() - entry["started"]
            data_dir = os.path.join(DATA_DIR, job.base_dataset,
                                    *job.dataset_name.split("/")[1:])
            n_done_now = count_done_quad(data_dir, job.nd, job.mode)
            n_avail    = n_synth_available(data_dir, job.nd)
            # Generative SD2 jobs always target N_TRIALS; non-SD2 capped by available synth
            n_target   = N_TRIALS if job.is_generative else min(N_TRIALS, n_avail)

            if rc == 0:
                total_completed += 1
                print(f"[sweep] ✓ {job.label}  "
                      f"({n_done_now}/{n_target} done, {elapsed:.0f}s)", flush=True)
                # Requeue if more trials still needed
                if n_done_now < n_target and not shutting_down[0]:
                    pending.insert(0, job)   # prioritise re-run at front of queue
            elif is_oom_kill(rc) and job.retries < MAX_OOM_RETRIES:
                total_failed += 1
                job.retries += 1
                new_pw = max(1, job.parallel_workers // 2)
                job.parallel_workers = new_pw
                # Rewrite config with reduced workers
                cfg_dir = os.path.dirname(job.cfg_path)
                job.cfg_path = write_config(
                    job.dataset_name, job.base_dataset, job.nd, job.mode,
                    cfg_dir, parallel_workers=new_pw,
                )
                print(f"[sweep] ⚠ OOM kill: {job.label}  "
                      f"retry {job.retries}/{MAX_OOM_RETRIES} with {new_pw} workers",
                      flush=True)
                oom_requeue.append(job)
            else:
                total_failed += 1
                print(f"[sweep] ✗ FAILED: {job.label}  "
                      f"rc={rc}  log={proc._log_path}", flush=True)
                failed.append({
                    "label":      job.label,
                    "dataset":    job.dataset_name,
                    "nd":         job.nd,
                    "mode":       job.mode,
                    "returncode": rc,
                    "retries":    job.retries,
                    "log":        proc._log_path,
                    "cfg":        job.cfg_path,
                })

        # ---- Launch new jobs if resources allow ----
        if not shutting_down[0]:
            for job in list(pending):
                if can_launch(job.nd, len(running), max_concurrent):
                    print(f"[sweep] → launching: {job.label}  "
                          f"(mem_avail={get_available_memory_gb():.1f}GB, "
                          f"running={len(running)})", flush=True)
                    proc = launch_job(job, log_dir)
                    running[proc.pid] = {
                        "proc":    proc,
                        "job":     job,
                        "started": time.time(),
                    }
                    pending.remove(job)
                    total_launched += 1
                    time.sleep(2)  # brief gap so launched process can claim memory

        time.sleep(5)

        # ---- Periodic summary (every ~5 min) ----
        if total_launched % 20 == 0 and total_launched > 0:
            print(f"[sweep] progress: {total_launched} launched, "
                  f"{total_completed} completed, {total_failed} failed, "
                  f"{len(pending)} pending, {len(running)} running",
                  flush=True)

    # ----------------------------------------------------------------
    # Final report
    # ----------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"  Sweep finished.  "
          f"Launched: {total_launched}  "
          f"Completed: {total_completed}  "
          f"Failed: {total_failed}")

    if failed:
        os.makedirs(log_dir, exist_ok=True)
        with open(failed_jobs_path, "w") as fh:
            json.dump(failed, fh, indent=2)
        print(f"\n  {len(failed)} FAILED jobs written to: {failed_jobs_path}")
        print("  Failed jobs:")
        for f in failed:
            print(f"    [{f['returncode']}] {f['label']}  log: {f['log']}")

    print(f"{'=' * 70}\n")
    print_status()


if __name__ == "__main__":
    main()
