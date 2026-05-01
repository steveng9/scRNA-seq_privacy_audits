#!/usr/bin/env python3
"""
run_baselines_sweep.py — Sweep DOMIAS-style baseline MIAs across
{ok, aida, cg}/scdesign2/no_dp at all donor counts up to 200d.

Each pending trial runs MC, GAN-Leaks, GAN-Leaks-Cal, GAN-Leaks-SC, LOGAN-D1,
and DOMIAS-KDE via src/run_baselines.py. The distance baselines use the
batched implementations in src/attacks/baselines/batched_baselines.py
(exact match to the unbatched optimised versions, bounded RAM); the KDE
baseline uses subsampled fits, matching the DOMIAS reference protocol.

Completion is tracked by the presence of
    {trial_dir}/results/baseline_mias/baselines_evaluation_results.csv

Usage
-----
    python experiments/sdg_comparison/run_baselines_sweep.py [options]

      --status            Print completion table and exit
      --dry-run           Print all pending jobs without running them
      --dataset DATASET   Filter by dataset name substring (e.g. "ok", "aida")
      --nd N              Filter to a single donor count
      --max-jobs N        Stop after running N trials (0 = unlimited)
      --log-dir PATH      Where to write per-job log files (default: _baseline_logs/)
      --max-concurrent N  Concurrent jobs (default: 1; baselines are heavy at high nd)
      --skip-nd-above N   Skip donor counts above N (default: 200)

Notes
-----
This sweep deliberately skips:
  - 490d (deferred — see notes/PRIORITY_TODO.md)
  - non-scDesign2 SDG baselines (deferred — see notes/PRIORITY_TODO.md)
  - DP scDesign2 epsilon variants (deferred)
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
DATA_DIR  = "/home/golobs/data/scMAMAMIA"
RUNNER    = os.path.join(SRC_DIR, "run_baselines.py")
N_TRIALS  = 5

# ---------------------------------------------------------------------------
# Sweep definition: scDesign2/no_dp on {ok, aida, cg}.
# All other targets are intentionally deferred.
# ---------------------------------------------------------------------------
SWEEP = [
    ("ok",   "scdesign2/no_dp", [2, 5, 10, 20, 50, 100, 200, 490]),
    ("aida", "scdesign2/no_dp", [2, 5, 10, 20, 50, 100, 200]),
    #("aida", "scdesign2/no_dp", [100, 200]),
    #("ok",   "scdesign2/no_dp", [200]),
    ("cg",   "scdesign2/no_dp", [2, 5, 10, 11, 20]),
    # 490d on OneK1K — synthetic data + copulas exist; baseline runs gated by
    # --skip-nd-above (default 200) so they appear in --status without being
    # auto-launched.
    #("ok",   "scdesign2/no_dp", [490]),
]

# ---------------------------------------------------------------------------
# scMAMA-MIA params block.  run_baselines.py inherits the same config schema as
# run_experiment.py; the mia_setting and mamamia_params blocks are required by
# the Box config loader even though run_baselines.py does not use them.
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
# Memory tiers: minimum free RAM (GB) required before launching a baseline job
# at the given donor count. Distance baselines are batched, but each trial
# still has to hold one copy of train+holdout+aux+synth in memory and run
# scanpy normalisation; the dense HVG-subset matrices for ok 200d are ~10-15 GB.
# ---------------------------------------------------------------------------
MEM_TIERS = [
    (200, 40),
    (100, 20),
    (50,  12),
    (20,   8),
    (0,    5),
]


# ===========================================================================
# Tracking helpers
# ===========================================================================

def baselines_done(data_dir, nd, trial):
    """True iff baselines_evaluation_results.csv exists for this (nd, trial)."""
    return os.path.exists(os.path.join(
        data_dir, f"{nd}d", str(trial), "results", "baseline_mias",
        "baselines_evaluation_results.csv",
    ))


def synth_exists(data_dir, nd, trial):
    return os.path.exists(os.path.join(
        data_dir, f"{nd}d", str(trial), "datasets", "synthetic.h5ad"
    ))


def n_baselines_done(data_dir, nd):
    return sum(1 for t in range(1, N_TRIALS + 1) if baselines_done(data_dir, nd, t))


def n_synth(data_dir, nd):
    return sum(1 for t in range(1, N_TRIALS + 1) if synth_exists(data_dir, nd, t))


def splits_present(base_dataset, nd, trial):
    p = os.path.join(DATA_DIR, base_dataset, "splits", f"{nd}d", str(trial))
    return all(os.path.exists(os.path.join(p, f))
               for f in ("train.npy", "holdout.npy", "auxiliary.npy"))


# ===========================================================================
# Tracking-CSV management
# ===========================================================================
# run_baselines.py reads tracking.csv at DATA/{dataset_name}/{nd}d/tracking.csv
# and walks rows where baselines==0 to pick the next trial. We need to ensure
# such a row exists (baselines==0) for any trial that has synth + splits but
# no baselines result yet.

import pandas as pd


def ensure_tracking_row(data_dir, nd, trial):
    """Ensure tracking.csv has a row for `trial` with baselines=0 (if no result)."""
    tracking_path = os.path.join(data_dir, f"{nd}d", "tracking.csv")
    if os.path.exists(tracking_path):
        df = pd.read_csv(tracking_path)
    else:
        cols = ["trial", "tm:000", "tm:001", "tm:010", "tm:011",
                "tm:100", "tm:101", "tm:110", "tm:111",
                "baselines", "quality"]
        df = pd.DataFrame(columns=cols)

    if "baselines" not in df.columns:
        df["baselines"] = 0
    if "quality" not in df.columns:
        df["quality"] = 0

    if (df["trial"] == trial).any():
        # If the row exists but baselines=1 while no results CSV is present,
        # fix it (covers an aborted prior run that pre-set the flag).
        idx = df.index[df["trial"] == trial][0]
        if not baselines_done(data_dir, nd, trial):
            df.loc[idx, "baselines"] = 0
    else:
        new_row = {c: 0 for c in df.columns}
        new_row["trial"] = trial
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    df.sort_values("trial", inplace=True, kind="stable")
    df.to_csv(tracking_path, index=False)


# ===========================================================================
# Memory check
# ===========================================================================

def get_available_memory_gb():
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 16.0


def min_free_gb_for(nd):
    for threshold, gb in MEM_TIERS:
        if nd >= threshold:
            return gb
    return 4


# ===========================================================================
# Config writer
# ===========================================================================

def write_config(dataset_name, nd, cfg_dir):
    """Write a minimal run_baselines.py-compatible YAML config."""
    cfg = {
        "dir_list": {
            "local":  {"home": REPO_ROOT, "data": DATA_DIR},
            "server": {"home": REPO_ROOT, "data": DATA_DIR},
        },
        "dataset_name":    dataset_name,
        "generator_name":  "scdesign2",
        "plot_results":    False,
        "parallelize":     False,
        "parallel_workers": 1,
        "min_aux_donors":  10,
        "mamamia_params":  dict(MAMAMIA_PARAMS),
        "mia_setting": {
            "sample_donors_strategy_fn": "sample_donors_strategy_2",
            "num_donors": nd,
            "white_box":  False,
            "use_wb_hvgs": True,
            "use_aux":    True,
        },
    }
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{nd}d.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


# ===========================================================================
# Job model
# ===========================================================================

class Job:
    def __init__(self, dataset_name, base_dataset, nd, trial, cfg_path, label):
        self.dataset_name = dataset_name
        self.base_dataset = base_dataset
        self.nd = nd
        self.trial = trial            # logical trial we're targeting; run_baselines
                                      # picks the next baselines==0 row, which may
                                      # be a different trial if some are concurrent.
        self.cfg_path = cfg_path
        self.label = label

    def __repr__(self):
        return f"Job({self.label})"


def build_queue(args):
    """Return a list of Job objects: one per (dataset, nd, trial) needing baselines."""
    cfg_root = os.path.join(REPO_ROOT, "experiments", "sdg_comparison",
                            "_baseline_sweep_cfgs")

    jobs = []
    for base_dataset, sdg_subpath, donor_counts in SWEEP:
        dataset_name = f"{base_dataset}/{sdg_subpath}"
        if args.dataset and args.dataset not in dataset_name:
            continue
        data_dir = os.path.join(DATA_DIR, base_dataset, *sdg_subpath.split("/"))
        cfg_dir  = os.path.join(cfg_root, dataset_name.replace("/", "_"))

        for nd in donor_counts:
            if args.nd and args.nd != nd:
                continue
            if nd > args.skip_nd_above:
                continue

            cfg_path = write_config(dataset_name, nd, cfg_dir)

            for trial in range(1, N_TRIALS + 1):
                if not synth_exists(data_dir, nd, trial):
                    continue
                if not splits_present(base_dataset, nd, trial):
                    continue
                if baselines_done(data_dir, nd, trial):
                    continue

                ensure_tracking_row(data_dir, nd, trial)
                label = f"{dataset_name}  {nd}d  t{trial}"
                jobs.append(Job(dataset_name, base_dataset, nd, trial, cfg_path, label))

    # Smaller nd first (cheaper warm-ups), then per-dataset trial order is
    # already preserved by insertion.
    jobs.sort(key=lambda j: (j.nd, j.dataset_name, j.trial))
    return jobs


# ===========================================================================
# Job execution
# ===========================================================================

def launch_job(job, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    safe_label = job.label.replace("/", "_").replace(" ", "_")
    log_path = os.path.join(log_dir, f"{safe_label}.log")

    cmd = [sys.executable, RUNNER, "F", job.cfg_path]
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        cwd=REPO_ROOT,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    proc._log_path = log_path
    proc._log_fh   = log_fh
    return proc


def kill_all(running):
    for pid, entry in list(running.items()):
        try:
            os.killpg(os.getpgid(entry["proc"].pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            entry["proc"]._log_fh.close()
        except Exception:
            pass


# ===========================================================================
# Status display
# ===========================================================================

def print_status():
    print("\n" + "=" * 78)
    print("  Baseline-MIA Sweep Completion")
    print("=" * 78 + "\n")
    print(f"  {'Dataset':<28} {'nd':>4}   {'baselines':>10}   {'synth':>8}")
    print(f"  {'-'*28} {'-'*4}   {'-'*10}   {'-'*8}")
    for base_dataset, sdg_subpath, donor_counts in SWEEP:
        dataset_name = f"{base_dataset}/{sdg_subpath}"
        data_dir = os.path.join(DATA_DIR, base_dataset, *sdg_subpath.split("/"))
        for nd in donor_counts:
            n_b = n_baselines_done(data_dir, nd)
            n_s = n_synth(data_dir, nd)
            sym = "✓" if n_b == N_TRIALS else (f"~{n_b}" if n_b > 0 else "·")
            print(f"  {dataset_name:<28} {nd:>3}d   {sym:>10}   {n_s}/{N_TRIALS}")
        print()
    print("  ✓ = all 5 trials done   ~N = N done   · = none\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--status",       action="store_true")
    parser.add_argument("--dry-run",      action="store_true")
    parser.add_argument("--dataset",      default=None)
    parser.add_argument("--nd",           type=int, default=None)
    parser.add_argument("--max-jobs",     type=int, default=0)
    parser.add_argument("--log-dir",      default=None)
    parser.add_argument("--max-concurrent", type=int, default=1,
                        help="Concurrent jobs (default: 1)")
    parser.add_argument("--skip-nd-above", type=int, default=200,
                        help="Skip donor counts above this value (default: 200)")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    log_dir = args.log_dir or os.path.join(
        REPO_ROOT, "experiments", "sdg_comparison", "_baseline_logs"
    )
    failed_jobs_path = os.path.join(log_dir, "failed_jobs.json")

    print("[baselines] Building job queue…", flush=True)
    queue = build_queue(args)
    if not queue:
        print("[baselines] No pending jobs.")
        print_status()
        return
    print(f"[baselines] {len(queue)} pending trials.")

    if args.dry_run:
        for j in queue:
            print(f"  [DRY-RUN] {j.label}")
        return

    pending = list(queue)
    running = {}
    failed  = []
    launched = completed = failures = 0

    shutting_down = [False]
    def _sigint(_sig, _frame):
        print("\n[baselines] signal — killing all running jobs.")
        shutting_down[0] = True
        kill_all(running)
        sys.exit(1)
    signal.signal(signal.SIGINT,  _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    print(f"[baselines] starting.  max_concurrent={args.max_concurrent}  log_dir={log_dir}\n",
          flush=True)

    while pending or running:
        # Reap finished jobs
        for pid in list(running.keys()):
            entry = running[pid]
            rc = entry["proc"].poll()
            if rc is None:
                continue
            entry["proc"]._log_fh.close()
            del running[pid]
            elapsed = time.time() - entry["started"]
            j = entry["job"]
            if rc == 0:
                completed += 1
                print(f"[baselines] ✓ {j.label}  ({elapsed:.0f}s)", flush=True)
            else:
                failures += 1
                print(f"[baselines] ✗ FAILED: {j.label}  rc={rc}  log={entry['proc']._log_path}",
                      flush=True)
                failed.append({
                    "label": j.label,
                    "dataset": j.dataset_name,
                    "nd": j.nd,
                    "trial": j.trial,
                    "returncode": rc,
                    "log": entry["proc"]._log_path,
                    "cfg": j.cfg_path,
                })

        # Launch new jobs
        if not shutting_down[0]:
            launched_this_tick = set()
            for j in list(pending):
                if len(running) >= args.max_concurrent:
                    break
                if args.max_jobs and launched >= args.max_jobs:
                    break
                # Avoid two jobs racing on the same (dataset_name, nd) — they
                # share a tracking.csv and would pick the same trial.
                key = (j.dataset_name, j.nd)
                if key in launched_this_tick or any(
                    (e["job"].dataset_name, e["job"].nd) == key
                    for e in running.values()
                ):
                    continue
                # Skip if baselines for this trial were completed by a prior job
                data_dir = os.path.join(DATA_DIR, j.base_dataset,
                                        *j.dataset_name.split("/")[1:])
                if baselines_done(data_dir, j.nd, j.trial):
                    pending.remove(j)
                    continue

                mem_gb = get_available_memory_gb()
                if mem_gb < min_free_gb_for(j.nd):
                    continue

                print(f"[baselines] → launching: {j.label}  "
                      f"(mem_avail={mem_gb:.1f}GB)", flush=True)
                proc = launch_job(j, log_dir)
                running[proc.pid] = {
                    "proc": proc, "job": j, "started": time.time(),
                }
                launched_this_tick.add(key)
                pending.remove(j)
                launched += 1
                time.sleep(2)

        if args.max_jobs and launched >= args.max_jobs and not running:
            break
        time.sleep(5)

    print("\n" + "=" * 70)
    print(f"  Sweep finished.  Launched: {launched}  "
          f"Completed: {completed}  Failed: {failures}")

    if failed:
        os.makedirs(log_dir, exist_ok=True)
        with open(failed_jobs_path, "w") as fh:
            json.dump(failed, fh, indent=2)
        print(f"\n  {len(failed)} FAILED jobs written to: {failed_jobs_path}")
        for f in failed:
            print(f"    [{f['returncode']}] {f['label']}  log: {f['log']}")
    print("=" * 70 + "\n")
    print_status()


if __name__ == "__main__":
    main()
