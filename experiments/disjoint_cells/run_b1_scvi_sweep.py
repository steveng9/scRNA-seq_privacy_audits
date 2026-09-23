#!/usr/bin/env python3
"""B1 disjoint-cells sweep for the scVI generator (transferability of the
attacker-cell-budget finding to a NON-copula neural generator).

Same protocol as run_b1_sweep.py but --generator scvi. The attack still uses the
scDesign2 proxy shadow model (black-box transfer); only the target generator that
produces synthetic.h5ad is scVI. No defense/noise is applied to scVI.

Tags: scvi_v1 (disjoint), scvi_v1_overlap (overlap control). Reuses the shared
donor splits {ds}/splits/{nd}d/{trial}. GPU-bound → low concurrency.

Monitor:  python run_b1_scvi_sweep.py --status
Kill:     bash kill_b1.sh   (matches run_b1_trial.py / *sweep* patterns)
"""
import os, sys, time, subprocess, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
CONDA_ENV = "tabddpm_"
LOGDIR = os.path.join(HERE, "_sweep_logs")

DATASETS = ["ok"]
SIZES = {"ok": [10, 20, 50, 100]}
TRIALS = [1, 2, 3]
VARIANTS = ["disjoint", "overlap"]
G, B, GEN = 200, 200, "scvi"

# GPU-bound: keep it gentle (also the B2 fine sweep may still be using CPU).
MAX_CONCURRENT = 2
MIN_FREE_GB = 30
MAX_LOAD_FRAC = 0.9
POLL_SEC = 30


def _tag(v):
    return "scvi_v1" if v == "disjoint" else "scvi_v1_overlap"


def build_schedule():
    jobs = []
    for t in TRIALS:
        for ds in DATASETS:
            for nd in SIZES[ds]:
                for v in VARIANTS:
                    jobs.append((ds, nd, t, v))
    return jobs


def job_key(j):
    ds, nd, t, v = j
    return f"{ds}_{nd}d_t{t}_{v}_scvi"


def _res(j):
    ds, nd, t, v = j
    return os.path.join(DATA, ds, "b1_disjoint", _tag(v), f"{nd}d", str(t),
                        "results", "mamamia_results_classb.csv")


def is_done(j):
    r = _res(j)
    if not os.path.exists(r):
        return False
    try:
        return "auc" in pd.read_csv(r)["metric"].values
    except Exception:
        return False


def resources_ok():
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
    free_gb = mem.get("MemAvailable", 0) / 1024 / 1024
    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 48
    return (free_gb >= MIN_FREE_GB) and (load1 <= MAX_LOAD_FRAC * ncpu), free_gb, load1


def launch(j):
    ds, nd, t, v = j
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{job_key(j)}.log")
    cmd = (f"conda run --no-capture-output -n {CONDA_ENV} python "
           f"{HERE}/run_b1_trial.py --dataset {ds} --nd {nd} --trial {t} "
           f"--generator {GEN} --variant {v} --tag {_tag(v)} "
           f"--G {G} --B {B} --workers 4")
    p = subprocess.Popen(["setsid", "bash", "-c", f"{cmd} > {log} 2>&1"],
                         cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, log


def status():
    jobs = build_schedule()
    done = [j for j in jobs if is_done(j)]
    print(f"B1-scVI sweep: {len(done)}/{len(jobs)} configs complete")
    for j in jobs:
        auc = ""
        r = _res(j)
        if os.path.exists(r):
            try:
                df = pd.read_csv(r); row = df[df.metric == "auc"]
                if len(row):
                    auc = " ".join(f"{c}={row[c].values[0]:.3f}" for c in df.columns if c.startswith("tm:"))
            except Exception:
                pass
        if is_done(j):
            print(f"  [DONE] {job_key(j)}  {auc}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.status:
        status(); return
    jobs = build_schedule()
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] B1-scVI sweep: {len(jobs)} configs "
          f"(MAX_CONCURRENT={MAX_CONCURRENT})", flush=True)
    if a.dry_run:
        for j in jobs:
            print("  ", job_key(j), "DONE" if is_done(j) else "queued")
        return
    running = {}
    pending = [j for j in jobs if not is_done(j)]
    while pending or running:
        for k in list(running):
            p, log = running[k]
            if p.poll() is not None:
                del running[k]
                print(f"[{time.strftime('%H:%M:%S')}] finished {k} (rc={p.returncode})", flush=True)
        while pending and len(running) < MAX_CONCURRENT:
            ok, free_gb, load1 = resources_ok()
            if not ok:
                print(f"[{time.strftime('%H:%M:%S')}] hold (free={free_gb:.0f}GB load={load1:.1f})", flush=True)
                break
            j = pending.pop(0)
            if is_done(j):
                continue
            p, log = launch(j)
            running[job_key(j)] = (p, log)
            print(f"[{time.strftime('%H:%M:%S')}] launched {job_key(j)} -> {log}", flush=True)
            time.sleep(8)
        time.sleep(POLL_SEC)
    print(f"[{time.strftime('%H:%M:%S')}] B1-scVI sweep complete.", flush=True)


if __name__ == "__main__":
    main()
