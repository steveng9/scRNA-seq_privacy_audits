#!/usr/bin/env python3
"""
B1 sweep scheduler — launches disjoint-cells trials in a TRIAL-MAJOR,
dataset-prioritised order with a concurrency cap and resource guards, fully
detached so it survives logout.

Priority order (breadth > depth, per user):
    OneK1K t1 (all sizes) -> OneK1K t2 -> OneK1K t3
    -> AIDA t1 -> HFRA(cg) t1 -> AIDA t2 -> cg t2 -> AIDA t3 -> cg t3
Each config runs BOTH the 'disjoint' experiment and the 'overlap' control.

Resource safety: never launch if free RAM < MIN_FREE_GB or 1-min loadavg per
core > MAX_LOAD_FRAC; at most MAX_CONCURRENT trials at once (each trial itself
uses 4 internal workers). Does not touch other users' jobs.

Monitor:   python run_b1_sweep.py --status
Kill all:  bash kill_b1.sh
"""
import os, sys, time, subprocess, argparse, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
CONDA_ENV = "tabddpm_"
LOGDIR = os.path.join(HERE, "_sweep_logs")
PROGRESS = os.path.join(HERE, "_sweep_logs", "progress.csv")

# --- schedule ---
SIZES = {"ok": [2, 5, 10, 20, 50, 100, 200], "aida": [2, 5, 10, 20, 50, 100], "cg": [2, 5, 10, 20]}
VARIANTS = ["disjoint", "overlap"]
G, B, TAG = 200, 200, "v1"

# --- resource guards ---
MAX_CONCURRENT = 3
MIN_FREE_GB = 25
MAX_LOAD_FRAC = 0.85     # of nproc
POLL_SEC = 30


def build_schedule():
    # Priority-ordered (dataset, trial) blocks — breadth first, then depth:
    #   OneK1K trials 1-3 (all sizes) -> AIDA t1, cg t1 -> OneK1K t4-5
    #   -> AIDA/cg t2-3 -> AIDA/cg t4-5. 5 trials total; trial-major so the
    #   essential 3-trial breadth across all datasets completes first.
    blocks = [("ok", 1), ("ok", 2), ("ok", 3),
              ("aida", 1), ("cg", 1),
              ("ok", 4), ("ok", 5),
              ("aida", 2), ("cg", 2), ("aida", 3), ("cg", 3),
              ("aida", 4), ("cg", 4), ("aida", 5), ("cg", 5)]
    jobs = []
    for ds, t in blocks:
        for nd in SIZES[ds]:
            for v in VARIANTS:
                jobs.append((ds, nd, t, v))
    return jobs


def job_key(j):
    ds, nd, t, v = j
    return f"{ds}_{nd}d_t{t}_{v}"


def is_done(j):
    ds, nd, t, v = j
    # 'disjoint' writes to tag v1; 'overlap' to tag v1_overlap (separate namespace)
    tag = TAG if v == "disjoint" else f"{TAG}_overlap"
    res = os.path.join(DATA, ds, "b1_disjoint", tag, f"{nd}d", str(t), "results", "mamamia_results_classb.csv")
    if not os.path.exists(res):
        return False
    try:
        df = pd.read_csv(res)
        return "auc" in df["metric"].values
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
    tag = TAG if v == "disjoint" else f"{TAG}_overlap"
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{job_key(j)}.log")
    cmd = (f"conda run --no-capture-output -n {CONDA_ENV} python "
           f"{HERE}/run_b1_trial.py --dataset {ds} --nd {nd} --trial {t} "
           f"--G {G} --B {B} --tag {tag} --variant {v} --workers 4")
    # detached; own process group
    p = subprocess.Popen(["setsid", "bash", "-c", f"{cmd} > {log} 2>&1"],
                         cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, log


def status():
    jobs = build_schedule()
    done = [j for j in jobs if is_done(j)]
    print(f"B1 sweep: {len(done)}/{len(jobs)} configs complete")
    rows = []
    for j in jobs:
        ds, nd, t, v = j
        tag = TAG if v == "disjoint" else f"{TAG}_overlap"
        res = os.path.join(DATA, ds, "b1_disjoint", tag, f"{nd}d", str(t), "results", "mamamia_results_classb.csv")
        auc = ""
        if os.path.exists(res):
            try:
                df = pd.read_csv(res); r = df[df.metric == "auc"]
                if len(r): auc = " ".join(f"{c}={r[c].values[0]:.3f}" for c in df.columns if c.startswith("tm:"))
            except Exception: pass
        rows.append((job_key(j), "DONE" if is_done(j) else "pending", auc))
    for k, st, auc in rows:
        if st == "DONE":
            print(f"  [{st}] {k}  {auc}")
    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.status:
        status(); return

    jobs = build_schedule()
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] B1 sweep: {len(jobs)} configs "
          f"(MAX_CONCURRENT={MAX_CONCURRENT}, MIN_FREE_GB={MIN_FREE_GB})", flush=True)
    if a.dry_run:
        for j in jobs:
            print("  ", job_key(j), "DONE" if is_done(j) else "queued")
        return

    running = {}   # key -> (Popen, log)
    pending = [j for j in jobs if not is_done(j)]
    while pending or running:
        # reap
        for k in list(running):
            p, log = running[k]
            if p.poll() is not None:
                del running[k]
                print(f"[{time.strftime('%H:%M:%S')}] finished {k} (rc={p.returncode})", flush=True)
        # launch
        while pending and len(running) < MAX_CONCURRENT:
            ok, free_gb, load1 = resources_ok()
            if not ok:
                print(f"[{time.strftime('%H:%M:%S')}] hold (free={free_gb:.0f}GB load={load1:.1f}) "
                      f"running={len(running)}", flush=True)
                break
            j = pending.pop(0)
            if is_done(j):
                continue
            p, log = launch(j)
            running[job_key(j)] = (p, log)
            print(f"[{time.strftime('%H:%M:%S')}] launched {job_key(j)} -> {log}", flush=True)
            time.sleep(5)  # stagger
        time.sleep(POLL_SEC)
    print(f"[{time.strftime('%H:%M:%S')}] B1 sweep complete.", flush=True)


if __name__ == "__main__":
    main()
