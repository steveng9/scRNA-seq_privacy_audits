#!/usr/bin/env python3
"""
Aux-disjointness fix — sweep scheduler.

Re-runs the manuscript's *contaminated* +aux configs with a provably-disjoint
auxiliary set (build_disjoint_aux_split.py) and the reuse-synth attack
(run_auxfix_trial.py). Detached, resource-guarded, per-experiment kill script.

Scope (scDesign2; the other-SDG 490d rows are handled separately once the
OneK1K-490d split strategy is confirmed):
  * AIDA-200d   — complement aux (508-400 = 108 disjoint), 5 trials.
  * OneK1K-490d — added below once the shrink-vs-rebalance decision is final.
  (HFRA-10d/20d are DROPPED per decision — the 22-donor pool cannot support a
   disjoint aux; those +aux cells become "—" in the paper.)

Monitor:  python run_auxfix_sweep.py --status
Kill:     bash kill_auxfix.sh   (run ALONE)
"""
import os, sys, time, subprocess, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
CONDA_ENV = "tabddpm_"
LOGDIR = os.path.join(HERE, "_logs")

# (dataset, nd, split_mode, split_kwargs, trials)
JOBS = [
    ("aida", 200, "complement", {}, [1, 2, 3, 4, 5]),
]

# AIDA-200d trials load ~225k cells × 35k genes → ~80 GB each; keep to 1 at a
# time (also protects other users). Bump only for lighter datasets.
MAX_CONCURRENT = 1
MIN_FREE_GB = 50
MAX_LOAD_FRAC = 0.9
POLL_SEC = 30


def _key(ds, nd, t):
    return f"{ds}_{nd}d_t{t}"


def _res_dir(ds, nd, t):
    return os.path.join(DATA, ds, "aux_disjoint", "scdesign2", f"{nd}d", str(t), "results")


def is_done(ds, nd, t):
    rd = _res_dir(ds, nd, t)
    need = ["bb_results.csv", "bb_results_classb.csv", "wb_results.csv", "wb_results_classb.csv"]
    if not all(os.path.exists(os.path.join(rd, f)) for f in need):
        return False
    try:
        for f in need:
            if "auc" not in pd.read_csv(os.path.join(rd, f))["metric"].values:
                return False
        return True
    except Exception:
        return False


def _build_split(ds, nd, t, mode, kw):
    out = os.path.join(DATA, ds, "aux_disjoint", "splits", f"{nd}d", str(t))
    if all(os.path.exists(os.path.join(out, f)) for f in ["train.npy", "holdout.npy", "auxiliary.npy"]):
        return
    cmd = [f"{HERE}/build_disjoint_aux_split.py", "--dataset", ds, "--nd", str(nd),
           "--trial", str(t), "--mode", mode]
    for k, v in kw.items():
        cmd += [f"--{k}", str(v)]
    subprocess.run(["conda", "run", "--no-capture-output", "-n", CONDA_ENV, "python"] + cmd,
                   cwd=REPO, check=True)


def resources_ok():
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
    free_gb = mem.get("MemAvailable", 0) / 1024 / 1024
    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 48
    return (free_gb >= MIN_FREE_GB) and (load1 <= MAX_LOAD_FRAC * ncpu), free_gb, load1


def launch(ds, nd, t):
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{_key(ds,nd,t)}.log")
    cmd = (f"conda run --no-capture-output -n {CONDA_ENV} python "
           f"{HERE}/run_auxfix_trial.py --dataset {ds} --nd {nd} --trial {t} --workers 4")
    p = subprocess.Popen(["setsid", "bash", "-c", f"{cmd} > {log} 2>&1"],
                         cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, log


def all_units():
    units = []
    for ds, nd, mode, kw, trials in JOBS:
        for t in trials:
            units.append((ds, nd, t, mode, kw))
    return units


def status():
    units = all_units()
    done = [u for u in units if is_done(u[0], u[1], u[2])]
    print(f"aux-disjoint sweep: {len(done)}/{len(units)} configs complete")
    for ds, nd, t, _, _ in units:
        if is_done(ds, nd, t):
            rd = _res_dir(ds, nd, t)
            bits = []
            for f, lab in [("bb_results_classb.csv", "CB-BB"), ("wb_results_classb.csv", "CB-WB")]:
                df = pd.read_csv(os.path.join(rd, f)); row = df[df.metric == "auc"]
                if len(row):
                    bits.append(" ".join(f"{c}={row[c].values[0]:.3f}" for c in df.columns if c.startswith("tm:")))
            print(f"  [DONE] {_key(ds,nd,t)}  {' | '.join(bits)}")
        else:
            print(f"  [ ... ] {_key(ds,nd,t)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.status:
        status(); return
    units = all_units()
    print(f"[{time.strftime('%H:%M:%S')}] aux-disjoint sweep: {len(units)} configs "
          f"(MAX_CONCURRENT={MAX_CONCURRENT})", flush=True)
    if a.dry_run:
        for ds, nd, t, m, kw in units:
            print("  ", _key(ds, nd, t), m, kw, "DONE" if is_done(ds, nd, t) else "queued")
        return
    pending = [u for u in units if not is_done(u[0], u[1], u[2])]
    # pre-build all splits (cheap, serial)
    for ds, nd, t, mode, kw in pending:
        _build_split(ds, nd, t, mode, kw)
    running = {}
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
            ds, nd, t, mode, kw = pending.pop(0)
            if is_done(ds, nd, t):
                continue
            p, log = launch(ds, nd, t)
            running[_key(ds, nd, t)] = (p, log)
            print(f"[{time.strftime('%H:%M:%S')}] launched {_key(ds,nd,t)} -> {log}", flush=True)
            time.sleep(10)
        time.sleep(POLL_SEC)
    print(f"[{time.strftime('%H:%M:%S')}] aux-disjoint sweep complete.", flush=True)


if __name__ == "__main__":
    main()
