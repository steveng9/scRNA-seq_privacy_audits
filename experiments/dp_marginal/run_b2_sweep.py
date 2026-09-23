#!/usr/bin/env python3
"""
B2 sweep scheduler — marginal-noise defense, detached & resource-guarded.

For OneK1K (primary, A6-comparable) then AIDA, at 50 donors, 3 trials, sweeps
noise modes × levels and records standard + enhanced BB AUC against the defended
generator.  Trial-major, dataset-prioritised (ok first) so the 3-trial breadth
on OneK1K lands before AIDA.

Schedule per (ds, trial):
    none                       (baseline, no noise)
    {cov, marg2, marg, both} × {0.1, 0.5, 1.0}
= 1 + 4*3 = 13 configs; × 3 trials × 2 datasets = 78 configs.

Modes recap: cov=covariance only (reproduces A6), marg2=secondary marginals only
(the Class B LLR genes — causal test), marg=all marginals, both=cov+all marginals.

Resource safety mirrors B1: MAX_CONCURRENT trials, MIN_FREE_GB, load guard.
Runs gently alongside the B1 sweep (lower concurrency); the guard prevents
choking daniilf/sikha.

Monitor:   python run_b2_sweep.py --status
Kill all:  bash kill_b2.sh
"""
import os, sys, time, subprocess, argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
CONDA_ENV = "tabddpm_"
LOGDIR = os.path.join(HERE, "_sweep_logs")

# --- schedule ---
DATASETS = ["ok", "aida"]
ND = 50
TRIALS = [1, 2, 3, 4, 5]  # 4,5 added 2026-07-26 (user: extend frontier 3→5 trials); is_done() skips 1–3
NOISE_MODES = ["cov", "marg", "both"]
# Per-mode noise grid. FINE added 2026-07-25 (user course-correction): `s` is a
# tunable knob exactly like the covariance η; the paper has an 11-point η sweep for
# cov but we had only 4 coarse marg points, so we could not see the graceful part
# of the marginal privacy-utility curve. We densify marg (and both) to matched
# resolution to trace the full frontier and find tunable operating points. cov
# stays coarse — it floors at AUC≈0.70 and its utility (LISI) barely moves across
# the range, so density there reveals no hidden regime.
COARSE = [0.1, 0.5, 1.0, 2.0]                         # already on disk
FINE   = [0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.5]     # new, fills the cliff
MODE_LEVELS = {"cov": COARSE, "marg": COARSE + FINE, "both": COARSE + FINE}
LEVELS = COARSE  # back-compat (unused by build_schedule now)

# --- resource guards (B1 done → box free; still cap so we don't choke daniilf/sikha) ---
MAX_CONCURRENT = 4
MIN_FREE_GB = 30         # quality LISI/ARI on all cells is RAM-heavy
MAX_LOAD_FRAC = 0.85     # of nproc
POLL_SEC = 30


def tag_for(mode, s):
    return "none" if mode == "none" else f"{mode}_s{s}"


def build_schedule():
    """Trial-major, ok-before-aida; baseline first within each (ds, trial).
    Per-mode levels via MODE_LEVELS. is_done() skips configs already on disk, so
    relaunching after adding FINE levels runs only the new (mode, s) combos."""
    jobs = []
    for t in TRIALS:
        for ds in DATASETS:
            jobs.append((ds, t, "none", None))
            for mode in NOISE_MODES:
                for s in MODE_LEVELS[mode]:
                    jobs.append((ds, t, mode, s))
    return jobs


def job_key(j):
    ds, t, mode, s = j
    return f"{ds}_{ND}d_t{t}_{tag_for(mode, s)}"


def _res_path(j):
    ds, t, mode, s = j
    tag = tag_for(mode, s)
    return os.path.join(DATA, ds, "b2_marginal", tag, f"{ND}d", str(t),
                        "results", "mamamia_results_classb.csv")


def is_done(j):
    """Done iff BOTH the privacy AUC and the quality metrics are present."""
    res = _res_path(j)
    qual = os.path.join(os.path.dirname(res), "statistics_evals.csv")
    if not (os.path.exists(res) and os.path.exists(qual)):
        return False
    try:
        return "auc" in pd.read_csv(res)["metric"].values
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
    ds, t, mode, s = j
    os.makedirs(LOGDIR, exist_ok=True)
    log = os.path.join(LOGDIR, f"{job_key(j)}.log")
    s_arg = "" if mode == "none" else f"--s {s}"
    cmd = (f"conda run --no-capture-output -n {CONDA_ENV} python "
           f"{HERE}/run_b2_trial.py --dataset {ds} --nd {ND} --trial {t} "
           f"--mode {mode} {s_arg} --workers 4")
    p = subprocess.Popen(["setsid", "bash", "-c", f"{cmd} > {log} 2>&1"],
                         cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p, log


def status():
    jobs = build_schedule()
    done = [j for j in jobs if is_done(j)]
    print(f"B2 sweep: {len(done)}/{len(jobs)} configs complete")
    for j in jobs:
        res = _res_path(j)
        auc = ""
        if os.path.exists(res):
            try:
                df = pd.read_csv(res); r = df[df.metric == "auc"]
                if len(r):
                    auc = " ".join(f"{c}={r[c].values[0]:.3f}" for c in df.columns if c.startswith("tm:"))
            except Exception:
                pass
        qual = os.path.join(os.path.dirname(res), "statistics_evals.csv")
        qstr = ""
        if os.path.exists(qual):
            try:
                q = pd.read_csv(qual).iloc[0]
                qstr = f" | mmd={q['mmd']:.5f} lisi={q['lisi']:.3f} ari={q['ari_real_vs_syn']:.3f}"
            except Exception:
                pass
        if is_done(j):
            print(f"  [DONE] {job_key(j)}  {auc}{qstr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.status:
        status(); return

    jobs = build_schedule()
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] B2 sweep: {len(jobs)} configs "
          f"(MAX_CONCURRENT={MAX_CONCURRENT}, MIN_FREE_GB={MIN_FREE_GB})", flush=True)
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
                print(f"[{time.strftime('%H:%M:%S')}] hold (free={free_gb:.0f}GB load={load1:.1f}) "
                      f"running={len(running)}", flush=True)
                break
            j = pending.pop(0)
            if is_done(j):
                continue
            p, log = launch(j)
            running[job_key(j)] = (p, log)
            print(f"[{time.strftime('%H:%M:%S')}] launched {job_key(j)} -> {log}", flush=True)
            time.sleep(5)
        time.sleep(POLL_SEC)
    print(f"[{time.strftime('%H:%M:%S')}] B2 sweep complete.", flush=True)


if __name__ == "__main__":
    main()
