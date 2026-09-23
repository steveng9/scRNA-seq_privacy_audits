#!/usr/bin/env python3
"""
run_sweep.py — multi-trial PrivBayes-MAMA-MIA sweep on scDesign2 + side-by-side
comparison vs scMAMA-MIA (Experiment #4; see docs/privbayes_mamamia_plan.md).

For each (nd, trial) it stages the scDesign2 data (stage_scdesign2_to_mamamia.py,
env tabddpm_) then runs the PrivBayes-MAMA-MIA attack (run_privbayes_mamamia.py,
env sdg), collecting a donor-level AUC. `--report` aggregates the AUCs to
mean+-std per donor count and prints them next to scMAMA-MIA's BB+aux AUC
(standard tm:100 and enhanced/classb tm:100) on the SAME splits — the intended
contrast: generic tabular attack ~0.5 vs. copula-specialised scMAMA-MIA >> 0.5.

Detached + resumable (skips any (nd, trial) whose result CSV exists) and
resource-guarded (per unit; interleaves safely with b2 / the 440d regen).

    setsid bash -c 'python experiments/privbayes_mamamia/run_sweep.py \
        > experiments/privbayes_mamamia/_logs/sweep.log 2>&1' &

    python experiments/privbayes_mamamia/run_sweep.py --status
    python experiments/privbayes_mamamia/run_sweep.py --report
"""
import argparse
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/golobs/scRNA-seq_privacy_audits"
DATA = "/home/golobs/data/scMAMAMIA"
STAGED = os.path.join(HERE, "_staged")
LOGDIR = os.path.join(HERE, "_logs")

STAGE = os.path.join(HERE, "stage_scdesign2_to_mamamia.py")
ATTACK = os.path.join(HERE, "run_privbayes_mamamia.py")
STAGE_ENV, ATTACK_ENV = "tabddpm_", "sdg"

# --- sweep scope (defaults; overridable via CLI) ---
DATASET, SDG = "ok", "scdesign2/no_dp"
ND_LIST = [10, 50, 200]
TRIALS = [1, 2, 3, 4, 5]
GENE_SET, N_GENES, N_BINS = "copula", 50, 10
FP_EPS, FP_SAMPLE = 1000.0, 10000

# --- resource guard (per unit) ---
MIN_FREE_GB = 55
MAX_LOAD_FRAC = 0.7
POLL_SEC = 120


def _tag(nd, t):
    return f"{DATASET}_{SDG.replace('/', '_')}_{nd}d_t{t}"


def _staged_dir(nd, t):
    return os.path.join(STAGED, _tag(nd, t))


def _result_csv(nd, t):
    return os.path.join(_staged_dir(nd, t), "results", "privbayes_mamamia.csv")


def _synth(nd, t):
    return os.path.join(DATA, DATASET, *SDG.split("/"), f"{nd}d", str(t),
                        "datasets", "synthetic.h5ad")


def _staged_ready(nd, t):
    d = _staged_dir(nd, t)
    return all(os.path.exists(os.path.join(d, f))
               for f in ("aux.parquet", "targets.parquet", "synth.parquet", "meta.json"))


def resources_ok():
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
    free_gb = mem.get("MemAvailable", 0) / 1024 / 1024
    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 48
    return (free_gb >= MIN_FREE_GB) and (load1 <= MAX_LOAD_FRAC * ncpu), free_gb, load1


def _wait_resources():
    while True:
        ok, free_gb, load1 = resources_ok()
        if ok:
            return
        print(f"[{time.strftime('%H:%M:%S')}] hold (free={free_gb:.0f}GB load={load1:.1f})",
              flush=True)
        time.sleep(POLL_SEC)


def _run(env, script, extra):
    cmd = ["conda", "run", "--no-capture-output", "-n", env, "python", "-u", script] + extra
    return subprocess.run(cmd, cwd=REPO).returncode


def stage(nd, t):
    return _run(STAGE_ENV, STAGE, [
        "--dataset", DATASET, "--sdg", SDG, "--nd", str(nd), "--trial", str(t),
        "--gene-set", GENE_SET, "--n-genes", str(N_GENES), "--n-bins", str(N_BINS),
        "--out-dir", _staged_dir(nd, t)])


def attack(nd, t):
    return _run(ATTACK_ENV, ATTACK, [
        "--staged", _staged_dir(nd, t), "--eps", str(FP_EPS), "--fp-sample", str(FP_SAMPLE)])


def _units():
    return [(nd, t) for nd in ND_LIST for t in TRIALS]


def status():
    print(f"\n  PrivBayes-MAMA-MIA sweep — {DATASET}/{SDG}  nd={ND_LIST} trials={TRIALS}")
    done = sum(os.path.exists(_result_csv(nd, t)) for nd, t in _units())
    print(f"  {done}/{len(_units())} (nd,trial) complete")
    for nd in ND_LIST:
        marks = "".join("O" if os.path.exists(_result_csv(nd, t)) else "." for t in TRIALS)
        print(f"    {nd:>4}d  [{marks}]")


# ---------------------------------------------------------------------------
def _scmama_bb(nd, t, classb):
    """scMAMA-MIA BB+aux AUC (tm:100) for the same split; classb=enhanced."""
    fn = "mamamia_results_classb.csv" if classb else "mamamia_results.csv"
    p = os.path.join(DATA, DATASET, *SDG.split("/"), f"{nd}d", str(t), "results", fn)
    if not os.path.exists(p):
        return np.nan
    try:
        df = pd.read_csv(p)
        row = df[df["metric"] == "auc"]
        return float(row["tm:100"].values[0]) if len(row) and "tm:100" in row else np.nan
    except Exception:
        return np.nan


def report():
    rows = []
    for nd in ND_LIST:
        pv, sc_std, sc_enh = [], [], []
        for t in TRIALS:
            rc = _result_csv(nd, t)
            if os.path.exists(rc):
                try:
                    pv.append(float(pd.read_csv(rc)["auc"].values[0]))
                except Exception:
                    pass
            sc_std.append(_scmama_bb(nd, t, classb=False))
            sc_enh.append(_scmama_bb(nd, t, classb=True))
        def ms(x):
            x = [v for v in x if v == v]
            return (np.mean(x), np.std(x), len(x)) if x else (np.nan, np.nan, 0)
        pm, ps, pn = ms(pv)
        sm, ss, _ = ms(sc_std)
        em, es, _ = ms(sc_enh)
        rows.append({"nd": nd, "privbayes_mamamia_auc": pm, "pb_std": ps, "pb_n": pn,
                     "scmama_bb_std_auc": sm, "scmama_bb_std_std": ss,
                     "scmama_bb_enh_auc": em, "scmama_bb_enh_std": es})
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "sweep_summary.csv")
    df.to_csv(out, index=False)
    print("\n  === PrivBayes-MAMA-MIA vs scMAMA-MIA (BB+aux) — donor-level AUC ===")
    print(f"  {'nd':>5}  {'PrivBayes-MAMA-MIA':>20}  {'scMAMA BB+aux':>16}  {'scMAMA BB+aux(enh)':>20}")
    for r in rows:
        print(f"  {r['nd']:>4}d  {r['privbayes_mamamia_auc']:>13.3f}+-{r['pb_std']:.3f}"
              f"  {r['scmama_bb_std_auc']:>10.3f}+-{r['scmama_bb_std_std']:.3f}"
              f"  {r['scmama_bb_enh_auc']:>13.3f}+-{r['scmama_bb_enh_std']:.3f}"
              f"   (pb n={r['pb_n']})")
    print(f"\n  wrote {out}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nd", nargs="+", type=int, default=None)
    ap.add_argument("--trials", nargs="+", type=int, default=None)
    ap.add_argument("--fp-eps", type=float, default=None)
    ap.add_argument("--fp-sample", type=int, default=None)
    ap.add_argument("--no-guard", action="store_true", help="skip the resource guard")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    global ND_LIST, TRIALS, FP_EPS, FP_SAMPLE
    if a.nd:
        ND_LIST = a.nd
    if a.trials:
        TRIALS = a.trials
    if a.fp_eps is not None:
        FP_EPS = a.fp_eps
    if a.fp_sample is not None:
        FP_SAMPLE = a.fp_sample

    os.makedirs(LOGDIR, exist_ok=True)
    if a.status:
        status(); return
    if a.report:
        report(); return

    units = _units()
    print(f"[{time.strftime('%H:%M:%S')}] sweep start: {len(units)} (nd,trial) units "
          f"nd={ND_LIST} trials={TRIALS} fp_eps={FP_EPS}", flush=True)
    for nd, t in units:
        if os.path.exists(_result_csv(nd, t)):
            print(f"[{time.strftime('%H:%M:%S')}] skip {_tag(nd,t)} (done)", flush=True)
            continue
        if not os.path.exists(_synth(nd, t)):
            print(f"[{time.strftime('%H:%M:%S')}] SKIP {_tag(nd,t)} — no scDesign2 synth", flush=True)
            continue
        if not a.no_guard:
            _wait_resources()
        print(f"[{time.strftime('%H:%M:%S')}] === {_tag(nd,t)} ===", flush=True)
        if not _staged_ready(nd, t):
            print(f"  staging ({STAGE_ENV}) ...", flush=True)
            rc = stage(nd, t)
            if rc != 0 or not _staged_ready(nd, t):
                print(f"  [FAIL] staging rc={rc}; skipping", flush=True)
                continue
        print(f"  attacking ({ATTACK_ENV}) ...", flush=True)
        rc = attack(nd, t)
        if rc != 0 or not os.path.exists(_result_csv(nd, t)):
            print(f"  [FAIL] attack rc={rc}", flush=True)
            continue
        try:
            print(f"  AUC = {float(pd.read_csv(_result_csv(nd,t))['auc'].values[0]):.4f}", flush=True)
        except Exception:
            pass

    print(f"\n[{time.strftime('%H:%M:%S')}] sweep complete.", flush=True)
    status()
    report()


if __name__ == "__main__":
    main()
