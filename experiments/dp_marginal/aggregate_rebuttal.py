#!/usr/bin/env python3
"""Aggregate B1 (disjoint-cells) and B2 (marginal-noise) sweeps into multi-trial
mean±std summary tables for the rebuttal. Writes CSVs + prints markdown tables.

B1: enhanced BB+aux (tm:100) AUC, disjoint vs overlap, per (dataset, nd), over 5 trials.
B2: standard & enhanced BB+aux/BB-aux AUC + MMD/LISI/ARI, per (dataset, mode, s), over 3 trials.
"""
import os, glob
import numpy as np
import pandas as pd

DATA = "/home/golobs/data/scMAMAMIA"
OUT = "/home/golobs/scRNA-seq_privacy_audits/experiments/dp_marginal/_agg"
os.makedirs(OUT, exist_ok=True)


def _auc(path, tm):
    try:
        df = pd.read_csv(path)
        return float(df.loc[df.metric == "auc", tm].values[0])
    except Exception:
        return np.nan


def _ms(vals):
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return np.nan, np.nan, 0
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


# ---------------- B1 ----------------
def agg_b1():
    SIZES = {"ok": [2, 5, 10, 20, 50, 100, 200], "aida": [2, 5, 10, 20, 50, 100], "cg": [2, 5, 10, 20]}
    rows = []
    for ds, sizes in SIZES.items():
        for nd in sizes:
            for variant, tag in [("disjoint", "v1"), ("overlap", "v1_overlap")]:
                trials = sorted(glob.glob(os.path.join(DATA, ds, "b1_disjoint", tag, f"{nd}d", "*")))
                enh, std = [], []
                for t in trials:
                    enh.append(_auc(os.path.join(t, "results", "mamamia_results_classb.csv"), "tm:100"))
                    std.append(_auc(os.path.join(t, "results", "mamamia_results.csv"), "tm:100"))
                em, es, en = _ms(enh)
                sm, ss, sn = _ms(std)
                rows.append(dict(dataset=ds, nd=nd, variant=variant, n_trials=en,
                                 enh_mean=em, enh_std=es, std_mean=sm, std_std=ss))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "b1_summary.csv"), index=False)
    return df


# ---------------- B2 ----------------
def agg_b2():
    DATASETS = ["ok", "aida"]
    ND = 50
    rows = []
    for ds in DATASETS:
        # auto-discover every (mode, s) level present on disk (coarse + fine grid)
        combos = [("none", None)]
        for m in ["cov", "marg", "both"]:
            levels = set()
            for d in glob.glob(os.path.join(DATA, ds, "b2_marginal", f"{m}_s*")):
                try:
                    levels.add(float(os.path.basename(d).split("_s")[1]))
                except (IndexError, ValueError):
                    pass
            combos += [(m, s) for s in sorted(levels)]
        for mode, s in combos:
            tag = "none" if mode == "none" else f"{mode}_s{s}"
            trials = sorted(glob.glob(os.path.join(DATA, ds, "b2_marginal", tag, f"{ND}d", "*")))
            enh100, enh101, std100, std101 = [], [], [], []
            mmd, lisi, ari = [], [], []
            for t in trials:
                r = os.path.join(t, "results")
                enh100.append(_auc(os.path.join(r, "mamamia_results_classb.csv"), "tm:100"))
                enh101.append(_auc(os.path.join(r, "mamamia_results_classb.csv"), "tm:101"))
                std100.append(_auc(os.path.join(r, "mamamia_results.csv"), "tm:100"))
                std101.append(_auc(os.path.join(r, "mamamia_results.csv"), "tm:101"))
                q = os.path.join(r, "statistics_evals.csv")
                if os.path.exists(q):
                    qd = pd.read_csv(q).iloc[0]
                    mmd.append(float(qd["mmd"])); lisi.append(float(qd["lisi"])); ari.append(float(qd["ari_real_vs_syn"]))
            row = dict(dataset=ds, mode=mode, s=(0.0 if s is None else s))
            for name, vals in [("enh_bbaux", enh100), ("enh_bbnoaux", enh101),
                               ("std_bbaux", std100), ("std_bbnoaux", std101),
                               ("mmd", mmd), ("lisi", lisi), ("ari", ari)]:
                m, sd, n = _ms(vals)
                row[f"{name}_mean"] = m; row[f"{name}_std"] = sd
            row["n_trials"] = len([v for v in enh100 if not np.isnan(v)])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "b2_summary.csv"), index=False)
    return df


def md_b1(df):
    print("\n### B1 — disjoint-cells (enhanced BB+aux AUC, mean±std over trials)\n")
    for ds in ["ok", "aida", "cg"]:
        sub = df[df.dataset == ds]
        if sub.empty:
            continue
        print(f"**{ds}**")
        print("| nd | overlap (control) | disjoint | Δ |")
        print("|---|---|---|---|")
        for nd in sorted(sub.nd.unique()):
            o = sub[(sub.nd == nd) & (sub.variant == "overlap")]
            d = sub[(sub.nd == nd) & (sub.variant == "disjoint")]
            if o.empty or d.empty:
                continue
            om, os_, dm, ds_ = o.enh_mean.values[0], o.enh_std.values[0], d.enh_mean.values[0], d.enh_std.values[0]
            print(f"| {nd} | {om:.3f}±{os_:.3f} | {dm:.3f}±{ds_:.3f} | {om-dm:+.3f} |")
        print()


def md_b2(df):
    print("\n### B2 — marginal-noise (mean over trials): privacy (enh/std BB+aux) + utility\n")
    for ds in ["ok", "aida"]:
        sub = df[df.dataset == ds]
        print(f"**{ds}**")
        print("| mode | s | std BB+aux | enh BB+aux | MMD↓ | LISI↑ | ARI |")
        print("|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            slabel = "—" if r["mode"] == "none" else f"{r['s']:.1f}"
            print(f"| {r['mode']} | {slabel} | {r['std_bbaux_mean']:.3f} | {r['enh_bbaux_mean']:.3f} | "
                  f"{r['mmd_mean']:.5f} | {r['lisi_mean']:.3f} | {r['ari_mean']:.3f} |")
        print()


if __name__ == "__main__":
    b1 = agg_b1()
    b2 = agg_b2()
    md_b1(b1)
    md_b2(b2)
    print(f"\nCSVs -> {OUT}/b1_summary.csv , b2_summary.csv")
