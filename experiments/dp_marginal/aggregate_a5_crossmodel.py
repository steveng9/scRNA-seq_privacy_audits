#!/usr/bin/env python3
"""A5 — consolidate scMAMA-MIA cross-model transferability from existing saved
results into one table. NO new compute; pure re-tabulation.

Primary metric: enhanced (ClassB) BB+aux AUC (tm:100), mean±std over trials, at
several donor sizes. Standard BB+aux shown for context. Attacks on non-copula
generators run via the scDesign2 proxy shadow model — the point of the table.

scDiffusion: v1 is invalid (missing Stage 3 + wrong hyperparams; see memory
project_scdiffusion_v2). Use scdiffusion_v2 only; report 'pending' if empty.
"""
import os, glob
import numpy as np
import pandas as pd

DATA = "/home/golobs/data/scMAMAMIA"
OUT = "/home/golobs/scRNA-seq_privacy_audits/experiments/dp_marginal/_agg"
FIG = "/home/golobs/scRNA-seq_privacy_audits/figures"
os.makedirs(OUT, exist_ok=True)

# (display name, sdg dir, variant subdir)
MODELS = [
    ("scDesign2",     "scdesign2",      "no_dp"),
    ("scDesign3-G",   "scdesign3",      "gaussian"),
    ("scDesign3-V",   "scdesign3",      "vine"),
    ("scVI",          "scvi",           "no_dp"),
    ("ZINB-WaVE",     "zinbwave",       "no_dp"),
    ("scDiffusion",   "scdiffusion_v2", "no_dp"),
    ("NMF",           "nmf",            "no_dp"),
]
DATASETS = ["ok", "aida"]
SIZES = [10, 20, 50, 100]


def _auc(path, tm):
    try:
        df = pd.read_csv(path)
        return float(df.loc[df.metric == "auc", tm].values[0])
    except Exception:
        return np.nan


def _ms(vals):
    vals = [v for v in vals if not np.isnan(v)]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else (np.nan, np.nan, 0)


def collect():
    rows = []
    for ds in DATASETS:
        for name, sdg, variant in MODELS:
            for nd in SIZES:
                base = os.path.join(DATA, ds, sdg, variant, f"{nd}d")
                enh, std = [], []
                for t in sorted(glob.glob(os.path.join(base, "*"))):
                    enh.append(_auc(os.path.join(t, "results", "mamamia_results_classb.csv"), "tm:100"))
                    std.append(_auc(os.path.join(t, "results", "mamamia_results.csv"), "tm:100"))
                em, es, en = _ms(enh)
                sm, ss, sn = _ms(std)
                rows.append(dict(dataset=ds, model=name, nd=nd, n_trials=en,
                                 enh_mean=em, enh_std=es, std_mean=sm, std_std=ss))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "a5_crossmodel.csv"), index=False)
    return df


def _lz(x):
    if np.isnan(x):
        return "--"
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else s


def md(df):
    for ds in DATASETS:
        print(f"\n### A5 cross-model — {ds}: enhanced BB+aux AUC (mean±std over trials)\n")
        print("| model | " + " | ".join(f"{nd}d" for nd in SIZES) + " |")
        print("|" + "---|" * (len(SIZES) + 1))
        for name, _, _ in MODELS:
            cells = []
            for nd in SIZES:
                r = df[(df.dataset == ds) & (df.model == name) & (df.nd == nd)]
                if r.empty or r.n_trials.values[0] == 0:
                    cells.append("--")
                else:
                    cells.append(f"{_lz(r.enh_mean.values[0])}±{_lz(r.enh_std.values[0])} (n={int(r.n_trials.values[0])})")
            print(f"| {name} | " + " | ".join(cells) + " |")


def write_tex(df, ds="ok"):
    lines = [
        r"\begin{table}[ht]", r"\centering", r"\scriptsize",
        r"\begin{tabular}{l|" + "c" * len(SIZES) + "}", r"\toprule",
        r"\textbf{Generator} & " + " & ".join(rf"\textbf{{{nd}d}}" for nd in SIZES) + r" \\",
        r"\midrule",
    ]
    for name, _, _ in MODELS:
        cells = [name]
        for nd in SIZES:
            r = df[(df.dataset == ds) & (df.model == name) & (df.nd == nd)]
            if r.empty or r.n_trials.values[0] == 0:
                cells.append("--")
            else:
                cells.append(_lz(r.enh_mean.values[0]) + r"{\tiny$\,\pm\,$" + _lz(r.enh_std.values[0]) + "}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{\textbf{Cross-model transferability of scMAMA-MIA (enhanced BB+aux AUC, " + ds + ").} "
        r"The attack fits a scDesign2 proxy copula on each generator's synthetic output, so it applies "
        r"regardless of the true generator's architecture. It is above chance on all high-fidelity "
        r"generators; NMF and scDiffusion resist only because their synthetic data is low-fidelity "
        r"(see quality tables), not because they protect privacy. Mean$\,\pm\,$std over trials.}",
        rf"\label{{tab:a5_crossmodel_{ds}}}", r"\end{table}",
    ]
    out = os.path.join(FIG, f"a5_crossmodel_{ds}.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    df = collect()
    md(df)
    for ds in DATASETS:
        write_tex(df, ds)
    print(f"\nCSV -> {OUT}/a5_crossmodel.csv")
