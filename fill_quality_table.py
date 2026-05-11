#!/usr/bin/env python3
"""
Generate the scDesign2 quality table (LISI, ARI, MMD×10²) across datasets and donor counts.

Reads statistics_evals.csv produced by run_quality_evals.py.
Rows: donor counts. Columns: OneK1K, AIDA, HFRA × {LISI, ARI, MMD×10²}.

Usage:
    python fill_quality_table.py
"""

import math
import glob
import os
import sys
import numpy as np
import pandas as pd

DATA     = os.path.expanduser("~/data/scMAMAMIA")
N_TRIALS = 5

# (display name, data key, valid donor counts)
DATASETS = [
    ("OneK1K", "ok",   [2, 5, 10, 20, 50, 100, 200, 490]),
    ("AIDA",   "aida", [2, 5, 10, 20, 50, 100, 200]),
    ("HFRA",   "cg",   [2, 5, 10, 20]),
]

ALL_DONORS = [2, 5, 10, 20, 50, 100, 200, 490]


def collect(dataset_key, nd):
    lisi_v, ari_v, mmd_v = [], [], []
    pattern = os.path.join(
        DATA, dataset_key, "scdesign2", "no_dp", f"{nd}d", "*",
        "results", "quality_eval_results", "results", "statistics_evals.csv",
    )
    for csv_path in glob.glob(pattern):
        try:
            row = pd.read_csv(csv_path).iloc[0]
            lisi_v.append(float(row["lisi"]))
            ari_val = row.get("ari_real_vs_syn")
            if pd.notna(ari_val):
                ari_v.append(float(ari_val))
            mmd_v.append(float(row["mmd"]))
        except Exception:
            continue
    return lisi_v, ari_v, mmd_v


def fmt(vals, scale=1.0):
    clean = [v * scale for v in vals if not math.isnan(float(v))]
    if not clean:
        return "--"
    mean = np.mean(clean)
    std  = np.std(clean, ddof=1) if len(clean) > 1 else 0.0
    n    = len(clean)

    mean_str = f"{mean:.2f}"
    std_str  = f"{std:.2f}"

    cell = mean_str + r" {\tiny$\,\pm\,$" + std_str + "}"
    if n < N_TRIALS:
        cell += rf"$^{{{n}}}$"
    return cell


def main():
    valid_nds = {dk: set(nds) for _, dk, nds in DATASETS}

    data = {}
    for _, dk, nds in DATASETS:
        for nd in nds:
            data[(dk, nd)] = collect(dk, nd)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{cccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"    & \multicolumn{3}{c}{\textbf{OneK1K}} & \multicolumn{3}{c}{\textbf{AIDA}}"
        r" & \multicolumn{3}{c}{\textbf{HFRA}} \\"
    )
    lines.append(r"    \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    lines.append(
        r"    \textbf{Donors}"
        r" & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times 10^2$ $\downarrow$)}"
        r" & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times 10^2$ $\downarrow$)}"
        r" & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times 10^2$ $\downarrow$)} \\"
    )
    lines.append(r"\midrule")

    for d in ALL_DONORS:
        cols = []
        for _, dk, nds in DATASETS:
            if d not in nds:
                cols += ["--", "--", "--"]
            else:
                lisi_v, ari_v, mmd_v = data[(dk, d)]
                cols += [fmt(lisi_v), fmt(ari_v), fmt(mmd_v, scale=100)]
        lines.append(f"{d} & " + " & ".join(cols) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Quality metrics (LISI, ARI, and MMD) for scDesign2 across datasets and donor counts. "
        r"Values are mean $\pm$ std over 5 trials. $\uparrow$ = higher is better; "
        r"$\downarrow$ = lower is better. MMD values scaled by $10^2$. "
        r"HFRA (CG) donor pool is limited to 22 donors.}"
    )
    lines.append(r"\label{tab:qualities}")
    lines.append(r"\end{table*}")

    tex = "\n".join(lines)
    print(tex)

    out_dir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quality_table_all_datasets.tex")
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
