#!/usr/bin/env python3
"""Read quality eval CSVs, compute mean±std, and reprint the LaTeX quality table with missing values filled in."""

import os
import csv
import numpy as np

DATA_ROOT = os.path.expanduser("~/data")

def read_metrics(dataset, donors, trials=range(1, 6)):
    """Read statistics_evals.csv for each trial, return list of (mmd, lisi, ari) tuples."""
    results = []
    for trial in trials:
        path = os.path.join(DATA_ROOT, dataset, f"{donors}d", str(trial),
                            "results", "quality_eval_results", "results", "statistics_evals.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                mmd  = float(row["mmd"])
                lisi = float(row["lisi"])
                ari  = float(row["ari_real_vs_syn"])
                results.append((mmd, lisi, ari))
    return results

def fmt(vals, scale=1.0, decimals=2):
    """Return 'mean ± std' formatted string."""
    arr = np.array(vals) * scale
    mean, std = arr.mean(), arr.std()
    fmt_str = f"{{:.{decimals}f}}"
    # drop leading zero on std if < 1
    std_str = f"{std:.{decimals}f}".lstrip("0") or "0"
    return f"{fmt_str.format(mean)} $\\pm$ {std_str}"

# ── Collect the new data points ───────────────────────────────────────────────

hfra_20  = read_metrics("cg",   20)
aida_50  = read_metrics("aida", 50)
aida_ari = {d: read_metrics("aida", d) for d in [2, 5, 10, 20, 50]}

print(f"HFRA 20d:  {len(hfra_20)} trials found")
print(f"AIDA 50d:  {len(aida_50)} trials found")
for d, vals in aida_ari.items():
    print(f"AIDA {d:>2}d:  {len(vals)} trials found")
print()

# ── Existing values from the paper ────────────────────────────────────────────
# Format: (lisi, ari, mmd_x100)  -- already formatted strings

existing = {
    # dataset: {donors: (lisi_str, ari_str, mmd_str)}
    "OneK1K": {
        2:   ("0.91 $\\pm$ .01", "0.52 $\\pm$ .19", "0.08 $\\pm$ .01"),
        5:   ("0.90 $\\pm$ .01", "0.48 $\\pm$ .04", "0.04 $\\pm$ .00"),
        10:  ("0.88 $\\pm$ .01", "0.41 $\\pm$ .09", "0.02 $\\pm$ .00"),
        20:  ("0.88 $\\pm$ .00", "0.47 $\\pm$ .04", "0.01 $\\pm$ .00"),
        50:  ("0.87 $\\pm$ .00", "0.43 $\\pm$ .03", "0.01 $\\pm$ .00"),
        100: ("0.86 $\\pm$ .00", "0.43 $\\pm$ .04", "0.01 $\\pm$ .00"),
        200: ("0.86 $\\pm$ .00", "0.42 $\\pm$ .05", "0.01 $\\pm$ .00"),
    },
    "AIDA": {
        2:   ("0.87 $\\pm$ .01", None,               "0.06 $\\pm$ .01"),
        5:   ("0.83 $\\pm$ .02", None,               "0.02 $\\pm$ .01"),
        10:  ("0.81 $\\pm$ .01", None,               "0.01 $\\pm$ .00"),
        20:  ("0.78 $\\pm$ .02", None,               "0.01 $\\pm$ .00"),
        50:  (None,              None,               None             ),
        100: (None,              None,               None             ),
        200: (None,              None,               None             ),
    },
    "HFRA": {
        2:   ("0.56 $\\pm$ .03", "0.29 $\\pm$ .12", "0.02 $\\pm$ .00"),
        5:   ("0.39 $\\pm$ .08", "0.32 $\\pm$ .06", "0.01 $\\pm$ .00"),
        10:  ("0.28 $\\pm$ .04", "0.26 $\\pm$ .02", "0.01 $\\pm$ .00"),
        20:  (None,              None,               None             ),
    },
}

# ── Fill in new values ────────────────────────────────────────────────────────

def cell(vals, metric_idx, scale=1.0):
    """Return formatted cell or '--' if no data."""
    if not vals:
        return "--"
    return fmt([v[metric_idx] for v in vals], scale=scale)

# AIDA ARI for existing rows
for d in [2, 5, 10, 20]:
    lisi_s, _, mmd_s = existing["AIDA"][d]
    existing["AIDA"][d] = (lisi_s, cell(aida_ari[d], metric_idx=2), mmd_s)

# AIDA 50d full row
existing["AIDA"][50] = (
    cell(aida_50, metric_idx=1),
    cell(aida_50, metric_idx=2),
    cell(aida_50, metric_idx=0, scale=100),
)

# HFRA 20d full row
existing["HFRA"][20] = (
    cell(hfra_20, metric_idx=1),
    cell(hfra_20, metric_idx=2),
    cell(hfra_20, metric_idx=0, scale=100),
)

# ── Render the LaTeX table ────────────────────────────────────────────────────

all_donors = [2, 5, 10, 20, 50, 100, 200]

def get(dataset, donors, idx):
    row = existing[dataset].get(donors)
    if row is None or row[idx] is None:
        return "--"
    return row[idx]

print(r"""\begin{table*}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{cccccccccc}
\toprule
& \multicolumn{3}{c}{\textbf{OneK1K}} & \multicolumn{3}{c}{\textbf{AIDA}} & \multicolumn{3}{c}{\textbf{HFRA}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}
\textbf{Donors} & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times$10$^2$ $\downarrow$)} & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times$10$^2$ $\downarrow$)} & \textbf{LISI ($\uparrow$)} & \textbf{ARI ($\uparrow$)} & \textbf{MMD ($\times$10$^2$ $\downarrow$)} \\
\midrule""")

for d in all_donors:
    cols = []
    for ds in ["OneK1K", "AIDA", "HFRA"]:
        if d not in existing[ds]:
            cols += ["--", "--", "--"]
        else:
            cols += [get(ds, d, 0), get(ds, d, 1), get(ds, d, 2)]
    print(f"{d} & " + " & ".join(cols) + r" \\")

print(r"""\bottomrule
\end{tabular}
\caption{Quality metrics (LISI, ARI, and MMD) across datasets and donor sizes. Values shown as mean $\pm$ standard deviation. $\uparrow$ indicates higher is better, $\downarrow$ indicates lower is better. MMD values are scaled by 100 for readability. Implementations taken from CAMDA2025 could not execute over larger settings on the AIDA and HFRA data.}
\label{tab:qualities}
\end{table*}""")
