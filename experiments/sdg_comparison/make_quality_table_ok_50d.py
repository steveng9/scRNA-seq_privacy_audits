"""
Generate a LaTeX table of synthetic data quality metrics (LISI, ARI, MMD) for all SDG methods.

OneK1K dataset, 50 donors only. Averages over all available trials.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

DATA = "/home/golobs/data/scMAMAMIA"

SOURCES = [
    # --- scDesign2 ---
    ("scDesign2",        "",         f"{DATA}/ok/scdesign2/no_dp/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDesign2 + DP (ok only) ---
    ("scDesign2+DP",     r"$\varepsilon=10^{0}$",  f"{DATA}/ok/scdesign2/eps_1/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{1}$",  f"{DATA}/ok/scdesign2/eps_10/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{2}$",  f"{DATA}/ok/scdesign2/eps_100/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{3}$",  f"{DATA}/ok/scdesign2/eps_1000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{4}$",  f"{DATA}/ok/scdesign2/eps_10000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{5}$",  f"{DATA}/ok/scdesign2/eps_100000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{6}$",  f"{DATA}/ok/scdesign2/eps_1000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{7}$",  f"{DATA}/ok/scdesign2/eps_10000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{8}$",  f"{DATA}/ok/scdesign2/eps_100000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon=10^{9}$",  f"{DATA}/ok/scdesign2/eps_1000000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDesign3 Gaussian ---
    ("scDesign3-G",      "",         f"{DATA}/ok/scdesign3/gaussian/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDesign3 Vine ---
    ("scDesign3-V",      "",         f"{DATA}/ok/scdesign3/vine/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scVI ---
    ("scVI",             "",         f"{DATA}/ok/scvi/no_dp/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDiffusion ---
    ("scDiffusion",      "",         f"{DATA}/ok/scdiffusion/no_dp/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- ZINBWave ---
    ("ZINBWave",         "",         f"{DATA}/ok/zinbwave/no_dp/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- NMF ---
    ("NMF",              "",         f"{DATA}/ok/nmf/no_dp/50d/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- NMF + DP (ok only) ---
    ("NMF+DP",           r"$\varepsilon=10^{8}$",  f"{DATA}/ok/nmf/eps_100000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{7}$",  f"{DATA}/ok/nmf/eps_10000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{6}$",  f"{DATA}/ok/nmf/eps_1000000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{5}$",  f"{DATA}/ok/nmf/eps_100000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{4}$",  f"{DATA}/ok/nmf/eps_10000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{3}$",  f"{DATA}/ok/nmf/eps_1000/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{2}$",  f"{DATA}/ok/nmf/eps_100/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{1}$",  f"{DATA}/ok/nmf/eps_10/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=2.8$",     f"{DATA}/ok/nmf/eps_2.8/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("NMF+DP",           r"$\varepsilon=10^{0}$",  f"{DATA}/ok/nmf/eps_1/50d/*/results/quality_eval_results/results/statistics_evals.csv"),
]

N_TRIALS = 5


def collect():
    """Collect quality metrics, returning dict of (method, sub) -> (lisi, ari, mmd) lists."""
    records = {}
    for method, sub, pattern in SOURCES:
        for csv_path in sorted(glob.glob(pattern)):
            try:
                row = pd.read_csv(csv_path).iloc[0]
                lisi = float(row["lisi"])
                ari = float(row["ari_real_vs_syn"]) if pd.notna(row["ari_real_vs_syn"]) else np.nan
                mmd = float(row["mmd"])
            except Exception:
                continue

            key = (method, sub)
            if key not in records:
                records[key] = {"lisi": [], "ari": [], "mmd": []}
            records[key]["lisi"].append(lisi)
            records[key]["ari"].append(ari)
            records[key]["mmd"].append(mmd)

    return records


def fmt(vals, scale=1.0):
    """Format a list of values as mean, optionally scaled."""
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "---"
    return f"{np.mean(vals) * scale:.3f}"


def make_table(records):
    """Return a LaTeX table string."""
    METHOD_ORDER = [
        ("scDesign2",    ""),
        ("scDesign2+DP", r"$\varepsilon=10^{9}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{8}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{7}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{6}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{5}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{4}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{3}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{2}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{1}$"),
        ("scDesign2+DP", r"$\varepsilon=10^{0}$"),
        ("scDesign3-G",  ""),
        ("scDesign3-V",  ""),
        ("scVI",         ""),
        ("scDiffusion",  ""),
        ("ZINBWave",     ""),
        ("NMF",          ""),
        ("NMF+DP",       r"$\varepsilon=10^{8}$"),
        ("NMF+DP",       r"$\varepsilon=10^{7}$"),
        ("NMF+DP",       r"$\varepsilon=10^{6}$"),
        ("NMF+DP",       r"$\varepsilon=10^{5}$"),
        ("NMF+DP",       r"$\varepsilon=10^{4}$"),
        ("NMF+DP",       r"$\varepsilon=10^{3}$"),
        ("NMF+DP",       r"$\varepsilon=10^{2}$"),
        ("NMF+DP",       r"$\varepsilon=10^{1}$"),
        ("NMF+DP",       r"$\varepsilon=2.8$"),
        ("NMF+DP",       r"$\varepsilon=10^{0}$"),
    ]

    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{5}{c}{OK1K — 50 donors} \\")
    lines.append(r"Method & & LISI$\uparrow$ & ARI$\uparrow$ & MMD$\times 10^3\downarrow$ \\")
    lines.append(r"\midrule")

    prev_method = None
    for method, sub in METHOD_ORDER:
        if method != prev_method:
            if prev_method is not None and prev_method != "scDesign2":
                lines.append(r"\addlinespace[2pt]")
            method_label = method
        else:
            method_label = ""
        prev_method = method

        key = (method, sub)
        if key in records:
            r = records[key]
            lisi = fmt(r["lisi"])
            ari = fmt(r["ari"])
            mmd = fmt(r["mmd"], scale=1e3)
        else:
            lisi = ari = mmd = "---"

        row = f"{method_label} & {sub} & {lisi} & {ari} & {mmd}" + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    return "\n".join(lines)


def main():
    records = collect()
    table = make_table(records)
    print(table)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quality_table_ok_50d_all_methods.tex")

    with open(out_path, "w") as f:
        f.write(table + "\n")

    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
