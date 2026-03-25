"""
Generate a LaTeX table of synthetic data quality metrics (LISI, ARI, MMD×10³).

Reads statistics_evals.csv files produced by run_quality_evals.py and
run_dp_quality_eval.py, averages over all available (donor-count, trial) pairs,
and prints a LaTeX booktabs table.

An asterisk (*) is appended to any cell whose average is computed from fewer
than the expected 5 trials for at least one donor-count setting.

Usage:
    python experiments/sdg_comparison/make_quality_table.py
"""

import os
import glob
import numpy as np
import pandas as pd

DATA = "/home/golobs/data"

# ---------------------------------------------------------------------------
# Source definitions
# Each entry: (row_label, col1, glob_pattern_for_csvs)
# col1 is the first column (method name), col2 is sub-label (e.g. epsilon)
# ---------------------------------------------------------------------------

# (display_name, sub_label, glob_pattern)
SOURCES = [
    # --- scDesign2 ---
    ("scDesign2",        "",         f"{DATA}/ok/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2",        "",         f"{DATA}/cg/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2",        "",         f"{DATA}/aida/*/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDesign2 + DP (ok only) ---
    ("scDesign2+DP",     r"$\varepsilon$=1",     f"{DATA}/ok_dp/eps_1/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon$=10",    f"{DATA}/ok_dp/eps_10/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon$=100",   f"{DATA}/ok_dp/eps_100/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon$=1000",  f"{DATA}/ok_dp/eps_1000/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign2+DP",     r"$\varepsilon$=10000", f"{DATA}/ok_dp/eps_10000/*/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDesign3 Gaussian ---
    ("scDesign3-G",      "",         f"{DATA}/ok_sd3g/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign3-G",      "",         f"{DATA}/aida_sd3g/*/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDesign3 Vine ---
    ("scDesign3-V",      "",         f"{DATA}/ok_sd3v/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDesign3-V",      "",         f"{DATA}/aida_sd3v/*/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scVI ---
    ("scVI",             "",         f"{DATA}/ok_scvi/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scVI",             "",         f"{DATA}/aida_scvi/*/*/results/quality_eval_results/results/statistics_evals.csv"),

    # --- scDiffusion ---
    ("scDiffusion",      "",         f"{DATA}/ok_scdiff/*/*/results/quality_eval_results/results/statistics_evals.csv"),
    ("scDiffusion",      "",         f"{DATA}/aida_scdiff/*/*/results/quality_eval_results/results/statistics_evals.csv"),
]

DATASETS = ["ok", "aida", "cg"]
METRICS  = ["lisi", "ari_real_vs_syn", "mmd"]
N_TRIALS = 5


def dataset_of(path):
    """Infer which dataset a CSV path belongs to."""
    p = path.replace(DATA + "/", "")
    if p.startswith("ok_dp"):      return "ok"
    if p.startswith("ok_sd3g"):    return "ok"
    if p.startswith("ok_sd3v"):    return "ok"
    if p.startswith("ok_scvi"):    return "ok"
    if p.startswith("ok_scdiff"):  return "ok"
    if p.startswith("ok/"):        return "ok"
    if p.startswith("aida_sd3g"):  return "aida"
    if p.startswith("aida_sd3v"):  return "aida"
    if p.startswith("aida_scvi"):  return "aida"
    if p.startswith("aida_scdiff"):return "aida"
    if p.startswith("aida/"):      return "aida"
    if p.startswith("cg/"):        return "cg"
    return None


def collect():
    """
    Returns a dict:
      records[(method, sub, dataset)] = DataFrame with columns [nd, lisi, ari, mmd]

    All raw rows are kept so callers can filter by donor count.
    """
    records = {}   # (method, sub, dataset) -> list of (nd, lisi, ari, mmd)

    for method, sub, pattern in SOURCES:
        for csv_path in sorted(glob.glob(pattern)):
            ds = dataset_of(csv_path)
            if ds is None:
                continue
            try:
                row = pd.read_csv(csv_path).iloc[0]
                lisi = float(row["lisi"])
                ari  = float(row["ari_real_vs_syn"]) if pd.notna(row["ari_real_vs_syn"]) else np.nan
                mmd  = float(row["mmd"])
            except Exception:
                continue

            # Extract nd from path
            parts = csv_path.split(os.sep)
            nd = next((p for p in parts if p.endswith("d") and p[:-1].isdigit()), "?")

            key = (method, sub, ds)
            if key not in records:
                records[key] = []
            records[key].append((nd, lisi, ari, mmd))

    # Convert to DataFrames
    dfs = {}
    for key, rows in records.items():
        dfs[key] = pd.DataFrame(rows, columns=["nd", "lisi", "ari", "mmd"])
    return dfs


def aggregate(dfs, nd_filter=None):
    """
    Aggregate raw DataFrames into summary dicts suitable for make_table().

    nd_filter : str or None — if given (e.g. "20d"), restrict to that donor count only.
    """
    results = {}
    for key, df in dfs.items():
        if nd_filter is not None:
            df = df[df["nd"] == nd_filter]
        if df.empty:
            continue
        nd_counts = df.groupby("nd").size()
        any_incomplete = bool((nd_counts < N_TRIALS).any())
        results[key] = {
            "lisi":           df["lisi"].dropna().tolist(),
            "ari":            df["ari"].dropna().tolist(),
            "mmd":            df["mmd"].dropna().tolist(),
            "any_incomplete": any_incomplete,
            "n":              len(df),
        }
    return results


def fmt(vals, scale=1.0, incomplete=False):
    """Format a list of values as mean, optionally scaled, with asterisk."""
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "---"
    mean = np.mean(vals) * scale
    star = "*" if incomplete else ""
    return f"{mean:.3f}{star}"


def make_ok_only_table(results, nd_label):
    """
    Single-dataset (OK1K only) table for a specific donor-count setting.
    Columns: LISI | ARI | MMD×10³
    """
    METHOD_ORDER = [
        ("scDesign2",    ""),
        ("scDesign2+DP", r"$\varepsilon$=1"),
        ("scDesign2+DP", r"$\varepsilon$=10"),
        ("scDesign2+DP", r"$\varepsilon$=100"),
        ("scDesign2+DP", r"$\varepsilon$=1000"),
        ("scDesign2+DP", r"$\varepsilon$=10000"),
        ("scDesign3-G",  ""),
        ("scDesign3-V",  ""),
        ("scVI",         ""),
        ("scDiffusion",  ""),
    ]

    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{5}{c}{OK1K — " + nd_label + r" donors} \\")
    lines.append(r"Method & & LISI & ARI & MMD$\times 10^3$ \\")
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

        key = (method, sub, "ok")
        if key in results:
            r = results[key]
            inc = r["any_incomplete"]
            lisi = fmt(r["lisi"],            incomplete=inc)
            ari  = fmt(r["ari"],             incomplete=inc)
            mmd  = fmt(r["mmd"], scale=1e3,  incomplete=inc)
        else:
            lisi = ari = mmd = "---"

        row = f"{method_label} & {sub} & {lisi} & {ari} & {mmd}" + r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"% * = fewer than 5 trials")
    lines.append(r"% MMD values multiplied by 10^3")
    return "\n".join(lines)


def make_table(results):
    """Return a LaTeX table string."""

    # Row order: method display names (de-duped, preserving order)
    METHOD_ORDER = [
        ("scDesign2",    ""),
        ("scDesign2+DP", r"$\varepsilon$=1"),
        ("scDesign2+DP", r"$\varepsilon$=10"),
        ("scDesign2+DP", r"$\varepsilon$=100"),
        ("scDesign2+DP", r"$\varepsilon$=1000"),
        ("scDesign2+DP", r"$\varepsilon$=10000"),
        ("scDesign3-G",  ""),
        ("scDesign3-V",  ""),
        ("scVI",         ""),
        ("scDiffusion",  ""),
    ]

    DS_LABEL = {"ok": "OK1K", "aida": "AIDA", "cg": "CG"}

    lines = []
    lines.append(r"\begin{tabular}{ll" + "|ccc" * len(DATASETS) + "}")
    lines.append(r"\toprule")

    # Header row 1 — dataset names spanning 3 cols each (2 label cols + 3×n_datasets data cols)
    ds_headers = " & ".join(
        r"\multicolumn{3}{c" + ("|" if i < len(DATASETS)-1 else "") + "}{" + DS_LABEL[ds] + "}"
        for i, ds in enumerate(DATASETS)
    )
    lines.append(r" & & " + ds_headers + r" \\")

    # Header row 2 — metric names
    metric_block = r"LISI & ARI & MMD$\times 10^3$"
    metric_headers = " & ".join(metric_block for _ in DATASETS)
    lines.append(r"Method & & " + metric_headers + r" \\")
    lines.append(r"\midrule")

    prev_method = None
    for method, sub in METHOD_ORDER:
        # Method label: only show on first row of each method group
        if method != prev_method:
            if prev_method is not None and prev_method != "scDesign2":
                lines.append(r"\addlinespace[2pt]")
            method_label = method
        else:
            method_label = ""
        prev_method = method

        # Sub-label display
        sub_label = sub if sub else ""

        row = f"{method_label} & {sub_label}"
        for ds in DATASETS:
            key = (method, sub, ds)
            if key in results:
                r = results[key]
                inc = r["any_incomplete"]
                lisi = fmt(r["lisi"],         incomplete=inc)
                ari  = fmt(r["ari"],          incomplete=inc)
                mmd  = fmt(r["mmd"], scale=1e3, incomplete=inc)
            else:
                lisi = ari = mmd = "---"
            row += f" & {lisi} & {ari} & {mmd}"
        row += r" \\"
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"% * = fewer than 5 trials available for at least one donor-count setting")
    lines.append(r"% MMD values multiplied by 10^3")

    return "\n".join(lines)


def main():
    dfs = collect()

    # Print coverage summary
    print("Coverage summary (all donor counts):")
    results_all = aggregate(dfs)
    for key, r in sorted(results_all.items()):
        method, sub, ds = key
        print(f"  {method:20s} {sub:20s} {ds:6s}  n={r['n']:3d}  incomplete={'*' if r['any_incomplete'] else ' '}")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)

    # Table 1: all donor counts, all datasets
    print("\n--- Table 1: all donor counts ---")
    table1 = make_table(results_all)
    print(table1)
    path1 = os.path.join(out_dir, "quality_table.tex")
    with open(path1, "w") as f:
        f.write(table1 + "\n")
    print(f"\nSaved to {path1}")

    # Table 2: OK1K only, 10d
    results_10d = aggregate(dfs, nd_filter="10d")
    print("\n--- Table 2: OK1K, 10d only ---")
    table2 = make_ok_only_table(results_10d, nd_label="10")
    print(table2)
    path2 = os.path.join(out_dir, "quality_table_ok_10d.tex")
    with open(path2, "w") as f:
        f.write(table2 + "\n")
    print(f"\nSaved to {path2}")

    # Table 3: OK1K only, 20d
    results_20d = aggregate(dfs, nd_filter="20d")
    print("\n--- Table 3: OK1K, 20d only ---")
    table3 = make_ok_only_table(results_20d, nd_label="20")
    print(table3)
    path3 = os.path.join(out_dir, "quality_table_ok_20d.tex")
    with open(path3, "w") as f:
        f.write(table3 + "\n")
    print(f"\nSaved to {path3}")


if __name__ == "__main__":
    main()
