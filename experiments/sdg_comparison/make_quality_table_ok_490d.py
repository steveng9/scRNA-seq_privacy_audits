"""
Generate a LaTeX table of synthetic data quality metrics (LISI, ARI, MMD) for available SDG methods.

OneK1K dataset, 490 donors. Dynamically detects available methods.
Averages over all available trials.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

DATA = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5


def find_available_methods():
    """Scan for available 490d quality results and return (method, sub, pattern) tuples."""
    methods = []
    ok_dir = os.path.join(DATA, "ok")

    # Scan for all 490d directories with results
    for sdg_dir in glob.glob(os.path.join(ok_dir, "*", "*", "490d")):
        # Extract method and variant from path
        # Path is like: .../ok/{sdg}/{variant}/490d
        parts = sdg_dir.split(os.sep)
        sdg_type = parts[-3]  # e.g., scdesign2, nmf
        variant = parts[-2]   # e.g., no_dp, eps_1

        # Check if any results exist
        pattern = os.path.join(sdg_dir, "*/results/quality_eval_results/results/statistics_evals.csv")
        if glob.glob(pattern):
            # Determine display name and sub-label
            if sdg_type == "scdesign2":
                method_name = "scDesign2" if variant == "no_dp" else "scDesign2+DP"
                if variant == "no_dp":
                    sub_label = ""
                else:
                    # Extract epsilon
                    eps_str = variant.replace("eps_", "")
                    sub_label = format_epsilon(eps_str)
            elif sdg_type == "nmf":
                method_name = "NMF" if variant == "no_dp" else "NMF+DP"
                if variant == "no_dp":
                    sub_label = ""
                else:
                    eps_str = variant.replace("eps_", "")
                    sub_label = format_epsilon(eps_str)
            elif sdg_type == "scdesign3":
                method_name = "scDesign3"
                sub_label = "-G" if variant == "gaussian" else "-V"
            else:
                method_name = sdg_type.capitalize()
                sub_label = ""

            methods.append((method_name, sub_label, pattern))

    return methods


def format_epsilon(eps_str):
    """Convert epsilon string to LaTeX notation."""
    if eps_str == "2.8":
        return r"$\varepsilon=2.8$"
    try:
        val = int(eps_str)
        # Find power of 10
        if val == 1:
            return r"$\varepsilon=10^{0}$"
        import math
        power = round(math.log10(val))
        if 10 ** power == val:
            return rf"$\varepsilon=10^{{{power}}}$"
    except:
        pass
    return rf"$\varepsilon={eps_str}$"


def collect(methods):
    """Collect quality metrics from available sources."""
    records = {}
    for method, sub, pattern in methods:
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


def make_table(methods, records):
    """Return a LaTeX table string."""
    lines = []
    lines.append(r"\begin{tabular}{llccc}")
    lines.append(r"\toprule")
    lines.append(r"\multicolumn{5}{c}{OK1K — 490 donors} \\")
    lines.append(r"Method & & LISI$\uparrow$ & ARI$\uparrow$ & MMD$\times 10^3\downarrow$ \\")
    lines.append(r"\midrule")

    prev_method = None
    for method, sub, _ in methods:
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
    methods = find_available_methods()

    # Sort methods by name for consistent ordering
    # Sort by: method name, then by epsilon value (extracted from sub_label)
    def sort_key(item):
        method, sub, _ = item
        # Define order for method names
        method_order = {"scDesign2": 0, "scDesign3": 1, "NMF": 2}
        method_idx = method_order.get(method.split("+")[0], 999)
        # For subvariants, extract numeric value if possible
        if r"\varepsilon" in sub:
            import re
            match = re.search(r'\^{?(\d+)', sub)
            if match:
                return (method_idx, 0, -int(match.group(1)))  # Negative to reverse sort (higher eps first)
        return (method_idx, 1, 0)

    methods.sort(key=sort_key)

    records = collect(methods)
    table = make_table(methods, records)
    print(table)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "quality_table_ok_490d_all_methods.tex")

    with open(out_path, "w") as f:
        f.write(table + "\n")

    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
