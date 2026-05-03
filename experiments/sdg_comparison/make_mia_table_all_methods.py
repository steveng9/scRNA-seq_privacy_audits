"""
Generate a LaTeX table of scMAMA-MIA BB+aux attack results for all SDG methods.

Columns: 10d, 20d, 50d
Variants: non-ClassB (first group) and ClassB (second group)

Scans for all available SDG methods dynamically.
"""

import os
import glob
import sys
import numpy as np
import pandas as pd

DATA_DIR = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5
TM_CODE = "100"  # BB+aux


def find_sdg_methods():
    """Scan for all SDG methods with results and return list of (display_name, sub_label, dataset_path)."""
    methods = []
    ok_dir = os.path.join(DATA_DIR, "ok")

    # Scan for all method directories
    for sdg_dir in glob.glob(os.path.join(ok_dir, "*")):
        if not os.path.isdir(sdg_dir):
            continue
        sdg_type = os.path.basename(sdg_dir)

        # Scan for variants within this method
        for variant_dir in glob.glob(os.path.join(sdg_dir, "*")):
            if not os.path.isdir(variant_dir):
                continue
            variant = os.path.basename(variant_dir)

            # Check if we have results for 10d, 20d, or 50d
            has_results = False
            for nd in ["10d", "20d", "50d"]:
                test_path = os.path.join(variant_dir, nd, "1", "results", "mamamia_results.csv")
                if os.path.exists(test_path):
                    has_results = True
                    break

            if not has_results:
                continue

            # Determine display name and sub-label
            if sdg_type == "scdesign2":
                method_name = "scDesign2" if variant == "no_dp" else "scDesign2+DP"
                dataset_path = f"ok/scdesign2/{variant}"
                if variant == "no_dp":
                    sub_label = ""
                else:
                    eps_str = variant.replace("eps_", "")
                    sub_label = format_epsilon(eps_str)
            elif sdg_type == "nmf":
                method_name = "NMF" if variant == "no_dp" else "NMF+DP"
                dataset_path = f"ok/nmf/{variant}"
                if variant == "no_dp":
                    sub_label = ""
                else:
                    eps_str = variant.replace("eps_", "")
                    sub_label = format_epsilon(eps_str)
            elif sdg_type == "scdesign3":
                method_name = "scDesign3"
                dataset_path = f"ok/scdesign3/{variant}"
                if variant == "gaussian":
                    sub_label = "-G"
                elif variant == "vine":
                    sub_label = "-V"
                else:
                    sub_label = ""
            elif sdg_type == "scvi":
                method_name = "scVI"
                dataset_path = f"ok/scvi/{variant}"
                sub_label = ""
            elif sdg_type == "scdiffusion":
                method_name = "scDiffusion"
                dataset_path = f"ok/scdiffusion/{variant}"
                sub_label = ""
            elif sdg_type == "zinbwave":
                method_name = "ZINBWave"
                dataset_path = f"ok/zinbwave/{variant}"
                sub_label = ""
            else:
                continue

            methods.append((method_name, sub_label, dataset_path))

    return methods


def format_epsilon(eps_str):
    """Convert epsilon string to LaTeX notation."""
    # Handle v2 variants
    if eps_str.startswith("v2_"):
        inner = eps_str[3:]  # Remove "v2_" prefix
        if inner == "no_dp":
            return r"$\varepsilon_{\text{v2}}=\infty$"
        try:
            val = int(inner)
            if val == 1:
                return r"$\varepsilon_{\text{v2}}=10^{0}$"
            import math
            power = round(math.log10(val))
            if 10 ** power == val:
                return rf"$\varepsilon_{{\text{{v2}}}}=10^{{{power}}}$"
        except:
            pass
        return rf"$\varepsilon_{{\text{{v2}}}}={inner}$"

    if eps_str == "2.8":
        return r"$\varepsilon=2.8$"
    try:
        val = int(eps_str)
        if val == 1:
            return r"$\varepsilon=10^{0}$"
        import math
        power = round(math.log10(val))
        if 10 ** power == val:
            return rf"$\varepsilon=10^{{{power}}}$"
    except:
        pass
    return rf"$\varepsilon={eps_str}$"


def collect_auc(dataset_path, nd, use_classb=False):
    """Collect BB+aux AUC values across trials."""
    values = []
    metric_col = f"tm:{TM_CODE}"
    results_file = "mamamia_results_classb.csv" if use_classb else "mamamia_results.csv"

    for trial in range(1, N_TRIALS + 1):
        csv_path = os.path.join(DATA_DIR, dataset_path, nd, str(trial), "results", results_file)
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            rows = df[df["metric"] == "auc"]
            if rows.empty or metric_col not in rows.columns:
                continue
            val = rows[metric_col].iloc[0]
            if pd.notna(val):
                values.append(float(val))
        except Exception:
            continue

    return values


def fmt_cell(values):
    """Format a list of values as mean with trial count superscript if incomplete."""
    if not values:
        return "---"
    mean = np.mean(values)
    n_trials = len(values)
    formatted = f"{mean:.3f}"
    if n_trials < N_TRIALS:
        formatted += f"$^{{{n_trials}}}$"
    return formatted


def make_table(methods):
    """Generate LaTeX table."""
    donor_counts = ["10d", "20d", "50d"]
    n_variants = 2  # non-ClassB and ClassB

    lines = []
    lines.append(r"\begin{tabular}{l" + "|".join(["ccc"] * n_variants) + "}")
    lines.append(r"\toprule")

    # Header row 1: variant names
    variant_labels = ["Non-ClassB", "ClassB"]
    variant_headers = " & ".join(
        r"\multicolumn{3}{c}{" + label + "}" for label in variant_labels
    )
    lines.append("Method & " + variant_headers + r" \\")

    # Header row 2: donor counts (repeated for each variant)
    nd_headers = " & ".join(["10d", "20d", "50d"] * n_variants)
    lines.append(" & " + nd_headers + r" \\")
    lines.append(r"\midrule")

    # Data rows: one per method
    prev_method_name = None
    for method_name, sub_label, dataset_path in methods:
        # Add spacing between method groups
        if method_name != prev_method_name:
            if prev_method_name is not None and prev_method_name not in ["scDesign2", "NMF"]:
                lines.append(r"\addlinespace[2pt]")
            row_label = method_name
        else:
            row_label = ""
        prev_method_name = method_name

        if sub_label:
            row_label = f"{row_label} {sub_label}".strip()
            row = [row_label]
        else:
            row = [row_label]

        # Collect AUC values for all donor counts, both variants
        for variant_idx, use_classb in enumerate([False, True]):
            for nd in donor_counts:
                values = collect_auc(dataset_path, nd, use_classb)
                row.append(fmt_cell(values))

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"% BB+aux threat model only")
    lines.append(r"% $^n$ = fewer than 5 trials (n trials averaged)")

    return "\n".join(lines)


def main():
    methods = find_sdg_methods()

    # Sort methods for consistent ordering
    def sort_key(item):
        method, sub, _ = item
        method_order = {"scDesign2": 0, "scDesign3": 1, "scVI": 2, "scDiffusion": 3, "ZINBWave": 4, "NMF": 5}
        method_idx = method_order.get(method.split("+")[0], 999)
        # For subvariants, extract numeric value if possible
        if r"\varepsilon" in sub:
            import re
            match = re.search(r'\^{?(-?\d+)', sub)
            if match:
                return (method_idx, 0, -int(match.group(1)))
        return (method_idx, 1, 0)

    methods.sort(key=sort_key)

    table = make_table(methods)
    print(table)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mia_table_all_methods_bb_aux.tex")

    with open(out_path, "w") as f:
        f.write(table + "\n")

    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
