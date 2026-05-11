"""
Generate a LaTeX table of scDesign2/no_dp attack results (scMAMA-MIA and baselines).

Rows: scMAMA-MIA (4 threat models × 2 variants: regular + ClassB), then baselines.
Columns: Three dataset groups (ok, aida, cg), each with available donor counts as subcolumns.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

DATA_DIR = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5

# Threat model order and labels
TM_ORDER = ["wb_aux", "wb_noaux", "bb_aux", "bb_noaux"]
TM_LABEL = {
    "wb_aux":   "WB+aux",
    "wb_noaux": "WB-aux",
    "bb_aux":   "BB+aux",
    "bb_noaux": "BB-aux",
}
TM_CODE = {
    "wb_aux":   "000",
    "wb_noaux": "001",
    "bb_aux":   "100",
    "bb_noaux": "101",
}

# Baselines (display name -> CSV method name)
BASELINES = [
    ("LOGAN", "LOGAN_D1"),
    ("GAN-Leaks (Cal.)", "gan_leaks_cal"),
    ("GAN-Leaks (SC)", "gan_leaks_sc"),
    ("DOMIAS+KDE", "domias_kde"),
    ("MC", "MC"),
]

# Dataset info
DATASETS = [
    ("ok", ["2d", "5d", "10d", "20d", "50d", "100d", "200d", "490d"]),
    ("aida", ["2d", "5d", "10d", "20d", "50d", "100d", "200d"]),
    ("cg", ["2d", "5d", "10d", "11d", "20d", "50d", "100d", "200d"]),
]


def collect_mamamia_auc(dataset, nd, tm_code, use_classb=False):
    """Collect scMAMA-MIA AUC values across trials."""
    values = []
    metric_col = f"tm:{tm_code}"
    results_file = "mamamia_results_classb.csv" if use_classb else "mamamia_results.csv"

    for trial in range(1, N_TRIALS + 1):
        csv_path = os.path.join(
            DATA_DIR, dataset, "scdesign2", "no_dp", nd, str(trial),
            "results", results_file
        )
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


def collect_baseline_auc(dataset, nd, baseline_name):
    """Collect baseline MIA AUC values across trials."""
    values = []

    for trial in range(1, N_TRIALS + 1):
        eval_path = os.path.join(
            DATA_DIR, dataset, "scdesign2", "no_dp", nd, str(trial),
            "results", "baseline_mias", "baselines_evaluation_results.csv"
        )
        if not os.path.exists(eval_path):
            continue
        try:
            df = pd.read_csv(eval_path)
            # Find row for this method
            rows = df[df["method"] == baseline_name]
            if rows.empty:
                continue
            auc = float(rows["aucroc"].iloc[0])
            if pd.notna(auc):
                values.append(auc)
        except Exception:
            continue

    return values


def fmt_cell(values):
    """Format a list of values as mean with trial count superscript if incomplete."""
    if not values:
        return "---"
    mean = np.mean(values)
    n_trials = len(values)
    if n_trials < N_TRIALS:
        return f"{mean:.3f}$^{{{n_trials}}}$"
    return f"{mean:.3f}"


def make_table():
    """Generate LaTeX table."""
    # Build column spec
    n_cols_per_ds = [len(nds) for _, nds in DATASETS]
    col_spec = "l" + "|".join(["c" * n for n in n_cols_per_ds])

    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row 1: dataset names with multicolumn
    ds_headers = []
    for i, (ds_name, nds) in enumerate(DATASETS):
        ds_label = {"ok": "OK1K", "aida": "AIDA", "cg": "CG"}[ds_name]
        sep = "|" if i < len(DATASETS) - 1 else ""
        ds_headers.append(r"\multicolumn{" + str(len(nds)) + "}{c" + sep + "}{" + ds_label + "}")
    lines.append("Method & " + " & ".join(ds_headers) + r" \\")

    # Header row 2: donor counts
    nd_headers = []
    for _, nds in DATASETS:
        for nd in nds:
            nd_headers.append(nd.replace("d", ""))
    lines.append(" & " + " & ".join(nd_headers) + r" \\")
    lines.append(r"\midrule")

    # scMAMA-MIA rows (4 threat models × 2 variants)
    for variant_label, use_classb in [("", False), ("(ClassB)", True)]:
        for tm_name in TM_ORDER:
            row_label = TM_LABEL[tm_name]
            if variant_label:
                row_label += f" {variant_label}"

            row = [row_label]
            tm_code = TM_CODE[tm_name]
            for ds_name, nds in DATASETS:
                for nd in nds:
                    values = collect_mamamia_auc(ds_name, nd, tm_code, use_classb)
                    row.append(fmt_cell(values))

            lines.append(" & ".join(row) + r" \\")

        if not use_classb:
            lines.append(r"\addlinespace[2pt]")

    # Baseline rows
    lines.append(r"\addlinespace[2pt]")
    for baseline_display, baseline_csv in BASELINES:
        row = [baseline_display]
        for ds_name, nds in DATASETS:
            for nd in nds:
                values = collect_baseline_auc(ds_name, nd, baseline_csv)
                row.append(fmt_cell(values))

        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")
    lines.append(r"% scDesign2/no_dp only")
    lines.append(r"% $^n$ = fewer than 5 trials (n trials averaged)")

    return "\n".join(lines)


def main():
    table = make_table()
    print(table)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sd2_nodp_table.tex")

    with open(out_path, "w") as f:
        f.write(table + "\n")

    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
