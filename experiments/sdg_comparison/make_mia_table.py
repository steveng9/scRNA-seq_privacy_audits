"""
make_mia_table.py — Generate a LaTeX table of scMAMA-MIA AUC results.

Reads mamamia_results.csv files produced by run_experiment.py / run_mia_sweep.py,
computes mean ± std over available trials for each (SDG, donor-count, threat-model)
combination, and emits a booktabs LaTeX table.

Result paths follow the pattern:
    /home/golobs/data/{dataset_name}/{nd}d/{trial}/results/mamamia_results.csv

where dataset_name uses "/" for DP sub-directories (e.g. ok_dp/eps_1).

Usage
-----
    python experiments/sdg_comparison/make_mia_table.py [options]

Options
-------
  --nd N [N ...]
        Donor counts to include (default: 10 20 50).

  --tm {bb_aux,bb_noaux,wb_aux,wb_noaux} [...]
        One or more threat models to show per donor-count column.
        Use "all" for all four (WB+aux, WB-aux, BB+aux, BB-aux).
        Default: bb_aux  (BB+aux — primary metric in the paper).
        For non-scDesign2 SDGs, WB threat models are marked "N/A".

  --metric {auc,tpr01,tpr05}
        Which metric to report (default: auc).
          auc   — ROC AUC
          tpr01 — TPR at FPR=0.01
          tpr05 — TPR at FPR=0.05

  --decimals N
        Decimal places in the formatted cells (default: 2).

  --no-std
        Suppress the ±std subscript; show mean only.

  --min-trials N
        Minimum number of trials required to show a cell value instead
        of "---" (default: 1).

  --output FILE
        Path for the output .tex file (default: figures/mia_table.tex).

  --no-save
        Print to stdout only; do not write to file.

Examples
--------
    # Default: BB+aux only, 10/20/50 donors, 2 decimals, with ±std
    python experiments/sdg_comparison/make_mia_table.py

    # Both BB threat models at 10 and 50 donors, 3 decimals, no std
    python experiments/sdg_comparison/make_mia_table.py \\
        --tm bb_aux bb_noaux --nd 10 50 --decimals 3 --no-std

    # Full four-TM table (WB+aux, WB-aux, BB+aux, BB-aux)
    python experiments/sdg_comparison/make_mia_table.py --tm all --nd 10 20 50

    # Require all 5 trials before showing a cell
    python experiments/sdg_comparison/make_mia_table.py --min-trials 5
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Threat-model codes used in the CSV columns
TM_CODE = {
    "wb_aux":   "000",
    "wb_noaux": "001",
    "bb_aux":   "100",
    "bb_noaux": "101",
}
TM_CODE_ORDER = ["000", "001", "100", "101"]

TM_LABEL = {
    "000": "WB+aux",
    "001": "WB-aux",
    "100": "BB+aux",
    "101": "BB-aux",
}

# Metric names as stored in the CSV "metric" column
METRIC_ROW = {
    "auc":   "auc",
    "tpr01": "tpr@fpr=0.01",
    "tpr05": "tpr@fpr=0.05",
}

# ---------------------------------------------------------------------------
# SDG method registry
# (sdg_key, display_name, sub_label, dataset_path, wb_supported)
#   wb_supported — True only for scDesign2, which has actual white-box access
# ---------------------------------------------------------------------------
SDG_METHODS = [
    # key          display_name      sub_label                dataset_path                       wb_supported
    ("sd2",        "scDesign2",      "",                      "ok/scdesign2/no_dp",              True),
    ("sd2_dp_1e9", "scDesign2+DP",   r"$\varepsilon=10^{9}$", "ok/scdesign2/eps_1000000000",     False),
    ("sd2_dp_1e8", "scDesign2+DP",   r"$\varepsilon=10^{8}$", "ok/scdesign2/eps_100000000",      False),
    ("sd2_dp_1e7", "scDesign2+DP",   r"$\varepsilon=10^{7}$", "ok/scdesign2/eps_10000000",       False),
    ("sd2_dp_1e6", "scDesign2+DP",   r"$\varepsilon=10^{6}$", "ok/scdesign2/eps_1000000",        False),
    ("sd2_dp_1e5", "scDesign2+DP",   r"$\varepsilon=10^{5}$", "ok/scdesign2/eps_100000",         False),
    ("sd2_dp_1e4", "scDesign2+DP",   r"$\varepsilon=10^{4}$", "ok/scdesign2/eps_10000",          False),
    ("sd2_dp_1e3", "scDesign2+DP",   r"$\varepsilon=10^{3}$", "ok/scdesign2/eps_1000",           False),
    ("sd2_dp_1e2", "scDesign2+DP",   r"$\varepsilon=10^{2}$", "ok/scdesign2/eps_100",            False),
    ("sd2_dp_1e1", "scDesign2+DP",   r"$\varepsilon=10^{1}$", "ok/scdesign2/eps_10",             False),
    ("sd2_dp_1e0", "scDesign2+DP",   r"$\varepsilon=10^{0}$", "ok/scdesign2/eps_1",              False),
    ("sd3g",       "scDesign3-G",    "",                      "ok/scdesign3/gaussian",           False),
    ("sd3v",       "scDesign3-V",    "",                      "ok/scdesign3/vine",               False),
    ("scvi",       "scVI",           "",                      "ok/scvi/no_dp",                   False),
    ("scdiff",     "scDiffusion",    "",                      "ok/scdiffusion/no_dp",            False),
    ("zinbwave",   "ZINBWave",       "",                      "ok/zinbwave/no_dp",                False),
    ("nmf",        "NMF",            "",                      "ok/nmf/no_dp",                    False),
    ("nmf_dp_1e8", "NMF+DP",        r"$\varepsilon=10^{8}$", "ok/nmf/eps_100000000",             False),
    ("nmf_dp_1e7", "NMF+DP",        r"$\varepsilon=10^{7}$", "ok/nmf/eps_10000000",              False),
    ("nmf_dp_1e6", "NMF+DP",        r"$\varepsilon=10^{6}$", "ok/nmf/eps_1000000",               False),
    ("nmf_dp_1e5", "NMF+DP",        r"$\varepsilon=10^{5}$", "ok/nmf/eps_100000",                False),
    ("nmf_dp_1e4", "NMF+DP",        r"$\varepsilon=10^{4}$", "ok/nmf/eps_10000",                 False),
    ("nmf_dp_1e3", "NMF+DP",        r"$\varepsilon=10^{3}$", "ok/nmf/eps_1000",                  False),
    ("nmf_dp_1e2", "NMF+DP",        r"$\varepsilon=10^{2}$", "ok/nmf/eps_100",                   False),
    ("nmf_dp_1e1", "NMF+DP",        r"$\varepsilon=10^{1}$", "ok/nmf/eps_10",                    False),
    ("nmf_dp_2p8", "NMF+DP",        r"$\varepsilon=2.8$",    "ok/nmf/eps_2.8",                   False),
    ("nmf_dp_1e0", "NMF+DP",        r"$\varepsilon=10^{0}$", "ok/nmf/eps_1",                     False),
]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def collect_values(
    dataset_path: str, nd: int, tm_code: str, metric_key: str,
    results_filename: str = "mamamia_results.csv",
) -> list[float]:
    """
    Return a list of metric values (one per completed trial) for the given
    (dataset, donor-count, threat-model, metric) combination.
    """
    metric_row_name = METRIC_ROW.get(metric_key, metric_key)
    data_dir = os.path.join(DATA_DIR, *dataset_path.split("/"))
    values = []

    for trial in range(1, N_TRIALS + 1):
        csv_path = os.path.join(
            data_dir, f"{nd}d", str(trial), "results", results_filename
        )
        if not os.path.exists(csv_path):
            continue
        try:
            df = pd.read_csv(csv_path)
            rows = df[df["metric"] == metric_row_name]
            if rows.empty:
                continue
            col = f"tm:{tm_code}"
            if col not in rows.columns:
                continue
            val = rows[col].iloc[0]
            if pd.notna(val):
                values.append(float(val))
        except Exception:
            continue

    return values


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------

def fmt_cell(
    values: list[float],
    decimals: int = 2,
    show_std: bool = True,
    min_trials: int = 1,
) -> str:
    """Format a list of metric values as a LaTeX cell string."""
    if len(values) < min_trials:
        return "---"
    mean = np.mean(values)
    incomplete_star = "*" if 0 < len(values) < N_TRIALS else ""
    if show_std and len(values) > 1:
        std = np.std(values)
        return (
            f"{mean:.{decimals}f}"
            r"{\scriptsize$\pm$"
            f"{std:.{decimals}f}"
            "}"
            f"{incomplete_star}"
        )
    return f"{mean:.{decimals}f}{incomplete_star}"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def make_table(
    nd_list: list[int],
    tm_display: list[str],   # ordered list of tm_codes to show per nd
    metric_key: str = "auc",
    decimals: int = 2,
    show_std: bool = True,
    min_trials: int = 1,
    results_filename: str = "mamamia_results.csv",
) -> str:
    """Return a complete LaTeX tabular string."""

    n_tm   = len(tm_display)
    n_data = len(nd_list) * n_tm

    # --- column spec --------------------------------------------------------
    col_spec = "ll" + "c" * n_data
    lines = []
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # --- header row 1: donor counts ----------------------------------------
    nd_headers = []
    for i, nd in enumerate(nd_list):
        # Add column-separator decoration between donor-count groups
        sep = "|" if (n_tm > 1 and i < len(nd_list) - 1) else ""
        nd_headers.append(
            r"\multicolumn{" + str(n_tm) + r"}{c" + sep + r"}{"
            + str(nd) + r" donors}"
        )
    lines.append(r"Method & & " + " & ".join(nd_headers) + r" \\")

    # --- header row 2: threat-model names (always shown so the TM is never ambiguous) ---
    tm_header_cells = [TM_LABEL[code] for code in tm_display] * len(nd_list)
    lines.append(r" & & " + " & ".join(tm_header_cells) + r" \\")

    lines.append(r"\midrule")

    # --- data rows ----------------------------------------------------------
    prev_method = None
    for sdg_key, method_name, sub_label, dataset_path, wb_supported in SDG_METHODS:

        # Spacing between method groups (but not inside the DP sub-rows)
        if method_name != prev_method:
            if prev_method is not None:
                lines.append(r"\addlinespace[3pt]")
            display_method = method_name
        else:
            display_method = ""
        prev_method = method_name

        row_parts = [display_method, sub_label]

        for nd in nd_list:
            for tm_code in tm_display:
                is_wb = tm_code in ("000", "001")
                if is_wb and not wb_supported:
                    row_parts.append(r"N/A")
                    continue
                values = collect_values(dataset_path, nd, tm_code, metric_key, results_filename)
                row_parts.append(
                    fmt_cell(values, decimals=decimals, show_std=show_std,
                             min_trials=min_trials)
                )

        lines.append(" & ".join(row_parts) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("")

    # --- footnote comments --------------------------------------------------
    metric_label = {"auc": "ROC AUC", "tpr01": "TPR@FPR=0.01", "tpr05": "TPR@FPR=0.05"}.get(
        metric_key, metric_key
    )
    lines.append(f"% Metric: {metric_label} at donor level, mean over up to {N_TRIALS} trials")
    if show_std:
        lines.append(r"% Subscript shows \pm std over trials")
    lines.append(f"% * = fewer than {N_TRIALS} trials available; --- = no data yet")
    lines.append("% N/A = white-box access not applicable to this SDG")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------------

def print_coverage(
    nd_list: list[int], tm_display: list[str], metric_key: str,
    results_filename: str = "mamamia_results.csv",
) -> None:
    """Print a plain-text coverage summary to stderr."""
    print("\nCoverage summary:", file=sys.stderr)
    for sdg_key, method_name, sub_label, dataset_path, wb_supported in SDG_METHODS:
        label = f"{method_name} {sub_label}".strip()
        parts = []
        for nd in nd_list:
            for tm_code in tm_display:
                is_wb = tm_code in ("000", "001")
                if is_wb and not wb_supported:
                    continue
                n = len(collect_values(dataset_path, nd, tm_code, metric_key, results_filename))
                parts.append(f"{nd}d/{TM_LABEL[tm_code]}:{n}/{N_TRIALS}")
        print(f"  {label:<30}  {',  '.join(parts)}", file=sys.stderr)
    print("", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--nd", nargs="+", type=int, default=[10, 20, 50],
        metavar="N",
        help="Donor counts to include (default: 10 20 50)",
    )
    parser.add_argument(
        "--tm", nargs="+",
        choices=["bb_aux", "bb_noaux", "wb_aux", "wb_noaux", "all"],
        default=["bb_aux"],
        metavar="TM",
        help=(
            "Threat model(s) to show: bb_aux, bb_noaux, wb_aux, wb_noaux, or all. "
            "Default: bb_aux"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=["auc", "tpr01", "tpr05"],
        default="auc",
        help="Metric to report (default: auc)",
    )
    parser.add_argument(
        "--decimals", type=int, default=2,
        help="Decimal places in formatted cells (default: 2)",
    )
    parser.add_argument(
        "--no-std", action="store_true",
        help="Suppress ±std subscript; show mean only",
    )
    parser.add_argument(
        "--min-trials", type=int, default=1,
        metavar="N",
        help="Minimum trials required to show a cell value (default: 1)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output .tex file path (default: figures/mia_table.tex)",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Print to stdout only; do not write to file",
    )
    parser.add_argument(
        "--classb", action="store_true",
        help=(
            "Read from mamamia_results_classb.csv (Class B / Mahalanobis+LLR attack) "
            "instead of mamamia_results.csv (standard Mahalanobis). "
            "Default output path becomes figures/mia_table_classb.tex."
        ),
    )
    args = parser.parse_args()

    # Resolve threat-model codes, preserving order and de-duplicating
    if "all" in args.tm:
        tm_display = TM_CODE_ORDER  # [000, 001, 100, 101]
    else:
        seen: set[str] = set()
        tm_display = []
        for name in args.tm:
            code = TM_CODE[name]
            if code not in seen:
                tm_display.append(code)
                seen.add(code)

    results_filename = "mamamia_results_classb.csv" if args.classb else "mamamia_results.csv"

    # Coverage check to stderr
    print_coverage(args.nd, tm_display, args.metric, results_filename)

    # Build table
    table = make_table(
        nd_list=args.nd,
        tm_display=tm_display,
        metric_key=args.metric,
        decimals=args.decimals,
        show_std=not args.no_std,
        min_trials=args.min_trials,
        results_filename=results_filename,
    )

    print(table)

    if not args.no_save:
        default_tex = "mia_table_classb.tex" if args.classb else "mia_table.tex"
        out_path = args.output or os.path.join(REPO_ROOT, "figures", default_tex)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(table + "\n")
        print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
