"""
make_subgroup_table.py — LaTeX table of scMAMA-MIA scores broken down by
demographic subgroup (sex / ethnicity) on AIDA.

Reads per-cell scores from
    {DATA_DIR}/{dataset_path}/{nd}d/{trial}/results/mamamia_all_scores.csv
(or _classb.csv with --classb), aggregates to donor level the same way
src/attacks/scmamamia/scoring.py:aggregate_scores_by_donor does (mean cell
score per donor; activation is a monotonic transform so AUC is unaffected),
restricts to a subgroup, and computes the chosen metric.

Output columns are ordered: SDG × donor-count × threat-model.
Output rows are subgroup levels (e.g. female, male).

Usage
-----
    python experiments/sdg_comparison/make_subgroup_table.py [options]

Examples
--------
    # Default: AIDA, by-sex, BB+aux, scDesign2 + scDiffusion + NMF, 10/20/50d
    python experiments/sdg_comparison/make_subgroup_table.py

    # By ethnicity, both BB threat models, Class B attack file
    python experiments/sdg_comparison/make_subgroup_table.py \\
        --subgroup ethnicity --tm bb_aux bb_noaux --classb

    # AUC per subgroup with stricter coverage
    python experiments/sdg_comparison/make_subgroup_table.py \\
        --metric auc --min-trials 3 --min-donors-per-class 2

    # Just scDesign2, all four threat models
    python experiments/sdg_comparison/make_subgroup_table.py \\
        --sdg sd2 --tm all
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

TM_CODE = {
    "wb_aux":   "000",
    "wb_noaux": "001",
    "bb_aux":   "100",
    "bb_noaux": "101",
}
TM_CODE_ORDER = ["000", "001", "100", "101"]
TM_LABEL = {
    "000": "WB+aux", "001": "WB-aux", "100": "BB+aux", "101": "BB-aux",
}

# (sdg_key, display_name, sub_label, dataset_path_template, wb_supported)
# dataset_path_template uses {dataset} placeholder so we can swap aida/ok/cg.
SDG_METHODS = [
    ("sd2",        "scDesign2",     "",                      "{dataset}/scdesign2/no_dp",          True),
    ("sd2_dp_1e9", "scDesign2+DP",  r"$\varepsilon=10^{9}$", "{dataset}/scdesign2/eps_1000000000", False),
    ("sd2_dp_1e6", "scDesign2+DP",  r"$\varepsilon=10^{6}$", "{dataset}/scdesign2/eps_1000000",    False),
    ("sd2_dp_1e3", "scDesign2+DP",  r"$\varepsilon=10^{3}$", "{dataset}/scdesign2/eps_1000",       False),
    ("sd2_dp_1e0", "scDesign2+DP",  r"$\varepsilon=10^{0}$", "{dataset}/scdesign2/eps_1",          False),
    ("sd3g",       "scDesign3-G",   "",                      "{dataset}/scdesign3/gaussian",       False),
    ("sd3v",       "scDesign3-V",   "",                      "{dataset}/scdesign3/vine",           False),
    ("scvi",       "scVI",          "",                      "{dataset}/scvi/no_dp",               False),
    ("scdiff",     "scDiffusion",   "",                      "{dataset}/scdiffusion/no_dp",        False),
    ("zinbwave",   "ZINBWave",      "",                      "{dataset}/zinbwave/no_dp",           False),
    ("nmf",        "NMF",           "",                      "{dataset}/nmf/no_dp",                False),
    ("nmf_dp_2p8", "NMF+DP",        r"$\varepsilon=2.8$",    "{dataset}/nmf/eps_2.8",              False),
]
SDG_KEYS = [m[0] for m in SDG_METHODS]


# ---------------------------------------------------------------------------
# Per-trial donor aggregation + subgroup metric
# ---------------------------------------------------------------------------

def _read_scores_file(csv_path: str, tm_code: str, subgroup_col: str) -> pd.DataFrame | None:
    """
    Load a mamamia_all_scores[_classb].csv and return columns
    [donor id, membership, score, subgroup]. Returns None if the file or the
    required score column is missing.

    If the file is a *_classb.csv that lacks the subgroup column, fall back to
    reading the donor→subgroup mapping from the sibling mamamia_all_scores.csv
    in the same results dir. This avoids regenerating runs just to pick up
    metadata.
    """
    if not os.path.exists(csv_path):
        return None
    score_col = f"score:{tm_code}"
    try:
        cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    except Exception as e:
        print(f"  [WARN] could not read header of {csv_path}: {e}", file=sys.stderr)
        return None

    if score_col not in cols:
        return None  # this TM wasn't run for this trial

    has_subgroup_inline = subgroup_col in cols
    use_cols = ["donor id", "membership", score_col]
    if has_subgroup_inline:
        use_cols.append(subgroup_col)

    df = pd.read_csv(csv_path, usecols=use_cols)
    df["membership"] = pd.to_numeric(df["membership"], errors="coerce").astype("Int64")
    df = df.rename(columns={score_col: "score"})

    if has_subgroup_inline:
        df = df.rename(columns={subgroup_col: "subgroup"})
        return df

    # Fallback: pull subgroup from sibling mamamia_all_scores.csv (same dir).
    sibling = os.path.join(os.path.dirname(csv_path), "mamamia_all_scores.csv")
    if not os.path.exists(sibling):
        print(f"  [WARN] {csv_path} lacks '{subgroup_col}' and sibling "
              f"mamamia_all_scores.csv not found — skipping.", file=sys.stderr)
        return None
    sib_cols = pd.read_csv(sibling, nrows=0).columns.tolist()
    if subgroup_col not in sib_cols:
        print(f"  [WARN] sibling {sibling} also lacks '{subgroup_col}' — skipping.",
              file=sys.stderr)
        return None
    sib = pd.read_csv(sibling, usecols=["donor id", subgroup_col])
    sib = sib.drop_duplicates("donor id").rename(columns={subgroup_col: "subgroup"})
    merged = df.merge(sib, on="donor id", how="left")
    n_missing = merged["subgroup"].isna().sum()
    if n_missing > 0:
        print(f"  [WARN] {csv_path}: {n_missing} cells had no subgroup match "
              f"in sibling and were dropped.", file=sys.stderr)
        merged = merged.dropna(subset=["subgroup"])
    return merged


def _donor_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse cells → donors. One row per donor with mean cell score, the
    donor's membership label, and the donor's subgroup. Asserts internal
    consistency (single membership / single subgroup per donor).
    """
    g = df.groupby("donor id", observed=True)
    out = g.agg(
        score=("score", "mean"),
        membership=("membership", "first"),
        membership_n=("membership", "nunique"),
        subgroup=("subgroup", "first"),
        subgroup_n=("subgroup", "nunique"),
    ).reset_index()
    bad = out[(out["membership_n"] > 1) | (out["subgroup_n"] > 1)]
    if not bad.empty:
        raise ValueError(
            f"Inconsistent membership/subgroup within donors: {bad['donor id'].tolist()}"
        )
    out = out.drop(columns=["membership_n", "subgroup_n"])
    out["membership"] = out["membership"].astype(int)
    return out


def _subgroup_metric(
    donor_df: pd.DataFrame,
    subgroup_value: str,
    metric_key: str,
    min_donors_per_class: int,
) -> float | None:
    """
    Compute the chosen metric for one subgroup level in a single trial.
    Returns None when the data is insufficient (caller treats as missing).
    """
    sub = donor_df[donor_df["subgroup"] == subgroup_value]
    if sub.empty:
        return None

    n_pos = int((sub["membership"] == 1).sum())
    n_neg = int((sub["membership"] == 0).sum())

    if metric_key == "auc":
        if n_pos < min_donors_per_class or n_neg < min_donors_per_class:
            return None
        return float(roc_auc_score(sub["membership"].values, sub["score"].values))
    if metric_key == "mean_score":
        return float(sub["score"].mean())
    if metric_key == "mean_score_member":
        if n_pos < 1:
            return None
        return float(sub.loc[sub["membership"] == 1, "score"].mean())
    if metric_key == "mean_score_nonmember":
        if n_neg < 1:
            return None
        return float(sub.loc[sub["membership"] == 0, "score"].mean())
    if metric_key == "score_gap":
        if n_pos < 1 or n_neg < 1:
            return None
        return (
            float(sub.loc[sub["membership"] == 1, "score"].mean())
            - float(sub.loc[sub["membership"] == 0, "score"].mean())
        )
    raise ValueError(f"Unknown metric: {metric_key}")


def collect_subgroup_values(
    dataset_path: str,
    nd: int,
    tm_code: str,
    subgroup_col: str,
    metric_key: str,
    results_filename: str,
    min_donors_per_class: int,
) -> dict[str, list[float]]:
    """
    Walk trials and return {subgroup_level: [per-trial metric values]}.
    """
    out: dict[str, list[float]] = {}
    for trial in range(1, N_TRIALS + 1):
        csv_path = os.path.join(
            DATA_DIR, dataset_path, f"{nd}d", str(trial), "results", results_filename
        )
        df = _read_scores_file(csv_path, tm_code, subgroup_col)
        if df is None:
            continue
        try:
            donors = _donor_table(df)
        except ValueError as e:
            print(f"  [WARN] {csv_path}: {e}", file=sys.stderr)
            continue
        for level in donors["subgroup"].dropna().unique().tolist():
            v = _subgroup_metric(donors, level, metric_key, min_donors_per_class)
            if v is not None:
                out.setdefault(level, []).append(v)
    return out


# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------

def fmt_cell(values: list[float], decimals: int, show_std: bool, min_trials: int) -> str:
    if len(values) < min_trials:
        return "---"
    mean = float(np.mean(values))
    star = "*" if 0 < len(values) < N_TRIALS else ""
    if show_std and len(values) > 1:
        std = float(np.std(values))
        return f"{mean:.{decimals}f}{{\\scriptsize$\\pm${std:.{decimals}f}}}{star}"
    return f"{mean:.{decimals}f}{star}"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(
    dataset: str,
    subgroup_col: str,
    sdg_keys: list[str],
    nd_list: list[int],
    tm_codes: list[str],
    metric_key: str,
    results_filename: str,
    decimals: int,
    show_std: bool,
    min_trials: int,
    min_donors_per_class: int,
) -> str:
    methods = [m for m in SDG_METHODS if m[0] in set(sdg_keys)]
    # Preserve user-supplied SDG ordering
    methods.sort(key=lambda m: sdg_keys.index(m[0]))

    # ---------------------- gather data ----------------------
    # data[(sdg_key, nd, tm_code)][subgroup_level] = [per-trial values]
    data: dict[tuple[str, int, str], dict[str, list[float]]] = {}
    all_levels: set[str] = set()
    for sdg_key, _name, _sub, path_tmpl, wb_ok in methods:
        dataset_path = path_tmpl.format(dataset=dataset)
        for nd in nd_list:
            for tm_code in tm_codes:
                if tm_code in ("000", "001") and not wb_ok:
                    data[(sdg_key, nd, tm_code)] = {}
                    continue
                d = collect_subgroup_values(
                    dataset_path, nd, tm_code, subgroup_col,
                    metric_key, results_filename, min_donors_per_class,
                )
                data[(sdg_key, nd, tm_code)] = d
                all_levels.update(d.keys())

    if not all_levels:
        print(
            f"[ERROR] No subgroup data found for dataset={dataset}, "
            f"subgroup={subgroup_col}. Check that mamamia_all_scores"
            f"{'_classb' if 'classb' in results_filename else ''}.csv files "
            f"contain the '{subgroup_col}' column.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Stable row ordering: known sex order first, otherwise alphabetical
    if subgroup_col == "sex":
        preferred = ["female", "male"]
    else:
        preferred = sorted(all_levels)
    levels = [lv for lv in preferred if lv in all_levels]
    levels += sorted(lv for lv in all_levels if lv not in levels)

    # ---------------------- header ----------------------
    n_tm = len(tm_codes)
    n_nd = len(nd_list)
    n_data_cols = len(methods) * n_nd * n_tm
    col_spec = "l" + "c" * n_data_cols

    # Build caption + label that fully describe every dimension of the table.
    metric_phrase = {
        "auc":                  "donor-level ROC AUC restricted to the subgroup",
        "mean_score":           "mean donor-level score (members and non-members combined)",
        "mean_score_member":    "mean donor-level score for members only",
        "mean_score_nonmember": "mean donor-level score for non-members only",
        "score_gap":            "mean(member) $-$ mean(non-member) of donor-level scores",
    }[metric_key]
    attack_phrase = (
        "Class~B Mahalanobis$+$LLR attack" if "classb" in results_filename
        else "standard Mahalanobis attack"
    )
    sdg_phrase = ", ".join(
        (name + (f" ({sub})" if sub else "")) for _k, name, sub, *_ in methods
    )
    tm_phrase = ", ".join(TM_LABEL[c] for c in tm_codes)
    nd_phrase = ", ".join(f"{nd}" for nd in nd_list)
    std_clause = (
        r" Subscript: $\pm$~std over trials. " if show_std else " "
    )
    caption = (
        f"scMAMA-MIA disparate-impact breakdown by {subgroup_col} on the "
        f"\\textsc{{{dataset}}} dataset. Cells report {metric_phrase}, "
        f"averaged over up to {N_TRIALS} trials.{std_clause}"
        f"Attack: {attack_phrase}. "
        f"SDG methods: {sdg_phrase}. "
        f"Donor counts (training-set size): {nd_phrase}. "
        f"Threat models: {tm_phrase} "
        r"(WB: white-box copula access; BB: black-box; "
        r"$\pm$aux: with/without auxiliary reference dataset). "
        f"\\texttt{{---}}: cell below \\texttt{{--min-trials={min_trials}}} "
        "(or, for AUC, fewer than "
        f"\\texttt{{--min-donors-per-class={min_donors_per_class}}} member or "
        "non-member donors in the subgroup); "
        "$*$: fewer than the full "
        f"{N_TRIALS} trials available; \\texttt{{N/A}}: white-box not applicable to this SDG. "
        f"Source: \\texttt{{{results_filename}}}."
    )
    label_attack = "_classb" if "classb" in results_filename else ""
    label = (
        f"tab:subgroup_{dataset}_{subgroup_col}{label_attack}_"
        f"{'-'.join(sdg_keys)}_{'-'.join(str(n) for n in nd_list)}d_"
        f"{'-'.join(tm_codes)}_{metric_key}"
    )

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # row 1: SDG groups
    sdg_hdr = []
    for i, (_, name, sub, *_rest) in enumerate(methods):
        sep = "|" if (i < len(methods) - 1) else ""
        label = name + (f" {sub}" if sub else "")
        sdg_hdr.append(
            r"\multicolumn{" + str(n_nd * n_tm) + r"}{c" + sep + r"}{" + label + r"}"
        )
    lines.append(r"Subgroup & " + " & ".join(sdg_hdr) + r" \\")

    # row 2: donor counts (within each SDG block)
    if n_nd > 1 or n_tm > 1:
        nd_hdr = []
        for _ in methods:
            for j, nd in enumerate(nd_list):
                sep = "|" if (n_tm > 1 and j < n_nd - 1) else ""
                nd_hdr.append(
                    r"\multicolumn{" + str(n_tm) + r"}{c" + sep + r"}{" + str(nd) + r"d}"
                )
        lines.append(r" & " + " & ".join(nd_hdr) + r" \\")

    # row 3: threat model labels
    if n_tm > 1 or len(methods) * n_nd > 1:
        tm_hdr = [TM_LABEL[c] for c in tm_codes] * (len(methods) * n_nd)
        lines.append(r" & " + " & ".join(tm_hdr) + r" \\")

    lines.append(r"\midrule")

    # ---------------------- body ----------------------
    for level in levels:
        row = [_escape_latex(level)]
        for sdg_key, *_rest in methods:
            wb_ok = next(m for m in methods if m[0] == sdg_key)[4]
            for nd in nd_list:
                for tm_code in tm_codes:
                    if tm_code in ("000", "001") and not wb_ok:
                        row.append("N/A")
                        continue
                    vs = data[(sdg_key, nd, tm_code)].get(level, [])
                    row.append(fmt_cell(vs, decimals, show_std, min_trials))
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _escape_latex(s: str) -> str:
    return s.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


# ---------------------------------------------------------------------------
# Coverage report
# ---------------------------------------------------------------------------

def print_coverage(
    dataset: str, sdg_keys: list[str], nd_list: list[int], tm_codes: list[str],
    subgroup_col: str, results_filename: str,
) -> None:
    print(f"\nCoverage summary (dataset={dataset}, subgroup={subgroup_col}, file={results_filename}):", file=sys.stderr)
    methods = [m for m in SDG_METHODS if m[0] in set(sdg_keys)]
    methods.sort(key=lambda m: sdg_keys.index(m[0]))
    for sdg_key, name, sub, path_tmpl, wb_ok in methods:
        dataset_path = path_tmpl.format(dataset=dataset)
        label = (name + (f" {sub}" if sub else "")).strip()
        parts = []
        for nd in nd_list:
            for tm_code in tm_codes:
                if tm_code in ("000", "001") and not wb_ok:
                    continue
                n_trials_with_data = 0
                for trial in range(1, N_TRIALS + 1):
                    csv_path = os.path.join(
                        DATA_DIR, dataset_path, f"{nd}d", str(trial),
                        "results", results_filename,
                    )
                    df = _read_scores_file(csv_path, tm_code, subgroup_col)
                    if df is not None:
                        n_trials_with_data += 1
                parts.append(f"{nd}d/{TM_LABEL[tm_code]}:{n_trials_with_data}/{N_TRIALS}")
        print(f"  {label:<26}  {',  '.join(parts)}", file=sys.stderr)
    print("", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default="aida",
                        help="Dataset key under DATA_DIR (default: aida — only one with sex/ethnicity)")
    parser.add_argument("--subgroup", choices=["sex", "ethnicity"], default="sex",
                        help="Subgroup column to break results down by (default: sex)")
    parser.add_argument("--sdg", nargs="+", default=["sd2", "scdiff", "nmf"],
                        choices=SDG_KEYS,
                        help=f"SDG method keys (default: sd2 scdiff nmf). Choices: {SDG_KEYS}")
    parser.add_argument("--nd", nargs="+", type=int, default=[10, 20, 50],
                        help="Donor counts (default: 10 20 50)")
    parser.add_argument("--tm", nargs="+",
                        choices=["bb_aux", "bb_noaux", "wb_aux", "wb_noaux", "all"],
                        default=["bb_aux"],
                        help="Threat models (default: bb_aux). 'all' expands to WB+aux,WB-aux,BB+aux,BB-aux.")
    parser.add_argument("--metric",
                        choices=["auc", "mean_score", "mean_score_member",
                                 "mean_score_nonmember", "score_gap"],
                        default="auc",
                        help="Per-subgroup metric (default: auc — donor-level AUC restricted to subgroup)")
    parser.add_argument("--decimals", type=int, default=2)
    parser.add_argument("--no-std", action="store_true",
                        help="Suppress ±std subscript")
    parser.add_argument("--min-trials", type=int, default=1,
                        help="Min trials a (subgroup, SDG, nd, tm) cell needs to render (default: 1)")
    parser.add_argument("--min-donors-per-class", type=int, default=1,
                        help="For AUC: min member and non-member donors required in subgroup (default: 1)")
    parser.add_argument("--classb", action="store_true",
                        help="Read mamamia_all_scores_classb.csv instead of mamamia_all_scores.csv. "
                             "NOTE: classb file may lack subgroup columns; falls back gracefully.")
    parser.add_argument("--output", default=None,
                        help="Output .tex path (default: figures/subgroup_table_{dataset}_{subgroup}[_classb].tex)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print to stdout only; do not write to file")
    args = parser.parse_args()

    # Resolve TM codes
    if "all" in args.tm:
        tm_codes = TM_CODE_ORDER
    else:
        seen, tm_codes = set(), []
        for n in args.tm:
            c = TM_CODE[n]
            if c not in seen:
                seen.add(c)
                tm_codes.append(c)

    results_filename = "mamamia_all_scores_classb.csv" if args.classb else "mamamia_all_scores.csv"

    print_coverage(args.dataset, args.sdg, args.nd, tm_codes, args.subgroup, results_filename)

    table = build_table(
        dataset=args.dataset,
        subgroup_col=args.subgroup,
        sdg_keys=args.sdg,
        nd_list=args.nd,
        tm_codes=tm_codes,
        metric_key=args.metric,
        results_filename=results_filename,
        decimals=args.decimals,
        show_std=not args.no_std,
        min_trials=args.min_trials,
        min_donors_per_class=args.min_donors_per_class,
    )
    print(table)

    if not args.no_save:
        suffix = "_classb" if args.classb else ""
        default_name = f"subgroup_table_{args.dataset}_{args.subgroup}{suffix}.tex"
        out_path = args.output or os.path.join(REPO_ROOT, "figures", default_name)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as fh:
            fh.write(table + "\n")
        print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
