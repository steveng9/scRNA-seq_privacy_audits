"""
Summarise the Class B ablation results and recommend the best variant.

Reads experiments/ablation/results/summary.csv produced by run_class_b_ablation.py.

Outputs
-------
- Console table: mean AUC (BB+aux, BB-aux) per variant, averaged across all
  SD2 (dataset, nd, trial) combinations, plus delta vs baseline.
- Separate regression table for non-SD2 targets.
- Recommended variant printed at the end.

Usage
-----
  python experiments/ablation/report_ablation.py
  python experiments/ablation/report_ablation.py --metric auc_aux
  python experiments/ablation/report_ablation.py --top 10
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_REPO   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULT_DIR = os.path.join(_REPO, "experiments", "ablation", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="auc_aux",
                        choices=["auc_aux", "auc_noaux"],
                        help="Primary metric for ranking (default: auc_aux)")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N variants in the ranking table")
    parser.add_argument("--csv", default=os.path.join(RESULT_DIR, "summary.csv"))
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"No summary file at {args.csv}. Run run_class_b_ablation.py first.")
        sys.exit(1)

    df = pd.read_csv(args.csv)
    print(f"\nLoaded {len(df)} rows from {args.csv}")
    print(f"Datasets: {sorted(df['dataset'].unique())}")
    print(f"Variants: {len(df['variant'].unique())}")
    print(f"Trials: {df.groupby(['dataset','nd'])['trial'].max().to_dict()}")

    # ---------------------------------------------------------------------------
    # SD2 targets: primary ablation
    # ---------------------------------------------------------------------------
    sd2 = df[df["label"] == "SD2"].copy()
    if sd2.empty:
        print("\nNo SD2 results yet.")
    else:
        _report_table(sd2, args.metric, args.top, "SD2 primary ablation")
        _recommend(sd2, args.metric)

    # ---------------------------------------------------------------------------
    # Non-SD2 targets: regression check
    # ---------------------------------------------------------------------------
    nsd2 = df[df["label"] == "non-SD2"].copy()
    if nsd2.empty:
        print("\nNo non-SD2 regression results yet.")
    else:
        _report_table(nsd2, args.metric, args.top, "non-SD2 regression check")
        _regression_warning(nsd2, args.metric)


def _report_table(df, metric, top_n, title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    baseline_vals = df[df["variant"] == "baseline"][metric].dropna()
    baseline_mean = baseline_vals.mean() if len(baseline_vals) > 0 else float("nan")
    print(f"  Baseline ({metric}) mean = {baseline_mean:.4f}  (n={len(baseline_vals)} trials)")

    summary = (
        df.groupby("variant")[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
        .assign(delta=lambda d: d["mean"] - baseline_mean)
        .sort_values("mean", ascending=False)
    )

    print(f"\n  {'Variant':<28}  {'Mean':>7}  {'Std':>6}  {'Delta':>7}  {'N':>4}")
    print(f"  {'-'*28}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*4}")
    for _, row in summary.head(top_n).iterrows():
        marker = " ←" if row["variant"] == "baseline" else ""
        print(f"  {row['variant']:<28}  {row['mean']:>7.4f}  {row['std']:>6.4f}"
              f"  {row['delta']:>+7.4f}  {int(row['n']):>4}{marker}")


def _recommend(df, metric):
    summary = (
        df.groupby("variant")[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean"})
        .sort_values("mean", ascending=False)
    )
    best = summary.iloc[0]
    baseline = summary[summary["variant"] == "baseline"]["mean"].values
    baseline_mean = baseline[0] if len(baseline) > 0 else float("nan")

    print(f"\n  ★ Recommended variant: {best['variant']}")
    print(f"    Mean {metric} = {best['mean']:.4f}  "
          f"(Δ = {best['mean'] - baseline_mean:+.4f} vs baseline)")

    # Parse recommended settings from name
    vname = best["variant"]
    if vname == "baseline":
        print("    No Class B improvement found — use pure Mahalanobis.")
        return

    # Lookup in ABLATION_VARIANTS
    sys.path.insert(0, os.path.join(_REPO, "experiments", "ablation"))
    from run_class_b_ablation import ABLATION_VARIANTS
    match = [v for v in ABLATION_VARIANTS if v[0] == vname]
    if match:
        _, gene_set, scoring, gamma = match[0]
        print(f"\n  Suggested mamamia_params additions:")
        print(f"    class_b_gene_set: \"{gene_set}\"")
        print(f"    class_b_scoring:  \"{scoring}\"")
        print(f"    class_b_gamma:    {gamma!r}")


def _regression_warning(df, metric):
    summary = (
        df.groupby("variant")[metric]
        .mean()
        .reset_index()
        .rename(columns={metric: "mean"})
    )
    baseline_mean = summary[summary["variant"] == "baseline"]["mean"].values
    if len(baseline_mean) == 0:
        return
    baseline_mean = baseline_mean[0]

    degraded = summary[summary["mean"] < baseline_mean - 0.02]
    if degraded.empty:
        print(f"\n  ✓ No variant degrades non-SD2 performance by >0.02 AUC.")
    else:
        print(f"\n  ⚠ Variants that degrade non-SD2 performance by >0.02 AUC:")
        for _, row in degraded.iterrows():
            print(f"    {row['variant']:<28}  Δ = {row['mean'] - baseline_mean:+.4f}")


if __name__ == "__main__":
    main()
