"""
Spot-check stale (pre-MMD-fix) quality CSVs for ok/* by re-running the
evaluator on a handful of representative trials and comparing fresh values
to the stored stale ones — without overwriting any existing CSV.

Output: /tmp/spot_check_quality.csv with one row per trial:
    trial_label, metric, stale_value, fresh_value, delta_abs, delta_rel

If fresh and stale values agree, the remaining stale CSVs can be trusted as-is.
"""

import os
import sys
import csv

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from run_quality_evals import make_cfg, DATA  # noqa: E402
from evaluation.sc_evaluate import SingleCellEvaluator  # noqa: E402

import pandas as pd  # noqa: E402

# (label, synth_relpath_under_DATA, dataset_name)
# Donor splits inferred as DATA/{ds}/splits/{nd}d/{trial}/train.npy
# Synth at DATA/{relpath}; full data at DATA/{ds}/full_dataset_cleaned.h5ad
TRIALS = [
    ("ok/scdesign2/no_dp/100d/1",      "ok/scdesign2/no_dp/100d/1/datasets/synthetic.h5ad",      "ok"),
    ("ok/scdesign2/eps_1/50d/1",       "ok/scdesign2/eps_1/50d/1/datasets/synthetic.h5ad",       "ok"),
    ("ok/scdesign2/eps_10000/50d/1",   "ok/scdesign2/eps_10000/50d/1/datasets/synthetic.h5ad",   "ok"),
    ("ok/scvi/no_dp/50d/1",            "ok/scvi/no_dp/50d/1/datasets/synthetic.h5ad",            "ok"),
    ("ok/scdesign3/gaussian/50d/1",    "ok/scdesign3/gaussian/50d/1/datasets/synthetic.h5ad",    "ok"),
]


def stale_values(label):
    csv_path = os.path.join(DATA, label, "results", "quality_eval_results",
                            "results", "statistics_evals.csv")
    if not os.path.exists(csv_path):
        return None
    row = pd.read_csv(csv_path).iloc[0].to_dict()
    return {k: row[k] for k in ("mmd", "lisi", "ari_real_vs_syn") if k in row}


def fresh_values(label, synth_relpath, dataset_name):
    parts = label.split("/")
    nd_tag = parts[-2]   # e.g. "50d"
    trial  = parts[-1]   # e.g. "1"

    synth_path     = os.path.join(DATA, synth_relpath)
    full_data_path = os.path.join(DATA, dataset_name, "full_dataset_cleaned.h5ad")
    train_npy      = os.path.join(DATA, dataset_name, "splits", nd_tag, trial, "train.npy")

    # Use a /tmp results_dir so we don't touch the real trial dir at all.
    tmp_results_dir = f"/tmp/spot_check_{label.replace('/', '_')}"
    os.makedirs(os.path.join(tmp_results_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(tmp_results_dir, "figures"), exist_ok=True)

    cfg = make_cfg(synth_path, train_npy, full_data_path, dataset_name, tmp_results_dir)
    evaluator = SingleCellEvaluator(config=cfg)
    return evaluator.get_statistical_evals()


def main():
    out_path = "/tmp/spot_check_quality.csv"
    rows = []
    for label, synth_relpath, dataset_name in TRIALS:
        print(f"\n=== {label} ===", flush=True)
        stale = stale_values(label)
        if stale is None:
            print(f"  [SKIP] stale CSV missing"); continue

        try:
            fresh = fresh_values(label, synth_relpath, dataset_name)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        for metric in ("mmd", "lisi", "ari_real_vs_syn"):
            sv, fv = stale.get(metric), fresh.get(metric)
            if sv is None or fv is None:
                continue
            d_abs = fv - sv
            d_rel = d_abs / sv if sv else float("inf")
            print(f"  {metric:>16}  stale={sv:.6e}  fresh={fv:.6e}  Δ={d_abs:+.3e}  Δ%={d_rel*100:+.3f}%")
            rows.append({
                "trial": label, "metric": metric,
                "stale": sv, "fresh": fv,
                "delta_abs": d_abs, "delta_rel": d_rel,
            })

    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nWrote {len(rows)} comparison rows → {out_path}")


if __name__ == "__main__":
    main()
