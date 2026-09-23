"""
run_evals.py — quality + MIA evaluation for lowrank spot-check outputs only.

Walks  ~/data/scMAMAMIA/{dataset}/scdesign2/lowrank_r*_no_dp/  and
       ~/data/scMAMAMIA/{dataset}/scdesign2/lowrank_r*_eps_*/   and for each
(nd, trial) trial dir that has datasets/synthetic.h5ad:

  Quality   — invokes evaluation.sc_evaluate.SingleCellEvaluator, writes
              {trial}/results/quality_eval_results/results/statistics_evals.csv
  MIA       — writes a YAML config and invokes src/run_experiment.py with
              run_quad_bb=True (BB+aux + BB-aux, standard + Class-B in one
              pass). Results land at {trial}/results/mamamia_results.csv
              (and ..._classb.csv).

Isolated from v1/v2 orchestration (run_quality_evals.py, run_mia_sweep.py,
experiments/dp/v2/run_evals.py are not touched), per the established
isolation contract.
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import traceback

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
DATA_ROOT = "/home/golobs/data/scMAMAMIA"
RUNNER    = os.path.join(SRC_DIR, "run_experiment.py")
CFG_DIR   = os.path.join(REPO_ROOT, "experiments", "dp", "lowrank", "_cfgs")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

CELL_TYPE_COL = {"ok": "cell_type", "aida": "cell_type", "cg": "cell_type"}
MIN_AUX_DONORS = 10
MAMAMIA_PARAMS = {
    "IMPORTANCE_OF_CLASS_B_FPs":   0.17,
    "epsilon":                     0.0001,
    "mahalanobis":                 True,
    "uniform_remapping_fn":        "zinb_cdf",
    "lin_alg_inverse_fn":          "pinv_gpu",
    "closeness_to_correlation_fn": "closeness_to_correlation_1",
    "class_b_gene_set":            "secondary",
    "class_b_scoring":             "llr",
    "class_b_gamma":               "auto",
    "class_b_gamma_noaux":         "auto",
}

VARIANT_RE = re.compile(r"^lowrank_r\d+_(no_dp|eps_\d+)$")


def discover_trials(dataset, only_nds=None, only_trials=None):
    base = os.path.join(DATA_ROOT, dataset, "scdesign2")
    out = []
    if not os.path.isdir(base):
        return out
    for variant in sorted(os.listdir(base)):
        if not VARIANT_RE.match(variant):
            continue
        for synth in sorted(glob.glob(os.path.join(base, variant, "*d", "*", "datasets", "synthetic.h5ad"))):
            parts = synth.split(os.sep)
            nd_tag = parts[-4]
            trial  = parts[-3]
            try:
                nd = int(nd_tag.rstrip("d"))
            except ValueError:
                continue
            if only_nds and nd not in only_nds:
                continue
            if only_trials and int(trial) not in only_trials:
                continue
            trial_dir = os.path.dirname(os.path.dirname(synth))
            out.append((variant, nd, int(trial), trial_dir))
    return out


def run_quality_one(dataset, variant, nd, trial, trial_dir, force=False):
    full_data = os.path.join(DATA_ROOT, dataset, "full_dataset_cleaned.h5ad")
    train_npy = os.path.join(DATA_ROOT, dataset, "splits", f"{nd}d", str(trial), "train.npy")
    if not os.path.exists(train_npy):
        print(f"[lowrank/qual] skip {variant}/{nd}d/t{trial} — no splits train.npy")
        return None

    synth_path = os.path.join(trial_dir, "datasets", "synthetic.h5ad")
    results_dir = os.path.join(trial_dir, "results", "quality_eval_results")
    out_csv     = os.path.join(results_dir, "results", "statistics_evals.csv")

    if os.path.exists(out_csv) and not force:
        print(f"[lowrank/qual] skip {variant}/{nd}d/t{trial} — CSV exists")
        return out_csv

    os.makedirs(os.path.join(results_dir, "results"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)

    cfg = {
        "dir_list":        {"home": results_dir, "figures": "figures", "res_files": "results"},
        "full_data_path":  full_data,
        "synthetic_file":  synth_path,
        "dataset_config":  {
            "name":               dataset,
            "test_count_file":    train_npy,
            "synthetic_file":     synth_path,
            "cell_type_col_name": CELL_TYPE_COL[dataset],
            "cell_label_col_name": "cell_label",
            "celltypist_model":   "",
        },
        "evaluator_config": {"random_seed": 1},
        "n_hvgs":           1000,
    }

    try:
        from evaluation.sc_evaluate import SingleCellEvaluator
        ev = SingleCellEvaluator(config=cfg)
        results = ev.get_statistical_evals()
        ev.save_results_to_csv(results, out_csv)
        print(f"[lowrank/qual] {variant}/{nd}d/t{trial}  "
              f"lisi={results.get('lisi'):.4f} "
              f"ari={results.get('ari_real_vs_syn'):.4f} "
              f"mmd={results.get('mmd'):.6f}", flush=True)
        return out_csv
    except Exception as e:
        print(f"[lowrank/qual] FAIL {variant}/{nd}d/t{trial}: {e}")
        traceback.print_exc()
        return None


def write_mia_config(dataset, variant, nd):
    cfg_dir = os.path.join(CFG_DIR, f"{dataset}_{variant}")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{nd}d_bb_quad.yaml")

    dataset_name = f"{dataset}/scdesign2/{variant}"
    hvg_path = os.path.join(DATA_ROOT, dataset, "hvg.csv")
    cfg = {
        "dir_list": {
            "local":  {"home": REPO_ROOT, "data": DATA_ROOT},
            "server": {"home": REPO_ROOT, "data": DATA_ROOT},
        },
        "dataset_name":     dataset_name,
        "hvg_path":         hvg_path,
        "generator_name":   "scdesign2",
        "plot_results":     False,
        "parallelize":      True,
        "parallel_workers": 2,
        "min_aux_donors":   MIN_AUX_DONORS,
        "mamamia_params":   dict(MAMAMIA_PARAMS),
        "mia_setting": {
            "sample_donors_strategy_fn": "sample_donors_strategy_2",
            "num_donors":   nd,
            "white_box":    False,
            "use_wb_hvgs":  True,
            "use_aux":      True,
            "run_quad_bb":  True,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path


def mia_already_done(trial_dir):
    import pandas as pd
    for fname in ("mamamia_results.csv", "mamamia_results_classb.csv"):
        p = os.path.join(trial_dir, "results", fname)
        if not os.path.exists(p):
            return False
        try:
            df = pd.read_csv(p)
            row = df[df["metric"] == "auc"]
            if row.empty:
                return False
            for col in ("tm:100", "tm:101"):
                if col not in row.columns or pd.isna(row[col].values[0]):
                    return False
        except Exception:
            return False
    return True


def run_mia_one(dataset, variant, nd, trial, trial_dir, force=False, dry_run=False):
    if mia_already_done(trial_dir) and not force:
        print(f"[lowrank/mia]  skip {variant}/{nd}d/t{trial} — already complete")
        return True
    cfg_path = write_mia_config(dataset, variant, nd)
    cmd = [sys.executable, RUNNER, cfg_path]
    if dry_run:
        print(f"[lowrank/mia]  DRY-RUN  {variant}/{nd}d/t{trial}: {' '.join(cmd)}")
        return True
    print(f"[lowrank/mia]  RUN  {variant}/{nd}d/t{trial}", flush=True)
    rc = subprocess.run(cmd, cwd=REPO_ROOT).returncode
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd",      type=int, nargs="+", default=None)
    ap.add_argument("--trial",   type=int, nargs="+", default=None)
    ap.add_argument("--quality-only", action="store_true")
    ap.add_argument("--mia-only",     action="store_true")
    ap.add_argument("--force",        action="store_true")
    ap.add_argument("--dry-run",      action="store_true")
    args = ap.parse_args()

    trials = discover_trials(args.dataset, args.nd, args.trial)
    if not trials:
        print("[lowrank/eval] no lowrank trial dirs found.")
        return

    print(f"[lowrank/eval] {len(trials)} lowrank trial dirs found.")
    for variant, nd, trial, trial_dir in trials:
        print(f"  {variant} / {nd}d / t{trial}")

    if not args.mia_only:
        print("\n=== Quality evaluations ===")
        for variant, nd, trial, trial_dir in trials:
            run_quality_one(args.dataset, variant, nd, trial, trial_dir, force=args.force)

    if not args.quality_only:
        print("\n=== MIA attacks ===")
        for variant, nd, trial, trial_dir in trials:
            run_mia_one(args.dataset, variant, nd, trial, trial_dir,
                        force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
