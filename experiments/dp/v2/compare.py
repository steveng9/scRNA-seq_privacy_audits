"""
compare.py — side-by-side v1 vs v2 spot-check table.

Reads quality + MIA CSVs for each (eps, trial) pair and prints a markdown
table comparing v1 (eps_*) and v2 (v2_eps_* / v2_no_dp).

Output goes to stdout AND to experiments/dp/v2/results/spot_check_{dataset}_{nd}d.md
"""

import argparse
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_ROOT = "/home/golobs/data/scMAMAMIA"
OUT_DIR   = os.path.join(REPO_ROOT, "experiments", "dp", "v2", "results")


def _quality(trial_dir):
    p = os.path.join(trial_dir, "results", "quality_eval_results", "results", "statistics_evals.csv")
    if not os.path.exists(p):
        return {"lisi": np.nan, "ari": np.nan, "mmd": np.nan}
    try:
        df = pd.read_csv(p)
        # statistics_evals.csv format: one-row CSV with named columns
        row = df.iloc[0]
        return {
            "lisi": float(row.get("lisi", np.nan)),
            "ari":  float(row.get("ari_real_vs_syn", np.nan)),
            "mmd":  float(row.get("mmd", np.nan)),
        }
    except Exception:
        return {"lisi": np.nan, "ari": np.nan, "mmd": np.nan}


def _mia(trial_dir, classb=False):
    fname = "mamamia_results_classb.csv" if classb else "mamamia_results.csv"
    p = os.path.join(trial_dir, "results", fname)
    out = {"bb_aux": np.nan, "bb_noaux": np.nan}
    if not os.path.exists(p):
        return out
    try:
        df = pd.read_csv(p)
        auc = df[df["metric"] == "auc"]
        if auc.empty:
            return out
        if "tm:100" in auc.columns:
            v = auc["tm:100"].values[0]
            out["bb_aux"] = float(v) if pd.notna(v) else np.nan
        if "tm:101" in auc.columns:
            v = auc["tm:101"].values[0]
            out["bb_noaux"] = float(v) if pd.notna(v) else np.nan
    except Exception:
        pass
    return out


def _gather(dataset, nd, variant_pattern):
    """Return dict[variant][trial] = {quality, mia, mia_classb}."""
    base = os.path.join(DATA_ROOT, dataset, "scdesign2")
    out = defaultdict(dict)
    for variant in sorted(os.listdir(base)):
        if not re.match(variant_pattern, variant):
            continue
        trial_pattern = os.path.join(base, variant, f"{nd}d", "*")
        for trial_dir in sorted(glob.glob(trial_pattern)):
            if not os.path.isdir(trial_dir):
                continue
            try:
                trial = int(os.path.basename(trial_dir))
            except ValueError:
                continue
            q  = _quality(trial_dir)
            m  = _mia(trial_dir, classb=False)
            mb = _mia(trial_dir, classb=True)
            out[variant][trial] = {"q": q, "mia": m, "mia_b": mb}
    return out


def _eps_value(variant):
    """Map 'eps_100' / 'v2_eps_100' / 'no_dp' / 'v2_no_dp' to a sortable epsilon."""
    if variant.endswith("no_dp"):
        return float("inf")
    m = re.search(r"eps_(\d+)", variant)
    if m:
        return float(m.group(1))
    return float("nan")


def _agg(per_trial_metrics):
    """Mean + std across trials for a single variant + epsilon."""
    if not per_trial_metrics:
        return None
    vals = np.array([v for v in per_trial_metrics if not np.isnan(v)])
    if len(vals) == 0:
        return (np.nan, np.nan, 0)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals))


def _fmt(agg):
    if agg is None:
        return "—"
    m, s, n = agg
    if np.isnan(m):
        return "—"
    return f"{m:.3f} ± {s:.3f}  (n={n})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd",      type=int, default=20)
    args = ap.parse_args()

    v1 = _gather(args.dataset, args.nd, r"^(no_dp|eps_\d+)$")
    v2 = _gather(args.dataset, args.nd, r"^(v2_no_dp|v2_eps_\d+)$")

    # Find paired epsilons present in both v2 and v1
    v1_eps = {_eps_value(v): v for v in v1.keys()}
    v2_eps = {_eps_value(v): v for v in v2.keys()}
    paired = sorted(set(v1_eps) & set(v2_eps))

    if not paired:
        print("[compare] no overlapping (epsilon, variant) pairs found.")
        return

    rows = []
    for eps in paired:
        v1_name = v1_eps[eps]
        v2_name = v2_eps[eps]
        for label, key in [("LISI", "lisi"), ("ARI", "ari"), ("MMD", "mmd")]:
            v1_vals = [t["q"][key] for t in v1[v1_name].values()]
            v2_vals = [t["q"][key] for t in v2[v2_name].values()]
            rows.append({
                "epsilon": "∞ (no DP)" if eps == float("inf") else f"{int(eps):,}",
                "metric":  label,
                "v1":      _fmt(_agg(v1_vals)),
                "v2":      _fmt(_agg(v2_vals)),
            })
        for label, src_key in [("BB+aux (std)", ("mia", "bb_aux")),
                                ("BB-aux (std)", ("mia", "bb_noaux")),
                                ("BB+aux (Bcl)", ("mia_b", "bb_aux")),
                                ("BB-aux (Bcl)", ("mia_b", "bb_noaux"))]:
            cat, col = src_key
            v1_vals = [t[cat][col] for t in v1[v1_name].values()]
            v2_vals = [t[cat][col] for t in v2[v2_name].values()]
            rows.append({
                "epsilon": "∞ (no DP)" if eps == float("inf") else f"{int(eps):,}",
                "metric":  label,
                "v1":      _fmt(_agg(v1_vals)),
                "v2":      _fmt(_agg(v2_vals)),
            })

    df = pd.DataFrame(rows)

    def _to_md(df):
        cols = list(df.columns)
        widths = [max(len(c), df[c].astype(str).map(len).max()) for c in cols]
        header = "| " + " | ".join(c.ljust(w) for c, w in zip(cols, widths)) + " |"
        sep    = "|"  + "|".join("-" * (w + 2) for w in widths) + "|"
        body   = "\n".join("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |"
                           for row in df.itertuples(index=False))
        return "\n".join([header, sep, body])

    md_lines = [
        f"# v1 vs v2 spot check — {args.dataset} {args.nd}d",
        "",
        _to_md(df),
        "",
        "Notes:",
        "- v2 σ is 4× smaller than v1 σ at the same nominal ε.",
        "- v2 trains on the uncentered second moment (no column-centering); v1 on the centered correlation.",
        "- Lower MMD is better; higher LISI/ARI generally better. Lower MIA AUC means stronger empirical privacy.",
    ]
    txt = "\n".join(md_lines)
    print(txt)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"spot_check_{args.dataset}_{args.nd}d.md")
    with open(out_path, "w") as f:
        f.write(txt + "\n")
    print(f"\nWrote: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
