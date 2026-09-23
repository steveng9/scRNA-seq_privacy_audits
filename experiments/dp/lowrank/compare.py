"""
compare.py — lowrank (rank x epsilon) vs. v2 (full-rank) spot-check table.

Reads quality + MIA CSVs for each (rank, eps, trial) lowrank variant and the
matching v2 (full-rank) variant, prints a markdown table, and writes it to
experiments/dp/lowrank/results/spot_check_{dataset}_{nd}d.md
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
OUT_DIR   = os.path.join(REPO_ROOT, "experiments", "dp", "lowrank", "results")

LOWRANK_RE = re.compile(r"^lowrank_r(\d+)_(no_dp|eps_(\d+))$")
V2_RE      = re.compile(r"^(v2_no_dp|v2_eps_(\d+))$")


def _quality(trial_dir):
    p = os.path.join(trial_dir, "results", "quality_eval_results", "results", "statistics_evals.csv")
    if not os.path.exists(p):
        return {"lisi": np.nan, "ari": np.nan, "mmd": np.nan}
    try:
        df = pd.read_csv(p)
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


def _gather(dataset, nd, regex):
    base = os.path.join(DATA_ROOT, dataset, "scdesign2")
    out = defaultdict(dict)
    if not os.path.isdir(base):
        return out
    for variant in sorted(os.listdir(base)):
        if not regex.match(variant):
            continue
        for trial_dir in sorted(glob.glob(os.path.join(base, variant, f"{nd}d", "*"))):
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
    if variant.endswith("no_dp"):
        return float("inf")
    m = re.search(r"eps_(\d+)$", variant)
    if m:
        return float(m.group(1))
    return float("nan")


def _rank_value(variant):
    m = re.match(r"^lowrank_r(\d+)_", variant)
    return int(m.group(1)) if m else None


def _agg(vals):
    vals = np.array([v for v in vals if not np.isnan(v)])
    if len(vals) == 0:
        return (np.nan, np.nan, 0)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals))


def _fmt(agg):
    m, s, n = agg
    if np.isnan(m):
        return "—"
    return f"{m:.3f} ± {s:.3f} (n={n})"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd",      type=int, default=20)
    args = ap.parse_args()

    lowrank = _gather(args.dataset, args.nd, LOWRANK_RE)
    v2      = _gather(args.dataset, args.nd, V2_RE)

    v2_eps = {_eps_value(v): v for v in v2.keys()}

    # Build rows keyed by (rank, epsilon)
    lr_keys = []
    for variant in lowrank.keys():
        rank = _rank_value(variant)
        eps = _eps_value(variant)
        lr_keys.append((rank, eps, variant))
    lr_keys.sort(key=lambda t: (t[0], t[1]))

    if not lr_keys:
        print("[compare] no lowrank trial dirs found.")
        return

    rows = []
    for rank, eps, variant in lr_keys:
        eps_label = "∞ (no DP)" if eps == float("inf") else f"{int(eps):,}"
        v2_name = v2_eps.get(eps)
        for label, key in [("LISI", "lisi"), ("ARI", "ari"), ("MMD", "mmd")]:
            lr_vals = [t["q"][key] for t in lowrank[variant].values()]
            v2_vals = [t["q"][key] for t in v2[v2_name].values()] if v2_name else []
            rows.append({
                "rank": rank, "epsilon": eps_label, "metric": label,
                "lowrank": _fmt(_agg(lr_vals)),
                "v2 (full-rank)": _fmt(_agg(v2_vals)) if v2_name else "—",
            })
        for label, src_key in [("BB+aux (std)", ("mia", "bb_aux")),
                                ("BB-aux (std)", ("mia", "bb_noaux")),
                                ("BB+aux (Bcl)", ("mia_b", "bb_aux")),
                                ("BB-aux (Bcl)", ("mia_b", "bb_noaux"))]:
            cat, col = src_key
            lr_vals = [t[cat][col] for t in lowrank[variant].values()]
            v2_vals = [t[cat][col] for t in v2[v2_name].values()] if v2_name else []
            rows.append({
                "rank": rank, "epsilon": eps_label, "metric": label,
                "lowrank": _fmt(_agg(lr_vals)),
                "v2 (full-rank)": _fmt(_agg(v2_vals)) if v2_name else "—",
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
        f"# lowrank vs v2 spot check — {args.dataset} {args.nd}d",
        "",
        _to_md(df),
        "",
        "Notes:",
        "- lowrank sigma is calibrated to n_genes=rank (not full G) -- the actual sensitivity win.",
        "- 'rank' rows with epsilon=∞ (no DP) isolate the cost of rank truncation alone (no noise).",
        "- v2 (full-rank) column reused/regenerated with the corrected TRUE_CLIP_VALUE "
        "(see notes/DP_clip_value_bug.txt) so this comparison is apples-to-apples.",
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
