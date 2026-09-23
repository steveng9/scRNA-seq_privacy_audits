#!/usr/bin/env python3
"""
run_privbayes_mamamia.py — full PrivBayes-MAMA-MIA attack against scDesign2 synth
(Experiment #4; see docs/privbayes_mamamia_plan.md). Consumes the frames written
by stage_scdesign2_to_mamamia.py and produces a donor-level ROC AUC.

Pipeline (original MAMA-MIA, adapted to inject scDesign2 synth):
  1. PrivBayes learns focal points (conditionals) on the REAL aux reference.
  2. For each conditional (child | parents) we accumulate, per target CELL,
     P_synth(child|parents) / P_aux(child|parents)  — synth = scDesign2 synth.
  3. score_attack aggregates per HHID(=donor) and computes AUC over our
     member(train)/non-member(holdout) split (set membership inference).

ENV: run in `sdg` (has the classic `mbi` + deps). We import ONLY the reprosyn
PrivBayes generator; the MAMA-MIA scorer (custom_privbayes_attack / score_attack /
helpers) is VENDORED below verbatim so we avoid conduct_attacks/util (which drag in
mst/gsd/jax and a global C.n_bins). privBayesSelect.*.so must be compiled once:
    cd reprosyn-main/src/reprosyn/methods/mbi/privBayes && python setup.py build_ext --inplace

    conda run -n sdg python experiments/privbayes_mamamia/run_privbayes_mamamia.py \
        --staged experiments/privbayes_mamamia/_staged/ok_no_dp_50d_t1 \
        --eps 1000 --fp-sample 10000
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_curve, auc

MAMAMIA_REPO = "/home/golobs/SyntheticData_MIA"


# ===========================================================================
# VENDORED from the MAMA-MIA repo (util.py / conduct_attacks.py), verbatim, so
# we do not import conduct_attacks/util (mst, gsd, jax, global C.n_bins).
# ===========================================================================
def activate_3(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    return 1 / (1 + np.exp(-1 * confidence * (zscores - median)))


def membership_advantage(y_true, scores):
    y_pred = scores > .5
    sample_weight = 2 * np.abs(0.5 - scores)
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (tn + fp)
    return ((tpr - fpr) + 1) / 2


def area_under_curve(y_true, predictions):
    try:
        fpr, tpr, _ = roc_curve(y_true, predictions)
        return auc(fpr, tpr)
    except ValueError:
        return None


def tpr_at_fpr(y_true, scores, target_fpr):
    """TPR at the largest operating point with FPR <= target_fpr (Carlini-style)."""
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
    except ValueError:
        return None
    ok = fpr <= target_fpr + 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def determine_weight_threshold(cfg, eps, fp_weights):
    return max(fp_weights.values()) * max(
        [t for e, t in cfg.fp_weight_thresholds.items() if eps >= e])


def score_attack(cfg, A, num_queries_used, targets, target_ids, membership,
                 activation_fn=activate_3):
    predictions = pd.DataFrame({
        'hhid': targets['HHID'].values if cfg.set_MI else targets.index.values,
        'A': pd.Series(A / np.maximum(np.array([1] * targets.shape[0]), num_queries_used)),
    })
    scores = []
    grouped = predictions.groupby('hhid')
    for hhid in target_ids.tolist():
        scores.append(grouped.get_group(hhid).A.mean())
    activated = activation_fn(np.array(scores))
    return predictions, membership_advantage(membership, activated), \
        area_under_curve(membership, activated), (membership, activated)


def custom_privbayes_attack(cfg, eps, aux, synth, targets, target_ids, membership,
                            conditionals_weights):
    """VECTORISED equivalent of the MAMA-MIA per-cell scorer. For each conditional
    (child|parents) we accumulate, per target CELL, P_synth(child|parents) /
    P_aux(child|parents). The original loops in Python with pandas .get per cell
    (O(#cond x #cells) — intractable at 10^5 cells); here we merge the conditional
    tables onto the targets (same math: synth default 1e-10; aux max(.,1e-10))."""
    A = np.zeros(targets.shape[0])
    num_queries_used = np.zeros(targets.shape[0], dtype=int)
    threshold = determine_weight_threshold(cfg, eps, conditionals_weights)
    default_val = 1e-10
    for conditional, weight in conditionals_weights.items():
        if weight < threshold:
            continue
        cl = list(conditional)
        child, parents = cl[0], cl[1:]
        key = parents + [child]

        def _cond(df):
            s = (df.groupby(parents)[child].value_counts(normalize=True) if parents
                 else df[child].value_counts(normalize=True))
            return s.rename("p").reset_index()          # cols: *key, "p"

        ps = _cond(synth).rename(columns={"p": "ps"})
        pa = _cond(aux).rename(columns={"p": "pa"})

        t = targets[key].reset_index(drop=True)
        t["_row"] = np.arange(len(t))
        t = (t.merge(ps, on=key, how="left")
              .merge(pa, on=key, how="left")
              .sort_values("_row"))                      # restore target order
        syn = np.where(np.isnan(t["ps"].to_numpy()), default_val, t["ps"].to_numpy())
        ax = np.where(np.isnan(t["pa"].to_numpy()), default_val, t["pa"].to_numpy())
        ax = np.maximum(ax, default_val)
        A += syn / ax
        num_queries_used += 1
    return score_attack(cfg, A, num_queries_used, targets, target_ids, membership)


class _Cfg:
    """Minimal stand-in for MAMA-MIA's Config (only the fields the vendored
    scorer reads)."""
    def __init__(self, set_MI=True, household_min_size=5):
        self.set_MI = set_MI
        self.household_min_size = household_min_size
        # weight threshold (value) chosen by epsilon (key) — MAMA-MIA default.
        self.fp_weight_thresholds = {.01: .5, 1: .6, 10: .7, 100: .8, 1000: .85}


# ===========================================================================
def _import_privbayes():
    """Import ONLY the reprosyn PrivBayes generator (needs the compiled
    privBayesSelect.so + mbi + in-repo matrix/generator/dataset on the path)."""
    for p in ("reprosyn-main/src/reprosyn/methods/mbi",
              "reprosyn-main/src/reprosyn",
              "reprosyn-main/src"):
        sys.path.insert(0, os.path.join(MAMAMIA_REPO, p))
    import privbayes
    return privbayes


def _load_frames(staged):
    aux = pd.read_parquet(os.path.join(staged, "aux.parquet"))
    targets = pd.read_parquet(os.path.join(staged, "targets.parquet"))
    synth = pd.read_parquet(os.path.join(staged, "synth.parquet"))
    with open(os.path.join(staged, "meta.json")) as f:
        domain = json.load(f)
    memb = pd.read_csv(os.path.join(staged, "membership.csv"), dtype={"HHID": str})
    with open(os.path.join(staged, "manifest.json")) as f:
        manifest = json.load(f)
    return aux, targets, synth, domain, memb, manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staged", required=True, help="dir from stage_scdesign2_to_mamamia.py")
    ap.add_argument("--eps", type=float, default=1000.0,
                    help="PrivBayes eps for FP extraction (default 1000 ~ non-private FPs)")
    ap.add_argument("--fp-sample", type=int, default=10000,
                    help="rows sampled from aux for PrivBayes FP learning")
    ap.add_argument("--household-min-size", type=int, default=5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    privbayes = _import_privbayes()
    aux, targets, synth, domain, memb, manifest = _load_frames(a.staged)
    cols = list(domain.keys())

    for df in (aux, targets, synth):
        df[cols] = df[cols].astype(int).astype(str)     # finite/ordered categories
    aux["HHID"] = aux["HHID"].astype(str)
    targets["HHID"] = targets["HHID"].astype(str)

    meta = [{"name": c, "type": "finite/ordered",
             "representation": [str(i) for i in range(int(domain[c]))]} for c in cols]

    counts = targets["HHID"].value_counts()
    cand = counts[counts >= a.household_min_size].index
    memb_map = dict(zip(memb["HHID"], memb["member"]))
    cand = [d for d in cand if d in memb_map]
    target_ids = pd.Series(cand)
    membership = np.array([int(memb_map[d]) for d in cand])
    targets = targets[targets["HHID"].isin(cand)].copy()
    print(f"targets: {len(cand)} donors "
          f"({int(membership.sum())} member / {len(cand) - int(membership.sum())} non-member), "
          f"{targets.shape[0]} cells; genes={len(cols)}", flush=True)
    if membership.sum() in (0, len(membership)):
        sys.exit("[FATAL] degenerate membership (all one class).")

    cfg = _Cfg(set_MI=True, household_min_size=a.household_min_size)

    # ---- 1. PrivBayes focal points on real aux ----
    n_size = int(min(a.fp_sample, len(aux)))
    print(f"[FP] PrivBayes on {n_size} aux cells, eps={a.eps} ...", flush=True)
    gen = privbayes.PRIVBAYES(dataset=aux[cols].sample(n=n_size),
                              metadata=meta, size=n_size, epsilon=a.eps)
    gen.run()
    conds = []
    for FP in gen.conditionals:
        attrs = np.array(FP).tolist()
        child, parents = attrs[0], sorted(attrs[1:])
        conds.append(tuple([child] + parents))
    fps = dict(Counter(conds))
    print(f"[FP] {len(conds)} conditionals ({len(fps)} unique)", flush=True)

    # ---- 2+3. score targets against scDesign2 synth, aggregate per donor ----
    predictions, MA, AUC, roc = custom_privbayes_attack(
        cfg, a.eps, aux, synth, targets, target_ids, membership, fps)
    y_true, y_score = roc
    tpr10 = tpr_at_fpr(y_true, y_score, 0.10)
    tpr01 = tpr_at_fpr(y_true, y_score, 0.01)
    print(f"\n=== PrivBayes-MAMA-MIA on scDesign2 ===\n"
          f"  donor-level AUC = {AUC:.4f}   membership-advantage = {MA:.4f}\n"
          f"  TPR@10%FPR = {tpr10:.4f}   TPR@1%FPR = {tpr01:.4f}", flush=True)

    out = a.out or os.path.join(a.staged, "results", "privbayes_mamamia.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame([{
        "dataset": manifest["dataset"], "sdg": manifest["sdg"],
        "nd": manifest["nd"], "trial": manifest["trial"],
        "gene_set": manifest["gene_set"], "n_genes": manifest["n_genes"],
        "n_bins": manifest["n_bins"], "fp_eps": a.eps, "fp_sample": n_size,
        "n_conditionals": len(conds), "n_donors": len(cand),
        "n_member": int(membership.sum()), "auc": AUC, "membership_advantage": MA,
        "tpr_at_10fpr": tpr10, "tpr_at_1fpr": tpr01,
    }]).to_csv(out, index=False)
    print(f"[ok] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
