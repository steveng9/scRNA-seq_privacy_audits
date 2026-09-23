#!/usr/bin/env python3
"""
Aux-disjointness fix — STEP 1: build a donor split whose auxiliary set is
provably disjoint from the target set (train ∪ holdout).

Motivation (rebuttal, meta pt.1 / kPMB W1 / oT6m): a handful of *presented*
+aux values were computed on splits where the donor pool was exhausted and
D_aux overlapped the target set (AIDA-200d: 92/200; OneK1K-490d: aux⊂holdout).
We re-draw a clean, fully-disjoint auxiliary set and re-run only those configs.

Strategy (per config, decided with the user):
  * complement       — keep train & holdout as-is; aux = ALL remaining donors
                       (used when the leftover pool is adequate, e.g. AIDA-200d
                       → 508-400 = 108 disjoint aux donors).
  * shrink_holdout   — keep all `train` members; shrink holdout to `--holdout`
                       donors (subset of the original holdout) and draw a
                       disjoint aux of `--aux-size` from the freed holdout + any
                       leftover pool (used for OneK1K-490d if pool-exhausted).

The train members are ALWAYS preserved unchanged, so the on-disk synthetic data
(which depends only on the training donors) remains valid and is reused — the
re-run is a cheap aux-refit + re-score, no regeneration.

Writes {ds}/aux_disjoint/splits/{nd}d/{trial}/{train,holdout,auxiliary}.npy and
verifies disjointness before writing.
"""
import os, argparse
import numpy as np
import anndata as ad

DATA = "/home/golobs/data/scMAMAMIA"


def _all_donors(ds):
    f = os.path.join(DATA, ds, "full_dataset_cleaned.h5ad")
    a = ad.read_h5ad(f, backed="r")
    d = np.unique(a.obs["individual"].astype(str).values)
    a.file.close()
    return d


def build(ds, nd, trial, mode, holdout_size=None, aux_size=None, seed=0):
    src = os.path.join(DATA, ds, "splits", f"{nd}d", str(trial))
    train = np.load(os.path.join(src, "train.npy"), allow_pickle=True).astype(str)
    holdout = np.load(os.path.join(src, "holdout.npy"), allow_pickle=True).astype(str)
    all_d = _all_donors(ds)
    rng = np.random.default_rng(seed + int(trial) * 100003 + nd)

    train_set = set(train.tolist())
    hold_set = set(holdout.tolist())

    if mode == "complement":
        new_train = train
        new_holdout = holdout
        pool = sorted(set(all_d.tolist()) - train_set - hold_set)
        aux = np.array(pool, dtype=object)
        if aux_size is not None:
            aux = np.array(sorted(rng.permutation(pool)[:aux_size].tolist()), dtype=object)

    elif mode == "shrink_holdout":
        assert holdout_size is not None and aux_size is not None
        new_train = train
        # keep the first `holdout_size` holdout donors as non-members
        keep_hold = sorted(holdout.tolist())[:holdout_size]
        new_holdout = np.array(keep_hold, dtype=object)
        freed = sorted(set(holdout.tolist()) - set(keep_hold))          # dropped holdout donors
        leftover = sorted(set(all_d.tolist()) - train_set - hold_set)   # never-used donors
        aux_pool = freed + leftover
        assert len(aux_pool) >= aux_size, f"aux_pool {len(aux_pool)} < aux_size {aux_size}"
        aux = np.array(sorted(rng.permutation(aux_pool)[:aux_size].tolist()), dtype=object)
    else:
        raise ValueError(mode)

    # --- verify disjointness ---
    tset, hset, aset = set(new_train.tolist()), set(new_holdout.tolist()), set(aux.tolist())
    assert aset.isdisjoint(tset), "aux overlaps train!"
    assert aset.isdisjoint(hset), "aux overlaps holdout!"

    out = os.path.join(DATA, ds, "aux_disjoint", "splits", f"{nd}d", str(trial))
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "train.npy"), new_train)
    np.save(os.path.join(out, "holdout.npy"), new_holdout)
    np.save(os.path.join(out, "auxiliary.npy"), aux)
    print(f"[{ds} {nd}d t{trial} {mode}] train={len(new_train)} holdout={len(new_holdout)} "
          f"aux={len(aux)} (disjoint ✓) -> {out}", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nd", type=int, required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--mode", choices=["complement", "shrink_holdout"], required=True)
    ap.add_argument("--holdout", type=int, default=None)
    ap.add_argument("--aux-size", type=int, default=None)
    a = ap.parse_args()
    build(a.dataset, a.nd, a.trial, a.mode, a.holdout, a.aux_size)
