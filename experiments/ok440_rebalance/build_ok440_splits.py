#!/usr/bin/env python3
"""
Build OneK1K 440d rebalanced splits: 440 train / 440 holdout / 100 aux, all
mutually DISJOINT (440+440+100 = 980 <= 981 donor pool).

Motivation: the manuscript's 490d config draws aux as a subsample of holdout
(strategy_490 -> aux subset of holdout = contamination of the non-member target
side). Shrinking holdout to restore a disjoint aux would leave an *unbalanced*
440 member vs <440 non-member target, which is not apples-to-apples with the
balanced setups used elsewhere in the paper. Rebalancing to 440/440 with a fully
disjoint 100-donor aux keeps the target balanced AND the aux provably disjoint.

Writes to the canonical shared location so generate_trial.py / the 490d-style
driver pick them up unchanged:
    ~/data/scMAMAMIA/ok/splits/440d/{trial}/{train,holdout,auxiliary}.npy

Per-trial deterministic seed -> reproducible. 5 trials to match the 490d protocol.
"""
import os
import numpy as np
import anndata as ad

DATA_ROOT = "/home/golobs/data/scMAMAMIA"
DATASET = "ok"
DONOR_COL = "individual"
N_TRAIN = 440
N_HOLDOUT = 440
N_AUX = 100
N_TRIALS = 5
BASE_SEED = 4402026


def main():
    full = os.path.join(DATA_ROOT, DATASET, "full_dataset_cleaned.h5ad")
    a = ad.read_h5ad(full, backed="r")
    donors = np.array(sorted(a.obs[DONOR_COL].unique()))
    a.file.close()
    n = len(donors)
    print(f"donor pool: {n}")
    assert n >= N_TRAIN + N_HOLDOUT + N_AUX, \
        f"pool {n} < {N_TRAIN + N_HOLDOUT + N_AUX} required"

    for t in range(1, N_TRIALS + 1):
        rng = np.random.default_rng(BASE_SEED + t)
        perm = rng.permutation(donors)
        train = perm[:N_TRAIN]
        holdout = perm[N_TRAIN:N_TRAIN + N_HOLDOUT]
        aux = perm[N_TRAIN + N_HOLDOUT:N_TRAIN + N_HOLDOUT + N_AUX]

        # verify disjointness
        s_tr, s_ho, s_ax = set(train), set(holdout), set(aux)
        assert not (s_tr & s_ho), "train/holdout overlap"
        assert not (s_tr & s_ax), "train/aux overlap"
        assert not (s_ho & s_ax), "holdout/aux overlap"

        out = os.path.join(DATA_ROOT, DATASET, "splits", "440d", str(t))
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "train.npy"), train)
        np.save(os.path.join(out, "holdout.npy"), holdout)
        np.save(os.path.join(out, "auxiliary.npy"), aux)
        print(f"t{t}: train={len(train)} holdout={len(holdout)} aux={len(aux)} "
              f"disjoint=OK -> {out}")


if __name__ == "__main__":
    main()
