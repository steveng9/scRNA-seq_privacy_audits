"""
Pre-build donor splits for the HLCA representative sweep and VERIFY that the
auxiliary set is fully disjoint from the target (train ∪ holdout) set.

Sizes 10/20/35 with the standard strategy_2 rule (aux size = nd). Disjointness
is guaranteed because 3*nd <= 107 (the HLCA-core donor pool) for nd <= 35 — at
nd=40 the pool is exhausted and strategy_2 would pad aux with target donors, so
we cap at 35 to keep every "+aux" split clean (mirrors the paper's W1 fix).

Pre-building also removes the sd2/scVI race: generate_trial (scVI) needs splits
to already exist, and both generators then share these on-disk splits.

Output: scMAMAMIA/hlca/splits/{nd}d/{trial}/{train,holdout,auxiliary}.npy
Run:
    conda run --no-capture-output -n tabddpm_ \
        python experiments/hlca_onboard/build_hlca_splits.py
"""
import os
import numpy as np
import anndata as ad

BASE = "/home/golobs/data/scMAMAMIA/hlca"
ND = [10, 20, 35]
TRIALS = [1, 2, 3]
MIN_AUX = 10
SEED0 = 20260731


def main():
    a = ad.read_h5ad(os.path.join(BASE, "full_dataset_cleaned.h5ad"), backed="r")
    donors = np.array(a.obs["individual"].unique())
    a.file.close()
    print(f"donor pool: {len(donors)}")

    all_ok = True
    for nd in ND:
        for t in TRIALS:
            rng = np.random.RandomState(SEED0 + nd * 100 + t)
            n_used = min(nd, len(donors) // 2)
            target = rng.choice(donors, size=n_used * 2, replace=False)
            train, holdout = target[:n_used], target[n_used:]
            non_target = np.array(list(set(donors) - set(target)))
            num_aux = max(MIN_AUX, n_used)
            aux = np.concatenate(
                (rng.permutation(non_target), rng.permutation(target))
            )[:num_aux]

            overlap = len(set(aux) & set(target))
            all_ok &= (overlap == 0)

            d = os.path.join(BASE, "splits", f"{nd}d", str(t))
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "train.npy"), train)
            np.save(os.path.join(d, "holdout.npy"), holdout)
            np.save(os.path.join(d, "auxiliary.npy"), aux)
            flag = "OK" if overlap == 0 else "*** OVERLAP ***"
            print(f"nd={nd:>2} t={t}: train={len(train)} holdout={len(holdout)} "
                  f"aux={len(aux)} aux∩target={overlap}  {flag}")

    print("\nALL SPLITS DISJOINT" if all_ok else "\n!!! SOME SPLITS CONTAMINATED !!!")


if __name__ == "__main__":
    main()
