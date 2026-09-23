#!/usr/bin/env python3
"""
B1 (disjoint-cells / equal-cell-budget) — STEP 1: build cell-level partitions.

Reuses the EXISTING donor split at:
    {ROOT}/{dataset}/splits/{nd}d/{trial}/{train,holdout,auxiliary}.npy

and produces, for a chosen (G, B):
  - gen_train      : G cells per MEMBER donor  -> the generator trains ONLY on these
  - tgt_disjoint   : B cells per MEMBER donor, DISJOINT from gen_train (attacker sees these)
  - tgt_overlap    : the SAME G training cells per MEMBER (overlap control)
  - tgt_nonmember  : B cells per each NON-MEMBER (holdout) donor  (equal budget)
  - aux donors     : passed through unchanged (disjoint reference)

Every target donor (member via tgt_disjoint, non-member via tgt_nonmember) contributes
exactly B cells -> equal cell budget. Members' attack cells are disjoint from the cells
that trained the generator. Donors with < G+B (members) or < B (non-members) cells are
DROPPED and reported.

Output: a manifest .npz of cell-id arrays + a printed validation report. No X is loaded
here (obs-only, fast); materialisation of h5ads is done by the orchestrator (step 2).
"""
import os, argparse, sys
import numpy as np
import anndata as ad

ROOT = "/home/golobs/data/scMAMAMIA"


def build(dataset, nd, trial, G, B, seed=None, verbose=True):
    base = os.path.join(ROOT, dataset)
    split_dir = os.path.join(base, "splits", f"{nd}d", str(trial))
    train_donors   = set(np.load(os.path.join(split_dir, "train.npy"),     allow_pickle=True).tolist())
    holdout_donors = set(np.load(os.path.join(split_dir, "holdout.npy"),   allow_pickle=True).tolist())
    aux_donors     = np.load(os.path.join(split_dir, "auxiliary.npy"), allow_pickle=True)

    # deterministic per (dataset, nd, trial, G, B)
    if seed is None:
        seed = abs(hash((dataset, nd, str(trial), G, B))) % (2**32)
    rng = np.random.default_rng(seed)

    full = ad.read_h5ad(os.path.join(base, "full_dataset_cleaned.h5ad"), backed="r")
    obs = full.obs
    donor_col = "individual"
    cell_ids = np.asarray(full.obs_names)
    donors_of_cell = obs[donor_col].to_numpy()
    full.file.close()

    # map donor -> array of cell ids
    from collections import defaultdict
    by_donor = defaultdict(list)
    for cid, d in zip(cell_ids, donors_of_cell):
        by_donor[d].append(cid)

    gen_train, tgt_disjoint, tgt_overlap, tgt_nonmember = [], [], [], []
    dropped_members, dropped_nonmembers = [], []

    for d in train_donors:
        cells = np.array(by_donor.get(d, []))
        if len(cells) < G + B:
            dropped_members.append((d, len(cells)))
            continue
        perm = rng.permutation(len(cells))
        g_idx = perm[:G]
        b_idx = perm[G:G + B]
        gen_train.extend(cells[g_idx])
        tgt_disjoint.extend(cells[b_idx])
        tgt_overlap.extend(cells[g_idx])   # control: attack the training cells themselves

    for d in holdout_donors:
        cells = np.array(by_donor.get(d, []))
        if len(cells) < B:
            dropped_nonmembers.append((d, len(cells)))
            continue
        perm = rng.permutation(len(cells))
        tgt_nonmember.extend(cells[perm[:B]])

    manifest = dict(
        gen_train=np.array(gen_train), tgt_disjoint=np.array(tgt_disjoint),
        tgt_overlap=np.array(tgt_overlap), tgt_nonmember=np.array(tgt_nonmember),
        aux_donors=np.array(aux_donors),
        train_donors=np.array(sorted(train_donors)),
        holdout_donors=np.array(sorted(holdout_donors)),
        G=G, B=B, seed=seed, dataset=dataset, nd=nd, trial=trial,
    )

    n_mem = len(train_donors) - len(dropped_members)
    n_non = len(holdout_donors) - len(dropped_nonmembers)

    # ---- validation ----
    gset, dset = set(gen_train), set(tgt_disjoint)
    overlap_leak = len(gset & dset)
    ok = True
    if verbose:
        print(f"[{dataset} {nd}d t{trial}] G={G} B={B} seed={seed}")
        print(f"  members kept:     {n_mem}/{len(train_donors)}  (dropped {len(dropped_members)} with <G+B={G+B} cells)")
        print(f"  non-members kept: {n_non}/{len(holdout_donors)}  (dropped {len(dropped_nonmembers)} with <B={B} cells)")
        print(f"  gen_train cells:  {len(gen_train)}  (expect {n_mem*G})")
        print(f"  tgt_disjoint:     {len(tgt_disjoint)} (expect {n_mem*B})   tgt_overlap: {len(tgt_overlap)}")
        print(f"  tgt_nonmember:    {len(tgt_nonmember)} (expect {n_non*B})")
        print(f"  DISJOINTNESS gen_train ∩ tgt_disjoint = {overlap_leak}  {'OK' if overlap_leak==0 else 'FAIL!!'}")
        # per-donor budget check
        budgets_ok = (len(tgt_disjoint) == n_mem*B) and (len(tgt_nonmember) == n_non*B)
        print(f"  EQUAL-BUDGET (all target donors = B cells): {'OK' if budgets_ok else 'MISMATCH'}")
        if dropped_members[:3]:
            print(f"  e.g. dropped members: {dropped_members[:3]}")
    ok = (overlap_leak == 0) and n_mem >= 2 and n_non >= 2
    return manifest, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ok")
    ap.add_argument("--nd", type=int, default=10)
    ap.add_argument("--trial", default="1")
    ap.add_argument("--G", type=int, default=200)
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--out", default=None, help="optional .npz path to save manifest")
    a = ap.parse_args()
    manifest, ok = build(a.dataset, a.nd, a.trial, a.G, a.B)
    if a.out:
        np.savez(a.out, **manifest)
        print(f"  manifest -> {a.out}")
    sys.exit(0 if ok else 1)
