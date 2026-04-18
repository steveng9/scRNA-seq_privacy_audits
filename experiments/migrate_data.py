"""
migrate_data.py — Reorganize ~/data/ into ~/data/scMAMAMIA/ with clean structure.

New layout:
  scMAMAMIA/
    {dataset}/                             # ok, aida, cg
      full_dataset_cleaned.h5ad            # canonical source data
      hvg_full.csv                         # shared HVGs
      scdesign2/
        no_dp/{nd}d/{trial}/               # from ok/{nd}d/{trial}/
        eps_{e}/{nd}d/{trial}/             # from ok_dp/eps_{e}/
      scdesign3/
        gaussian/{nd}d/{trial}/            # from ok_sd3g/
        vine/{nd}d/{trial}/               # from ok_sd3v/
      scvi/
        no_dp/{nd}d/{trial}/              # from ok_scvi/
      scdiffusion/
        no_dp/{nd}d/{trial}/              # from ok_scdiff/
      nmf/
        no_dp/{nd}d/{trial}/              # from ok_nmf/
        eps_{e}/{nd}d/{trial}/            # from ok_nmf_dp/eps_{e}/

Run with --dry-run first to verify, then run for real.
"""

import argparse
import glob
import os
import shutil
import sys

DATA  = "/home/golobs/data"
NEW   = "/home/golobs/data/scMAMAMIA"

MOVES = []   # list of (src, dst) tuples, built below


def add(src, dst):
    MOVES.append((src, dst))


def build_moves():
    """Populate MOVES list — each (src, dst) is a directory or file to mv."""

    # ------------------------------------------------------------------
    # OK1K
    # ------------------------------------------------------------------
    D = f"{DATA}/ok"
    N = f"{NEW}/ok"

    # Dataset-level files
    add(f"{D}/full_dataset_cleaned.h5ad",       f"{N}/full_dataset_cleaned.h5ad")
    add(f"{D}/hvg_full.csv",                    f"{N}/hvg_full.csv")

    # scDesign2 no-DP: all donor-count subdirectories
    for nd_dir in glob.glob(f"{D}/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdesign2/no_dp/{nd}")
    # exp_cfgs alongside donor dirs
    if os.path.isdir(f"{D}/exp_cfgs"):
        add(f"{D}/exp_cfgs", f"{N}/scdesign2/no_dp/exp_cfgs")

    # scDesign2 DP: all eps_* subdirectories
    for eps_dir in glob.glob(f"{DATA}/ok_dp/eps_*"):
        eps = os.path.basename(eps_dir)
        add(eps_dir, f"{N}/scdesign2/{eps}")
    for eps_dir in glob.glob(f"{DATA}/ok_dp/eps_noclip*"):
        eps = os.path.basename(eps_dir)
        add(eps_dir, f"{N}/scdesign2/{eps}")  # already caught above, harmless dedup

    # scDesign3-Gaussian (ok_sd3g) — strip symlinks then move donor dirs
    for nd_dir in glob.glob(f"{DATA}/ok_sd3g/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdesign3/gaussian/{nd}")
    if os.path.isdir(f"{DATA}/ok_sd3g/exp_cfgs"):
        add(f"{DATA}/ok_sd3g/exp_cfgs", f"{N}/scdesign3/gaussian/exp_cfgs")

    # scDesign3-Vine (ok_sd3v)
    for nd_dir in glob.glob(f"{DATA}/ok_sd3v/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdesign3/vine/{nd}")

    # scVI (ok_scvi)
    for nd_dir in glob.glob(f"{DATA}/ok_scvi/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scvi/no_dp/{nd}")

    # scDiffusion (ok_scdiff)
    for nd_dir in glob.glob(f"{DATA}/ok_scdiff/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdiffusion/no_dp/{nd}")

    # NMF no-DP (ok_nmf)
    for nd_dir in glob.glob(f"{DATA}/ok_nmf/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/nmf/no_dp/{nd}")

    # NMF DP sweep (ok_nmf_dp)
    for eps_dir in glob.glob(f"{DATA}/ok_nmf_dp/eps_*"):
        eps = os.path.basename(eps_dir)
        add(eps_dir, f"{N}/nmf/{eps}")

    # ------------------------------------------------------------------
    # AIDA
    # ------------------------------------------------------------------
    D = f"{DATA}/aida"
    N = f"{NEW}/aida"

    add(f"{D}/full_dataset_cleaned.h5ad",       f"{N}/full_dataset_cleaned.h5ad")
    add(f"{D}/hvg_full.csv",                    f"{N}/hvg_full.csv")

    for nd_dir in glob.glob(f"{D}/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdesign2/no_dp/{nd}")
    if os.path.isdir(f"{D}/exp_cfgs"):
        add(f"{D}/exp_cfgs", f"{N}/scdesign2/no_dp/exp_cfgs")

    for nd_dir in glob.glob(f"{DATA}/aida_scvi/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scvi/no_dp/{nd}")

    for nd_dir in glob.glob(f"{DATA}/aida_scdiff/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdiffusion/no_dp/{nd}")

    for nd_dir in glob.glob(f"{DATA}/aida_nmf/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/nmf/no_dp/{nd}")

    # ------------------------------------------------------------------
    # CG (HFRA)
    # ------------------------------------------------------------------
    D = f"{DATA}/cg"
    N = f"{NEW}/cg"

    add(f"{D}/full_dataset_cleaned.h5ad",       f"{N}/full_dataset_cleaned.h5ad")
    # cg has hvg.csv only (no hvg_full.csv)
    if os.path.isfile(f"{D}/hvg_full.csv"):
        add(f"{D}/hvg_full.csv", f"{N}/hvg_full.csv")
    elif os.path.isfile(f"{D}/hvg.csv"):
        add(f"{D}/hvg.csv", f"{N}/hvg_full.csv")

    for nd_dir in glob.glob(f"{D}/[0-9]*d"):
        nd = os.path.basename(nd_dir)
        add(nd_dir, f"{N}/scdesign2/no_dp/{nd}")
    if os.path.isdir(f"{D}/exp_cfgs"):
        add(f"{D}/exp_cfgs", f"{N}/scdesign2/no_dp/exp_cfgs")


def remove_symlinks_in(directory):
    """Delete all symlinks directly inside a directory (not recursive)."""
    if not os.path.isdir(directory):
        return
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.islink(path):
            os.unlink(path)


SYMLINK_DIRS = [
    f"{DATA}/ok_sd3g",
    f"{DATA}/ok_sd3v",
    f"{DATA}/ok_scvi",
    f"{DATA}/ok_scdiff",
    f"{DATA}/ok_nmf",
    f"{DATA}/aida_scvi",
    f"{DATA}/aida_scdiff",
    f"{DATA}/aida_nmf",
    # DP eps dirs each have their own symlinks
    *glob.glob(f"{DATA}/ok_dp/eps_*"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print moves without executing them")
    args = ap.parse_args()
    dry = args.dry_run

    build_moves()

    if dry:
        print(f"DRY RUN — {len(MOVES)} moves:")
        for src, dst in MOVES:
            exists = "✓" if os.path.exists(src) else "✗ MISSING"
            print(f"  {exists}  {src}\n       → {dst}")
        return

    # 1. Remove symlinks from SDG directories
    print("\n[1/3] Removing symlinks…")
    for d in SYMLINK_DIRS:
        remove_symlinks_in(d)
        print(f"  cleaned {d}")

    # 2. Execute moves
    print(f"\n[2/3] Moving {len(MOVES)} items…")
    errors = []
    for src, dst in MOVES:
        if not os.path.exists(src):
            print(f"  [SKIP] not found: {src}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst):
            print(f"  [SKIP] dst exists: {dst}")
            continue
        try:
            shutil.move(src, dst)
            print(f"  moved  {os.path.basename(src)} → {dst}")
        except Exception as e:
            print(f"  [ERROR] {src}: {e}")
            errors.append((src, dst, e))

    # 3. Report
    print(f"\n[3/3] Done. {len(errors)} errors.")
    for src, dst, e in errors:
        print(f"  ERROR: {src} → {dst}: {e}")

    if not errors:
        print("\nMigration complete. Old source dirs can be removed manually once verified.")


if __name__ == "__main__":
    main()
