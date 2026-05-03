#!/usr/bin/env python3
"""
print_unified_status.py — Single-pane completion grid across all sweeps.

Walks DATA_DIR to discover every (base_dataset, sdg, variant, nd) combination
that has at least one trial directory, and reports per-trial completion counts
for the five tracked artifacts:

    synth     — datasets/synthetic.h5ad
    MIA       — mamamia_results.csv (BB = tm:100/tm:101, WB = tm:000/tm:001)
    Class-B   — mamamia_results_classb.csv (same tm codes)
    Baselines — results/baseline_mias/baselines_evaluation_results.csv
    Quality   — results/quality_eval_results/results/statistics_evals.csv
                (split into fresh vs stale by the MMD-fix mtime cutoff)

WB columns are only meaningful for scDesign2; shown as "—" elsewhere.

Usage:
    python experiments/sdg_comparison/print_unified_status.py
    python experiments/sdg_comparison/print_unified_status.py --dataset ok
    python experiments/sdg_comparison/print_unified_status.py --sdg scdesign2
    python experiments/sdg_comparison/print_unified_status.py --nd 50
"""

import argparse
import os
import re

import pandas as pd

DATA_DIR = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5

# Quality CSVs written before this timestamp used the broken MMD path.
# Same cutoff used by run_quality_evals.py (commit d9ae732, 2026-03-25 19:51 UTC).
MMD_FIX_TS = 1774417865

_ND_RE = re.compile(r"^(\d+)d$")
_SKIP_TOP = {"splits", "aux_artifacts"}


# ---------------------------------------------------------------------------
# Per-artifact trial counters
# ---------------------------------------------------------------------------

def _auc_present(csv_path, tm_code):
    if not os.path.exists(csv_path):
        return False
    try:
        df = pd.read_csv(csv_path)
        auc_row = df[df["metric"] == "auc"]
        if auc_row.empty:
            return False
        col = f"tm:{tm_code}"
        return col in auc_row.columns and pd.notna(auc_row[col].values[0])
    except Exception:
        return False


def count_quad(data_dir, nd, mode, results_filename):
    """Trials where the given results CSV has both AUCs for the requested quad mode."""
    tm_aux, tm_noaux = ("000", "001") if mode == "wb_quad" else ("100", "101")
    n = 0
    for t in range(1, N_TRIALS + 1):
        f = os.path.join(data_dir, f"{nd}d", str(t), "results", results_filename)
        if _auc_present(f, tm_aux) and _auc_present(f, tm_noaux):
            n += 1
    return n


def n_synth(data_dir, nd):
    return sum(
        1 for t in range(1, N_TRIALS + 1)
        if os.path.exists(os.path.join(
            data_dir, f"{nd}d", str(t), "datasets", "synthetic.h5ad"
        ))
    )


def n_baselines(data_dir, nd):
    return sum(
        1 for t in range(1, N_TRIALS + 1)
        if os.path.exists(os.path.join(
            data_dir, f"{nd}d", str(t), "results", "baseline_mias",
            "baselines_evaluation_results.csv",
        ))
    )


def n_quality(data_dir, nd):
    """Return (fresh, stale) trial counts based on MMD-fix mtime cutoff."""
    fresh = stale = 0
    for t in range(1, N_TRIALS + 1):
        f = os.path.join(
            data_dir, f"{nd}d", str(t), "results",
            "quality_eval_results", "results", "statistics_evals.csv",
        )
        if not os.path.exists(f):
            continue
        try:
            mt = os.path.getmtime(f)
        except OSError:
            continue
        if mt >= MMD_FIX_TS:
            fresh += 1
        else:
            stale += 1
    return fresh, stale


def n_umaps(data_dir, nd):
    """Count trials that have a paper_umap.png in their umaps/ subdir."""
    return sum(
        1 for t in range(1, N_TRIALS + 1)
        if os.path.exists(os.path.join(data_dir, f"{nd}d", str(t), "umaps", "paper_umap.png"))
    )


# ---------------------------------------------------------------------------
# Disk walker
# ---------------------------------------------------------------------------

def discover_combos():
    """Return sorted list of (base_dataset, sdg_path, nd) tuples found on disk."""
    combos = []
    if not os.path.isdir(DATA_DIR):
        return combos
    for base in sorted(os.listdir(DATA_DIR)):
        base_dir = os.path.join(DATA_DIR, base)
        if not os.path.isdir(base_dir):
            continue
        # Treat anything with a splits/ subdir as a real base dataset.
        if not os.path.isdir(os.path.join(base_dir, "splits")):
            continue
        for sdg in sorted(os.listdir(base_dir)):
            if sdg in _SKIP_TOP:
                continue
            sdg_dir = os.path.join(base_dir, sdg)
            if not os.path.isdir(sdg_dir):
                continue
            for variant in sorted(os.listdir(sdg_dir)):
                vdir = os.path.join(sdg_dir, variant)
                if not os.path.isdir(vdir):
                    continue
                for entry in sorted(os.listdir(vdir)):
                    m = _ND_RE.match(entry)
                    if not m or not os.path.isdir(os.path.join(vdir, entry)):
                        continue
                    combos.append((base, f"{sdg}/{variant}", int(m.group(1))))
    combos.sort(key=lambda t: (t[0], t[1], t[2]))
    return combos


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(n):
    if n >= N_TRIALS:
        return "✓"
    if n > 0:
        return f"~{n}"
    return "·"


def fmt_quality(fresh, stale, synth):
    if synth == 0 and fresh == 0 and stale == 0:
        return "·"
    if fresh >= max(synth, N_TRIALS) and stale == 0:
        return "✓"
    parts = [f"~{fresh}" if fresh > 0 else "·"]
    if stale > 0:
        parts.append(f"!{stale}")
    return "/".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset", default=None,
                    help="Filter: only base datasets containing this substring (e.g. 'ok', 'aida')")
    ap.add_argument("--sdg", default=None,
                    help="Filter: only sdg/variant paths containing this substring "
                         "(e.g. 'scdesign2', 'nmf/eps_', 'no_dp')")
    ap.add_argument("--nd", type=int, default=None,
                    help="Filter: only this donor count")
    ap.add_argument("--hide-empty", action="store_true",
                    help="Hide rows where every count is zero")
    args = ap.parse_args()

    combos = discover_combos()
    rows = []
    for base, sdg_path, nd in combos:
        if args.dataset and args.dataset not in base:
            continue
        if args.sdg and args.sdg not in sdg_path:
            continue
        if args.nd is not None and args.nd != nd:
            continue

        ddir = os.path.join(DATA_DIR, base, *sdg_path.split("/"))
        ns = n_synth(ddir, nd)
        is_sd2 = sdg_path.startswith("scdesign2/")
        mia_bb = count_quad(ddir, nd, "bb_quad", "mamamia_results.csv")
        mia_wb = count_quad(ddir, nd, "wb_quad", "mamamia_results.csv") if is_sd2 else None
        cb_bb  = count_quad(ddir, nd, "bb_quad", "mamamia_results_classb.csv")
        cb_wb  = count_quad(ddir, nd, "wb_quad", "mamamia_results_classb.csv") if is_sd2 else None
        nb     = n_baselines(ddir, nd)
        qf, qs = n_quality(ddir, nd)
        nu     = n_umaps(ddir, nd)

        if args.hide_empty and not any([ns, mia_bb, mia_wb or 0, cb_bb,
                                        cb_wb or 0, nb, qf, qs, nu]):
            continue
        rows.append((f"{base}/{sdg_path}", nd, ns, mia_wb, cb_wb, mia_bb, cb_bb, nb, qf, qs, nu))

    hdr = (
        f"  {'Dataset / SDG':<40} {'nd':>4}  {'synth':>5}  "
        f"{'MIA WB':>6} {'CB WB':>5}  {'MIA BB':>6} {'CB BB':>5}  "
        f"{'Base':>4}  {'Quality':>9}  {'UMAPs':>5}"
    )
    sep = (
        f"  {'-'*40} {'-'*4}  {'-'*5}  "
        f"{'-'*6} {'-'*5}  {'-'*6} {'-'*5}  "
        f"{'-'*4}  {'-'*9}  {'-'*5}"
    )
    print()
    print("=" * len(hdr))
    print(f"  Unified Sweep Completion (counts out of {N_TRIALS} trials)")
    print("=" * len(hdr))
    print()
    print(hdr)
    print(sep)

    last_label = None
    for label, nd, ns, mia_wb, cb_wb, mia_bb, cb_bb, nb, qf, qs, nu in rows:
        if last_label is not None and label != last_label:
            print()
        last_label = label
        wb_str    = " " if mia_wb is None else fmt(mia_wb)
        cb_wb_str = " " if cb_wb is None else fmt(cb_wb)
        umap_str  = "·" if nu == 0 else ("✓" if nu >= N_TRIALS else str(nu))
        print(
            f"  {label:<40} {nd:>3}d  {fmt(ns):>5}  "
            f"{wb_str:>6}  {cb_wb_str:>5} {fmt(mia_bb):>6}  {fmt(cb_bb):>5}  "
            f"{fmt(nb):>4}  {fmt_quality(qf, qs, ns):>9}  {umap_str:>5}"
        )

    print()
    print("  Legend:")
    print(f"    ✓ = all {N_TRIALS} trials done    ~N = N done    · = none    "
          f"' ' = N/A (WB only meaningful for scDesign2)")
    print("    Quality column: '~F' (fresh) plus '!S' (stale, pre-MMD-fix; "
          "mtime < 2026-03-25 19:51 UTC, commit d9ae732)")
    print(f"    Total combos shown: {len(rows)}")
    print()


if __name__ == "__main__":
    main()
