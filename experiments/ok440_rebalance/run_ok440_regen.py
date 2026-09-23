#!/usr/bin/env python3
"""
run_ok440_regen.py — OneK1K 440/440/100 rebalanced regeneration + re-attack.

WHY
---
The manuscript's OneK1K-490d rows draw the auxiliary set as a 200-donor subsample
of the 490-donor holdout (aux is a subset of the scored target -> contamination
flagged by reviewers kPMB W1 / oT6m / meta-pt.1).  Rather than shrink the holdout
(which would make the 490d AUC no longer apples-to-apples with the paper's other
*balanced* splits), we rebalance to a fully balanced, fully disjoint

    440 train  |  440 holdout  |  100 auxiliary        (all pairwise disjoint)

split (built by build_ok440_splits.py, seed 4402026+trial, verified disjoint).
440+440+100 = 980 <= 981 OneK1K donors.  We then regenerate + re-attack ONLY the
six rows that were actually affected (the +aux / enhanced-+aux columns of
tab:combined_ok_490d and the OneK1K row of tab:scdesign2_all) and relabel
"490d" -> "440d" in the camera-ready.

SCOPE (six affected rows)
-------------------------
  Phase A  (run_full_sweep — its proven MEM_TIERS / OOM-retry / exclusivity guards):
      scdesign2/no_dp   WB+BB quad   <- ALSO writes no_dp/440d/*/models for Phase B
      scdesign3/vine    BB quad
      scvi/no_dp        BB quad
      zinbwave/no_dp    BB quad
  Phase B  (explicit reuse-copula DP gen, then BB-quad attack):
      scdesign2/eps_1000000   (eta = 1e-4)
      scdesign2/eps_10        (eta = 1e1)
    DP synth is made by run_490d_generation.run_sd2_dp (reuse the no_dp copulas +
    apply_gaussian_dp) — the SAME mechanism that produced the 490d DP data being
    replaced.  We deliberately do NOT route DP through the generic path, which does
    not apply the copula noise and would silently write non-DP data.

AUTO-GATING
-----------
Waits until the b2 marginal-noise sweep has finished AND the box has freed
(MemAvailable and 1-min load below thresholds) before starting, so the heavy 440d
regen never contends with b2 or chokes other users (daniilf/sikha).  Detached,
resumable (skips any trial whose results/synth already exist).

USAGE
-----
    # detached, auto-gated:
    setsid bash -c 'conda run --no-capture-output -n tabddpm_ \
        python experiments/ok440_rebalance/run_ok440_regen.py \
        > experiments/ok440_rebalance/_logs/orchestrator.log 2>&1' &

    python experiments/ok440_rebalance/run_ok440_regen.py --status
    python experiments/ok440_rebalance/run_ok440_regen.py --no-wait   # skip the gate
"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SDG_DIR = os.path.join(HERE, "..", "sdg_comparison")
sys.path.insert(0, os.path.abspath(SDG_DIR))

DATA = "/home/golobs/data/scMAMAMIA"
DATASET = "ok"
ND = 440
N_TRIALS = 5
LOGDIR = os.path.join(HERE, "_logs")

# --- auto-gate thresholds (start only when the box is genuinely free) ---
GATE_MIN_FREE_GB = 50            # a 440d job's MEM_TIER needs 45; leave headroom
GATE_MAX_LOAD_FRAC = 0.6         # of nproc (48) -> ~29
GATE_POLL_SEC = 180
GATE_STABLE_CHECKS = 3           # require N consecutive OK polls before starting


# ---------------------------------------------------------------------------
# Path / completion helpers
# ---------------------------------------------------------------------------
def _synth(subpath, trial):
    return os.path.join(DATA, DATASET, *subpath.split("/"), f"{ND}d",
                        str(trial), "datasets", "synthetic.h5ad")


def _no_dp_models(trial):
    return os.path.join(DATA, DATASET, "scdesign2", "no_dp", f"{ND}d",
                        str(trial), "models")


def _splits_ok():
    for t in range(1, N_TRIALS + 1):
        d = os.path.join(DATA, DATASET, "splits", f"{ND}d", str(t))
        if not all(os.path.exists(os.path.join(d, f))
                   for f in ("train.npy", "holdout.npy", "auxiliary.npy")):
            return False, d
    return True, None


# ---------------------------------------------------------------------------
# Auto-gate
# ---------------------------------------------------------------------------
def _b2_running():
    try:
        import subprocess
        out = subprocess.run(["pgrep", "-f", "run_b2_sweep.py"],
                             capture_output=True, text=True)
        return out.returncode == 0 and out.stdout.strip() != ""
    except Exception:
        return False


def _resources_free():
    with open("/proc/meminfo") as f:
        mem = {l.split(":")[0]: int(l.split()[1]) for l in f}
    free_gb = mem.get("MemAvailable", 0) / 1024 / 1024
    load1 = os.getloadavg()[0]
    ncpu = os.cpu_count() or 48
    ok = (free_gb >= GATE_MIN_FREE_GB) and (load1 <= GATE_MAX_LOAD_FRAC * ncpu) \
        and (not _b2_running())
    return ok, free_gb, load1


def wait_for_gate():
    print(f"[{time.strftime('%H:%M:%S')}] auto-gate: waiting for b2 to finish and "
          f"box to free (need >= {GATE_MIN_FREE_GB}GB free, load <= "
          f"{GATE_MAX_LOAD_FRAC:.0%} of {os.cpu_count()} cpus)...", flush=True)
    stable = 0
    while True:
        ok, free_gb, load1 = _resources_free()
        b2 = _b2_running()
        if ok:
            stable += 1
            print(f"[{time.strftime('%H:%M:%S')}] gate OK ({stable}/{GATE_STABLE_CHECKS}) "
                  f"free={free_gb:.0f}GB load={load1:.1f} b2={'yes' if b2 else 'no'}",
                  flush=True)
            if stable >= GATE_STABLE_CHECKS:
                print(f"[{time.strftime('%H:%M:%S')}] gate cleared — starting regen.",
                      flush=True)
                return
        else:
            stable = 0
            print(f"[{time.strftime('%H:%M:%S')}] hold  free={free_gb:.0f}GB "
                  f"load={load1:.1f} b2={'yes' if b2 else 'no'}", flush=True)
        time.sleep(GATE_POLL_SEC)


# ---------------------------------------------------------------------------
# Phase A — non-DP variants via run_full_sweep (all its guards intact)
# ---------------------------------------------------------------------------
def _run_full_sweep(sd2_sweep, other_sweep):
    """Drive run_full_sweep.main() with the module sweep tables overridden to our
    440d scope.  Every job is forced exclusive (EXCLUSIVE_ND=ND) so the ~1M-cell
    440d jobs run strictly one-at-a-time, protecting the shared box."""
    import importlib
    fs = importlib.import_module("run_full_sweep")
    fs.SD2_SWEEP = sd2_sweep
    fs.OTHER_SWEEP = other_sweep
    fs.OTHER_SWEEP_ND = [ND]
    fs.EXCLUSIVE_ND = ND          # 440d jobs are exclusive -> fully serial
    # Reset argv so fs's argparse sees only our log-dir (real run, no filters).
    old_argv = sys.argv
    sys.argv = ["run_full_sweep", "--log-dir", LOGDIR]
    try:
        fs.main()
    finally:
        sys.argv = old_argv


def phase_a():
    print(f"\n{'='*70}\n  PHASE A — non-DP variants (run_full_sweep, 440d, serial)\n"
          f"{'='*70}", flush=True)
    _run_full_sweep(
        sd2_sweep=[("ok", "scdesign2/no_dp", [ND])],
        other_sweep=[
            ("ok", "scdesign3/vine"),
            ("ok", "scvi/no_dp"),
            ("ok", "zinbwave/no_dp"),
        ],
    )


# ---------------------------------------------------------------------------
# Phase B — DP variants: reuse-copula gen (run_sd2_dp) then BB-quad attack
# ---------------------------------------------------------------------------
# eta = 100 / eps  ->  eps_1000000 == eta 1e-4 ; eps_10 == eta 1e1
DP_EPS = [1000000, 10]


def phase_b_generate():
    print(f"\n{'='*70}\n  PHASE B.1 — DP synth (reuse no_dp copulas + apply_gaussian_dp)\n"
          f"{'='*70}", flush=True)
    import importlib
    gen490 = importlib.import_module("run_490d_generation")
    gen490.ND = ND                # monkeypatch: resolved at call time in gen490 fns
    for eps in DP_EPS:
        subpath = f"scdesign2/eps_{eps}"
        for t in range(1, N_TRIALS + 1):
            if os.path.exists(_synth(subpath, t)):
                print(f"  [skip] {subpath} 440d/{t} — synth present", flush=True)
                continue
            if not os.path.isdir(_no_dp_models(t)):
                print(f"  [WARN] {subpath} 440d/{t} — no_dp models missing "
                      f"({_no_dp_models(t)}); Phase A must finish first. Skipping.",
                      flush=True)
                continue
            print(f"  [gen ] {subpath} 440d/{t} (eps={eps}) ...", flush=True)
            try:
                gen490.run_sd2_dp(subpath, t, epsilon=eps)
            except Exception as e:
                print(f"  [FAIL] {subpath} 440d/{t}: {e}", flush=True)


def phase_b_attack():
    print(f"\n{'='*70}\n  PHASE B.2 — attack DP synth (BB quad; synth already present)\n"
          f"{'='*70}", flush=True)
    # SAFETY: the eps_* path has gen_info=None, so run_experiment would inline-generate
    # any MISSING trial as NON-DP data (no copula noise) and mislabel it eps_*. Only
    # queue an eps variant whose synth is fully present for all trials, so no inline
    # (non-DP) generation can ever be triggered.
    other = []
    for eps in DP_EPS:
        subpath = f"scdesign2/eps_{eps}"
        n_synth = sum(1 for t in range(1, N_TRIALS + 1) if os.path.exists(_synth(subpath, t)))
        if n_synth == N_TRIALS:
            other.append(("ok", subpath))
        else:
            print(f"  [skip attack] {subpath} — only {n_synth}/{N_TRIALS} DP synth present; "
                  f"refusing to risk inline non-DP generation.", flush=True)
    if not other:
        print("  [PHASE B.2] no fully-generated DP variant to attack — skipping.", flush=True)
        return
    # run_full_sweep sees the eps synth on disk -> generation is skipped, attack only.
    _run_full_sweep(sd2_sweep=[], other_sweep=other)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------
def status():
    rows = [
        ("scdesign2/no_dp", "WB+BB"),
        ("scdesign3/vine", "BB"),
        ("scvi/no_dp", "BB"),
        ("zinbwave/no_dp", "BB"),
        ("scdesign2/eps_1000000", "BB"),
        ("scdesign2/eps_10", "BB"),
    ]
    print(f"\n  OneK1K 440d rebalance regen — status ({DATASET}, {ND}d, {N_TRIALS} trials)")
    ok, missing = _splits_ok()
    print(f"  splits/{ND}d: {'all present' if ok else 'MISSING ' + str(missing)}")
    print(f"  {'variant':<26} {'synth':>7}  {'no_dp models':>13}")
    print(f"  {'-'*26} {'-'*7}  {'-'*13}")
    for subpath, _ in rows:
        ns = sum(1 for t in range(1, N_TRIALS + 1) if os.path.exists(_synth(subpath, t)))
        nm = (sum(1 for t in range(1, N_TRIALS + 1) if os.path.isdir(_no_dp_models(t)))
              if subpath == "scdesign2/no_dp" else "-")
        print(f"  {subpath:<26} {ns:>3}/{N_TRIALS}    {str(nm):>13}")
    print()


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="print status and exit")
    ap.add_argument("--no-wait", action="store_true",
                    help="skip the auto-gate (start immediately)")
    ap.add_argument("--phase", choices=["a", "b", "all"], default="all",
                    help="run only phase a / b (default: all)")
    a = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)

    if a.status:
        status()
        return

    ok, missing = _splits_ok()
    if not ok:
        print(f"[FATAL] 440d splits missing: {missing}\n"
              f"        run build_ok440_splits.py first.", flush=True)
        sys.exit(1)

    print(f"[{time.strftime('%H:%M:%S')}] ok440 regen starting "
          f"(phase={a.phase}, wait={not a.no_wait})", flush=True)
    if not a.no_wait:
        wait_for_gate()

    if a.phase in ("a", "all"):
        phase_a()
    if a.phase in ("b", "all"):
        phase_b_generate()
        phase_b_attack()

    print(f"\n[{time.strftime('%H:%M:%S')}] ok440 regen orchestrator complete.", flush=True)
    status()


if __name__ == "__main__":
    main()
