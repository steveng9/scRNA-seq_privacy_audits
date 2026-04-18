"""
Display what synthetic data exists across all datasets and generators,
plus a live view of currently running SDG jobs and their progress.

Usage:
    python experiments/sdg_comparison/check_generated_data.py
"""

import os
import re
import subprocess

DATA    = "/home/golobs/data/scMAMAMIA"
LOG_DIR = "/tmp/sdg_comparison_logs"

# --- Dataset / generator definitions ---
# (display_name, data_root, donor_counts, is_dp, dp_epsilons)

GENERATORS = [
    # scDesign2 (original)
    ("ok  / scDesign2",   f"{DATA}/ok/scdesign2/no_dp",   [2, 5, 10, 20, 50, 100, 200], False, None),
    ("cg  / scDesign2",   f"{DATA}/cg/scdesign2/no_dp",   [2, 5, 10, 20],               False, None),
    ("aida/ scDesign2",   f"{DATA}/aida/scdesign2/no_dp", [5, 10, 20, 50, 100, 200],    False, None),

    # scDesign2 + DP (ok only)
    ("ok  / scDesign2+DP", f"{DATA}/ok/scdesign2",        [10, 20, 50],                 True,
     ["eps_0.1", "eps_0.5", "eps_1", "eps_2", "eps_5", "eps_10",
      "eps_100", "eps_1000", "eps_10000",
      "eps_100000", "eps_1000000", "eps_10000000", "eps_100000000", "eps_1000000000"]),

    # scDesign3 Gaussian
    ("ok  / scDesign3-Gauss", f"{DATA}/ok/scdesign3/gaussian",   [2, 5, 10, 20, 50, 100, 200], False, None),
    ("aida/ scDesign3-Gauss", f"{DATA}/aida/scdesign3/gaussian", [10, 20, 50, 100],             False, None),

    # scDesign3 Vine
    ("ok  / scDesign3-Vine",  f"{DATA}/ok/scdesign3/vine",   [10, 20, 50, 100],  False, None),
    ("aida/ scDesign3-Vine",  f"{DATA}/aida/scdesign3/vine", [10, 20, 50],        False, None),

    # scVI
    ("ok  / scVI",   f"{DATA}/ok/scvi/no_dp",   [5, 10, 20, 50, 100], False, None),
    ("aida/ scVI",   f"{DATA}/aida/scvi/no_dp", [10, 20, 50],          False, None),

    # scDiffusion
    ("ok  / scDiffusion",   f"{DATA}/ok/scdiffusion/no_dp",   [10, 20, 50], False, None),
    ("aida/ scDiffusion",   f"{DATA}/aida/scdiffusion/no_dp", [20, 50],      False, None),

    # NMF (SingleCellNMFGenerator — CAMDA 2024 co-winner)
    ("ok  / NMF",    f"{DATA}/ok/nmf/no_dp",   [10, 20, 50, 100], False, None),
    ("aida/ NMF",    f"{DATA}/aida/nmf/no_dp", [10, 20, 50],       False, None),

    # NMF + DP (ok only)
    ("ok  / NMF+DP", f"{DATA}/ok/nmf",         [50],               True,
     ["eps_1", "eps_10", "eps_100", "eps_1000", "eps_10000",
      "eps_100000", "eps_1000000", "eps_10000000", "eps_100000000"]),
]

N_TRIALS = 5


# ---------------------------------------------------------------------------
# Inventory helpers
# ---------------------------------------------------------------------------

def count_trials(root, nd, trial_range=range(1, N_TRIALS + 1)):
    """Count how many trials have synthetic.h5ad for a given donor count."""
    done = 0
    for t in trial_range:
        synth = os.path.join(root, f"{nd}d", str(t), "datasets", "synthetic.h5ad")
        if os.path.exists(synth):
            done += 1
    return done


def count_trials_dp(root, eps, nd, trial_range=range(1, N_TRIALS + 1)):
    """Count trials for a DP epsilon / donor count."""
    done = 0
    for t in trial_range:
        synth = os.path.join(root, eps, f"{nd}d", str(t), "datasets", "synthetic.h5ad")
        if os.path.exists(synth):
            done += 1
    return done


def bar(done, total, width=20):
    filled = int(width * done / total) if total > 0 else 0
    return f"[{'#' * filled}{'.' * (width - filled)}] {done}/{total}"


# ---------------------------------------------------------------------------
# Running-job detection helpers
# ---------------------------------------------------------------------------

def _get_running_jobs():
    """
    Parse `ps aux` for generate_trial.py processes and return a list of dicts:
      { label, generator, out_dir, log_path }
    Deduplicates by out_dir (each trial spawns multiple shell/python processes).
    """
    try:
        out = subprocess.check_output(
            ["ps", "aux"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []

    seen = set()
    jobs = []
    for line in out.splitlines():
        if "generate_trial.py" not in line or "grep" in line:
            continue
        # Only capture the top-level python invocation (not sh -c wrappers)
        if "python" not in line.split("generate_trial.py")[0].split()[-1:]:
            # Keep lines where the executable before generate_trial.py is python
            if not re.search(r"\bpython\b.*generate_trial\.py", line):
                continue

        gen_m    = re.search(r"--generator\s+(\S+)", line)
        out_m    = re.search(r"--out-dir\s+(\S+)", line)
        if not gen_m or not out_m:
            continue

        generator = gen_m.group(1)
        out_dir   = out_m.group(1)
        if "$" in out_dir:          # unexpanded shell variable — skip
            continue
        if out_dir in seen:
            continue
        seen.add(out_dir)

        # Derive a log label from out_dir, e.g. /home/golobs/data/ok_scdiff/10d/5
        # → ok_scdiff_10d_t5
        parts = out_dir.rstrip("/").split("/")
        try:
            dataset  = parts[-3]   # e.g. ok_scdiff
            nd_str   = parts[-2]   # e.g. 10d
            trial    = parts[-1]   # e.g. 5
            label    = f"{dataset}_{nd_str}_t{trial}"
        except IndexError:
            label = out_dir

        log_path = os.path.join(LOG_DIR, f"{label}.log")
        jobs.append(dict(label=label, generator=generator,
                         out_dir=out_dir, log_path=log_path))

    return sorted(jobs, key=lambda j: j["label"])


def _tail_log(log_path, n=200):
    """Return last n lines of the most recent run in a log file.

    Each time a job is (re)launched, run_all.py appends a new 'CMD:' block.
    We only want lines from the last block so stale EXIT codes don't confuse
    progress parsing.
    """
    if not os.path.exists(log_path):
        return []
    with open(log_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        # Read up to 64 KB from the end — enough for most jobs
        chunk = min(size, 65536)
        f.seek(-chunk, 2)
        raw = f.read()
    all_lines = raw.decode("utf-8", errors="replace").splitlines()
    # Find the last CMD: marker and return only lines after it
    last_cmd = max(
        (i for i, l in enumerate(all_lines) if l.startswith("CMD:")),
        default=-1,
    )
    lines = all_lines[last_cmd + 1:] if last_cmd >= 0 else all_lines
    return lines[-n:]


def _progress_scdiffusion(lines, out_dir):
    """
    Return a short progress string for a running scDiffusion job.
    Looks for VAE / diffusion step counters and phase indicators.
    """
    vae_dir  = os.path.join(out_dir, "models", "vae")
    diff_sub = os.path.join(out_dir, "models", "diff", "diffusion")

    # Determine phase
    generating  = any("Generating" in l or "generate" in l.lower() for l in lines[-20:])
    diff_started = os.path.isdir(diff_sub) and any(
        f.endswith(".pt") for f in os.listdir(diff_sub) if os.path.isdir(diff_sub)
    ) if os.path.isdir(diff_sub) else False
    vae_done = any("[SKIP] VAE" in l for l in lines)

    # Parse last reported step from table output: "| step       | 3e+03    |"
    step_val = None
    for line in reversed(lines):
        m = re.search(r"\|\s*step\s*\|\s*([\d.e+]+)\s*\|", line)
        if m:
            try:
                step_val = int(float(m.group(1)))
            except ValueError:
                pass
            break

    if generating:
        return "phase=generating"
    if step_val is not None:
        if vae_done:
            return f"phase=diffusion  step={step_val:,}/300,000  ({100*step_val//300000}%)"
        else:
            return f"phase=VAE        step={step_val:,}/150,000  ({100*step_val//150000}%)"
    if vae_done:
        return "phase=diffusion  (starting…)"
    return "phase=VAE  (starting…)"


def _progress_sd3(lines, out_dir):
    """
    Return a short progress string for a running scDesign3 job.
    Counts completed cell types from 'Finished training cell type:' lines.
    """
    # Count completed cell types in this run
    n_done = sum(1 for l in lines if "Finished training cell type:" in l)
    training_started = any("Launching parallel scDesign3" in l for l in lines)
    if not training_started:
        return "preparing…"
    if n_done == 0:
        return "training cell types… (0 done)"
    # Try to infer total from last-seen cell type count
    # (scDesign3 prints counts, but we don't have total easily — just show done)
    return f"training cell types… ({n_done} cell types done)"


def _progress_scvi(lines):
    """Return a short progress string for a running scVI job."""
    # scVI prints a rich progress table; look for epoch lines
    # Typical: "Epoch 12/400: 100%|..." or "Training: 100%|..."
    for line in reversed(lines):
        if re.search(r"Epoch\s+\d+/\d+", line):
            m = re.search(r"Epoch\s+(\d+)/(\d+)", line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                return f"epoch={cur}/{total}  ({100*cur//total}%)"
        if "training" in line.lower() and "%" in line:
            return line.strip()[-80:]
    return "training…"


def _summarize_job(job):
    """Return a one-line progress summary for a running job."""
    lines    = _tail_log(job["log_path"])
    gen      = job["generator"]

    if not lines:
        return "(no log yet)"

    # Check for recent crash
    last_exit = next((l for l in reversed(lines) if "[EXIT" in l), None)
    if last_exit and "[EXIT 0]" not in last_exit:
        return f"FAILED {last_exit.strip()}"

    if gen == "scdiffusion":
        return _progress_scdiffusion(lines, job["out_dir"])
    elif gen in ("sd3_gaussian", "sd3_vine"):
        return _progress_sd3(lines, job["out_dir"])
    elif gen == "scvi":
        return _progress_scvi(lines)
    elif gen == "nmf":
        # NMF prints phase tags: Fitting, KMeans, Sampling, Assigning, Saved
        for line in reversed(lines):
            if "[NMF]" in line:
                return line.strip()[-100:]
        return lines[-1].strip()[-100:] if lines else "(empty log)"
    else:
        return lines[-1].strip()[-100:] if lines else "(empty log)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'=' * 72}")
    print(f"  Synthetic Data Inventory")
    print(f"{'=' * 72}\n")

    for display, root, donor_counts, is_dp, epsilons in GENERATORS:
        if not os.path.isdir(root):
            print(f"  {display:<28}  [directory missing: {root}]")
            print()
            continue

        if is_dp:
            existing_eps = sorted(
                e for e in (epsilons or []) if os.path.isdir(os.path.join(root, e))
            )
            if not existing_eps:
                print(f"  {display:<28}  [no epsilon dirs found]")
                print()
                continue

            print(f"  {display}")
            for eps in existing_eps:
                eps_label = eps.replace("eps_", "ε=")
                parts = []
                total_done = total_all = 0
                for nd in donor_counts:
                    done = count_trials_dp(root, eps, nd)
                    total_done += done
                    total_all  += N_TRIALS
                    sym = "✓" if done == N_TRIALS else ("~" if done > 0 else "·")
                    parts.append(f"{nd}d:{sym}{done}")
                summary = "  ".join(parts)
                overall = bar(total_done, total_all)
                print(f"    {eps_label:<12}  {overall}   {summary}")
            print()

        else:
            total_done = total_all = 0
            parts = []
            for nd in donor_counts:
                done = count_trials(root, nd)
                total_done += done
                total_all  += N_TRIALS
                sym = "✓" if done == N_TRIALS else ("~" if done > 0 else "·")
                parts.append(f"{nd}d:{sym}{done}")
            summary = "  ".join(parts)
            overall = bar(total_done, total_all)
            print(f"  {display:<28}  {overall}")
            print(f"  {'':28}  {summary}")
            print()

    print(f"  Legend:  ✓ = all {N_TRIALS} trials done   ~ = partial   · = none")
    print(f"{'=' * 72}\n")

    # ------------------------------------------------------------------
    # Running jobs
    # ------------------------------------------------------------------
    jobs = _get_running_jobs()
    if not jobs:
        print("  No SDG generation jobs currently running.\n")
        return

    print(f"{'=' * 72}")
    print(f"  Running Jobs  ({len(jobs)} active)")
    print(f"{'=' * 72}\n")

    for job in jobs:
        progress = _summarize_job(job)
        print(f"  {job['label']:<28}  [{job['generator']}]")
        print(f"  {'':28}  {progress}")
        print()

    print(f"  (Logs: {LOG_DIR}/)")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    main()
