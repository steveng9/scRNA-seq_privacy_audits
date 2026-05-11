"""
Generate three LaTeX tables: OneK1K dataset, 50 donors, averaged over 5 trials.

Tables produced:
  1. combined_table_ok_50d.tex  — all SDG methods (no-DP + SD2/NMF DP sweep rows)
  2. dp_table_ok_50d.tex        — DP sweep only (scDesign2 + NMF, all epsilon variants)
  3. nodp_table_ok_50d.tex      — no-DP methods only (all generators, one row each)

Columns (same in all three tables):
  [SDG Method] [DP Param]  |  Fidelity: LISI  ARI  MMD×10³  |
  Privacy: AUC BB⁻  AUC BB⁺  AUC ClassB BB⁻  AUC ClassB BB⁺

Formatting:
  - Values rounded to 2 decimal places, leading zero removed (.87 not 0.87)
  - ±std in \tiny (visibly smaller than the \scriptsize table body)

Usage:
  python experiments/sdg_comparison/make_combined_table_ok_50d.py
"""

import os
import math
import sys
import numpy as np
import pandas as pd

DATA     = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5
ND       = "50d"

# Threat model columns
TM_BB_AUX   = "tm:100"   # black-box + aux
TM_BB_NOAUX = "tm:101"   # black-box, no aux


# ──────────────────────────────────────────────
# η label helpers
# ──────────────────────────────────────────────

def eta_label(eps_val: float) -> str:
    """LaTeX for η = 10²/ε (power-of-ten form when exact)."""
    eta = 100.0 / eps_val
    log = math.log10(eta)
    exp = round(log)
    if abs(10 ** exp - eta) < 1e-6:
        if exp == 0:
            return r"$\eta=1$"
        return rf"$\eta=10^{{{exp}}}$"
    return rf"$\eta={eta:.3g}$"


def eps_label(eps_str: str) -> str:
    """LaTeX for a raw epsilon value (used for NMF)."""
    try:
        val = float(eps_str)
        if val == int(val):
            val = int(val)
            exp = round(math.log10(val))
            if 10 ** exp == val:
                return rf"$\varepsilon=10^{{{exp}}}$"
        return rf"$\varepsilon={eps_str}$"
    except Exception:
        return rf"$\varepsilon={eps_str}$"


# ──────────────────────────────────────────────
# Row catalog
# ──────────────────────────────────────────────

SD2_EPS = [
    (1_000_000_000, "eps_1000000000"),
    (100_000_000,   "eps_100000000"),
    (10_000_000,    "eps_10000000"),
    (1_000_000,     "eps_1000000"),
    (100_000,       "eps_100000"),
    (10_000,        "eps_10000"),
    (1_000,         "eps_1000"),
    (100,           "eps_100"),
    (10,            "eps_10"),
    (1,             "eps_1"),
]

NMF_EPS = [
    ("100000000", "eps_100000000"),
    ("10000000",  "eps_10000000"),
    ("1000000",   "eps_1000000"),
    ("100000",    "eps_100000"),
    ("10000",     "eps_10000"),
    ("1000",      "eps_1000"),
    ("100",       "eps_100"),
    ("10",        "eps_10"),
    ("2.8",       "eps_2.8"),
    ("1",         "eps_1"),
]


def sd2_dp_rows():
    rows = []
    for eps_int, eps_dir in SD2_EPS:
        rows.append({
            "method": "scDesign2",
            "param":  eta_label(eps_int),
            "path":   f"ok/scdesign2/{eps_dir}",
        })
    return rows


def nmf_dp_rows():
    rows = []
    for eps_str, eps_dir in NMF_EPS:
        label = eps_label(eps_str)
        if eps_str == "2.8":
            label += r" {\tiny(default)}"
        rows.append({
            "method": "NMF",
            "param":  label,
            "path":   f"ok/nmf/{eps_dir}",
        })
    return rows


# All rows for the combined table (order: SD2, SD3, scVI, scDiff, ZINBWave, NMF)
ALL_ROWS = [
    # ── scDesign2 ──────────────────────────────────
    {"method": "scDesign2",   "param": r"(no DP)",             "path": "ok/scdesign2/no_dp"},
    *sd2_dp_rows(),
    # ── scDesign3 ──────────────────────────────────
    {"method": "scDesign3-G", "param": r"(no DP)",             "path": "ok/scdesign3/gaussian"},
    {"method": "scDesign3-V", "param": r"(no DP)",             "path": "ok/scdesign3/vine"},
    # ── scVI ───────────────────────────────────────
    {"method": "scVI",        "param": r"(no DP)",             "path": "ok/scvi/no_dp"},
    # ── scDiffusion ────────────────────────────────
    {"method": "scDiffusion", "param": r"(no DP)",             "path": "ok/scdiffusion_v3/v1_celltypist"},
    # ── ZINBWave ───────────────────────────────────
    {"method": "ZINBWave",    "param": r"(no DP)",             "path": "ok/zinbwave/no_dp"},
    # ── NMF ────────────────────────────────────────
    {"method": "NMF",         "param": r"(no DP)",             "path": "ok/nmf/no_dp"},
    {"method": "NMF",         "param": r"$\varepsilon=2.8$ {\tiny(default)}", "path": "ok/nmf/eps_2.8"},
]

DP_ROWS = [
    {"method": "scDesign2", "param": r"(no DP)", "path": "ok/scdesign2/no_dp"},
    *sd2_dp_rows(),
    {"method": "NMF",       "param": r"(no DP)", "path": "ok/nmf/no_dp"},
    *nmf_dp_rows(),
]

NODP_ROWS = [
    {"method": "scDesign2",   "param": r"—", "path": "ok/scdesign2/no_dp"},
    {"method": "scDesign3-G", "param": r"—", "path": "ok/scdesign3/gaussian"},
    {"method": "scDesign3-V", "param": r"—", "path": "ok/scdesign3/vine"},
    {"method": "scVI",        "param": r"—", "path": "ok/scvi/no_dp"},
    {"method": "scDiffusion", "param": r"—", "path": "ok/scdiffusion_v3/v1_celltypist"},
    {"method": "ZINBWave",    "param": r"—", "path": "ok/zinbwave/no_dp"},
    {"method": "NMF",         "param": r"—", "path": "ok/nmf/no_dp"},
]


# ──────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────

def collect_quality(path):
    lisi_v, ari_v, mmd_v = [], [], []
    for trial in range(1, N_TRIALS + 1):
        csv = os.path.join(DATA, path, ND, str(trial),
                           "results", "quality_eval_results", "results", "statistics_evals.csv")
        if not os.path.exists(csv):
            continue
        try:
            row = pd.read_csv(csv).iloc[0]
            lisi_v.append(float(row["lisi"]))
            ari_v.append(float(row["ari_real_vs_syn"]) if pd.notna(row.get("ari_real_vs_syn")) else float("nan"))
            mmd_v.append(float(row["mmd"]))
        except Exception:
            continue
    return lisi_v, ari_v, mmd_v


def collect_auc(path, tm_col, use_classb):
    fname = "mamamia_results_classb.csv" if use_classb else "mamamia_results.csv"
    vals = []
    for trial in range(1, N_TRIALS + 1):
        csv = os.path.join(DATA, path, ND, str(trial), "results", fname)
        if not os.path.exists(csv):
            continue
        try:
            df  = pd.read_csv(csv)
            row = df[df["metric"] == "auc"]
            if row.empty or tm_col not in row.columns:
                continue
            v = row[tm_col].iloc[0]
            if pd.notna(v):
                vals.append(float(v))
        except Exception:
            continue
    return vals


# ──────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────

def strip_lead(s: str) -> str:
    """Remove leading zero before decimal point: '0.87' → '.87', '45.69' → '45.69'."""
    if s.startswith("0."):
        return s[1:]   # → '.87'
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def fmt(vals, scale=1.0, missing="---"):
    """Format mean ± std.  ±std rendered in \\tiny; leading zeros stripped."""
    clean = [v * scale for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return missing
    mean = np.mean(clean)
    std  = np.std(clean, ddof=1) if len(clean) > 1 else 0.0
    n    = len(clean)

    mean_str = strip_lead(f"{mean:.2f}")
    std_str  = strip_lead(f"{std:.2f}")

    cell = mean_str + r"{\tiny$\,\pm\,$" + std_str + "}"
    if n < N_TRIALS:
        cell += rf"$^{{{n}}}$"
    return cell


# ──────────────────────────────────────────────
# Table builder
# ──────────────────────────────────────────────

HEADER_PART1 = (
    r"\multicolumn{2}{c|}{} & "
    r"\multicolumn{3}{c|}{\textbf{Fidelity}} & "
    r"\multicolumn{4}{c}{\textbf{Privacy (MIA AUC)}} \\"
)

HEADER_PART2 = (
    r"\textbf{SDG Method} & \textbf{DP Param} & "
    r"LISI$\uparrow$ & ARI$\uparrow$ & MMD${\times}10^3\downarrow$ & "
    r"BB$^{-}$ & BB$^{+}$ & "
    r"\textsc{CB} BB$^{-}$ & \textsc{CB} BB$^{+}$ \\"
)


def build_table(rows, caption, label):
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{ll|ccc|cccc}")
    lines.append(r"\toprule")
    lines.append(HEADER_PART1)
    lines.append(HEADER_PART2)
    lines.append(r"\midrule")

    prev_method = None
    for row in rows:
        method = row["method"]
        param  = row["param"]
        path   = row["path"]

        if prev_method is not None and method != prev_method:
            lines.append(r"\midrule")
        prev_method = method

        lisi_v, ari_v, mmd_v = collect_quality(path)
        bb_noaux = collect_auc(path, TM_BB_NOAUX, use_classb=False)
        bb_aux   = collect_auc(path, TM_BB_AUX,   use_classb=False)
        cb_noaux = collect_auc(path, TM_BB_NOAUX, use_classb=True)
        cb_aux   = collect_auc(path, TM_BB_AUX,   use_classb=True)

        cells = [
            method,
            param,
            fmt(lisi_v),
            fmt(ari_v),
            fmt(mmd_v, scale=1e3),
            fmt(bb_noaux),
            fmt(bb_aux),
            fmt(cb_noaux),
            fmt(cb_aux),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(caption)
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# Captions
# ──────────────────────────────────────────────

CAPTION_SHARED_DEFS = (
    r"  LISI$\uparrow$: mixing of real and synthetic cells (higher = better); "
    r"ARI$\uparrow$: cell-type structure preservation; "
    r"MMD${\times}10^3\downarrow$: distributional distance (lower = better). "
    r"BB$^{-}$/BB$^{+}$: scMAMA-MIA black-box attack without/with auxiliary data; "
    r"\textsc{CB}: \textsc{ClassB} variant adding per-gene log-likelihood ratio evidence. "
    r"AUC$=0.5$ = chance level; AUC$=1.0$ = perfect membership inference. "
    r"Values are mean$\,\pm\,$std over 5 random donor splits (OneK1K, 50 donors). "
    r"$^n$ = fewer than 5 trials."
)

CAPTION_COMBINED = (
    r"\caption{"
    "\n"
    r"  \textbf{Fidelity and privacy for all SDG methods (OneK1K, 50 donors).} "
    "\n"
    + CAPTION_SHARED_DEFS + "\n"
    r"  For scDesign2+DP, $\eta=10^2/\varepsilon$ (larger $\eta$ = stronger privacy). "
    r"scDiffusion: v3 implementation with v1\_celltypist annotations. "
    r"NMF $\varepsilon=2.8$ is the CAMDA 2024 default (three-stage budget)."
    "\n}"
)

CAPTION_DP = (
    r"\caption{"
    "\n"
    r"  \textbf{Privacy--utility tradeoff under DP for scDesign2 and NMF (OneK1K, 50 donors).} "
    "\n"
    + CAPTION_SHARED_DEFS + "\n"
    r"  scDesign2: $\eta=10^2/\varepsilon$ is the Gaussian noise-to-signal parameter added to"
    r"  the copula covariance (larger $\eta$ = stronger DP = more noise). "
    r"  NMF: $\varepsilon$ is the per-run DP budget across NMF, KMeans, and sampling stages "
    r"  (CAMDA 2024 proportional allocation; $\varepsilon=2.8$ is the CAMDA default)."
    "\n}"
)

CAPTION_NODP = (
    r"\caption{"
    "\n"
    r"  \textbf{Fidelity and privacy for all non-DP SDG methods (OneK1K, 50 donors).} "
    "\n"
    + CAPTION_SHARED_DEFS + "\n"
    r"  scDiffusion: v3 implementation with v1\_celltypist annotations."
    "\n}"
)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def save(tex, filename):
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"Saved: {out_path}", file=sys.stderr)


def main():
    t1 = build_table(ALL_ROWS,  CAPTION_COMBINED, "tab:combined_ok_50d")
    t2 = build_table(DP_ROWS,   CAPTION_DP,       "tab:dp_ok_50d")
    t3 = build_table(NODP_ROWS, CAPTION_NODP,     "tab:nodp_ok_50d")

    save(t1, "combined_table_ok_50d.tex")
    save(t2, "dp_table_ok_50d.tex")
    save(t3, "nodp_table_ok_50d.tex")

    print("\n=== TABLE 1: Combined (all methods) ===\n")
    print(t1)
    print("\n=== TABLE 2: DP sweep (SD2 + NMF) ===\n")
    print(t2)
    print("\n=== TABLE 3: No-DP methods only ===\n")
    print(t3)


if __name__ == "__main__":
    main()
