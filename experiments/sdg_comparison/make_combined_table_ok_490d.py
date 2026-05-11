"""
Generate a combined LaTeX table: OneK1K dataset, 490 donors.

Rows: all SDG methods with 490d data (scDesign2 no_dp + DP variants, scDesign3-V,
      scVI, scDiffusion, ZINBWave); partial trials shown with ^n superscript.

Columns:
  [SDG Method] [DP Param]  |  Fidelity: LISI  ARI  MMD×10³  |
  Privacy: BB⁻  BB⁺  CB BB⁻  CB BB⁺

No horizontal rules between data rows.

Usage:
  python experiments/sdg_comparison/make_combined_table_ok_490d.py
"""

import os
import math
import sys
import numpy as np
import pandas as pd

DATA     = "/home/golobs/data/scMAMAMIA"
N_TRIALS = 5
ND       = "490d"

TM_BB_AUX   = "tm:100"
TM_BB_NOAUX = "tm:101"


# ──────────────────────────────────────────────
# Label helpers (shared with 50d script)
# ──────────────────────────────────────────────

def eta_label(eps_val: float) -> str:
    eta = 100.0 / eps_val
    log = math.log10(eta)
    exp = round(log)
    if abs(10 ** exp - eta) < 1e-6:
        if exp == 0:
            return r"$\eta=1$"
        return rf"$\eta=10^{{{exp}}}$"
    return rf"$\eta={eta:.3g}$"


# ──────────────────────────────────────────────
# Row definitions (only variants with 490d data)
# ──────────────────────────────────────────────

ROWS = [
    # ── scDesign2 ──────────────────────────────────────────────────────────
    {"method": "scDesign2",   "param": r"(no DP)",          "path": "ok/scdesign2/no_dp"},
    {"method": "scDesign2",   "param": eta_label(100_000_000), "path": "ok/scdesign2/eps_100000000"},
    {"method": "scDesign2",   "param": eta_label(1_000_000),   "path": "ok/scdesign2/eps_1000000"},
    {"method": "scDesign2",   "param": eta_label(10_000),      "path": "ok/scdesign2/eps_10000"},
    {"method": "scDesign2",   "param": eta_label(100),         "path": "ok/scdesign2/eps_100"},
    {"method": "scDesign2",   "param": eta_label(10),          "path": "ok/scdesign2/eps_10"},
    {"method": "scDesign2",   "param": eta_label(1),           "path": "ok/scdesign2/eps_1"},
    # ── scDesign3-V (2/5 trials) ────────────────────────────────────────
    {"method": "scDesign3-V", "param": r"(no DP)",          "path": "ok/scdesign3/vine"},
    # ── scVI (3/5 trials) ───────────────────────────────────────────────
    {"method": "scVI",        "param": r"(no DP)",          "path": "ok/scvi/no_dp"},
    # ── scDiffusion v1 (4/5 MIA trials; quality not available) ──────────
    {"method": "scDiffusion", "param": r"(no DP)",          "path": "ok/scdiffusion/no_dp"},
    # ── ZINBWave (5/5) ──────────────────────────────────────────────────
    {"method": "ZINBWave",    "param": r"(no DP)",          "path": "ok/zinbwave/no_dp"},
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
            ari_v.append(float(row.get("ari_real_vs_syn", float("nan"))))
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
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def fmt(vals, scale=1.0, missing="---"):
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

def build_table(rows):
    lines = []
    lines.append(r"\begin{table*}[ht]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{ll|ccc|cccc}")
    lines.append(r"\toprule")

    # Header row 1
    lines.append(
        r"\multicolumn{2}{c|}{} & "
        r"\multicolumn{3}{c|}{\textbf{Fidelity}} & "
        r"\multicolumn{4}{c}{\textbf{Privacy (MIA AUC)}} \\"
    )
    # Header row 2
    lines.append(
        r"\textbf{SDG Method} & \textbf{DP Param} & "
        r"LISI$\uparrow$ & ARI$\uparrow$ & MMD${\times}10^3\downarrow$ & "
        r"BB$^{-}$ & BB$^{+}$ & \textsc{CB} BB$^{-}$ & \textsc{CB} BB$^{+}$ \\"
    )
    lines.append(r"\midrule")

    for row in rows:
        lisi_v, ari_v, mmd_v = collect_quality(row["path"])
        bb_noaux = collect_auc(row["path"], TM_BB_NOAUX, use_classb=False)
        bb_aux   = collect_auc(row["path"], TM_BB_AUX,   use_classb=False)
        cb_noaux = collect_auc(row["path"], TM_BB_NOAUX, use_classb=True)
        cb_aux   = collect_auc(row["path"], TM_BB_AUX,   use_classb=True)

        cells = [
            row["method"],
            row["param"],
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

    lines.append(r"\caption{")
    lines.append(
        r"  \textbf{Fidelity and privacy for all SDG methods with available 490-donor results (OneK1K).}"
    )
    lines.append(
        r"  LISI$\uparrow$: real/synthetic cell mixing; ARI$\uparrow$: cell-type structure;"
        r"  MMD${\times}10^3\downarrow$: distributional distance."
        r"  BB$^{-}$/BB$^{+}$: scMAMA-MIA black-box without/with auxiliary data;"
        r"  \textsc{CB}: \textsc{ClassB} variant with per-gene LLR evidence."
        r"  AUC$=0.5$ = chance; AUC$=1.0$ = perfect inference."
        r"  For scDesign2+DP, $\eta=10^2/\varepsilon$ (larger = stronger DP)."
        r"  $^n$ = fewer than 5 completed trials ($n$ averaged)."
        r"  scDiffusion quality metrics unavailable at 490d."
        r"  Values: mean$\,\pm\,$std over available trials."
    )
    lines.append(r"}")
    lines.append(r"\label{tab:combined_ok_490d}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    table = build_table(ROWS)
    print(table)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "combined_table_ok_490d.tex")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nSaved: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
