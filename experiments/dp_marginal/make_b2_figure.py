#!/usr/bin/env python3
"""Build B2 deliverables from _agg/b2_summary.csv:
  1. figures/b2_marginal_noise_ok_50d.tex  (+ aida) — same style as dp_table_ok_50d.tex
  2. figures/b2_frontier.png — privacy-utility frontier (enhanced BB+aux vs LISI), ok+aida panels
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AGG = os.path.join(HERE, "_agg", "b2_summary.csv")
FIG = "/home/golobs/scRNA-seq_privacy_audits/figures"
df = pd.read_csv(AGG)


def _lz(x):
    """Format 2dp, dropping a genuine leading zero (0.46 -> .46; 7.86 -> 7.86)."""
    s = f"{x:.2f}"
    return s[1:] if s.startswith("0.") else s


def _c(m, sd, scale=1.0, prec=2):
    """Paper cell: .XX{\tiny±.XX} at given scale."""
    if np.isnan(m):
        return "--"
    mm, ss = m * scale, (0.0 if np.isnan(sd) else sd * scale)
    return _lz(mm) + r"{\tiny$\,\pm\,$" + _lz(ss) + "}"


def mode_label(mode, s):
    if mode == "none":
        return "(no noise)"
    pretty = {"cov": "cov", "marg": "marg", "both": "both"}[mode]
    eq = "{=}"
    return rf"{pretty}, $s{eq}{s:.1f}$"


def write_tex(ds, dsname):
    sub = df[df.dataset == ds].copy()
    order = [("none", 0.0)] + [(m, s) for m in ["cov", "marg", "both"] for s in [0.1, 0.5, 1.0, 2.0]]
    lines = [
        r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
        r"\begin{tabular}{l|ccc|cccc}", r"\toprule",
        r"\multicolumn{1}{c|}{} & \multicolumn{3}{c|}{\textbf{Fidelity}} & \multicolumn{4}{c}{\textbf{Privacy (MIA AUC)}} \\",
        r"\textbf{Noise target} & LISI$\uparrow$ & ARI$\uparrow$ & MMD${\times}10^3\downarrow$ & BB$^{-}$ & BB$^{+}$ & \textsc{CB} BB$^{-}$ & \textsc{CB} BB$^{+}$ \\",
        r"\midrule",
    ]
    prev = None
    for mode, s in order:
        r = sub[(sub["mode"] == mode) & (np.isclose(sub["s"], s))]
        if r.empty:
            continue
        r = r.iloc[0]
        if prev is not None and mode != prev:
            lines.append(r"\midrule")
        prev = mode
        cells = [
            mode_label(mode, s),
            _c(r["lisi_mean"], r["lisi_std"]),
            _c(r["ari_mean"], r["ari_std"]),
            _c(r["mmd_mean"], r["mmd_std"], scale=1e3),
            _c(r["std_bbnoaux_mean"], r["std_bbnoaux_std"]),
            _c(r["std_bbaux_mean"], r["std_bbaux_std"]),
            _c(r["enh_bbnoaux_mean"], r["enh_bbnoaux_std"]),
            _c(r["enh_bbaux_mean"], r["enh_bbaux_std"]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{",
        rf"  \textbf{{Marginal- vs covariance-noise defense against the enhanced attack ({dsname}, 50 donors).}}",
        r"  Noise is injected into the copula covariance only (cov), the per-gene marginals only (marg), or both, at matched level $s$.",
        r"  LISI$\uparrow$/ARI$\uparrow$/MMD${\times}10^3\downarrow$ as in Table~\ref{tab:dp_ok_50d}. BB$^{-}$/BB$^{+}$: black-box attack without/with aux; \textsc{CB}: \textsc{ClassB} variant.",
        r"  Covariance noise cannot drive the enhanced attack (\textsc{CB} BB$^{+}$) below $\approx$0.70 at any level, while marginal noise reaches chance only once fidelity (LISI) has collapsed; ``both'' gains no privacy over ``marg''.",
        r"  Mean$\,\pm\,$std over 3 donor splits.",
        r"}",
        rf"\label{{tab:b2_marginal_{ds}_50d}}",
        r"\end{table*}",
    ]
    out = os.path.join(FIG, f"b2_marginal_noise_{ds}_50d.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


def make_frontier():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    colors = {"cov": "#1f77b4", "marg": "#d62728", "both": "#2ca02c"}
    markers = {"cov": "o", "marg": "s", "both": "^"}
    for ax, ds, title in [(axes[0], "ok", "OneK1K"), (axes[1], "aida", "AIDA")]:
        sub = df[df.dataset == ds]
        none = sub[sub["mode"] == "none"].iloc[0]
        # baseline point
        ax.scatter([none["lisi_mean"]], [none["enh_bbaux_mean"]], c="k", s=70, zorder=5,
                   marker="*", label="no noise")
        for mode in ["cov", "marg", "both"]:
            m = sub[sub["mode"] == mode].sort_values("s")
            xs = [none["lisi_mean"]] + list(m["lisi_mean"])
            ys = [none["enh_bbaux_mean"]] + list(m["enh_bbaux_mean"])
            ax.plot(xs, ys, marker=markers[mode], color=colors[mode], label=mode, lw=1.8, ms=5)
        ax.axhline(0.5, ls=":", c="gray", lw=1)
        ax.set_title(title)
        ax.set_xlabel("LISI (utility) $\\rightarrow$ better")
        ax.grid(alpha=0.25)
        ax.invert_xaxis()  # utility decreases left-to-right as noise grows
    axes[0].set_ylabel("Enhanced BB+aux AUC")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Privacy–utility frontier: covariance noise floors ($\\approx$0.70); "
                 "marginal noise reaches chance only at LISI$\\to$0", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(FIG, "b2_frontier.png")
    fig.savefig(out, dpi=160)
    print("wrote", out)


if __name__ == "__main__":
    write_tex("ok", "OneK1K")
    write_tex("aida", "AIDA")
    make_frontier()
