# Aux↔Target Contamination: Affected Manuscript Values + Clean-Aux Re-run

**Purpose.** Reviewers (meta pt.1, kPMB W1, oT6m) flagged that the auxiliary set may
overlap the target set (train ∪ holdout). Rather than explain the exceptions in the
rebuttal, we are re-running the *specific presented values* that are actually affected with
a provably-disjoint auxiliary set, so the tables carry clean numbers.

**Audit basis.** `scratchpad/a1_aux_overlap.csv` (script `a1_aux_overlap_audit.py`) measured
`|aux ∩ (train ∪ holdout)|` for every saved split. Only four (dataset, size) combos have any
overlap; every other presented size is provably disjoint:

| Config | aux size | overlap with target | cause |
|---|---|---|---|
| AIDA 200d | 200 | 92 / 200 | 508-donor pool exhausted (200+200 target leaves 108) |
| OneK1K 490d | 200 | 200 / 200 (all ⊂ holdout) | `strategy_490` draws aux as a subsample of holdout |
| HFRA 10d | 10 | 8 / 10 | 22-donor pool |
| HFRA 20d (=11/11) | 11 | 11 / 11 | 22-donor pool |

All other OneK1K (2–200d) and AIDA (2–100d) sizes: **0 overlap, provably disjoint.**

---

## Every affected value presented in the manuscript

### 1. Appendix Table `tab:scdesign2_all` (scDesign2, four threat models × donor sizes)
Only the **+aux** rows at the four contaminated sizes are affected (the `-aux` rows use no
auxiliary data and are untouched):

| Dataset/size | WB+aux | WB+aux (ClassB) | BB+aux | BB+aux (ClassB) |
|---|---|---|---|---|
| OneK1K 490d | 0.64 | 0.68 | 0.55 | 0.59 |
| AIDA 200d | 0.87 | 0.82 | 0.77 | 0.77 |
| HFRA 10d | 0.84 | 0.76 | 0.79 | 0.75 |
| HFRA 20d | 0.84 | 0.72 | 0.77 | 0.71 |

### 2. Appendix Table `tab:combined_ok_490d` (OneK1K 490d, all SDGs)
The **BB+aux** and **enhanced BB+aux** columns are affected (the BB-aux column is not):

| Row | BB+aux | enh BB+aux |
|---|---|---|
| scDesign2 (no DP) | 0.55 | 0.59 |
| scDesign2 η=10⁻⁴ | 0.51 | 0.50 |
| scDesign2 η=10¹ | 0.52 | 0.50 |
| scDesign3-V | 0.55 | 0.65 |
| scVI | 0.63 | 0.68 |
| ZINBWave | 0.59 | 0.61 |

### 3. Main Figure `fig:mia_aucs` (BB+aux, scDesign2)
The **HFRA 10d and 20d** points (same underlying runs as the HFRA BB+aux cells in Table 1).
AIDA (≤50d) and OneK1K (≤200d) points in this figure are all disjoint sizes → clean.

---

## Clean-aux sampling strategy (decided with author)

Draw D_aux uniformly from donors **not in D_train ∪ D_holdout** (disjoint from the full
scored target set). Handling of pool-exhausted configs:

- **AIDA 200d** → keep 200 train / 200 holdout; **aux = the 108 remaining donors** (all of
  them, complement). Train unchanged → on-disk synthetic data reused → cheap aux re-fit +
  re-score, no regeneration.
- **OneK1K 490d** → **[DECISION PENDING]** rebalance to 390/390 with 200 disjoint aux
  (requires FULL regeneration of all six generators at 390 donors and relabelling "490d"→
  "390d"), **or** shrink holdout to 490/291 with 200 disjoint aux (keeps 490 members, reuses
  synth, no regeneration). Author leaning to be confirmed.
- **HFRA 10d & 20d** → **DROP** the +aux cells (report "—"). The 22-donor pool cannot supply
  a disjoint auxiliary set at these sizes; stated honestly as a small-pool limitation. This
  removes 8 values from Table 1 and the two HFRA +aux points from `fig:mia_aucs`.

---

## Re-run status

- Isolated code: `experiments/aux_disjoint/` — `build_disjoint_aux_split.py` (writes a
  disjoint split, verifies disjointness), `run_auxfix_trial.py` (reuses synth + train
  copulas, refits ONLY the aux copula on the new disjoint aux, runs the quad attack twice:
  BB then WB → all four threat models + Class B), `run_auxfix_sweep.py` (scheduler,
  resource-guarded, detached), `kill_auxfix.sh`.
- **AIDA 200d** (scDesign2, disjoint aux=108 ✓): trials 1–4 DONE (t5 queued, low priority). Clean-aux
  4-trial means vs contaminated:
    - standard  **BB+aux 0.77 → 0.80** (0.771/0.791/0.805/0.850), **WB+aux 0.87 → 0.92** (0.906/0.904/0.918/0.933)
    - enhanced  **BB+aux 0.77 → 0.79** (0.720/0.732/0.857/0.838), **WB+aux 0.82 → 0.85** (0.797/0.742/0.915/0.925)
  ALL four clean numbers are ≥ the contaminated values (marginally higher), so the overlap did NOT inflate
  results — it slightly *understated* them. These are the camera-ready replacement values (t5 will barely move them).
- **OneK1K 490d**: ⏳ STILL awaiting shrink-vs-rebalance decision (drives cheap re-attack vs full
  6-generator regeneration). Recommended: shrink-holdout (490/291, aux=200) — keeps the 490 label,
  reuses synth, no regen. Marked [PENDING] in the rebuttal.
- **HFRA**: no compute (dropped; +aux cells become "—", stated as small-pool limitation).

Results land in `{ds}/aux_disjoint/scdesign2/{nd}d/{trial}/results/{bb,wb}_results*.csv`.
