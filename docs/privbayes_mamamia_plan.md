# PrivBayes–MAMA-MIA attack pipeline on scDesign2 data (Experiment #4)

**Goal.** Run the *full original MAMA-MIA attack* — PrivBayes focal-point (FP)
extraction, then focal-point marginal aggregation — against **scDesign2 synthetic
scRNA-seq data**, to study how well the generic tabular MAMA-MIA attack transfers
to single-cell data (vs. our copula-specialised scMAMA-MIA). This is *not*
PrivBayes-as-a-generator; PrivBayes is only the FP extractor.

Repo: `~/SyntheticData_MIA` (MAMA-MIA). Env: `mamamia_` (already built).

---

## Key finding: MAMA-MIA already does donor-level (set) MIA

`util.sample_experimental_data` and `util.score_attack` operate in **set
membership inference** mode when `cfg.set_MI = True`:

- `aux['HHID']` groups records into **households**; candidates are households with
  `>= cfg.household_min_size` records.
- `score_attack` iterates `for hhid in target_ids`, aggregates the per-record
  focal-point scores within each household, and computes `area_under_curve(
  membership, activated_scores)` at the **household level**.

This maps **directly** onto our setting:

| MAMA-MIA (tabular) | scRNA-seq (ours) |
|---|---|
| household `HHID`   | **donor** id |
| record (row)       | **cell** |
| set membership inference (`set_MI=True`) | **donor-level** MIA |
| `aux` reference    | real held-out cells (for FP extraction) |
| `synth`            | **scDesign2 synthetic cells** (the target) |

So we do **not** need a custom donor aggregator — we reuse MAMA-MIA's set-MI path
with `HHID = donor`. This is the main de-risking insight.

---

## Pipeline (what we build)

1. **Stage** scDesign2 data into MAMA-MIA's discretised-tabular contract
   (`stage_scdesign2_to_mamamia.py`):
   - rows = cells; columns = **discretised genes** + `HHID` (= donor id);
   - `meta.json` domain = `{gene: n_bins}` (categorical cardinalities);
   - three frames: `aux` (real held-out cells, for FP extraction & candidate
     sampling), `synth` (scDesign2 synthetic cells, the attack target),
     and the donor→membership map from our existing split.
   - Discretiser is **fit on aux** (quantile bins per gene) and applied identically
     to synth, so bins are shared.

2. **Focal points**: `determine_privbayes_conditionals(cfg, aux, columns,
   categorical, meta, eps, n_size)` → PrivBayes conditionals (the FPs).

3. **Attack (adapted)**: call `run_all_privbayes_experiments(cfg, columns, aux,
   synth = SCDESIGN2_SYNTH, eps, targets, target_ids, membership, fps)`.
   The one change vs. stock `attack_privbayes`: we **inject our staged scDesign2
   synth** instead of the PrivBayes-generated synth, and we build
   `targets/target_ids/membership` from **our** train(member)/holdout(non-member)
   donor split rather than MAMA-MIA's random `sample_experimental_data` resample
   (a thin `sample_experimental_data_from_split()` shim).

4. **Metric**: donor-level ROC AUC (MAMA-MIA's `area_under_curve`), 5 trials,
   compared against scMAMA-MIA's BB AUC on the same scDesign2 data/splits.

---

## Design choices (defaults chosen; flag for confirmation)

| Choice | Default | Rationale / alternative |
|---|---|---|
| **Gene set** for FP extraction | **group-2 copula genes** (per cell type) | Direct apples-to-apples with scMAMA-MIA's focal points. PrivBayes over all ~2000 HVGs is expensive and its clique structure degrades with dimension. Alt: top-K most-variable genes. |
| **Per-cell-type vs pooled** | **per cell type**, then aggregate to donor | Matches scMAMA-MIA (each cell type has its own copula/FPs). Pooling loses cell-type structure. |
| **Discretisation** | **quantile bins, n_bins=10**, fit on aux | PrivBayes needs categorical data. 10 bins ≈ standard MAMA-MIA cardinality. |
| **PrivBayes eps** (FP extraction) | **1000** (≈ non-private FPs) | We study attack *ceiling*; the FP extractor should not itself be privatised. Can sweep. |
| **Pilot config** | `ok/scdesign2/no_dp`, 50d, trial 1 | Cheapest meaningful setting; scale to 490d/440d + all trials after pilot. |

---

## Decisions locked (2026-07-27, user-approved)
- **Gene set = `copula`** (default): the scDesign2 COPULA-COVARIANCE genes
  (`primary_genes`/`gene_sel1` — the Mahalanobis focal genes scMAMA-MIA exploits),
  unioned across cell-type models, top-N by aux variance. NOTE: parse_copula's
  `primary` = IN copula, `secondary` = marginal-only; CLAUDE.md's group-1/2 wording
  is inverted — the code is authoritative.
- **Binning = zero-aware** (bin 0 = exact zero; bins 1..k = quantile bins of the
  non-zero tail). Restores resolution lost to zero-inflation: domain median 6
  (min/max 3/9) vs. 3 for a plain quantile grid.

## Status
- Data contract mapped (this doc). HHID=donor / set_MI insight validated in code.
- `stage_scdesign2_to_mamamia.py` — **BUILT + validated** end-to-end (dry-run) on
  ok/sd2/no_dp/50d/1: 512 copula genes found, 50 selected, zero-aware domains,
  balanced 50 member / 50 non-member. Writes aux/targets/synth parquet + meta.json
  + membership.csv + manifest.json.
- `run_privbayes_mamamia.py` — **BUILT** (attack driver). Wiring verified against
  MAMA-MIA source: PrivBayes on aux -> `gen.conditionals` -> Counter(weights) ->
  `custom_privbayes_attack` -> `score_attack` (donor-level AUC). Bypasses the
  save/load/`C.n_bins`/artifact machinery on purpose.
## ✅ WORKING END-TO-END (2026-07-27) + first result
- **Env solved** (no `mamamia_` build needed): the `sdg` conda env already has the
  classic `mbi` + deps. Compiled `privBayesSelect` once
  (`.../privBayes && python setup.py build_ext --inplace`; cp310 .so, ABI-matches
  `sdg`), `pip install validators` (one tiny dep). Driver imports ONLY the reprosyn
  PrivBayes generator and VENDORS the pure-numpy scorer -> no jax/mst/gsd.
- **Scorer vectorised** (merge-based) because the original per-cell Python loop is
  intractable at 10^5 cells. Verified `max|A_loop - A_vec| = 0.0` on real staged
  data (exact match to the MAMA-MIA math).
- **Pilot (ok/scdesign2/no_dp/50d/1, copula genes x50, zero-aware bins, FP eps=1000,
  10k aux cells):** donor-level **AUC = 0.464**, MA = 0.466 — i.e. ~RANDOM. 100
  donors balanced 50/50, 125,802 target cells, 50 PrivBayes conditionals.
- **Interpretation (preliminary, 1 trial):** the generic tabular PrivBayes-MAMA-MIA
  attack does NOT transfer to scDesign2 scRNA-seq, whereas the copula-specialised
  scMAMA-MIA does — supporting the paper's thesis that the leak comes from the
  copula structure. NEXT: multi-trial + larger nd (490/440) for a stable estimate,
  and report side-by-side vs scMAMA-MIA BB AUC on the SAME splits.

RUN:
    conda run -n tabddpm_ python .../stage_scdesign2_to_mamamia.py --dataset ok \
        --sdg scdesign2/no_dp --nd 50 --trial 1 --gene-set copula --n-genes 50 \
        --n-bins 10 --out-dir .../_staged/ok_no_dp_50d_t1
    conda run -n sdg python .../run_privbayes_mamamia.py \
        --staged .../_staged/ok_no_dp_50d_t1 --eps 1000 --fp-sample 10000

## (historical) env blocker — now resolved above
- **MAMA-MIA PrivBayes env.** The `privbayes`
  module (`reprosyn-main/.../mbi/privbayes.py`) needs a NATIVE build:
  `privBayesSelect.pyx/.cpp` (Cython + C++ compile) + `mbi` (private-pgm; reprosyn
  expects the OLD `Dataset/Domain/Factor` API, so pin an old mbi, not PyPI 1.x) +
  `jax==0.4.13`. `conduct_attacks`/`util` also import `mst`/`gsd`/jax at module top,
  so either install the full stack OR (lighter) vendor the ~40 self-contained lines
  of `custom_privbayes_attack`+`score_attack`+helpers and import ONLY the compiled
  PrivBayes generator. Next step once env resolves: run staging then
  `run_privbayes_mamamia.py` on the 50d/1 pilot, then scale to 490d/440d × trials.
