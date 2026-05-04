# scDiffusion: Correct Generation Spec for Paper-Faithful Implementation

**Written 2026-05-04. Read this before touching any scDiffusion code.**

---

## 1. What the Original Paper Actually Does

Paper: Luo et al. 2024, *Bioinformatics* ("scDiffusion: Conditional Generation of
High-Quality Single-Cell Data Using Diffusion Model").
Repo cloned at: `/home/golobs/scDiffusion/`

### Training pipeline (all 3 stages, per `train.sh` and README):

1. **Stage 1 — VAE**: `VAE/VAE_train.py`. Autoencoder: gene space → 128-dim latent.
   Paper: 200k steps, batch=128.

2. **Stage 2 — Diffusion backbone**: `cell_train.py`. Unconditional DDPM in latent
   space. Paper: 800k steps (`lr_anneal_steps=800000`), batch=128.

3. **Stage 3 — Classifier**: `classifier_train.py`. Cell_classifier trained on noisy
   latents for classifier-guided *conditional* generation. Paper: 200k–400k steps.

### Generation for the paper's main results (UMAPs, statistical evals):

**The paper's main evaluations use UNCONDITIONAL generation**, NOT classifier-guided.

From README: "Run `cell_sample.py`" → generates raw latent embeddings, no cell type
conditioning. The output is a `.npz` file of latent vectors with no cell type labels.

**Cell type labels are assigned post-hoc using CellTypist** — a logistic regression
classifier trained on the real training data and applied to the decoded generated cells.
See `celltypist_train.py` in the repo.

The paper's experiment reproduction scripts (`exp_script/script_diffusion_umap.ipynb`,
`exp_script/script_static_eval.ipynb`) all follow this unconditional + CellTypist flow.

### What the classifier IS for:

`classifier_sample.py` = **on-demand conditional generation**: "generate me N cells of
cell type X." This is presented as an *additional capability* of the model, not the
main evaluation approach. The `train.sh` includes classifier training so the full
feature set is available, but the paper's quality/UMAP results don't use it.

---

## 2. History of Our Implementation Mistakes

### v1 (original, all data in `scdiffusion/` directories — DISCARD):
- Ran only Stages 1+2 (unconditional DDPM). ✓ Correct generation method.
- Assigned cell types via **1-NN in VAE latent space** against training cells. ✗ WRONG.
  Paper uses CellTypist. 1-NN in latent space is ad hoc and not validated.
- Wrong hyperparameters: vae_steps=150k (paper: 200k), diff_steps=300k (paper: 800k),
  batch=512 (paper: 128).
- All data in `scdiffusion/` dirs is invalid. Do not use.

### v2 (current, data in `scdiffusion_v2/` directories — ALSO WRONG):
- Implemented all 3 stages. Stage 3 matches `classifier_train.py` faithfully.
- Generation uses **classifier-guided sampling** (`classifier_sample.py` approach). ✗
  This is NOT what the paper's main results use. It's the optional conditional feature.
- Correct hyperparameters: vae=200k, diff=800k, batch=128. ✓
- Cell type labels are assigned via guidance target (baked in). Defensible, but a
  reviewer can ask: "Why didn't you use the same generation mode as the original paper?"
- Code committed 2026-05-04 (commit c301fe7). Already running on GPU0 and GPU1.

### v3 (TARGET — what this spec describes):
- Stages 1+2 only (no classifier training needed).
- Generation: **unconditional** (all cells at once, no guidance, no classifier).
- Cell type labels: **CellTypist** trained on `D_train`, applied to decoded generated cells.
- Matches the paper's main evaluation pipeline exactly.
- Defensible to reviewers.

---

## 3. What Needs to Change

### 3a. `src/sdg/scdiffusion/run_scdiffusion_standalone.py`

This file lives at:
`/home/golobs/scRNA-seq_privacy_audits/src/sdg/scdiffusion/run_scdiffusion_standalone.py`

**Keep unchanged:**
- `train_vae` subcommand — already correct.
- `train` (diffusion backbone) subcommand — already correct.
- `_preprocess()`, `_load_vae()`, `_load_diffusion_model()` helpers — already correct.
- All Stage 1/2 training logic.

**Remove / replace:**
- `train_classifier` subcommand — not needed for paper-faithful generation. Can be kept
  but should be clearly marked optional/unused.
- `cmd_generate()` — currently does classifier-guided per-cell-type sampling. Must be
  replaced with unconditional generation + CellTypist annotation.

**New `generate` subcommand logic (paper-faithful):**

```python
def cmd_generate(args):
    """
    Generate cells unconditionally (paper-faithful, matching cell_sample.py),
    then assign cell types via CellTypist trained on D_train.

    Matches the paper's main evaluation pipeline:
      - Unconditional DDPM reverse process (no classifier, no guidance)
      - clip_denoised=False  (cell_sample.py default)
      - CellTypist for post-hoc cell type assignment
    """
    # 1. Load VAE + diffusion model (no classifier loaded)
    # 2. Run p_sample_loop unconditionally for n_cells total
    #      sample_fn = diffusion.p_sample_loop (or ddim_sample_loop)
    #      no cond_fn, no model_kwargs with y
    #      clip_denoised=False  (matches cell_sample.py default)
    # 3. Decode latents via VAE → log1p gene expression
    # 4. Load CellTypist model (trained on D_train, stored in args.celltypist_ckpt)
    # 5. Apply CellTypist: annotate decoded cells → cell type labels
    # 6. Apply expm1() to get pseudo-count space
    # 7. Save as h5ad with cell_type column
```

Key CellTypist usage:
```python
import celltypist

# Load model
ct_model = celltypist.models.Model.load(args.celltypist_ckpt)

# Annotate — needs AnnData with log1p-normalized expression (BEFORE expm1)
synth_log1p = ad.AnnData(X=gene_expr_log1p, var=pd.DataFrame(index=gene_names))
sc.pp.normalize_total(synth_log1p, target_sum=1e4)  # may already be normalized
# CellTypist expects log1p-normalized data
predictions = celltypist.annotate(synth_log1p, model=ct_model, majority_voting=False)
cell_type_labels = predictions.predicted_labels['predicted_labels'].values
```

**New `train_celltypist` subcommand** (train the CellTypist classifier on D_train):
```python
def cmd_train_celltypist(args):
    """
    Train a CellTypist model on D_train for post-hoc annotation of generated cells.
    Mirrors celltypist_train.py from Luo et al. repo.
    Saves model to args.celltypist_out.
    """
    import celltypist
    adata = sc.read_h5ad(args.train_h5ad)
    # normalize + log1p (CellTypist expects this)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    new_model = celltypist.train(adata, labels=args.cell_type_col, n_jobs=8,
                                  feature_selection=False)
    new_model.write(args.celltypist_out)
    print(f"CellTypist model saved to {args.celltypist_out}")
```

**Updated `generate` CLI args:** Remove `--classifier-ckpt`, `--classifier-scale`,
`--start-guide-steps`. Add `--celltypist-ckpt`. Keep `--generation-batch-size`.

---

### 3b. `experiments/sdg_comparison/generate_trial.py`

File: `/home/golobs/scRNA-seq_privacy_audits/experiments/sdg_comparison/generate_trial.py`

**Current `generate_scdiffusion_v2()` function** (lines ~423–534):
- Runs Stages 1, 2, 3 (VAE, diffusion, classifier)
- Calls `generate` with classifier_ckpt

**New logic:**
- Stage 1 (VAE): unchanged
- Stage 2 (Diffusion): unchanged  
- Stage 3: **Replace classifier training with CellTypist training**
  - Train CellTypist on `train_hvg.h5ad` → save to `models/celltypist/model.pkl`
  - Skip if `models/celltypist/model.pkl` already exists
- Generation: call `generate` WITHOUT classifier_ckpt, WITH `--celltypist-ckpt`

Remove: `classifier_dir`, `classifier_steps`, `classifier_scale`, `start_guide_steps`
Add: `celltypist_dir` (default: `<out_dir>/models/celltypist/`)

Remove from `generate_trial.py` argparse:
- `--classifier-steps`
- `--classifier-scale`
- `--start-guide-steps`

Add:
- (no new CLI args needed — CellTypist is always used, no options to tune)

---

### 3c. `src/sdg/scdiffusion/model.py`

File: `/home/golobs/scRNA-seq_privacy_audits/src/sdg/scdiffusion/model.py`

This is the high-level wrapper used by `run_all.py`. Update similarly:
- Remove `classifier_dir`, `classifier_steps`, `classifier_scale`, `start_guide_steps`
- Add `celltypist_dir`
- Stage 3 in `train()`: run `train_celltypist` instead of `train_classifier`
- `generate()`: pass `--celltypist-ckpt` instead of `--classifier-ckpt`

---

### 3d. `experiments/sdg_comparison/run_all.py`

The `_scdf_v2_jobs()` function (added 2026-05-04) builds the CLI commands for the
generation sweep. Update it to NOT pass `--classifier-steps`, `--classifier-scale`,
`--start-guide-steps`. No new args needed (CellTypist is automatic).

---

## 4. What Does NOT Change

- Directory convention: keep `scdiffusion_v2/`. The v2 name refers to the corrected
  implementation relative to v1's defects (wrong hyperparams, wrong cell type assigner).
  This is still an improvement over v1 even after switching from classifier-guided to
  unconditional + CellTypist.
- VAE architecture and training: unchanged.
- Diffusion backbone architecture and training: unchanged.
- Hyperparameters: vae=200k, diff=800k, batch=128 — keep all of these.
- Output h5ad format: unchanged. Cell type column = `cell_type` (or `--cell-type-col`).
- All MIA attack code, quality eval code, status scripts: unchanged.

---

## 5. Currently Running Jobs (as of 2026-05-04 ~05:43 UTC)

Two jobs are running with the v2 (classifier-guided) approach. They need to be killed
and re-launched after this spec is implemented:

- **GPU0 (spot-check):** ok dataset, 10 donors, trial 1
  - PID: 2897976
  - Log: `/tmp/scdfv2_spot_check.log`
  - Out-dir: `/home/golobs/data/scMAMAMIA/ok/scdiffusion_v2/no_dp/10d/1/`

- **GPU1 (serial sweep):** ok 50d trials 1–5 + cg 10d trials 1–5, interleaved
  - PID: 2898148 (bash wrapper)
  - Log: `/tmp/scdfv2_serial.log` (milestones) + `/tmp/scdfv2_serial_logs/` (per-trial)
  - Script: `/tmp/run_scdfv2_serial.sh`

**Before relaunching:** delete partial model dirs so stages restart cleanly:
```bash
rm -rf /home/golobs/data/scMAMAMIA/ok/scdiffusion_v2/no_dp/10d/1/models/
rm -rf /home/golobs/data/scMAMAMIA/ok/scdiffusion_v2/no_dp/50d/1/models/
# (and any other partially-started trials)
```

**Relaunch with the same commands** (once v3 implementation is done):
- GPU0: `CUDA_VISIBLE_DEVICES=0 nohup conda run -n tabddpm_ python generate_trial.py ...`
- GPU1: `nohup bash /tmp/run_scdfv2_serial.sh ... &` (update script to remove classifier args)

---

## 6. CellTypist Notes

- Package already installed in `scdiff_` conda env (listed in scDiffusion requirements).
- `celltypist.train()` is fast (logistic regression, runs in minutes on a CPU).
- `celltypist.annotate()` is fast too — no GPU needed.
- CellTypist expects: AnnData with **log1p-normalized** expression.
  - Input from our pipeline: VAE decoder output is in log1p space.
  - Call `celltypist.annotate()` BEFORE `np.expm1()`.
  - Do NOT re-run `sc.pp.normalize_total()` + `sc.pp.log1p()` if data is already in
    that space — just pass it directly.
- The trained CellTypist model should be saved per-trial (since each trial has different
  D_train donors → different cell type frequencies, possibly).
- `majority_voting=False` is fine for per-cell annotation.

---

## 7. Reference Files

Original paper's scripts (read these for ground truth):
- Generation (unconditional): `/home/golobs/scDiffusion/cell_sample.py`
- CellTypist training: `/home/golobs/scDiffusion/celltypist_train.py`
- Classifier training (NOT used for main results): `/home/golobs/scDiffusion/classifier_train.py`
- Classifier sampling (NOT used for main results): `/home/golobs/scDiffusion/classifier_sample.py`
- Full train pipeline: `/home/golobs/scDiffusion/train.sh`
- Experiment reproduction: `/home/golobs/scDiffusion/exp_script/script_static_eval.ipynb`

Our current code (to be modified):
- `/home/golobs/scRNA-seq_privacy_audits/src/sdg/scdiffusion/run_scdiffusion_standalone.py`
- `/home/golobs/scRNA-seq_privacy_audits/src/sdg/scdiffusion/model.py`
- `/home/golobs/scRNA-seq_privacy_audits/experiments/sdg_comparison/generate_trial.py`
- `/home/golobs/scRNA-seq_privacy_audits/experiments/sdg_comparison/run_all.py`
