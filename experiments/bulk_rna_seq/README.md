# Bulk RNA-seq Experiments (CAMDA Track I)

**NOT IMPLEMENTED** — preserved from the CAMDA 2025 competition for potential future
adaptation of scMAMA-MIA to bulk RNA-seq generators.

The original CAMDA Track I materials (configs, example zips) are in the
`blue_team/` and `red_team/` subdirectories here.

## What's here

- `blue_team/` — generation pipeline configs for TCGA-BRCA and TCGA-COMBINED datasets
- `red_team/` — baseline MIA configs for bulk RNA-seq
- `CAMDA_TRACK_I_README.md` — original CAMDA Track I README

## If you want to adapt scMAMA-MIA to bulk RNA-seq

1. Bulk generators (CTGAN, CVAE, multivariate) are partially implemented in
   `src/generators/models/` (legacy location — not yet migrated to `src/sdg/`)
2. Baseline MIAs in `src/attacks/baselines/` were originally designed for bulk data
3. The data loader for bulk data is in `src/sdg/utils/prepare_data.py`
4. A new attack module would be needed in `src/attacks/` since there is no per-cell-type
   Gaussian copula to exploit
