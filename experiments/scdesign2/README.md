# scDesign2 MIA Experiments

scMAMA-MIA applied to scDesign2-generated synthetic scRNA-seq data.
This is the primary set of experiments in the current paper revision.

## Running

```bash
# Single experiment
python src/run_experiment.py configs/onek1k/scdesign2_100d_bb_aux.yaml

# Batch
./create_experiment_config_files.sh
cd src/ && ./run_donor_level_mia.sh
```

## Legacy CAMDA configs

`legacy_camda_configs/` contains the original CAMDA Track II generation, evaluation,
and privacy configs.  These use the old `sc_dist` generator and path structure;
they are kept for reference but are superseded by `configs/template.yaml`.

## Config parameters

See `configs/template.yaml` for all options.  Key settings:

| Parameter | Description |
|-----------|-------------|
| `mia_setting.num_donors` | Donor pool size {2, 5, 10, 20, 50, 100, 200} |
| `mia_setting.white_box` | `true` = WB, `false` = BB |
| `mia_setting.use_aux` | Whether attacker has auxiliary data |
| `mamamia_params.mahalanobis` | `true` = Mahalanobis attack (recommended) |

## Output

Each trial produces:
- `results/mamamia_results.csv` — AUC per threat model
- `results/mamamia_all_scores.csv` — per-cell membership scores
- `results/figures/` — ROC plots
