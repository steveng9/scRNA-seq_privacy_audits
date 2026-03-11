# Baseline MIA Experiments

CAMDA2025 baseline MIA methods compared against scMAMA-MIA:
LOGAN, GAN-Leaks, GAN-Leaks (SC), DOMIAS+KDE, Monte Carlo.

## Running

Baselines require scMAMA-MIA trials to have already run (they reuse the same
donor splits and synthetic data).

```bash
cd src/mia/
python ../baseline_mias.py T <path-to-config>
# or for all configs:
./run_baseline_mias.sh
```

## Notes

These baselines fail to scale beyond ~20 donors on genomic data due to
O(n²) KDE distance computation.  See paper Section X for profiling details.
