# Differential Privacy Experiments

**HIGH PRIORITY** — Experiment 1 in the paper revision (requested by all 5 reviewers).

## Goal

Add calibrated noise to scDesign2's Gaussian copula parameters at training time,
implementing a donor-level DP guarantee, then measure the privacy-utility tradeoff:

- Privacy: scMAMA-MIA AUC at each epsilon
- Utility: LISI, ARI, MMD on the resulting synthetic data

## Epsilon values to test

ε ∈ {0.1, 0.5, 1.0, 2.0, 5.0, 10.0}

## Implementation plan

The natural noise injection points are in `src/sdg/scdesign2/copula.py`:

1. **Covariance matrix** `R`: add calibrated Gaussian noise
   (`R_dp = R + N(0, σ²I)` where `σ = Δ/ε` and `Δ` is the sensitivity of `R`)
2. **Marginal parameters** (pi, theta, mu): add Laplace or Gaussian noise

The DP guarantee should be at the **donor level** (group privacy / donor-level
sensitivity), consistent with the threat model in the paper.

### Relevant prior work
- Dwork et al. — algorithmic foundations of DP
- Chaudhuri et al. — differentially private ERM
- Abadi et al. — DP-SGD

## Output

For each epsilon:
- scMAMA-MIA AUC (5-trial average ± std)
- LISI / ARI / MMD of the DP-synthetic data

Combined privacy-utility tradeoff curve (Figure for paper).
