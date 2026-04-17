# Class B Attack: Theory Notes for Paper Writing

These notes explain the statistical reasoning behind the Class B (secondary gene LLR)
enhancement to scMAMA-MIA. Written for use when drafting the paper methods section.

---

## What are "Class B" / secondary genes?

scDesign2 divides genes into two groups:

- **Group 2 (primary / "Class A")**: genes that exceed a non-zero entry threshold.
  For these genes, scDesign2 fits a **Gaussian copula** capturing pairwise gene-gene
  covariances in addition to marginal distributions. These are the genes already used
  by the Mahalanobis attack.

- **Group 1 (secondary / "Class B")**: genes that fall below the threshold — they are
  sparser or less variable. scDesign2 fits **only marginal distributions** for these
  genes (ZINB, NB, Poisson, or ZIP), with no covariance structure.

The standard Mahalanobis attack uses only group-2 genes. The Class B enhancement
exploits the *remaining* signal in group-1 genes.

---

## Why use log-likelihood ratio (LLR)?

For each secondary gene g, scDesign2 has fitted a marginal distribution with
parameters (π, θ, μ) from the **synthetic data** (or from the training copula in the
white-box setting). We also have a fitted marginal from the **auxiliary data**
(non-members).

For a target cell with count x_g at gene g, we can compute:

    log p_synth(x_g)   — log-probability under the synthetic/training model
    log p_aux(x_g)     — log-probability under the auxiliary/non-member model

The **log-likelihood ratio (LLR)** for gene g is:

    llr_g = log p_synth(x_g) − log p_aux(x_g)

A positive value means this observation is MORE consistent with the training
distribution than the non-member distribution — evidence of membership.

Summed across all secondary genes:

    LLR_B(cell) = Σ_g [ log p_synth(x_g) − log p_aux(x_g) ]

This is the **Neyman-Pearson most powerful test statistic** for distinguishing two
distributions when observations are independent. It is theoretically optimal — no other
function of the data can produce a more powerful test at the same significance level
(Neyman-Pearson lemma, 1933).

---

## Why is the independence assumption (approximately) justified?

The sum Σ_g [LLR_g] is the correct joint LLR **only if genes are independent**. If
genes are correlated, the true joint LLR would require the joint density, which we
don't have for group-1 genes (that's why they're not in the copula).

Three reasons the independence approximation is defensible here:

1. **Selection bias works in our favor.** scDesign2 puts the *most correlated* genes
   into the Gaussian copula (group 2). Group-1 genes are excluded *because* they are
   lower-correlation. So by construction, the genes we apply the independence
   approximation to are those most likely to be approximately independent.

2. **γ acts as a regularizer.** If some group-1 genes are weakly correlated with each
   other, summing their LLRs slightly overcounts correlated evidence. But this is
   partially cancelled when γ is small — the combined logit is:

       combined_logit = log(d_aux/d_synth) + γ · LLR_B

   A small γ downweights Class B relative to the primary Mahalanobis signal, so
   correlated noise doesn't dominate.

3. **Auto-normalization γ = 1/√n_secondary echoes naive Bayes scaling.** In naive
   Bayes classifiers, which also assume feature independence, the effective per-feature
   weight is 1/n. The 1/√n scaling is intermediate — stronger than 1/n but still
   shrinks as more correlated features are added.

**Bottom line for the paper:** The independence assumption is an approximation, justified
by the fact that group-1 genes are by definition the lower-correlation subset. We treat
it as a reasonable approximation and validate it empirically via the ablation study.

---

## How Class B combines with the Mahalanobis score

The primary attack computes Mahalanobis distances d_synth and d_aux in the Gaussian
copula space of group-2 genes. In log-odds space, the membership log-evidence from
this primary signal is:

    log_primary = log(d_aux) − log(d_synth)

(Large d_synth = far from synthetic distribution = non-member; large d_aux = far from
aux distribution = member evidence.)

The combined logit is:

    combined_logit = log(d_aux) − log(d_synth)  +  γ_eff · LLR_B

Both terms are additive in log-odds space. This is principled: under the assumption
that the primary (copula) and secondary (marginals) signals are independent, summing
log-evidence is exact (Bayes' rule for independent likelihoods).

The final cell score is obtained by z-scoring and applying sigmoid:
`activate_from_logits(combined_logit)`.

---

## Why Class B helps BB+aux but NOT BB-aux

In the BB+aux setting, we have:
- A synthetic copula from D_synth → gives p_synth for secondary genes
- An auxiliary copula from D_aux → gives p_aux for secondary genes
- LLR = log(p_synth / p_aux) is a meaningful ratio: positive = member-like

In the BB-aux (no auxiliary data) setting:
- We have p_synth but no p_aux
- Using `log p_synth` alone (without a reference) is much weaker: high log probability
  doesn't reliably discriminate members from non-members because zero-inflated genes
  assign high probability at zero for ALL cells regardless of membership

Pilot result (ok 10d trial 1):
- BB+aux: baseline AUC 0.770 → +0.140 with llr_sec_auto → AUC 0.910
- BB-aux: AUC *decreased* by 0.10–0.17 when class_b_gamma_noaux was nonzero

Therefore: `class_b_gamma_noaux` defaults to 0 (Class B disabled for BB-aux).

---

## Auxiliary data sampling: justification for 490d strategy

For experiments with ~490 training donors (the CAMDA-scale setup), the 490d donor
sampling strategy is:

    train = 490 donors (from target cohort)
    holdout = 490 donors (disjoint from train, from same cohort)
    aux = 200-donor subsample of holdout

**Potential reviewer concern:** "The experiment assumes the attacker knows who is a
non-member — they're sampling aux from known held-out donors."

**Response for the paper:** In the threat model we are evaluating, the auxiliary
dataset D_aux represents a publicly available reference cohort (e.g., data from
CZ CELLxGENE or another public atlas with similar tissue/cell types). The attacker
does NOT need to know which donors are in the training set; they only need access to
*some* relevant cell atlas data from the same tissue type.

In our experimental setup, we *approximate* this public atlas using held-out donors
from the same study. This is standard in MIA literature (e.g., Carlini et al., Shokri
et al.) and is the cleanest possible approximation: the held-out donors have the same
technical properties as the training donors but are guaranteed non-members.

Sampling aux from all targets (train ∪ holdout) would mix members and non-members into
the reference, weakening the attack and producing a *lower bound* on real-world
attacker capability — not a more realistic estimate.

**Key sentence for paper:** "The auxiliary dataset D_aux approximates a publicly
available cell atlas; in experiments, we use held-out donors from the same cohort as a
clean non-member reference, consistent with standard MIA evaluation practice."
