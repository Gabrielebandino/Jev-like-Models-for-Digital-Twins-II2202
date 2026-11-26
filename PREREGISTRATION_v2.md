# Pre-Registration — NDT Policy-Selection Experiment v2 (II2202)

**Date:** 25 November 2026  
**Authors:** Gabriele Bandino (`bandino@kth.se`), Jago Revrenna (`revrenna@kth.se`)  
**Mentor:** Nadia Bekkali (KTH Royal Institute of Technology)  
**Compliance Base:** Approved Research Plan (`researchPlan5Nov/II2202-research-plan.tex`)  
**Engine & Notebook:** `ndt_v2_engine.py`, `ndt_v2_models.py`, `ndt_v2_kaggle_experiments.ipynb`

---

## 1. Pre-Registered Hypotheses

- **H0 (Headroom Verification):** On the primary Markov-modulated `regime_switching` workload (8-link shared-path topology) and external-validity `abilene_regime_switching` (SNDlib 12-node, 15-link OSPF failover), the hindsight counterfactual oracle switcher beats the best static policy ($\min(J_{\text{fixed}}, J_{\text{ca}})$) by strictly more than the pre-registered SESOI ($\delta = 0.10$ cost units). On the homogeneous `stable` control, headroom is below SESOI.
- **H1 (Threshold-Free Model Signal):** Across held-out reviews, a decision model's continuous probability $p(\text{change-aware})$ predicts the per-review oracle action better than chance ($\text{ROC-AUC} > 0.50$, 95% CI excluding 0.50). Evaluating ROC-AUC separates *absence of numerical signal* from *prior/threshold miscalibration*.
- **H2 (Primary Endpoint — Superiority or Equivalence vs. Strongest Deterministic Switcher):** Under strict information parity and zero prompt spoon-feeding, at equal monitoring cost, compare the pilot-frozen primary decision model against the strongest deterministic switcher (`rule_ewma_cusum`) on paired scalarized cost $J = \text{MAE} + \lambda \cdot (\text{KB}/N)$ and $\Delta\text{MAE}_{\text{iso-bytes}}$ (interpolated against the dense 15-point static fixed frontier):
  - **Superiority:** two-sided paired Wilcoxon signed-rank test, Holm-Bonferroni corrected across regimes ($\alpha = 0.05$).
  - **Equivalence (TOST, Lakens 2017):** Two One-Sided Tests within $[-\text{SESOI}, +\text{SESOI}]$ ($\text{SESOI} = 0.10$ cost units, $\alpha = 0.05$).
- **H3 (Practical Benefit under Operational Overhead):** Any gain observed under H2 remains positive after applying live GPU inference latency causally in-loop (`t_eff = t_review + rtt`), enforcing the 2000 ms deadline fallback, and evaluating elevated measurement-delivery delay (500 ms).

---

## 2. Primary & Secondary Factors

- **Primary Topology (Approved Plan §2.2):** 8-link shared-path backbone (`L0..L7`, 6 multi-hop OD flows).
- **Primary Switching Workload:** `regime_switching` (480 s, 30 s warm-up, 10 s review cadence, realistic ISP Markov transitions among `quiet` [45%], `macro_volatile` [25%], `localized_flap` [15%], `jitter_microburst` [15%]) × `{low_delay (50 ms), elevated_delay (500 ms)}`.
- **Homogeneous Controls (Approved Plan §2.2):** `stable`, `bursty`, `alternating`, `shifted` (240 s) × `{low_delay, elevated_delay}`.
- **External-Validity Extension (Mentor Amendment):** SNDlib Abilene (12 US backbone nodes, 15 bidirectional links, Dijkstra OSPF shortest-path routing with real single-link `IPLS-KSCY` failover + Pareto ON/OFF bursts), plus optional real SNDlib Abilene 5-min traffic matrices (explicitly documented as time-rescaled).
- **Primary Four Controllers:** `fixed@2.5s`, rate-normalized `change_aware`, deterministic switchers (`rule_simple` with change+AoI and `rule_ewma_cusum` with EWMA/CUSUM+oscillation/Gini+hysteresis), and one pilot-frozen primary `model_switcher`.
- **Pre-Registered Secondary Model-Sensitivity Backbones:** Evaluate 3 architectures sequentially in isolation:
  1. `laya` (`convaiinnovations/laya`, 421M ModernBERT System-1 encoder)
  2. `clm_8b` (`Contrastive-LM/CLM-v0.1-8B`, 4-bit Qwen3-8B + InfoNCE projection heads)
  3. `qwen25_7b` (`Qwen/Qwen2.5-7B-Instruct`, 4-bit NF4 single-pass exact A/B logit scoring — both zero-shot and training-free 4-shot pilot in-context examples).
- **Diagnostic Reference Points (not competing controllers):** Dense 15-point `fixed` frontier, hindsight `oracle`, `random` switcher floor, and pilot-trained `learnability_ceiling` (GBDT on pilot features/oracle labels to establish whether observable features carry sufficient signal).

---

## 3. Equal Tuning Budget, Pilot Freeze & Sample Size

- **Strict Seed Separation:** Pilot seeds `[101..105]`; Held-out seeds `[2001..2050]` (primary $N = 50$ for baselines + fast primary model, justified by pilot power analysis targeting 80% power at $\text{SESOI} = 0.10$; paired subset $N = 15\text{--}20$ for slower 8B secondary sensitivity models; $N = 15\text{--}20$ for controls & Abilene).
- **Equal Pilot Budget ($k = 20$ evaluations each):** Rate-normalized `change_aware`, `rule_simple`, and `rule_ewma_cusum` each receive an identical 20-candidate pilot grid search. Every decision model receives a 4-paraphrase neutral prompt screen (including option-order swap) + Zhao et al. (ICML 2021) training-free contextual calibration frozen strictly on pilot data.
- **Zero Prompt Spoon-Feeding:** No `if/else` regime classifications or decision hints are ever injected into any model prompt. Both rules and models receive the exact same causal temporal/spatial telemetry and monotone pilot-percentile scale context.
- **Reporting Commitment:** All regimes, all models, null/negative outcomes, raw vs. calibrated behavior, paraphrase spread, fallback rates, P50/P95 latencies, and GPU VRAM footprints will be reported transparently.
