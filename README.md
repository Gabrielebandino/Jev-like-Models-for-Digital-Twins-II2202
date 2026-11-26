# Decision-Model Policy Selection for Network Digital Twin Synchronization (II2202)

**Authors:** Gabriele Bandino (`bandino@kth.se`) & Jago Revrenna (`revrenna@kth.se`)  
**Mentor:** Nadia Bekkali (KTH Royal Institute of Technology)  
**Course:** KTH II2202 — Research Methodology and Scientific Writing (Autumn 2026)

---

## 1. Overview & Scientific Scope

This repository implements the pre-registered, trace-driven **v2 experimental suite** for **Decision-Model Policy Selection for Network Digital Twin (NDT) Synchronization**, strictly compliant with the approved research plan ([`researchPlan5Nov/II2202-research-plan.tex`](researchPlan5Nov/II2202-research-plan.tex)) and documented amendments ([`PREREGISTRATION_v2.md`](PREREGISTRATION_v2.md), [`DEVIATIONS.md`](DEVIATIONS.md)).

At preset review points, a controller observes strictly causal measurements already received by the twin (`arrival_time_s <= review_time_s`) and chooses which polling policy runs during the next review window:
- **`fixed`** — Periodic fixed-interval polling (`2.5s` frozen midpoint; dense 15-point sweep `0.5s–10.0s` for unbiased static Pareto frontier interpolation).
- **`change-aware`** — Rate-normalized ($\Delta / \Delta t^{0.5}$) adaptive polling inspired by OpenNetMon and IETF RFC 8641 / OpenConfig gNMI.

### Controllers & Diagnostic References
- **Primary Four Controllers (§2.1):**
  1. `fixed` (static frozen midpoint + dense 15-point Pareto frontier)
  2. `change_aware` (rate-normalized, equal-budget $k=20$ pilot-tuned)
  3. Deterministic switchers: `rule_simple` (observed change + AoI) and `rule_ewma_cusum` (EWMA control chart + CUSUM + spatial Gini / oscillation reversal guards + dwell hysteresis, $k=20$ pilot-tuned)
  4. `model_switcher` (strictly neutral numerical telemetry, zero prompt spoon-feeding, 4 pre-registered paraphrases, monotone pilot-percentile scale context, and Zhao et al. ICML 2021 training-free contextual calibration)
- **Evaluated Open-Weight Decision Models (Sequential Kaggle T4 GPU Isolation, Zero Cost):**
  - `convaiinnovations/laya` (421M ModernBERT-large System-1, fp16, raw vs. Zhao-calibrated)
  - `Contrastive-LM/CLM-v0.1-8B` (4-bit NF4 Qwen3-8B + InfoNCE projection heads)
  - `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4, single-pass exact A/B logit scoring — Zero-Shot & training-free Pilot 4-Shot)
- **Diagnostic Reference Points (Not Competing Controllers):**
  - Hindsight counterfactual `oracle` (verified against exact $2^R$ enumeration) → **Headroom $H$** & **Fraction of Headroom Captured (%)**
  - `random` switcher floor
  - Pilot-trained `learnability_ceiling` (GBDT diagnostic establishing feature learnability)
  - Threshold-free **ROC-AUC & Average Precision** of $p(\text{change-aware})$ vs. per-review oracle labels

### Workloads & Topologies
- **Primary (Approved 8-Link Shared-Path Backbone):** Markov-modulated `regime_switching` (480 s: quiet, macro Pareto volatility, localized optical flap, aliasing jitter) × `{low_delay (50 ms), elevated_delay (500 ms)}`, plus the four homogeneous §2.2 controls (`stable`, `bursty`, `alternating`, `shifted`).
- **External Validity:** **SNDlib Abilene** (12 US backbone nodes, 15 bidirectional links, Dijkstra OSPF routing + real single-link `IPLS-KSCY` failover) + optional real SNDlib Abilene 5-min traffic matrices (explicitly time-rescaled).

---

## 2. Repository Structure

| Path | Role |
| :--- | :--- |
| `ndt_v2_kaggle_experiments.ipynb` | **Main self-contained Kaggle GPU T4 x2 notebook** (upload & Run All). |
| `ndt_v2_engine.py` | Verified v2 simulator, rate-normalized policies, oracle/headroom engine, QA suite (7 checks), and statistics (Wilcoxon+Holm, TOST, bootstrap CIs, ROC-AUC). |
| `ndt_v2_models.py` | Live GPU server managers, single-pass Qwen2.5-7B logit scorer, 4-paraphrase screen, and Zhao et al. contextual calibration. |
| `build_v2_notebook.py` | Reproducibly regenerates `ndt_v2_kaggle_experiments.ipynb` from the Python modules. |
| `researchPlan5Nov/II2202-research-plan.tex` | Mentor-approved Research Plan (13 Nov 2026). |
| `PREREGISTRATION_v2.md` | Pre-registered hypotheses (H0–H3), SESOI, sample size, and statistical plan. |
| `DEVIATIONS.md` | Transparent dated audit log of all v1→v2 repairs and mentor amendments. |
| `RESEARCH_PLAN_REVIEW.md` | Peer-critique document motivating the v2 redesign. |
| `KNOWLEDGE_BASE.md` | Project provenance, roles, ethics boundaries, and source map. |
| `logs/v2/` | Output directory for v2 Kaggle CSVs, JSONL review logs, frozen protocol, and 6 figures. |
| `archive/` | Clearly labeled superseded v1 audit trail & unbiased Laya collapse diagnostic (not primary evidence). |

---

## 3. Running Locally & on Kaggle

### Local QA & Pilot Verification (CPU)
```bash
python3 ndt_v2_engine.py
```

### Full GPU Experiment Suite on Kaggle (Zero Cost)
1. Upload `ndt_v2_kaggle_experiments.ipynb` to Kaggle.
2. Set **Accelerator → GPU T4 x2** and **Internet → ON**.
3. Set `RUN_PROFILE = "FULL"` (complete evaluation run, ~30–40 min) or `"QUICK_VERIFY"` (~6 min dry run).
4. Click **Run All** and download `/kaggle/working/ndt_v2_complete_outputs.zip`, then unzip into `logs/v2/`.
