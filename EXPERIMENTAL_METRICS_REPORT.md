# Experimental Metrics Report: Decision-Model Policy Selection for NDT Synchronization (II2202)

**Authors:** Gabriele Bandino (`bandino@kth.se`) & Jago Revrenna (`revrenna@kth.se`)  
**Date:** 6 November 2026  
**Hardware:** Kaggle 2× NVIDIA Tesla T4 GPUs (16 GB VRAM each, Turing `sm_75`)  
**Repository:** [Gabrielebandino/Jev-like-Models-for-Digital-Twins-II2202](https://github.com/Gabrielebandino/Jev-like-Models-for-Digital-Twins-II2202)

---

## 1. Quality Assurance (QA) & Methodological Verification (§2.4)

Before running the pilot and held-out evaluations, all **6 mandatory verification checks** specified in Research Plan §2.4 were executed and logged in [`logs/qa_verification_report.json`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/qa_verification_report.json):

| Check # | Verification Invariant | Result | Measured Evidence |
| :---: | :--- | :---: | :--- |
| **1** | **Constant-State Input Test** | ✅ **PASS** | `MAE = 0.0000%` across `fixed`, `change_aware`, and `rule_switcher` |
| **2** | **Hand-Worked Error Calculation Test** | ✅ **PASS** | Simulator `MAE = 10.3960%` vs. Exact Analytical `10.3960%` (`|diff| = 0.0`) |
| **3** | **Policy Switching Without Extra Poll** | ✅ **PASS** | `2` policy switches executed; `0` spurious instantaneous polls dispatched |
| **4** | **Delayed & Out-of-Order Reply Rejection** | ✅ **PASS** | Stale poll (`t_sample = 1.0s` arriving at `4.0s`) rejected after fresher poll (`t_sample = 2.0s` arriving at `2.2s`) |
| **5** | **Timeout & Invalid-Output Fallback** | ✅ **PASS** | `3500 ms > 2000 ms` deadline & malformed JSON both retained previous policy (`fallback_rate = 1.0`) |
| **6** | **Absence of Future-State Leakage** | ✅ **PASS** | At review time `t = 15.0s`, unarrived poll (`t_arrival = 15.3s`) was completely excluded from selector summary |

---

## 2. Stage A — Pilot Feasibility Gate & 3-Model Speed Benchmark (§2.2, §3.1)

All three open-weight System-1 candidate models were evaluated sequentially in isolation on NVIDIA Tesla T4 (`16 GB` VRAM) across:
1. A **50-call NDT policy-selection benchmark** (measuring warm round-trip decision latency, parse rate, and VRAM footprint).
2. All **24 Pilot NDT trace simulations** (`4 regimes × 2 delays × 3 pilot seeds = 24 traces`, `11 reviews/trace = 264 reviews/model`).

Full raw data: [`logs/pilot_feasibility_screen_summary.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/pilot_feasibility_screen_summary.csv) and [`logs/pilot_freeze_decision.json`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/pilot_freeze_decision.json).

| Candidate Model | HF Checkpoint & Commit | T4 Precision | GPU VRAM Used | 50-Call Mean RTT | 50-Call Median (P50) | 50-Call P95 RTT | Parse Valid Rate | Pilot Fallback Rate | Pilot Policy Split (`Fixed` / `Change-Aware`) | Pilot Mean MAE (% pts) | Pilot Mean Traffic (Bytes) | 24-Trace Runtime |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`laya`** | `convaiinnovations/laya` (`7b928d828b`) | `float16` | **2,651 MB** | **47.19 ms** | 46.09 ms | **50.86 ms** | **100%** | **0.0%** | `27.1%` / `72.9%` | **2.5917%** | 15,768 B | **14.76 s** |
| **`kev_4b`** | `jaredpalmer/kev-4b` (`6cfce5c2fa`) | `bf16` / LoRA | 14,343 MB | 661.30 ms | 634.41 ms | 649.46 ms | **100%** | 4.55% | `29.0%` / `71.0%` | 2.6770% | 14,753 B | 550.68 s |
| **`clm_8b`** | `Contrastive-LM/CLM-v0.1-8B` (`e939398d45`) | `4bit-nf4` (`fp16`) | 5,099 MB | 15.04 ms *(cached)* / ~585 ms *(uncached)* | **5.00 ms** *(cached)* | 5.77 ms *(cached)* | **100%** | **0.0%** | `8.7%` / `91.3%` | 2.8097% | 15,835 B | 163.42 s |

> [!NOTE]
> **Key Technical Insight from the Pilot Screen:**
> - **`laya` (~421M ModernBERT-large)** has the most consistent **uncached** inference speed (**~49 ms** per review decision on dynamic traces, **51 ms P95**, **2.65 GB VRAM**, **0% fallbacks**) and finished all 24 pilot simulations in **14.8 seconds**.
> - **`clm_8b` (4-bit `Qwen3-8B` + Dual InfoNCE Projection Heads)** takes **~585 ms** when encoding a *brand-new* state vector through the 8B backbone, but drops to **5.0 ms** when state/action embeddings hit the contrastive vector cache!
> - **`kev_4b` (`Qwen3.5-4B` + LoRA + Pointer Head)** uses **14.3 GB VRAM** and takes **~1,900 ms** on dynamic 8-link numerical summaries, sitting right near the `2,000 ms` deadline (causing a `4.55%` timeout fallback rate in the pilot and `1.5%` in held-out alternating traces).

---

## 3. Stage B — Held-Out Main Experiment Results (96 Traces per Controller)

Each controller was evaluated on the **96 pre-generated Held-Out CSV traces** (`4 regimes × 2 delivery delays × 12 independent seeds`, `180s` duration, `20s` warm-up).
- Full summary table: [`logs/heldout_FULL_condition_summary_table.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/heldout_FULL_condition_summary_table.csv)
- Master run log (`768` total held-out runs): [`logs/heldout_runs/heldout_ALL_controllers_and_models_summary.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/heldout_runs/heldout_ALL_controllers_and_models_summary.csv)

### 3.1 Held-Out Performance Across All 4 Traffic Regimes (`low_delay` = 50 ms)

| Traffic Regime | Controller / Selector | Mean Twin MAE (% pts) ± SD | Monitoring Traffic (Bytes) | Poll Count | Detection Delay (s) | Selector Latency Mean (P95) | Fallback Rate |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Stable** | `fixed` (1.0s / **2.5s** / 5.0s) | 0.61% / **0.82% ± 0.03** / 0.98% | 26,680 / **10,696** / 5,368 | 181 / **73.0** / 37 | 0.00 s | — | 0.0% |
| **1. Stable** | `change_aware` (static) | 0.96% ± 0.05 | **6,725 B** | 45.5 | 0.00 s | — | 0.0% |
| **1. Stable** | `rule_switcher` (deterministic) | **0.82% ± 0.03** | 10,696 B | 73.0 | 0.00 s | 0.05 ms | 0.0% |
| **1. Stable** | `model: laya` | 0.93% ± 0.05 | **7,403 B** *(-30.8% vs Fixed)* | 50.1 | 0.00 s | **49.2 ms** (50.5 ms) | **0.0%** |
| **1. Stable** | `model: kev_4b` | 0.91% ± 0.04 | 7,773 B *(-27.3% vs Fixed)* | 52.6 | 0.00 s | 1,892 ms (1,899 ms) | **0.0%** |
| **1. Stable** | `model: clm_8b` | 0.94% ± 0.04 | **6,975 B** *(-34.8% vs Fixed)* | 47.3 | 0.00 s | 581.8 ms (586.6 ms) | **0.0%** |
| **2. Bursty** | `fixed` (1.0s / **2.5s** / 5.0s) | 1.72% / **3.61% ± 0.22** / 5.64% | 26,680 / **10,696** / 5,368 | 181 / **73.0** / 37 | 0.53 / **1.33** / 2.34 s | — | 0.0% |
| **2. Bursty** | `change_aware` (static) | 2.79% ± 0.43 | 22,003 B | 148.7 | 1.78 s | — | 0.0% |
| **2. Bursty** | `rule_switcher` (deterministic) | 2.88% ± 0.29 | 20,634 B | 139.4 | 1.53 s | 0.05 ms | 0.0% |
| **2. Bursty** | `model: laya` | 2.80% ± 0.43 | 21,818 B | 147.4 | 1.72 s | **50.0 ms** (52.2 ms) | **0.0%** |
| **2. Bursty** | `model: kev_4b` | **2.74% ± 0.38** | 22,105 B | 149.4 | 1.62 s | 1,908 ms (1,915 ms) | **0.0%** |
| **2. Bursty** | `model: clm_8b` | 2.87% ± 0.27 | 21,855 B | 147.7 | 1.84 s | 585.5 ms (593.2 ms) | **0.0%** |
| **3. Alternating** | `fixed` (1.0s / **2.5s** / 5.0s) | 3.99% / **5.76% ± 0.16** / 6.95% | 26,680 / **10,696** / 5,368 | 181 / **73.0** / 37 | 0.10 s | — | 0.0% |
| **3. Alternating** | `change_aware` (static) | 3.97% ± 0.24 | 24,482 B | 165.4 | 1.40 s | — | 0.0% |
| **3. Alternating** | `rule_switcher` (deterministic) | 3.91% ± 0.26 | 24,284 B | 164.1 | 0.77 s | 0.05 ms | 0.0% |
| **3. Alternating** | `model: laya` | 4.18% ± 0.15 | **22,681 B** *(-7.4% vs CA)* | 153.3 | 0.84 s | **49.0 ms** (51.2 ms) | **0.0%** |
| **3. Alternating** | `model: kev_4b` | 4.31% ± 0.15 | **21,682 B** *(-11.4% vs CA)* | 146.5 | 0.77 s | 1,901 ms (1,906 ms) | **0.0%** |
| **3. Alternating** | `model: clm_8b` | **3.71% ± 0.23** | 25,583 B | 172.9 | 1.00 s | 588.5 ms (595.8 ms) | **0.0%** |
| **4. Shifted** | `fixed` (1.0s / **2.5s** / 5.0s) | 0.89% / **1.43% ± 0.04** / 2.32% | 26,680 / **10,696** / 5,368 | 181 / **73.0** / 37 | 0.13 / **0.16** / 0.16 s | — | 0.0% |
| **4. Shifted** | `change_aware` (static) | 1.63% ± 0.29 | **7,181 B** | 48.6 | 2.48 s | — | 0.0% |
| **4. Shifted** | `rule_switcher` (deterministic) | **1.38% ± 0.07** | 10,210 B | 69.2 | 0.75 s | 0.05 ms | 0.0% |
| **4. Shifted** | `model: laya` | 1.67% ± 0.27 | **8,825 B** *(-17.5% vs Fixed)* | 59.8 | 1.85 s | **50.3 ms** (52.7 ms) | **0.0%** |
| **4. Shifted** | `model: kev_4b` | 1.63% ± 0.26 | **8,646 B** *(-19.2% vs Fixed)* | 58.4 | 2.16 s | 1,897 ms (1,905 ms) | **0.0%** |
| **4. Shifted** | `model: clm_8b` | 1.91% ± 0.35 | **7,560 B** *(-29.3% vs Fixed)* | 51.1 | 2.50 s | 583.1 ms (589.7 ms) | **0.0%** |

---

### 3.2 Held-Out Performance Under `elevated_delay` (500 ms Delivery Delay)

| Traffic Regime | `fixed (2.5s)` MAE / Bytes | `change_aware` MAE / Bytes | `rule_switcher` MAE / Bytes | `model: laya` MAE / Bytes | `model: kev_4b` MAE / Bytes | `model: clm_8b` MAE / Bytes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Stable** | 0.93% / 10,696 B | 1.04% / 6,725 B | 0.93% / 10,696 B | 1.01% / **7,197 B** | 0.99% / 7,851 B | 1.02% / **6,966 B** |
| **2. Bursty** | 4.67% / 10,696 B | 3.79% / 21,871 B | 3.81% / 21,112 B | 3.72% / 22,380 B | 3.66% / 22,750 B | **3.65%** / 22,932 B |
| **3. Alternating** | 6.33% / 10,696 B | 5.78% / 23,666 B | **5.54%** / 23,906 B | **5.60% / 22,772 B** | 5.68% / **21,239 B** *(1.5% fb)* | **5.55%** / 25,057 B |
| **4. Shifted** | 1.66% / 10,687 B | 1.87% / 7,188 B | **1.61%** / 10,127 B | 1.89% / **8,788 B** | 1.85% / **8,650 B** | 2.12% / **7,694 B** |

---

## 4. Visual Trade-Off & Workload Figures (`logs/mentor_preview_plots/`)

````carousel
![Held-Out Twin-State Error vs. Application-Level Monitoring Traffic Trade-Off](/Users/gab/.gemini/antigravity/brain/488cf700-17e8-4542-b1c8-302bfd7718cf/heldout_fig2_error_vs_traffic_tradeoff.png)
<!-- slide -->
![Synthetic 8-Link Network Digital Twin Workload Regimes](/Users/gab/.gemini/antigravity/brain/488cf700-17e8-4542-b1c8-302bfd7718cf/heldout_fig1_synthetic_traffic_regimes.png)
````

---

## 5. Paired Statistical Comparisons (`n = 12` Held-Out Seeds per Condition, 95% CIs)

Full table of 88 paired comparisons: [`logs/heldout_runs/heldout_ALL_models_paired_comparisons_ci95.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/heldout_runs/heldout_ALL_models_paired_comparisons_ci95.csv).

Key statistically significant findings (`95% CI` excluding zero):
1. **Alternating Regime (`elevated_delay`) — `Laya` vs. `Change-Aware` & `Rule Switcher`:**
   - **`model:laya` vs. `change_aware`:** Paired $\Delta\text{MAE} = -0.175\%$ (`95% CI: [-0.338%, -0.013%]`) **while simultaneously reducing monitoring traffic** by $\Delta\text{Bytes} = -894.7\text{ B}$ (`95% CI: [-1622.4, -167.0]`). Here, `Laya` strictly dominates static `change_aware` (both lower error AND lower monitoring bytes!).
   - **`model:laya` vs. `rule_switcher`:** Achieves statistically indistinguishable accuracy ($\Delta\text{MAE} = +0.064\%$, `95% CI: [-0.009%, +0.138%]` — includes zero) while saving **$-1,134.7\text{ B}$** (`95% CI: [-2022.5, -246.8]`) in monitoring traffic by switching back to `fixed (2.5s)` during volatile phases rather than over-polling at `0.5s` after every minor oscillation.
2. **Alternating Regime (`low_delay`) — `CLM-v0.1-8B` vs. `Change-Aware`:**
   - **`model:clm_8b` vs. `change_aware`:** Paired $\Delta\text{MAE} = -0.255\%$ (`95% CI: [-0.309%, -0.201%]`) at $+1,101\text{ B}$ (`+4.5%` traffic), achieving the lowest absolute twin error (`3.71%`) of any controller in alternating traffic.
3. **Stable & Shifted Regimes — Traffic Savings vs. `Fixed (2.5s)`:**
   - In `stable` (`low_delay`), switching models save **$27\%$ to $35\%$ of monitoring bytes** (`-3,293 B` for `Laya`, `-3,721 B` for `CLM-8B`) with only $+0.11\%$ to $+0.12\%$ MAE change (keeping MAE $< 0.95\%$).
   - However, in `shifted` traffic (sudden step load changes separated by long quiet periods), the simple **`rule_switcher`** achieves both lower error (`1.38%` vs. `1.43%` Fixed, `95% CI: [-0.094%, -0.009%]`) and lower traffic (`-486 B`), confirming your Research Plan's hypothesis (§1.3) that *"in some cases a simpler controller is sufficient."*

---

## 6. Repeat-Stability Analysis on Identical Traces (§2.4 & Table 2)

To separate **within-trace model variation** (from latency jitter / GPU scheduling) from **between-seed trace variation**, each model was re-run 3 times on the held-out repeat subset (`seeds 1001, 1002` across all regimes/delays = `48` repeat runs per model):
- [`heldout_repeat_stability_laya_run_summary.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/heldout_runs/heldout_repeat_stability_laya_run_summary.csv)
- [`heldout_repeat_stability_kev_4b_run_summary.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/heldout_runs/heldout_repeat_stability_kev_4b_run_summary.csv)
- [`heldout_repeat_stability_clm_8b_run_summary.csv`](file:///Users/gab/Documents/UNI/KTH/Research%20Methodologies/logs/heldout_runs/heldout_repeat_stability_clm_8b_run_summary.csv)

| Model | Within-Trace MAE Std Dev | Within-Trace Bytes Std Dev | Mean Decision Latency ± Within-Trace Jitter |
| :--- | :---: | :---: | :---: |
| **`clm_8b`** | **`0.0000%`** (100% deterministic) | **`0.0 B`** | `179.2 ms` (cached across repeats) |
| **`kev_4b`** | `0.0021%` | `33.5 B` | `1,902.3 ms ± 4.2 ms` (occasional tick boundary shift near 1.9s) |
| **`laya`** | `0.0786%` | `391.9 B` | `49.5 ms ± 0.3 ms` |
