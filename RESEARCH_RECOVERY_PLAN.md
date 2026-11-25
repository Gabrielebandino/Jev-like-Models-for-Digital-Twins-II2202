# Research Recovery & Redesign Plan (II2202 — Master's Level)

**Project:** *Decision-Model Policy Selection for Network Digital Twin (NDT) Synchronization*  
**Authors:** Gabriele Bandino & Jago Revrenna (KTH Royal Institute of Technology)  
**Status:** Draft Recovery Plan for Critical Peer Review (Claude + Co-Author Review)  
**Constraint:** Must remain strictly compliant with Professor Nadia Bekkali's approved Research Plan (`researchPlan5Nov/II2202-research-plan.tex`) — trace-driven simulation, 4-controller comparison (`fixed`, `change-aware`, `deterministic rule`, `decision model`), pilot-freeze protocol, paired held-out seeds, zero personal expenditure, no model training, no live network control.

---

## 1. Brutally Honest Post-Mortem: What Was Wrong With Our Current Research

Before redesigning anything, we explicitly diagnose the **six methodological flaws** in our earlier runs:

1. **Prompt Spoon-Feeding (Fatal Validity Flaw in Earlier Runs):**  
   In both Phase 1 (`ndt_sync/selectors.py`) and Phase 2 (`run_sweet_spot_live_laya.py`), Python code ran an `if/else` over telemetry numbers and injected pre-computed text labels (`HIGH_VOLATILITY_OR_BURST -> requires change-aware`, `LOCALIZED_EDGE_NOISE -> use fixed`) directly into the model prompt. The model was merely parroting our own deterministic `if/else`.
2. **Class Collapse on Raw Numbers (Revealed by Our 1,344-Call Unbiased Ablation):**  
   When we stripped out those `if/else` hints and gave `convaiinnovations/laya` (421M ModernBERT encoder) strictly raw numbers across 1,344 live GPU calls (`fallback_rate = 0.0%`, `~48 ms` RTT), it selected `'fixed'` **100% of the time (`change_aware_frac = 0.000`)**—even during massive 64-link `+35%` bursts. Small zero-shot text encoders cannot perform numerical threshold arithmetic.
3. **Discarding Temporal Structure (Orderless Scalars Instead of Time Series):**  
   Our 15-second lookback window was flattened into static, orderless scalars (`mean`, `max`, `std`). Neither the rule baselines nor the model saw **chronological order**, **moving averages (EWMA)**, **trend direction (ramping up vs. decaying tail)**, or **oscillation sign-reversal rates**.
4. **Strawman Deterministic Baseline:**  
   Comparing a neural model against a naive 1-line scalar threshold (`if max_delta >= 6.5`) is not Master's-level networking research. Real network telemetry controllers use **EWMA**, **variance tracking**, and **hysteresis/dwell-time** to avoid chattering.
5. **Toy Workloads & Arbitrary Topology:**  
   Our Phase 2 workloads used synthetic sine waves (`18*sin(2*pi*t/3)`) and an ad-hoc flat link array instead of a standard networking benchmark topology (e.g., **SNDlib Abilene**) and canonical networking traffic models (heavy-tailed **Pareto ON/OFF** self-similar traffic, **MMPP**, and **OSPF link-failure reroutes**).
6. **Non-Overlapping Pareto Comparisons:**  
   Several earlier comparisons highlighted lower traffic at higher error (e.g., `74 KB / 2.30% MAE` vs. `173 KB / 1.36% MAE`) without measuring **iso-budget improvement** relative to the continuous static `fixed` Pareto frontier.

---

## 2. Grounding the Redesign in Networking Standards (IETF / IRTF / IEEE)

To elevate this from a toy simulation to a rigorous Master's-level networking study, every component will map directly to established standards:

| System Component | Networking Standard / Canonical Reference | Concrete Implementation in Our Simulator |
| :--- | :--- | :--- |
| **NDT Architecture** | **IRTF NMRG** `draft-irtf-nmrg-network-digital-twin-arch-13` | Isolates the *Data Collection / Synchronization* interface between the Physical Network and the NDT Unified Data Repository. |
| **Telemetry Framework** | **IETF RFC 9232** (*Network Telemetry Framework*, May 2022) | Formalizes telemetry data sources, collector, pull-based polling vs. adaptive collection, and application-level telemetry payload overhead. |
| **Policy Dichotomy** | **IETF RFC 8641** (*YANG-Push*, Nov 2019) | Connects our two policies directly to the standard `update-trigger` choice: **`periodic`** (`period = 2.5s`) vs. dampened **`on-change` / adaptive** (`dampening-period` bounded within `[0.5s, 6.0s]`, inspired by **OpenNetMon**, IEEE NOMS 2014). |
| **Temporal Filtering Baseline** | **IETF RFC 6298** (*TCP RTO / Jacobson–Karels EWMA*) | Uses the standard dual-parameter EWMA ($\alpha = 1/8 = 0.125$ for smoothed mean, $\beta = 1/4 = 0.25$ for mean deviation) plus a fast EWMA ($\alpha_f = 0.5$) and hysteresis dwell timer for our serious deterministic baseline. |
| **Freshness Metric** | **Age of Information (AoI)** (Yates et al., *IEEE JSAC* 2021) | Tracks exact AoI ($t - t_{\text{sample}}$) alongside ground-truth MAE (utilization percentage points) and RFC 9232 payload bytes. |
| **Benchmark Topology** | **SNDlib Abilene** (12 US backbone nodes, 15 bidirectional links, OSPF shortest-path routing) | Replaces the arbitrary 8-link array with a canonical, reproducible networking topology (while also supporting an 8-link subset or scaled 30-directed-arc view for scale sensitivity). |
| **Traffic Models** | **Self-Similar Pareto ON/OFF** (Willinger et al., *IEEE/ACM ToN* 1997) + **2-State MMPP** + **OSPF Failover Reroute** | Replaces toy sine waves with real networking traffic processes routed via shortest-path OSPF matrices, including a real **link-down failover event** (where traffic instantaneously reroutes onto backup paths!). |

---

## 3. Proper Temporal Representation & Horizon Design (Zero Leakage)

All features are derived **strictly from measurements already received by the twin** (`arrival_time_s <= review_time_s`).

### 3.1 Pilot-Calibrated Lookback Window & Decision Horizon
Instead of arbitrarily fixing the review period at `15.0s`, the **pilot phase** will run a controlled horizon sensitivity sweep across $H \in \{5\text{ s}, 10\text{ s}, 15\text{ s}, 30\text{ s}\}$ (keeping the winning $H$ frozen before held-out runs) to justify the trade-off between responsiveness and decision overhead.

### 3.2 Full Temporal Feature Vector (Shared Equally by Rules & Decision Models)
At each review point $t$, over the received poll sequence $\{(t_k, \mathbf{u}_k, \boldsymbol{\Delta}_k)\}$ in the lookback window $[t - W, t]$:

1. **Ordered Chronological Time Series (Last $K = 6$ Received Polls):**
   $$[(t_{k}, \bar{\Delta}_k, \Delta^{\max}_k, \text{active\_frac}_k)]_{k=1}^{K}$$
   Preserves exact temporal order (allowing detection of rising bursts vs. decaying tails).
2. **RFC 6298 Dual-Timescale EWMA & Deviation:**
   * Slow baseline volatility: $\text{EWMA}_{\text{slow}} \leftarrow (1 - \frac{1}{8})\text{EWMA}_{\text{slow}} + \frac{1}{8}\bar{\Delta}_k$
   * Volatility deviation: $\text{DEV} \leftarrow (1 - \frac{1}{4})\text{DEV} + \frac{1}{4}|\bar{\Delta}_k - \text{EWMA}_{\text{slow}}|$
   * Fast volatility: $\text{EWMA}_{\text{fast}} \leftarrow (1 - 0.5)\text{EWMA}_{\text{fast}} + 0.5\bar{\Delta}_k$
   * **Momentum Ratio:** $R_{\text{mom}} = \text{EWMA}_{\text{fast}} / (\text{EWMA}_{\text{slow}} + \epsilon)$ (spikes $> 1.5$ at burst onset; drops $< 0.7$ during decay).
3. **Temporal Trend Slope ($\frac{d\bar{\Delta}}{dt}$):** Ordinary least-squares slope of $\bar{\Delta}_k$ over the window (positive = accelerating turbulence; negative = settling).
4. **Signed-Change Reversal Rate (Oscillation Index $\in [0, 1]$):** Fraction of consecutive polls where $\text{sign}(\mathbf{u}_k - \mathbf{u}_{k-1})$ flips sign — directly separating **zero-mean periodic jitter** (high reversal rate $\approx 0.8\text{--}1.0$) from **sustained OSPF reroute shifts** (low reversal rate $\approx 0.0\text{--}0.2$).
5. **Spatial Concentration (Active-Link Fraction & Gini/Max-to-Mean Ratio):** Distinguishes isolated single-link flap from multi-hop backbone congestion.

---

## 4. Fair Controller Lineup & Scientific Experimental Design

To ensure **zero strawmen** and **zero prompt spoon-feeding**, we compare **5 controllers** on identical paired traces:

1. **`fixed` (Static Pareto Frontier):** Swept across `1.0s`, `2.5s` (pilot-frozen midpoint), `5.0s`, `10.0s` to form a continuous reference curve $\text{MAE}_{\text{fixed}}(\text{Bytes})$.
2. **`change_aware` (Static Adaptive Baseline):** OpenNetMon / RFC 8641 dampened adaptive polling bounded in `[0.5s, 6.0s]`, frozen in pilot.
3. **`rule_simple` (Naive Scalar Threshold):** Simple threshold on `max_delta` / `mean_delta` (represents a basic engineering heuristic).
4. **`rule_ewma_hysteresis` (Master's-Level Networking Baseline):** Uses RFC 6298 dual-EWMA + deviation bound ($\text{EWMA}_{\text{fast}} > \text{EWMA}_{\text{slow}} + 2\cdot\text{DEV}$), oscillation-index guard, spatial concentration check, and minimum 1-window hysteresis dwell time to prevent policy chattering.
5. **Decision-Model Selectors (Controlled 2-Axis Ablation — ZERO Spoon-Feeding):**
   * **Axis 1 — Model Architecture & Scale (on Kaggle T4, zero cost):**
     * **`laya` (421M ModernBERT System-1 encoder)** — already tested; collapsed on raw numbers.
     * **`clm_8b` (8B Qwen3 contrastive decision model)** or **`Qwen2.5-7B-Instruct` / `Llama-3.1-8B-Instruct` (4-bit structured JSON Choice)** — tests whether a larger 8B reasoning backbone can actually interpret raw chronological time-series + EWMA numbers where a 421M encoder failed!
   * **Axis 2 — Input Representation Ablation (The Core Scientific Contribution):**
     * **Condition A (Pure Raw Numeric Time Series — Strictly Fair):** Model receives the chronological time series + RFC 6298 EWMA numbers with **zero semantic pre-labels**. Answers: *Can a zero-shot decision model beat a proper RFC 6298 EWMA rule on numeric telemetry alone?*
     * **Condition B (Optional Extension — Hybrid Numeric + Unstructured Operational Context):** Add realistic short NOC (Network Operations Center) text events (e.g., *"Planned OSPF weight change on Atlanta–Houston; monitor convergence"* vs. *"Known faulty optical transceiver flapping on leaf port; ignore jitter"*) that a numeric rule cannot parse. Answers: *If pure numeric telemetry favors EWMA rules, does unstructured operational context create the regime where a decision model genuinely outperforms deterministic rules?*

---

## 5. Evaluation Metrics & Statistical Rigor (Professor Nadia-Compliant)

1. **Iso-Budget Pareto Metric ($\Delta\text{MAE}_{\text{iso-bytes}}$):**  
   Instead of comparing controllers at wildly different byte counts, we interpolate the static `fixed` frontier (`1.0s → 2.5s → 5.0s → 10.0s`) at each controller's exact monitoring-byte usage $B_c$ and compute:
   $$\Delta\text{MAE}_{\text{iso-bytes}}(c) = \text{MAE}(c) - \text{MAE}_{\text{fixed}}(B_c)$$
   Negative values prove **true Pareto improvement** (strictly better accuracy than static polling *at the exact same byte budget*); positive values prove Pareto inferiority.
2. **Paired Non-Parametric Statistics:**  
   $n = 15$ independent held-out seeds per condition (separated from $n = 5$ pilot seeds). Report paired differences with **95% bootstrap confidence intervals** and **paired Wilcoxon signed-rank $p$-values** (Holm–Bonferroni corrected across regimes).
3. **Operational Overhead Reported Separately (§2.4):**  
   * Monitoring traffic (RFC 9232 application-layer payload bytes)
   * Model decision latency (mean, P50, P95 ms — applied causally in-loop)
   * Fallback / timeout rate ($> 2000\text{ ms}$ or invalid parse)
   * Compute footprint (GPU VRAM MB, inference time per run)
   * Repeat-stability variance on identical traces (`3` repeats on held-out subset)

---

## 6. Concrete Implementation Roadmap

1. **Step 1 (Topology & Traffic Upgrade):** Implement `sndlib_abilene.py` (12 nodes, 15 bidirectional links, OSPF shortest-path routing matrix) + canonical Pareto ON/OFF, MMPP alternating, OSPF link-failure reroute, and localized transceiver-flap traces.
2. **Step 2 (Temporal Engine & RFC 6298 Baseline):** Implement `temporal_features.py` (ordered sliding window, RFC 6298 EWMA/DEV, trend slope, oscillation reversal index, spatial concentration) and `EWMAHysteresisRuleSelector`.
3. **Step 3 (Pilot Calibration & Freeze):** Run pilot on 5 seeds to freeze horizon $H$, fixed midpoint interval, change-aware bounds, and RFC 6298 rule constants. Archive `frozen_protocol_v2.json`.
4. **Step 4 (Unbiased Kaggle GPU Evaluation):** Run `laya` (421M) AND an 8B model (`clm_8b` / `Qwen2.5-7B-Instruct` 4-bit) on Kaggle T4 across all 15 held-out seeds with **zero prompt spoon-feeding**, logging every raw prompt/reply to JSONL.
5. **Step 5 (Pareto & Statistical Analysis):** Compute $\Delta\text{MAE}_{\text{iso-bytes}}$, 95% bootstrap CIs, Wilcoxon tests, and generate publication-ready figures for the final II2202 report.

---

## 7. Review Questions for Claude (Peer Critique Checklist)

When reviewing this plan, please critique ruthlessly on:
1. **Scope Alignment:** Does switching from the 8-link synthetic graph to the **SNDlib Abilene** (12-node, 15-link) topology and adding **RFC 6298 EWMA** stay within Professor Nadia's approved Research Plan (`II2202-research-plan.tex`), or should we keep the 8-link graph as Primary and Abilene as a Secondary validation?
2. **Model Choice:** Since `laya` (421M) collapsed to 100% `'fixed'` on raw numbers, should we evaluate an **8B instruction-tuned model (`Qwen2.5-7B-Instruct` 4-bit JSON Choice)** alongside `laya` and `clm_8b` so our negative/nuanced finding isn't dismissed as *"you only tested a tiny 421M BERT model"*?
3. **Condition B (Hybrid NOC Text + Telemetry):** Is adding unstructured NOC text alerts a smart Phase 2 extension, or does it violate Research Plan §2.1 (*"Both selectors receive the same types of summary"*) and risk looking like a post-hoc shift?
4. **Iso-Budget Pareto Metric ($\Delta\text{MAE}_{\text{iso-bytes}}$):** Is interpolating the static `fixed` curve (`1s / 2.5s / 5s / 10s`) mathematically sound for quantifying Pareto dominance across controllers with non-overlapping byte counts?
5. **Missing Networking Standards:** Are there any other IETF/IEEE telemetry or NDT standards (e.g., gNMI, SNMPv3, IPFIX/RFC 7011) that should be referenced or bounded in the threat-to-validity section?
