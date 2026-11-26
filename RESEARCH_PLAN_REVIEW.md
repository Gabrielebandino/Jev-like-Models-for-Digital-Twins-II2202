# Critical Review of the Research Plan and `RESEARCH_RECOVERY_PLAN.md`

**Date:** 25 Nov 2026  
**Scope reviewed:** `researchPlan5Nov/II2202-research-plan.tex` (approved), `RESEARCH_RECOVERY_PLAN.md`, `KNOWLEDGE_BASE.md`, `ndt_sync/*.py`, `logs/heldout_FULL_condition_summary_table.csv`, `logs/pilot_freeze_decision.json`  
**Goal:** keep the approved topic and four-controller design, and make the analysis as rigorous and fair as possible.

---

## 0. Verdict in one paragraph

The approved plan is sound and honest. The recovery plan diagnoses real problems correctly: prompt spoon-feeding, a weak baseline, orderless features and toy workloads. Its fixes are mostly **cosmetic**, though. Citing more RFCs and adding an 8B model will not make the study rigorous. The real gap is that **the experiment currently has no way to tell whether a selector *could* help at all**. There is no oracle or headroom analysis. The change-aware arm is Pareto-dominated in some regimes. Most traces never change regime, so switching cannot matter by construction. And a null result cannot be distinguished from "no effect possible". The changes below fix that while staying inside the approved scope: a **headroom (oracle) analysis**, a **properly tuned change-aware arm**, **regime-switching traces**, **threshold-free diagnostics of the model's signal**, **equal tuning effort for every controller**, and **equivalence testing**. With these, a negative result becomes a strong, publishable finding rather than an inconclusive one.

---

## 1. What is already good (keep it)

- Narrow, testable claim: the model selects a policy; it does not predict state or pick links.
- Causality guard (`arrival_time_s <= review_time_s`), decision latency applied in the loop, single outstanding request, fallback keeps the previous policy. These are implemented correctly in `simulator.py`.
- Pilot/held-out seed separation and a pre-freeze.
- Honest self-audit: the spoon-feeding was found and reported. **Keep this audit in the final report.** The ablation "spoon-fed vs. raw" is itself a methodological contribution: it shows how easily LLM-in-the-loop networking results get inflated.

---

## 2. Critical issues, ranked by how much they threaten validity

### 2.1 No headroom/oracle analysis (most important)
The study asks whether the model improves on the alternatives, but never measures **the best any switcher could possibly do**. If the hindsight-optimal switching sequence beats the best static policy by only 0.05 pp, no selector can win, and the model's "failure" says nothing about models.

**Fix:** for every held-out trace, compute an **oracle switcher**:
- With *R* review points and 2 options there are 2^R switching sequences. At the current R = 12, that is **4,096 simulator runs per trace**, which is cheap and can be enumerated exactly. Because of path dependence (change-aware keeps internal state), only full enumeration is exact.
- For longer traces (see 2.3), use a **one-step counterfactual oracle**: at each review, fork the simulator state, run both policies over the next window, and label the one with lower cost. This approximates the optimum and also gives a **per-review "correct action" label**.
- Report, per regime, the **headroom** H = J(best static) − J(oracle) and the **fraction of headroom captured** by each switcher: (J(best static) − J(c)) / H.

This turns any result into an interpretable statement. "The model captured 3% of a 0.4 pp headroom, versus 61% for the EWMA rule" is a conclusion. "No significant difference" is not.

### 2.2 One of the two arms is dominated, so the experiment is rigged towards a null
I recomputed the iso-byte gap from the existing held-out table. I log-interpolated the 1.0/2.5/5.0 s fixed frontier. The baselines are unaffected by the spoon-feeding bug, so these numbers are valid.

| Regime | change_aware | threshold_rule | (spoon-fed) model switchers |
|---|---|---|---|
| alternating | −0.05 / −0.19 | −0.28 / −0.26 | −0.09 … −0.36 |
| bursty | **+0.50 / +0.67** | +0.46 / +0.63 | +0.45 … +0.73 |
| shifted | −0.30 / −0.32 | −0.12 / −0.11 | −0.08 … +0.04 |
| stable | +0.03 / +0.03 | 0.00 / 0.00 | +0.01 … +0.02 |

(ΔMAE in pp at equal bytes; elevated / low delay; negative = better than fixed)

- In **bursty** traffic, change-aware is clearly worse than simply polling faster at a fixed interval. Every switcher inherits this.
- In **shifted** traffic, change-aware alone beats every switcher, so switching only adds lag.
- Only **alternating** shows any switching benefit, and the simple rule already captures it.

**Root causes in the code** (`twin.py`, `ChangeAwarePolicy`):
1. The change signal is the raw delta between consecutive polls, which **grows with the polling interval itself** (≈√Δt for diffusive noise). Long intervals therefore look "volatile" and short intervals look "quiet", which pushes the policy to oscillate. Use a **rate-normalized** signal (Δ/√Δt or Δ/Δt), chosen in the pilot.
2. Its thresholds (4.5 / 1.8) were hand-set, not tuned against an objective.

**Fix:** in the pilot, **sweep the change-aware parameters** to trace its own Pareto frontier. Freeze the point on the lower convex hull at the byte budget of `fixed@2.5s`. The same tuning budget then goes to the rules and the model (see 2.6).

### 2.3 Traces are too short and mostly homogeneous
- 180 s with a 15 s review period gives **12 decisions per run**, which is too few to learn anything about switching behaviour.
- `stable`, `bursty` and `shifted` each keep one regime for the whole trace. **The best static policy is optimal by construction** there. Switching can only matter when the regime changes inside a trace.

**Fix:**
- Make the **primary workload a regime-switching (Markov-modulated) trace** that moves between quiet, bursty, level-shift and local-flap segments with random dwell times. Use 20–30 min traces, giving about 80–120 reviews. This stays within the approved plan's "alternating quiet and changing periods".
- Keep the four homogeneous regimes as **controls**, where the expected headroom is ≈ 0. They show that no selector does *harm*.

### 2.4 The model is tested unfairly in a way that is not spoon-feeding
Removing labels was correct. But the model now gets numbers with **no scale reference**. "max_abs_change = 4.3%" means nothing unless you know what is typical. The rule *does* have that knowledge, because its thresholds were tuned on pilot data. That is an information asymmetry *against* the model. Fair, non-leaking ways to remove it:

1. **Describe the policies completely.** The criteria text should state the change-aware thresholds and bounds. These are facts about the policy, not hints about which to choose.
2. **Pilot reference context.** Give percentiles of each feature from the pilot distribution (e.g. "mean change = 4.3 pp, 92nd percentile of pilot reviews"). This is a monotone transform with no policy label, so it is parity-preserving if the rule gets the same features.
3. **Contextual calibration** (Zhao et al., ICML 2021, *Calibrate Before Use*). Zero-shot choice models have strong prior biases towards one label. Laya's 100 % "fixed" output is consistent with that. Estimate the bias with a content-free input and correct it. This is training-free and must be frozen in the pilot.
4. **Prompt robustness.** Pre-register 3–5 paraphrases of the instructions and criteria, plus option-order swaps. Select one on pilot seeds and report the spread on held-out seeds. Single-prompt zero-shot results are not robust evidence.
5. *(Secondary, generative models only)* **Few-shot in-context examples** from *pilot* seeds, labelled by the oracle from 2.1. This mirrors exactly how the rule thresholds are calibrated on pilot data. It involves no weight updates, so the "no model training" rule holds.

### 2.5 Argmax decisions hide whether the model has *any* signal
"100 % fixed" can mean either "no information" or "information, but a badly placed decision boundary". These need to be told apart.

**Fix:** log `p(change-aware)` at every review (already done) and compute the **ROC-AUC of that score against the oracle's per-review correct action**. AUC ≈ 0.5 means no signal. AUC ≈ 0.8 with collapsed argmax means signal plus miscalibration. Report the same for the rule's internal score (e.g. EWMA_fast − EWMA_slow − 2·DEV). This is the most direct answer to "can a decision model play a part in the decision?"

Also add two **reference points** to bracket the model:
- a **random switcher** (p = 0.5, and p = the model's empirical rate), as a floor;
- a **learnability ceiling**: a small logistic-regression/GBDT trained on *pilot* features and oracle labels. **Report it as a diagnostic, not as a fifth controller.** It answers whether the information to make the decision is present in the features at all. If even a trained classifier can't beat the rule, the model never had a chance, and that is a finding.

### 2.6 Unequal tuning effort
The rule thresholds and the change-aware parameters were tuned; the model's prompt was not, or the reverse. It is well documented that unequal tuning budgets create spurious "wins" (Melis et al., ICLR 2018; Sculley et al., *Winner's Curse?*, ICLR-WS 2018).

**Fix:** pre-register an **equal tuning budget** (e.g. *k* = 20 pilot evaluations each) for:
- the rule thresholds;
- the change-aware parameters;
- the EWMA rule constants;
- the model's prompt/paraphrase plus calibration choice.

### 2.7 Statistics are underpowered, and a null result is uninterpretable
- Each model call takes ≈ 48 ms, and a run has about 12–120 calls. **Seeds are nearly free.** n = 15 is an arbitrary and unnecessary limit. Base n on a **power analysis** using the pilot SD of paired differences. With an expected SD of about 0.15 pp and δ = 0.10 pp, n ≈ 18 for superiority and ≈ 40–60 for equivalence. **Use n ≥ 50.**
- Pre-register a **smallest effect size of interest** (SESOI), e.g. 0.10 pp MAE or 10 % of headroom. Add **TOST equivalence tests**. "The model is *equivalent* to the rule within ±0.1 pp" is a positive, defensible claim. "Not significant" is not.
- Primary endpoint: **one** contrast per regime, *model vs. strongest non-model baseline*, with Holm correction across regimes. Everything else is secondary.
- Secondary: a mixed-effects model `cost ~ controller × regime × delay + (1|seed)`.

### 2.8 The iso-bytes interpolation is biased in your favour
The fixed frontier MAE(bytes) is convex. Linear interpolation between only 3–4 points gives a chord that lies **above** the true curve. That makes every controller look better than fixed.

**Fix:**
- Sweep fixed intervals densely (e.g. 15 log-spaced values from 0.5–10 s).
- Interpolate in log-bytes, paired per seed, and never extrapolate outside the range.
- Compare against the **lower convex hull of fixed ∪ tuned change-aware**, the true "best static" frontier.
- Also report a scalarized cost **J = MAE + λ·kB**, with λ set to the frontier slope at the reference operating point. It makes ranking unambiguous and is the same objective the oracle optimizes.

### 2.9 Protocol inconsistencies to fix and document
- `pilot_freeze_decision.json` froze **`clm_8b`**, but the unbiased runs used **`laya`**. The pilot was also run with the spoon-fed prompt, so **its selection is void**. Re-run the feasibility pilot under the unbiased protocol, freeze one model, and log the deviation.
- The approved plan says the rule uses "observed change **and age**". `rule_max_age_safety_s` is defined but never used, and the hard-coded `2.2` std threshold is not in the frozen protocol.
- The recovery plan's §6 lists 5 pilot / 15 held-out seeds, while `config.py` has 3 / 12. Pick one, justify it with the power analysis, and freeze it.

### 2.10 The standards grounding is partly decorative
- **RFC 8641 (YANG-Push periodic vs. on-change)** maps directly onto your two policies. Keep it.
- **gNMI** (OpenConfig) is an even better fit. Its `STREAM` subscriptions offer `SAMPLE` (periodic), `ON_CHANGE` and **`TARGET_DEFINED`**, where the device picks the mode. Your study is essentially asking *"what should `TARGET_DEFINED` do, and can a model do it?"* That makes a strong motivation sentence.
- **RFC 6298** is RTT estimation. Borrowing its α = 1/8, β = 1/4 for traffic volatility is arbitrary, and a reviewer will notice. Ground the strong baseline in **statistical change detection** instead: EWMA control charts (Roberts 1959; Lucas & Saccucci 1990) and **CUSUM / Page–Hinkley** (Page 1954). These are *the* canonical deterministic detectors. Tune their constants in the pilot.
- Missing theory that strengthens the framing: **sampling for remote estimation**. Sun, Polyanskiy & Uysal (IEEE T-IT 2020) prove that for a Wiener process over a random-delay channel, the MSE-optimal sampler is a **threshold policy on the estimation error**, which beats periodic and zero-wait sampling. This gives your change-aware arm a theoretical justification and predicts *when* adaptive beats periodic. Also cite Age of *Incorrect* Information (Maatouk et al., IEEE/ACM ToN 2020), which is closer to your MAE metric than AoI.

---

## 3. Answers to the five review questions in the recovery plan

1. **Abilene vs. 8-link:** keep the **8-link graph as the pre-registered primary** (the approved plan says "one small fixed topology, around eight links"). Add **Abilene (and real traces, §4.2) as a documented external-validity extension**, approved by Nadia via a short amendment.
2. **Add an 8B instruct model?** Yes, but as a **pre-registered secondary "model-sensitivity" analysis**, not a primary factor. The approved plan explicitly says model choice is *not* an experimental factor. The primary result uses the one model re-frozen in the unbiased pilot. The secondary analysis answers "is this just a small-model artefact?" Include a generative instruct model (e.g. Qwen2.5-7B-Instruct with constrained JSON output) **and** keep a System-1 choice model, since that is the class the plan is about.
3. **Condition B (NOC text):** **drop it from this study.**
   - It breaks the information parity in §2.1.
   - It is **circular**: you would write the text that implies the right answer. That is spoon-feeding again, just in natural language.
   - Mention it only as future work, ideally with real operator tickets.
4. **Iso-bytes metric:** the concept is sound, but the implementation is biased (see 2.8). Use a dense sweep, a convex hull, per-seed pairing and no extrapolation, and add the scalarized J.
5. **Missing standards:** add **gNMI subscription modes**, **IPFIX (RFC 7011)** only as the flow-export contrast in threats-to-validity, **SNMP polling** as the legacy pull baseline, and the remote-estimation and AoII theory above.

---

## 4. Proposed v2 design (rigorous, still within the approved scope)

### 4.1 Refined hypotheses (pre-register them)
- **H0 (headroom):** in regime-switching traces, the oracle switcher beats the best static policy by more than the SESOI. In homogeneous traces it does not.
- **H1 (signal):** the model's `p(change-aware)` predicts the oracle action better than chance (AUC > 0.5, CI excludes 0.5).
- **H2 (primary, approved RQ):** at equal monitoring cost, the model switcher's J is lower than (superiority) or equivalent to (TOST) the strongest deterministic switcher.
- **H3 (supporting RQ):** with decision latency, fallbacks and delivery delay included, any H2 benefit stays positive.

### 4.2 Data: synthetic plus real (document every source)
| Source | What it gives | Use | Caveats |
|---|---|---|---|
| **Synthetic 8-link** (current, upgraded) | Ground truth, controlled regimes | **Primary** | Synthetic |
| **Synthetic regime-switching** (MMPP over quiet/burst/shift/flap; Pareto ON/OFF sources) | Within-trace regime changes | **Primary workload** | Generator parameters set in the pilot |
| **Abilene topology + Abilene TMs** (Zhang; 5-min, 2004, via SNDlib) routed by OSPF | Real multi-link correlation structure | External validity | 5-min granularity, so it is a *time-rescaled* study (poll interval ↔ minutes). State this explicitly |
| **GÉANT TMs** (Uhlig et al., CCR 2006; 15-min, 23 nodes, via SNDlib) | Second real network | External validity | Same rescaling caveat |
| **MAWI samplepoint-F** (WIDE; daily 15-min pcaps, research use only) aggregated to 100 ms rates, flows hashed into K "member links" (LAG/ECMP model) | **Real sub-second burstiness** | External validity at native time scale | One physical link; the LAG split is a modelling choice |
| *(optional)* **CESNET-TimeSeries24** (Koumar et al., *Sci. Data* 2025; 40 weeks) | Long-term real drift | Robustness | Per-IP/institution series, not links; coarse |

Real traces are run through the **same simulator** (trace-driven), so the approved method is unchanged; only the trace source differs. Seed-equivalent on real data = **disjoint time windows**, split into pilot and held-out *before* anything is run.

### 4.3 Controllers
**Primary four (as approved):**
- `fixed@frozen`
- `change_aware` (tuned)
- deterministic switcher
- model switcher

The deterministic switcher is reported in two variants: **simple threshold** (approved) and **EWMA/CUSUM with hysteresis** (strong). The primary contrast is against the *stronger* of the two.

**References (not controllers):**
- `fixed` frontier sweep
- oracle switcher
- random switcher
- trained-classifier learnability ceiling

### 4.4 Shared feature vector (identical for rule and model)
- Ordered last-K polls (time, mean/max Δ, active-link fraction)
- Rate-normalized change
- EWMA fast/slow + deviation
- OLS slope
- Sign-reversal index
- Spatial concentration (Gini)
- AoI
- Poll counts
- Pilot percentile context

**No policy labels.**

### 4.5 Metrics
- **Primary:** MAE (pp) and monitoring bytes, giving ΔMAE_iso against the convex static frontier and J = MAE + λ·kB.
- **Secondary:**
  - P95 error
  - **congestion-event detection** (time for the twin to see a link cross 80 %; F1 of threshold crossings), which is the operational purpose of an NDT
  - AoI
  - detection delay
- **Selector diagnostics:**
  - AUC vs. oracle
  - headroom captured
  - switch rate / chattering
  - repeat-stability
  - latency P50/P95
  - fallback rate
  - VRAM and runtime

### 4.6 Statistics
- Unit = seed or real window, paired across controllers.
- n set by power analysis (≥ 50).
- BCa bootstrap CIs.
- Wilcoxon plus TOST with a pre-registered SESOI.
- Holm correction across regimes for the primary family.
- Mixed-effects model as secondary.
- **Report all regimes, including nulls.**

---

## 5. Compliance matrix (what needs Nadia's sign-off)

| Change | Status relative to the approved plan |
|---|---|
| Oracle/headroom, random switcher, AUC diagnostics | **Analysis addition**: no change to the design; strengthens §2.4 "evidence used to call an approach suitable" |
| Tuned change-aware, rate-normalized signal | **Within plan** ("thresholds and bounds chosen on pilot traces") |
| Regime-switching primary workload, longer traces | **Within plan** (alternating regime; durations "determined in pilot") |
| Stronger EWMA/CUSUM rule alongside the simple rule | **Within plan** (rule thresholds on observed change and age) |
| Re-run model pilot under the unbiased protocol | **Required repair**, document as a deviation (plan §3.3 anticipates "document the repair and refreeze") |
| Larger n, TOST, SESOI | **Within plan** ("feasible number of seeds"; uncertainty intervals) |
| Percentile context, calibration, prompt paraphrases | **Within plan** (request format "finalized in the pilot"); must be frozen |
| Secondary 8B / instruct model sensitivity | **Amendment**: plan says one model; present as a pre-registered secondary analysis |
| Abilene / GÉANT / MAWI real traces | **Amendment**: plan says synthetic and one topology; present as external validity |
| Trained-classifier learnability ceiling | **Amendment** (diagnostic only); justify that the *controller* is never trained |
| Condition B (NOC text) | **Drop** |

Send Nadia **one short amendment email** listing the "Amendment" rows before running held-out experiments, and archive her reply in the repo.

---

## 6. Documentation and pre-registration checklist
1. `PREREGISTRATION_v2.md`: hypotheses H0–H3, primary endpoint, SESOI, n with power calculation, exclusion rules, analysis code path. Commit it and **tag the git hash before any held-out run**.
2. `DEVIATIONS.md`: dated log of every change from the approved plan (the spoon-feeding audit, the void pilot, every row of §5) with the reason.
3. `DATA_SOURCES.md`: URL, version/date, licence/terms, preprocessing script, checksum and pilot/held-out split for every dataset.
4. `frozen_protocol_v2.json`: all parameters, model revision hash, prompt text, calibration constants and seed lists.
5. Raw logs: every prompt, reply and probability (JSONL), already done. Keep it.
6. One script reproduces every figure and table from the logs.

## 7. Suggested order of work
1. Fix change-aware (rate normalization) and run the parameter sweep, giving the convex static frontier. *(cheap, no GPU)*
2. Build the regime-switching generator and the oracle/headroom tool, then measure **headroom first**. If headroom is ~0 everywhere, redesign the workloads *before* touching models.
3. Implement the EWMA/CUSUM rule with an equal tuning budget.
4. Re-run the unbiased model pilot (calibration, paraphrases, percentile context), freeze it and write the pre-registration.
5. Run held-out experiments (synthetic primary, then real-trace external validity), then the secondary model sensitivity.
6. Analysis: headroom captured, AUC, ΔJ with TOST, operational overhead.

---

## 8. References to add (verify each before citing)
- Y. Sun, Y. Polyanskiy, E. Uysal, "Sampling of the Wiener Process for Remote Estimation Over a Channel With Random Delay," *IEEE T-IT* 66(2):1118–1135, 2020.
- A. Maatouk, S. Kriouile, M. Assaad, A. Ephremides, "The Age of Incorrect Information," *IEEE/ACM ToN*, 2020.
- E. S. Page, "Continuous Inspection Schemes," *Biometrika*, 1954 (CUSUM); J. M. Lucas, M. S. Saccucci, "EWMA Control Schemes," *Technometrics*, 1990.
- Z. Zhao et al., "Calibrate Before Use: Improving Few-Shot Performance of Language Models," *ICML* 2021.
- G. Melis, C. Dyer, P. Blunsom, "On the State of the Art of Evaluation in Neural Language Models," *ICLR* 2018.
- D. Lakens, "Equivalence Tests," *Social Psychological and Personality Science*, 2017 (TOST/SESOI).
- OpenConfig, *gNMI Specification* (subscription modes SAMPLE / ON_CHANGE / TARGET_DEFINED).
- IETF RFC 8641 (YANG-Push), RFC 9232 (Network Telemetry Framework), RFC 7011 (IPFIX).
- S. Uhlig et al., "Providing Public Intradomain Traffic Matrices to the Research Community," *ACM CCR*, 2006; SNDlib (Orlowski et al., *Networks*, 2010).
- MAWI Working Group Traffic Archive, WIDE Project (samplepoint-F).
- J. Koumar et al., "CESNET-TimeSeries24," *Scientific Data* 12:338, 2025.
- D. Wu et al., "NetLLM: Adapting Large Language Models for Networking," *ACM SIGCOMM* 2024 (motivation: off-the-shelf LLMs need adaptation for numeric networking tasks).
- P. Almasan et al., "Network Digital Twin: Context, Enabling Technologies, and Opportunities," *IEEE Commun. Mag.*, 2022.
