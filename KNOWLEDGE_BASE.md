# II2202 Project Knowledge Base

**Last updated:** 25 November 2026  
**Status:** active working record; includes verified mentor approval, methodological audit of Nov 5–25 runs, and the Master's-level recovery plan (`RESEARCH_RECOVERY_PLAN.md`).

## What changed compared with the previous version

The older [`II2202_Project_Knowledge_Base_2026-10-06.md`](II2202_Project_Knowledge_Base_2026-10-06.md) is historical. Its central design was Jev-specific and, at earlier points, described Jev selecting links. The current direction is broader and different: one **general decision model** acts as a controller that selects between two polling policies. Jev and an open-weight candidate such as Laya are possible implementations; the study will use one feasible model after a pilot. The 29 October qualitative-exercise source already reflects this broader formulation.

## 1. Project at a glance

### Verified current framing

The project concerns **state synchronization in a network digital twin (NDT)**. The twin stores the latest received utilization measurement for a fixed set of network links. That stored state may be inaccurate while the real network changes.

The experiment asks whether a decision model can decide which of two existing polling policies should run during the next period:

- **Fixed-interval polling:** poll at a constant interval.
- **Change-aware polling:** adapt the interval from changes already observed in received measurements.

At scheduled review points, the model receives a structured summary of information already available to the twin and selects one of the two policies. Ordinary local code calculates the summary and executes the chosen policy.

### Research question, working version

Under which tested traffic conditions, if any, does a decision-model-based policy selector improve the trade-off between twin-state error and monitoring traffic, compared with the two policies used alone and a simple rule-based switcher?

Supporting question: how do delivery delay, decision latency, and model/API communication affect any apparent benefit?

The exact final wording must be taken from the current Overleaf research plan before submission or reuse.

### Expected contribution

This is not a literature review as the primary output. The planned result is a reproducible, scenario-by-scenario comparison of four controllers on the same simulated workloads. It should show where a selector helps, where a conventional policy or simple rule is sufficient, and the selector's extra operational cost. A negative result is still a valid outcome.

## 2. Decisions already taken

| Decision | Current position | Why / boundary |
|---|---|---|
| Unit of decision | Select a polling **policy**, not a link. | Keeps the study focused on when to observe, not on routing, link prioritisation, or topology control. |
| Poll contents | Every poll samples the same fixed link set. | The controllers differ in schedule/policy only. |
| Controller candidates | Fixed only; change-aware only; deterministic rule switcher; one decision-model switcher. | Separates any value of switching from any additional value of using a model. |
| Model choice | General decision-model framing. Jev and an open-weight model such as Laya are candidates. | Avoids making the report oddly specific to Jev while retaining Jev references where they document its interface. |
| Model selection | Pilot selects one feasible model, then model/version/interface are frozen. | This is not a benchmark between Jev, Laya, or all decision models. |
| Data and environment | Synthetic, trace-driven simulation with known ground truth. | Enables controlled paired comparisons and avoids private telemetry or live-network risk. |
| Evaluation focus | Twin-state error versus monitoring traffic; record decision latency and model/API overhead. | More polling must not be mistaken for a better decision simply because it costs more. |
| Financial constraint | No personal expenditure. | Confirm access, credits/quotas, request volume and expected charges before main runs. |

## 3. What the decision model does and does not do

### It does

- At a review point, choose `fixed-interval` or `change-aware` for the next period.
- Use only summaries of received measurements, their timestamps/ages, observed changes, and recent polling activity.
- Produce a decision whose latency, failures, fallback behaviour, and communication overhead are part of the evaluation.

### It does not

- Choose which individual link to poll.
- Decide whether a given individual poll is push or pull.
- See the unmeasured current network state, future traffic, or evaluator ground truth.
- Create a new numeric interval, predict utilization, train/fine-tune itself, or control a live network.
- Establish that decision models generally work better: one selected model gives evidence only for the tested setup.

## 4. Planned experimental design

### Conditions and data

The planned workload families are stable traffic, short bursts, alternating quiet/volatile periods, and persistent load shifts. Delivery delay is also varied. The topology and set of monitored links are fixed during the main comparison. Numerical scenario parameters, seed count, review cadence, thresholds, timeouts, and trace duration are **da confermare** through the pilot and final research plan.

### Fair comparison rules

- Replay the same synthetic trace/seed across controllers.
- The evaluator sees ground truth only to calculate error; controllers see only data received by the twin.
- **Zero prompt spoon-feeding:** The decision model receives **strictly raw, neutral numerical telemetry** (timestamps, AoI, poll counts, observed deltas, per-link utilizations/deltas) with **no pre-computed `if/else` regime labels or hints** telling it which policy to pick.
- **Strict information parity:** Never withhold a summary feature from the rule baseline while giving it (or a pre-labeled version of it) to the decision model.
- **Audit note (24 Nov 2026):** Earlier Nov 5–24 experimental runs (`logs/heldout_runs/` and `logs/sweet_spot_live_laya/`) contained pre-classified `if/else` labels inside `to_prompt_text()` / `build_multinode_state_text()`. Those runs are superseded by the new unbiased protocol in `ndt_fair_unbiased_kaggle_experiments.ipynb` and `ndt_sync/selectors.py`.
- Freeze policy settings, model choice/version, request format, selector timing, and analysis choices before held-out main runs.
- Keep the previous policy active while a model decision is pending.
- Record fallback if a decision is late or invalid; do not turn this into a hidden successful choice.
- Do not add an extra poll solely because a switch occurs.

### Measures

Primary measures:

- Mean absolute difference between the true and stored twin state.
- Application-level monitoring bytes (and/or polling traffic as defined in the final protocol).

Supporting measures:

- Poll count and change-detection delay.
- Model decision latency, model/API bytes, failures, and fallback frequency.
- Any provider-reported usage/cost, if available and safe to record.

The study does **not** directly measure network-wide wire traffic, electricity use, or CO2 emissions. Those may be discussed only as possible implications, not measured findings.

## 5. People and roles

| Person | Role | Status |
|---|---|---|
| Jago Revrenna (`revrenna@kth.se`) | Co-author and student. | Verified in project documents. |
| Gabriele Bandino (`bandino@kth.se`) | Co-author and student. | Verified in project documents. |
| Nadia Bekkali | Mentor teacher. | Verified in feedback and project documents. |

An older plan proposed Jago leading simulation/conventional policies and Gabriele leading model integration/logging. This division is **historical working allocation, da confermare**, not a current binding split. Both authors should understand and verify the whole experiment.

## 6. Mentor feedback and how the current design responds

Nadia's written feedback on the initial proposal asked for:

1. A direct statement of the concrete contribution beyond summarising papers.
2. Research questions organised around approaches, network conditions, and metrics.
3. A clear account of the evidence used to call an approach suitable for a condition.

### Approval of Revised Research Plan (13 November 2026)
Nadia Bekkali reviewed the revised research plan (`researchPlan5Nov/II2202-research-plan.tex`) and confirmed:
- *"It is now much clearer and more rigorous in my opinion. Overall, the plan looks good."*
- **One small clarification requested (for reporting & presentation):** Section 2.2 mentioned three candidate fixed intervals (`1.0 s`, `2.5 s`, `5.0 s`), while Section 2.3 referred to one selected interval. We clarified both §2.2 and §2.3 in `researchPlan5Nov/II2202-research-plan.tex` so it is explicit that the pilot evaluates the three candidates and freezes **one** representative interval (`2.5 s`, matching `ca_init_interval_s`) for the primary 4-controller comparison, while the others serve only as static Pareto frontier anchors.

## 6b. Experimental Audit & Master's-Level Recovery Plan (25 November 2026)

A rigorous self-audit of our Nov 5–24 runs (`logs/heldout_runs/`, `logs/sweet_spot_live_laya/`) identified **six methodological flaws**:
1. **Prompt spoon-feeding (invalidates earlier model-switching claims):** Both `ndt_sync/selectors.py` and `run_sweet_spot_live_laya.py` ran a Python `if/else` over telemetry numbers and injected pre-computed text labels (`HIGH_VOLATILITY_OR_BURST -> requires change-aware`) into the prompt.
2. **Class collapse on raw numerical telemetry (1,344-call unbiased GPU ablation, 25 Nov 2026):** After removing all `if/else` hints (`ndt_fair_unbiased_kaggle_experiments.ipynb`, Tesla T4 CUDA, `0.0%` fallback rate, `~48 ms` flat $O(1)$ latency across 8–64 links), `convaiinnovations/laya` (421M ModernBERT) collapsed **100% onto `'fixed'` (`change_aware_frac = 0.000`)** across all 8 regimes, 2 delays, and 4 scales—empirically confirming TypeSafe's documented numerical-precision limitation (`\cite{limits}`).
3. **Orderless scalar flattening:** Compressing the 15-second lookback window into unordered `mean`/`max`/`std` scalars threw away chronological time-series order, EWMA momentum, trend slope, and oscillation sign-reversals.
4. **Strawman deterministic baseline:** Comparing against a trivial 1-threshold rule rather than a standard networking EWMA + hysteresis controller.
5. **Toy sine waves & arbitrary link array:** Phase 2 used ad-hoc sine waves instead of standard networking topologies and traffic models.
6. **Non-overlapping Pareto comparisons:** Earlier runs lacked an iso-budget metric ($\Delta\text{MAE}_{\text{iso-bytes}}$) interpolated against the continuous static `fixed` frontier.

### Active Redesign (`RESEARCH_RECOVERY_PLAN.md`)
We drafted **`RESEARCH_RECOVERY_PLAN.md`** (currently queued for critical peer review by Claude & Jago) to rebuild the experiments on real networking standards while remaining 100% compliant with Nadia's approved plan:
- **Standards:** IRTF NMRG NDT (`draft-irtf-nmrg-network-digital-twin-arch-13`), **IETF RFC 9232** (*Network Telemetry Framework*), **IETF RFC 8641** (*YANG-Push*: `periodic` vs. dampened `on-change`), **IETF RFC 6298** (*Jacobson–Karels EWMA*: $\alpha=1/8, \beta=1/4$), **SNDlib Abilene** (12 nodes, 15 bidirectional links, OSPF shortest-path routing), and canonical **Pareto ON/OFF + MMPP + OSPF failover** workloads.
- **Temporal features & non-strawman baseline:** Ordered sliding-window time series, RFC 6298 fast/slow EWMA + deviation + momentum ratio, linear trend slope, signed-change reversal rate, and **`rule_ewma_hysteresis`**.
- **Model scale ablation:** Evaluate **both** `laya` (421M) and an **8B backbone** (`clm_8b` / `Qwen2.5-7B-Instruct` 4-bit) with **zero prompt spoon-feeding** on raw temporal time series, evaluated via **$\Delta\text{MAE}_{\text{iso-bytes}}$**, 95% bootstrap CIs, and paired Wilcoxon tests ($n = 15$ held-out seeds).

## 7. Source map and precedence

| Source | Modified / status | Use it for | Limitation |
|---|---|---|---|
| Shared GitHub repo (`Gabrielebandino/Jev-like-Models-for-Digital-Twins-II2202`, `@jagorev` invited) | Active (`main`); authoritative for code, traces, logs, and recovery plan. | All code, notebooks, QA checks, unbiased logs, and documentation. | Synced with local folder; check `git status` before pushing. |
| `researchPlan5Nov/II2202-research-plan.tex` | 13 Nov 2026; approved by Nadia Bekkali and updated with §2.2–2.3 fixed-interval clarification. | Approved scope, RQs, method, quality checks, and bibliography. | Must be synced to Overleaf if re-compiling PDF. |
| `RESEARCH_RECOVERY_PLAN.md` | 25 Nov 2026; active redesign draft for Claude & Jago peer review. | Next-generation standards-grounded experiment design (RFC 9232/8641/6298, SNDlib Abilene, EWMA, 8B ablation). | Pending peer review before coding v2. |
| `RESEARCH_WORKLOADS_AND_REASONING.md` | 25 Nov 2026; chronicle including Phase 3 unbiased GPU ablation (`58710af`). | Historical audit trail + 1,344-call unbiased Laya class-collapse results (`logs/fair_unbiased_laya/`). | Earlier Phase 1/2 sections document superseded spoon-fed runs (flagged explicitly). |
| `ndt_sync/selectors.py` & `ndt_fair_unbiased_kaggle_experiments.ipynb` | 24–25 Nov 2026; cleaned of all prompt spoon-feeding; all 6 QA checks passing. | Verified unbiased harness & fair Pareto evaluation plots. | Uses flat 15s stats; to be upgraded with RFC 6298 EWMA + ordered time series in v2. |
| Current Overleaf research-plan project | Live; authoritative but not stored in this folder. | Final wording, tables, dates, and submission-ready content. | Must be opened before edits; local legacy source is not a substitute. |
| `/Users/jagorev/Downloads/II2202_Research_Plan_Updated.pdf` | 29 Oct 2026; newest research-plan artifact located. | Latest rendered plan and final content check. | PDF is outside this folder; editable Overleaf source remains the source of truth. |
| `II2202_Qualitative_Method_Exercise/main.tex` | 29 Oct 2026. | Latest local, editable confirmation of the general decision-model scope and the four-controller study. | Created for a qualitative-method exercise, not the main experimental protocol. |
| `II2202_Qualitative_Method_Exercise_Bandino_Revrenna (1).pdf` | 30 Oct 2026. | Submitted/rendered qualitative exercise. | Same scope limitation as its source. |
| `II2202_Project_Proposal_Bandino_Revrenna_Updated.pdf` | 7 Oct 2026. | Historical revised project proposal. | Earlier than the general-model update; do not use to narrow the project to Jev alone. |
| `II2202_Project_Proposal_Jev/II2202-proposal.tex` | 6 Oct 2026. | Citation history and earlier Jev-specific rationale. | Superseded in scope. |
| `II2202_Research_Plan_Jev/II2202-research-plan.tex` | 6 Oct 2026. | Earlier detailed protocol and risk checklist. | Superseded in scope; do not edit or submit as current plan. |
| `II2202_Project_Proposal_Jev/teacher-feedback-notes.md` | 2 Oct 2026. | Paraphrased record of the mentor conversation and original feedback. | Its side-conversation scope says Jev selects individual links, which conflicts with the later policy-selection design; it is historical only and not a quotation of Nadia. |
| `II2202_Project_Knowledge_Base_2026-10-06.md` | 6 Oct 2026. | History only. | Explicitly superseded by this file. |
| `II2202_full_course_and_project_reference_2026-09-15.docx`, `II2202_citation_check_2026-09-15.md`, and `II2202_verifica_fonti_e_metodologia_2026-09-16.docx` | September 2026. | Course and citation history. | Review-era scope and calendar; each citation and date must be rechecked before reuse. |
| `II2202_Video_Presentation/` | 6 Oct 2026. | Video script and slide-planning history. | Jev-specific wording; update before reuse. |

## 8. Sources and citation boundaries

Earlier drafts cite NDT architecture material, OpenNetMon, monitoring-frequency work, Age of Information material, and TypeSafe/Jev documentation. Use primary research papers for scientific claims. TypeSafe documentation can establish what its interfaces claim to do, but cannot establish that Jev or any decision model improves NDT synchronization.

The project must not claim that a free tier, student credits, a particular price, API access, or a model capability exists unless checked from the provider at the time of use. Open-weight availability/licensing and the suitability of Laya are **da confermare**.

## 9. Course deliverables and dates

- Project proposal: a revised version has been prepared/uploaded according to the working history. Exact Canvas status and deadline: **da confermare**.
- Research plan: latest local rendered artifact dated 29 October 2026; exact Canvas submission status and deadline: **da confermare**.
- Video presentation: five slides, maximum ten minutes, both authors visible and speaking roughly equally. The local script/planning materials are historical and need general-model wording before reuse.
- Qualitative-method exercise: local LaTeX source and PDF exist; it uses the current general decision-model framing. Canvas submission status: **da confermare**.
- Peer feedback tasks and quiz completion: **da confermare** in Canvas.

Do not rely on old course-reference files as a calendar. Verify deadlines, upload status, and any resubmission requirements directly in Canvas.

## 10. Ethics, sustainability, and AI-use boundaries

No human participants, private operational traces, credentials in logs, or live network actions are planned. A future deployment could expose telemetry to an external API or lead operators to trust a stale twin. Relevant mitigations are synthetic inputs, no live control, timeouts/fallbacks, decision/staleness logs, and human review before real deployment.

Fewer polling messages may reduce communication work, but this does not prove lower energy use or emissions. A decision model can add API communication and external compute. Report what is measured and keep wider claims conditional.

AI can assist with language, coding, and source discovery, but Jago and Gabriele remain responsible for checking references, code, experiment design, and final prose. Do not represent simulated qualitative answers as human evidence.

## 11. Current risks and open points

1. **Model feasibility:** access, quota/credit, license, hardware requirements, latency, and zero personal cost must be established before main experiments.
2. **Choice of model:** choose one candidate after a documented pilot; do not change model after looking at held-out results.
3. **Protocol freeze:** define/freeze all numerical parameters, seed split, timeout, fallbacks, and analysis plan before main runs.
4. **Reproducibility:** preserve code, seed/configuration files, model identifier, inputs/decisions where permitted, logs, and analysis scripts.
5. **Scope drift:** avoid reverting to individual-link selection, a generic LLM comparison, a literature-only contribution, or unsupported energy claims.
6. **Document divergence:** update the live Overleaf research plan and any proposal/video reuse together; local sources in this folder are partly historical.

## 12. Immediate next actions

1. **Peer-review `RESEARCH_RECOVERY_PLAN.md`** with Claude and Jago (resolving the 5 review questions in §7: 8-link vs. Abilene primary topology, adding an 8B instruction model alongside `laya`/`clm_8b`, whether hybrid NOC text is in scope, and $\Delta\text{MAE}_{\text{iso-bytes}}$).
2. **Implement v2 standards-grounded simulator:** SNDlib Abilene + Pareto ON/OFF / MMPP / OSPF failover traces, ordered sliding-window + RFC 6298 EWMA features, and `rule_ewma_hysteresis`.
3. **Run pilot horizon & parameter freeze** ($H \in \{5, 10, 15, 30\}\text{ s}$, 5 pilot seeds) and archive `frozen_protocol_v2.json`.
4. **Run unbiased Kaggle T4 GPU evaluation** (zero prompt spoon-feeding) across `laya` (421M) and an 8B model on $n = 15$ held-out seeds.
5. **Draft final II2202 report & 5-slide video presentation** incorporating the controlled ablation (spoon-fed vs. unbiased raw numbers), RFC 9232/8641/6298 framing, and full citation verification.

## Da confermare

- Decisione finale dopo review di Claude & Jago su `RESEARCH_RECOVERY_PLAN.md` (topologia primaria Abilene vs. 8-link, inclusione di un modello 8B instruction-tuned, e inclusione o meno di Condition B con testo NOC).
- Parametri congelati nel pilot v2 (`frozen_protocol_v2.json`): orizzonte $H$, soglie EWMA/hysteresis, seed split ($n=5$ pilot, $n=15$ held-out).
- Stato di upload e scadenze ufficiali su Canvas per report finale, presentazione video, quiz e peer feedback.
