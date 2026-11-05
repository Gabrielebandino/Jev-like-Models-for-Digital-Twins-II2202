# II2202 Project Knowledge Base

**Last updated:** 3 November 2026  
**Status:** active working record; not a source of experimental results.

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

The current experimental framing addresses these with a four-controller comparison, explicit scenario/metric separation, paired repeated simulation runs, and a pre-specified comparison at comparable communication levels. This is a design response, not evidence that Nadia has formally approved the final version.

## 7. Source map and precedence

| Source | Modified / status | Use it for | Limitation |
|---|---|---|---|
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

1. Open the live Overleaf research plan and verify it uses the general decision-model wording consistently.
2. Select and document the feasibility-gate criteria before spending time on implementation.
3. Agree the pilot protocol and freeze conditions before collecting main-run results.
4. Check Canvas for status, deadlines, and required AI-use disclosure.
5. Before final submission, perform a citation-by-citation source check and verify every reference is actually cited.

## Da confermare

- Quale decision model verrà selezionato dopo il pilot, inclusi versione, licenza/accesso e costo effettivo pari a zero per gli studenti.
- Parametri finali del pilot e dei main runs: topologia, scenari, seed, intervalli, soglie, review cadence, timeout e metriche esatte.
- Stato di upload e scadenze ufficiali su Canvas per proposal, research plan, esercizio qualitativo, video, quiz e peer feedback.
- Ruoli di implementazione finali tra Jago e Gabriele e repository condiviso per codice/log/configurazioni.
- Conferma esplicita di Nadia sulla fattibilità della versione generalizzata ai decision models.
