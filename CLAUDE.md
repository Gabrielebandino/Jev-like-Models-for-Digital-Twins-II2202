# II2202 project guide

Last updated: 3 November 2026. Detailed context: [KNOWLEDGE_BASE.md](KNOWLEDGE_BASE.md).

## What this is

KTH II2202 research project by Jago Revrenna and Gabriele Bandino on synchronizing a network digital twin (NDT). The current study is quantitative and simulation-based.

## Current scope

- A **general decision model** selects, at scheduled review points, the policy for the next period: `fixed-interval` or `change-aware` polling.
- Every poll reads the same fixed set of links. The selector does **not** choose individual links, predict unknown state, control a live network, or train a model.
- Compare fixed only, change-aware only, a deterministic switcher, and one selected decision-model switcher.
- Jev and open-weight candidates such as Laya are candidates, not the sole subject of the project. A pilot chooses one feasible model and freezes it.
- Main evidence: twin-state error versus monitoring traffic; report model/API overhead and decision latency separately.

## Source of truth

1. The current Overleaf projects and the latest submitted PDFs take priority.
2. Latest local artifact: `/Users/jagorev/Downloads/II2202_Research_Plan_Updated.pdf` (29 Oct 2026).
3. Latest editable local evidence of the final direction: `II2202_Qualitative_Method_Exercise/main.tex` (29 Oct 2026).
4. `KNOWLEDGE_BASE.md` records provenance, limits, unresolved items, and history.

## Historical files: do not silently reuse their scope

- `II2202_Project_Knowledge_Base_2026-10-06.md`
- `II2202_Project_Proposal_Jev/II2202-proposal.tex`
- `II2202_Research_Plan_Jev/II2202-research-plan.tex`

They describe a Jev-specific design and, in older phases, link selection. They are context only.

## Working rules

- Write course deliverables in natural, clear English; use Italian for discussion with Jago unless asked otherwise.
- Keep claims bounded by what is measured. Do not invent results, access, credits, costs, approvals, sources, or deadlines.
- Use synthetic data only; no private traces, credentials in logs, live-network control, model training, or personal spending.
- Preserve existing files. Do not replace PDFs or Overleaf content without an explicit request.
- Cite sources near the claims they support; product documentation supports interface facts, not performance claims.

## Useful checks

```sh
cd '/Users/jagorev/Documents/Università/5 anno/P1 - Research Metodology'
rg --files
rg -n 'Jev|Laya|decision model|polling' .
```

The local legacy LaTeX sources use LuaLaTeX; the live Overleaf project is authoritative for the current research plan.
