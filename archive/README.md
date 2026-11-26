# Archive — Superseded v1 Methodological Audit Trail

> **WARNING:** Nothing in `archive/` is primary evidence for the final II2202 study. All primary results come from the pre-registered v2 suite (`ndt_v2_kaggle_experiments.ipynb`, `ndt_v2_engine.py`, `logs/v2/`).

### Contents
1. **`v1_spoonfed_audit/` (INVALIDATED AS PRIMARY EVIDENCE):**
   - Summary tables and voided pilot freeze from Nov 5–24 runs where prompts inadvertently injected Python `if/else` regime hints. Retained **solely** for transparent reporting of the methodological self-audit documented in `DEVIATIONS.md`, plus the pre-review `RESEARCH_RECOVERY_PLAN.md` draft (superseded by `RESEARCH_PLAN_REVIEW.md` and `PREREGISTRATION_v2.md`).
2. **`v1_unbiased_collapse_ablation/` (VALID DIAGNOSTIC MOTIVATING v2):**
   - The 1,344-call live Kaggle T4 unbiased ablation of `convaiinnovations/laya` with all hints stripped, which revealed 100% raw-argmax class collapse onto `'fixed'` and directly motivated the v2 redesign (threshold-free ROC-AUC, Zhao et al. contextual calibration, pilot-percentile scale context, oracle headroom, and 7B/8B model sensitivity).
