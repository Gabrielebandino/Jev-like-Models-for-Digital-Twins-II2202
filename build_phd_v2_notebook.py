#!/usr/bin/env python3
"""Generate the self-contained PhD-Level v2 Kaggle T4 x2 Notebook (`ndt_phd_v2_kaggle_experiments.ipynb`).

Embeds the locally verified `ndt_v2_engine.py` and `ndt_v2_models.py` directly into `%%writefile`
cells so the notebook runs standalone on Kaggle GPU T4 x2 with zero external repo sync issues.
"""

from __future__ import annotations
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.splitlines()],
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": [line + "\n" for line in text.splitlines()],
    }


def build_v2_notebook() -> dict:
    with open(os.path.join(HERE, "ndt_v2_engine.py"), "r", encoding="utf-8") as f:
        engine_code = f.read().rstrip()
    with open(os.path.join(HERE, "ndt_v2_models.py"), "r", encoding="utf-8") as f:
        models_code = f.read().rstrip()

    cells = [
        md_cell(
            r"""# 🔬 PhD-Level NDT Policy-Selection Research Suite (v2 — Kaggle GPU T4 x2)
### Decision-Model Applicability to Network Digital Twin Synchronization (KTH II2202)
**Authors:** Gabriele Bandino (`bandino@kth.se`) & Jago Revrenna (`revrenna@kth.se`)  
**Compliance Base:** Approved Research Plan (`researchPlan5Nov/II2202-research-plan.tex`) + `PREREGISTRATION_v2.md` + `DEVIATIONS.md`

---

### What This Notebook Solves (Every Flaw Identified in Our Audit)
1. **Zero Prompt Spoon-Feeding & Strict Information Parity:** Models receive strictly neutral chronological time series + EWMA/CUSUM + spatial Gini + oscillation reversal rate + monotone pilot-percentile scale context — **zero** pre-classified `if/else` regime labels.
2. **Oracle Headroom & Fraction of Headroom Captured:** Every trace is evaluated against a hindsight counterfactual oracle (verified against exact $2^R$ enumeration in QA), turning any outcome (positive, equivalent, or negative) into a rigorous, interpretable result.
3. **Balanced, Non-Dominated Static Arms:** Rate-normalized `ChangeAwarePolicy` ($\Delta / \Delta t^{0.5}$) calibrated with an **equal pilot budget ($k = 20$)** alongside supporting-hyperplane $\lambda$ so neither static arm games the objective.
4. **Realistic Regime-Switching Workload + Controls + External Validity:**
   - **Primary (Approved 8-Link Topology):** Markov-modulated `regime_switching` (480 s: quiet, macro Pareto volatility, localized transceiver flap, aliasing jitter) × `{low_delay, elevated_delay}` ($N = 50$ held-out seeds).
   - **Homogeneous Controls (Approved §2.2):** `stable`, `bursty`, `alternating`, `shifted`.
   - **External Validity:** **SNDlib Abilene** (12 US backbone nodes, 15 bidirectional links, Dijkstra OSPF routing + real single-link failover reroute) + optional real SNDlib Abilene 5-min traffic matrices (time-rescaled).
5. **Fair Model Evaluation Across 3 Open-Weight Families (Sequential GPU Isolation, Zero Cost):**
   - **System-1 Encoder:** `convaiinnovations/laya` (421M ModernBERT-large, fp16, ~48 ms)
   - **System-1 Contrastive 8B:** `Contrastive-LM/CLM-v0.1-8B` (4-bit Qwen3-8B + InfoNCE projection heads)
   - **Generative 7B Instruct (Logit-Scored):** `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4, single-pass A/B logit scoring — both **Zero-Shot** and training-free **Pilot 4-Shot** in-context examples!)
   - **4 Pre-Registered Neutral Paraphrases** (including option-order swap) + **Zhao et al. (ICML 2021) Contextual Calibration** frozen strictly on Pilot seeds!
6. **Threshold-Free Signal Diagnostics & Publication Statistics:**
   - **ROC-AUC & Average Precision** of $p(\text{change-aware})$ vs. per-review oracle labels (separating *no signal* from *miscalibrated decision boundary*)
   - **Learnability Ceiling** (pilot-trained GBDT diagnostic) & **Random Switcher** floor
   - **Dense 15-Point Static Fixed Frontier** ($\Delta\text{MAE}_{\text{iso-bytes}}$, paired per seed, zero extrapolation)
   - **95% Paired Bootstrap CIs**, **Wilcoxon Signed-Rank + Holm-Bonferroni**, and **TOST Equivalence Tests** ($\text{SESOI} = 0.10$)

**Kaggle Settings:** Accelerator → **GPU T4 x2** (or T4 / P100), Internet → **ON**, then **Run All**."""
        ),
        md_cell("## ⚙️ 0. Run Profile & GPU Pre-Flight Check"),
        code_cell(
            """import os, sys, json, time, shutil, warnings, subprocess
warnings.filterwarnings("ignore")
import torch

assert torch.cuda.is_available(), "ERROR: GPU Accelerator is OFF! Enable Accelerator -> GPU T4 x2 in Kaggle right sidebar."
print("✓ GPU Detected:", torch.cuda.get_device_name(0), f"({torch.cuda.device_count()} GPU(s))")
!nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

# ==============================================================================
# EXPERIMENT PROFILE (Switch to "QUICK_VERIFY" for a ~6-min dry run,
#                     or keep "FULL_PHD" for the complete publication dataset!)
# ==============================================================================
RUN_PROFILE = "FULL_PHD"   # Options: "FULL_PHD" (~30-40 min on T4) or "QUICK_VERIFY" (~6 min on T4)

if RUN_PROFILE == "FULL_PHD":
    CFG = {
        "pilot_seeds": [101, 102, 103, 104, 105],
        "heldout_primary_n": 50,          # Power-justified N=50 for baselines, oracle & primary fast model (laya)
        "heldout_raw_ablation_n": 15,     # Raw uncalibrated laya ablation showing collapse vs Zhao calibration
        "heldout_secondary_8b_n": 15,     # Paired subset for slower 8B/7B secondary sensitivity models
        "heldout_control_n": 15,          # Homogeneous controls (stable, bursty, alternating, shifted)
        "heldout_abilene_n": 20,          # SNDlib Abilene external validity (laya + baselines)
        "heldout_abilene_7b_n": 10,       # SNDlib Abilene secondary 7B model sensitivity
        "pilot_screen_reviews": 100,      # Pilot reviews per paraphrase during 4-paraphrase screen
        "repeat_stability_seeds": [2001, 2002],
        "run_clm_8b": True,
        "run_qwen25_7b": True,
        "fetch_real_sndlib_tm": True,
    }
else:
    CFG = {
        "pilot_seeds": [101, 102, 103],
        "heldout_primary_n": 8,
        "heldout_raw_ablation_n": 4,
        "heldout_secondary_8b_n": 4,
        "heldout_control_n": 4,
        "heldout_abilene_n": 4,
        "heldout_abilene_7b_n": 3,
        "pilot_screen_reviews": 40,
        "repeat_stability_seeds": [2001],
        "run_clm_8b": True,
        "run_qwen25_7b": True,
        "fetch_real_sndlib_tm": True,
    }

OUT_DIR = "logs/v2_phd"
os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "run_profile_config.json"), "w") as f:
    json.dump({"profile": RUN_PROFILE, **CFG}, f, indent=2)
print(f"✓ Active Profile: {RUN_PROFILE} -> outputs saved in {OUT_DIR}/")

# Optional HF_TOKEN from Kaggle Secrets (all 3 models are public, token just avoids HF rate limits)
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
        print("✓ Loaded HF_TOKEN from Kaggle Secrets.")
except Exception:
    print("ℹ Running without Kaggle HF_TOKEN (all 3 HF models are public).")"""
        ),
        md_cell("## 📦 1. Conflict-Free Kaggle Dependency Installation (~20s)"),
        code_cell(
            """# Remove Kaggle's incompatible pre-installed torchao so peft/transformers/bitsandbytes load cleanly
!pip uninstall -y -q torchao
!rm -rf /usr/local/lib/python3.13/dist-packages/torchao* /usr/local/lib/python3.12/dist-packages/torchao* /usr/local/lib/python3.11/dist-packages/torchao*

!pip install -q --no-warn-script-location \\
    "laya[serve]>=0.1.0" \\
    "bitsandbytes>=0.45.0" \\
    "accelerate>=1.0.0" \\
    fastapi uvicorn requests tabulate scipy scikit-learn matplotlib pandas numpy

print("✓ All dependencies installed cleanly!")"""
        ),
        md_cell("## 🧱 2. Write Verified Core Simulation, Oracle & Statistics Engine (`ndt_v2_engine.py`)"),
        code_cell("%%writefile ndt_v2_engine.py\n" + engine_code),
        md_cell("## 🤖 3. Write Unified Live GPU Model Adapters & Calibration Module (`ndt_v2_models.py`)"),
        code_cell("%%writefile ndt_v2_models.py\n" + models_code),
        md_cell("## ✅ 4. Stage 0 — Mandatory 7-Check QA Invariant Suite (Research Plan §2.4 + Exact $2^R$ Oracle Check)"),
        code_cell(
            """import importlib
import ndt_v2_engine
import ndt_v2_models
importlib.reload(ndt_v2_engine)
importlib.reload(ndt_v2_models)

from ndt_v2_engine import (
    ProtocolV2, DENSE_FIXED_INTERVALS_S, FEATURE_NAMES,
    generate_trace_v2, simulate_trace, run_qa_suite_v2, run_pilot_calibration,
    build_neutral_prompt_paraphrases, format_neutral_telemetry_text,
    compute_iso_bytes_delta, compute_headroom_and_stats_table, compute_signal_diagnostics,
    paired_bootstrap_ci, tost_paired_equivalence,
)
from ndt_v2_models import (
    get_gpu_vram_mb, kill_all_model_servers, start_http_model_server,
    SystemOneV2Client, Qwen25LogitSelector, calibrate_and_select_paraphrase_on_pilot,
)

qa_report = run_qa_suite_v2(ProtocolV2())
with open(os.path.join(OUT_DIR, "qa_verification_v2.json"), "w") as f:
    json.dump(qa_report, f, indent=2)

print(f"QA Suite: {qa_report['num_passed']}/{qa_report['num_checks']} PASSED (oracle_vs_2R_gap={qa_report['oracle_vs_2R_gap']})")
for chk, ok in qa_report["details"].items():
    print(f"  [{'PASS' if ok else 'FAIL'}] {chk}")
assert qa_report["all_passed"], f"Mandatory QA check failed: {qa_report}"
print("✓ All mandatory QA invariants verified!")"""
        ),
        md_cell(
            """## 🧪 5. Stage 1 — Equal-Budget Pilot Calibration ($k=20$), Supporting-Hyperplane $\\lambda$ & Power Analysis
Runs **strictly on Pilot seeds** (`[101..105]`, zero held-out contact):
- Derives scale-invariant supporting-hyperplane $\\lambda_{\\text{KB/link}}$ at `fixed@2.5s` so no static interval can game $J$.
- Sweeps equal $k=20$ candidate grids for `change_aware`, `rule_simple` (change + AoI), and `rule_ewma_cusum`.
- Builds the label-free `PilotPercentileReference` and fits the diagnostic `learnability_ceiling` (GBDT).
- Extracts 4 balanced, high-margin oracle-labeled pilot examples for training-free few-shot prompting of `Qwen2.5-7B-Instruct`."""
        ),
        code_cell(
            """import numpy as np
import pandas as pd
from tabulate import tabulate

t0 = time.perf_counter()
proto, pct_ref, learnability_clf, pilot_report = run_pilot_calibration(CFG["pilot_seeds"])
print(f"✓ Pilot calibration finished in {time.perf_counter() - t0:.1f}s")
print(json.dumps(pilot_report, indent=2))

# Build pilot review dataset (strictly from pilot seeds) for model 4-paraphrase & Zhao calibration screen
pilot_review_items = []     # List[(neutral_telemetry_text, oracle_label, oracle_margin)]
pilot_oracle_df_rows = []
for dl in ("low_delay", "elevated_delay"):
    for s in CFG["pilot_seeds"]:
        tr = generate_trace_v2("regime_switching", dl, s, proto, topology="8link")
        _, o_logs = simulate_trace(tr, "oracle", proto, pct_ref=pct_ref, compute_oracle_labels=True, record_review_logs=True)
        # Re-extract neutral telemetry strings via a lightweight replay of oracle reviews
        for lg in o_logs:
            pilot_oracle_df_rows.append(lg)

# Capture exact neutral telemetry texts at pilot review ticks via a quick oracle-labeled pass
pilot_texts_and_labels = []
def _capture_pilot_review(feat):
    return {"choice": "fixed", "is_valid": True, "latency_ms": 0.05, "p_ca_raw": 0.5}

for dl in ("low_delay", "elevated_delay"):
    for s in CFG["pilot_seeds"]:
        tr = generate_trace_v2("regime_switching", dl, s, proto, topology="8link")
        captured_txts = []
        def _cap(feat):
            captured_txts.append(format_neutral_telemetry_text(feat))
            return {"choice": "change-aware" if feat.active_link_frac >= 0.35 or feat.mean_rate_delta <= 1.0 else "fixed",
                    "is_valid": True, "latency_ms": 0.05, "p_ca_raw": 0.5}
        _, logs_cap = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=_cap,
                                     compute_oracle_labels=True, record_review_logs=True)
        for txt, lg in zip(captured_txts, logs_cap):
            if lg["oracle_label"] in ("fixed", "change-aware"):
                pilot_texts_and_labels.append((txt, lg["oracle_label"], float(lg["oracle_margin"])))

# Deterministic balanced sample of pilot reviews for fast 4-paraphrase screening
rng_p = np.random.default_rng(42)
ca_items = [x for x in pilot_texts_and_labels if x[1] == "change-aware"]
fx_items = [x for x in pilot_texts_and_labels if x[1] == "fixed"]
n_half = min(CFG["pilot_screen_reviews"] // 2, len(ca_items), len(fx_items))
idx_ca = rng_p.choice(len(ca_items), size=n_half, replace=False)
idx_fx = rng_p.choice(len(fx_items), size=n_half, replace=False)
pilot_screen_items = [ca_items[i] for i in idx_ca] + [fx_items[i] for i in idx_fx]

# Select 4 clean, highest-margin pilot few-shot examples (2 change-aware, 2 fixed) for Qwen2.5-7B-Instruct
ca_sorted = sorted(ca_items, key=lambda x: x[2], reverse=True)
fx_sorted = sorted(fx_items, key=lambda x: x[2], reverse=False)
PILOT_FEW_SHOT_EXAMPLES = [
    (ca_sorted[0][0], "change-aware"),
    (fx_sorted[0][0], "fixed"),
    (ca_sorted[min(3, len(ca_sorted)-1)][0], "change-aware"),
    (fx_sorted[min(3, len(fx_sorted)-1)][0], "fixed"),
]
PARAPHRASES = build_neutral_prompt_paraphrases(proto)
print(f"✓ Prepared {len(pilot_screen_items)} balanced pilot reviews ({n_half} CA / {n_half} fixed) and 4 pilot few-shot examples.")"""
        ),
        md_cell(
            """## 🎯 6. Stage 2 — Unbiased Model Pilot Gate, 4-Paraphrase Robustness & Zhao et al. (2021) Calibration
Screens all 3 open-weight model families **sequentially in strict GPU isolation** on the Pilot reviews:
1. Measures live GPU VRAM, P50/P95 latency vs. 2000 ms deadline, and parse validity.
2. Evaluates all **4 pre-registered neutral paraphrases** (including option-order swap!) and records the paraphrase ROC-AUC spread.
3. Measures each model's **content-free prior bias** $p_0(\text{change-aware})$ (Zhao et al., ICML 2021) and freezes the winning paraphrase & prior calibration into `frozen_protocol_v2.json` **before any held-out run**."""
        ),
        code_cell(
            """pilot_model_screens = {}
vram_log = {}

# --- 1. LAYA (421M ModernBERT-large System-1) ---
print("\\n" + "=" * 72 + "\\n🚀 PILOT SCREEN 1/3: convaiinnovations/laya (421M ModernBERT)\\n" + "=" * 72)
try:
    proc, base_url, api_field = start_http_model_server("laya", OUT_DIR)
    vram_log["laya"] = get_gpu_vram_mb()
    # Warmup
    _w = SystemOneV2Client("laya", base_url, api_field, PARAPHRASES[0], use_calibration=False)
    for _ in range(3):
        _w.query_text(pilot_screen_items[0][0])
    scr = calibrate_and_select_paraphrase_on_pilot(
        "laya", pilot_screen_items, PARAPHRASES,
        lambda para: SystemOneV2Client("laya", base_url, api_field, para, use_calibration=False).query_text,
    )
    pilot_model_screens["laya"] = scr
    print(tabulate(pd.DataFrame(scr["all_paraphrases"]), headers="keys", tablefmt="github", showindex=False))
finally:
    kill_all_model_servers()

# --- 2. CLM-v0.1-8B (4-bit Qwen3-8B + InfoNCE Projection Heads) ---
if CFG["run_clm_8b"]:
    print("\\n" + "=" * 72 + "\\n🚀 PILOT SCREEN 2/3: Contrastive-LM/CLM-v0.1-8B (4-bit Qwen3-8B)\\n" + "=" * 72)
    try:
        proc, base_url, api_field = start_http_model_server("clm_8b", OUT_DIR)
        vram_log["clm_8b"] = get_gpu_vram_mb()
        _w = SystemOneV2Client("clm_8b", base_url, api_field, PARAPHRASES[0], use_calibration=False)
        for _ in range(2):
            _w.query_text(pilot_screen_items[0][0])
        scr = calibrate_and_select_paraphrase_on_pilot(
            "clm_8b", pilot_screen_items, PARAPHRASES,
            lambda para: SystemOneV2Client("clm_8b", base_url, api_field, para, use_calibration=False).query_text,
        )
        pilot_model_screens["clm_8b"] = scr
        print(tabulate(pd.DataFrame(scr["all_paraphrases"]), headers="keys", tablefmt="github", showindex=False))
    except Exception as ex:
        print(f"⚠ CLM-v0.1-8B pilot screen failed (continuing cleanly): {ex}")
    finally:
        kill_all_model_servers()

# --- 3. Qwen2.5-7B-Instruct (4-bit NF4, Single-Pass A/B Logit Scoring: Zero-Shot & Pilot 4-Shot) ---
if CFG["run_qwen25_7b"]:
    print("\\n" + "=" * 72 + "\\n🚀 PILOT SCREEN 3/3: Qwen/Qwen2.5-7B-Instruct (Zero-Shot & Pilot 4-Shot)\\n" + "=" * 72)
    qwen_eng = None
    try:
        qwen_eng = Qwen25LogitSelector("Qwen/Qwen2.5-7B-Instruct")
        vram_log["qwen25_7b"] = get_gpu_vram_mb()
        # Warmup
        _wq = qwen_eng.make_callable(PARAPHRASES[0], few_shot_examples=None, use_calibration=False)
        _wq.query_text(pilot_screen_items[0][0])

        scr_zs = calibrate_and_select_paraphrase_on_pilot(
            "qwen25_7b_zeroshot", pilot_screen_items, PARAPHRASES,
            lambda para: qwen_eng.make_callable(para, few_shot_examples=None, use_calibration=False).query_text,
        )
        pilot_model_screens["qwen25_7b_zeroshot"] = scr_zs
        print("Zero-Shot Paraphrase Screen:")
        print(tabulate(pd.DataFrame(scr_zs["all_paraphrases"]), headers="keys", tablefmt="github", showindex=False))

        # Pilot 4-Shot (uses winning zero-shot paraphrase + 4 pilot oracle examples)
        best_p_idx = scr_zs["best_paraphrase_idx"]
        scr_fs = calibrate_and_select_paraphrase_on_pilot(
            "qwen25_7b_fewshot", pilot_screen_items, [PARAPHRASES[best_p_idx]],
            lambda para: qwen_eng.make_callable(para, few_shot_examples=PILOT_FEW_SHOT_EXAMPLES, use_calibration=False).query_text,
        )
        scr_fs["best_paraphrase_idx"] = best_p_idx
        pilot_model_screens["qwen25_7b_fewshot"] = scr_fs
        print("Pilot 4-Shot Screen:")
        print(tabulate(pd.DataFrame(scr_fs["all_paraphrases"]), headers="keys", tablefmt="github", showindex=False))
    except Exception as ex:
        print(f"⚠ Qwen2.5-7B-Instruct pilot screen failed (continuing cleanly): {ex}")
    finally:
        if qwen_eng is not None:
            qwen_eng.unload()
        kill_all_model_servers()

# Freeze Primary Model per Research Plan §2.2 (meets <2000ms P95, >=95% valid, fastest / best pilot ROC-AUC)
feasible = [
    (k, v) for k, v in pilot_model_screens.items()
    if v["parse_valid_rate"] >= 0.95 and v["best_latency_p95_ms"] < proto.decision_deadline_ms
]
# Primary System-1 model freeze (prefer laya if feasible for fast N=50 primary, record best ROC-AUC model too)
primary_model_key = max(feasible, key=lambda kv: kv[1]["best_pilot_roc_auc"])[0] if feasible else "laya"
print(f"\\n★ PILOT FREEZE COMPLETE! Highest pilot-AUC feasible model: {primary_model_key}")

frozen_bundle = {
    "protocol": proto.to_dict(),
    "pilot_report": pilot_report,
    "primary_model_by_pilot_auc": primary_model_key,
    "pilot_model_screens": pilot_model_screens,
    "gpu_vram_log": vram_log,
}
with open(os.path.join(OUT_DIR, "frozen_protocol_v2.json"), "w") as f:
    json.dump(frozen_bundle, f, indent=2)
print(f"✓ Saved frozen protocol & pilot gate to {OUT_DIR}/frozen_protocol_v2.json")"""
        ),
        md_cell(
            """## 📊 7. Stage 3 — Held-Out CPU Baselines, Dense 15-Point Frontier, Oracle Headroom & Learnability Ceiling
Runs on strictly disjoint Held-Out seeds (`[2001..]`):
- **Primary 8-Link `regime_switching`** ($N = 50$ seeds × 2 delays)
- **Homogeneous Controls** (`stable`, `bursty`, `alternating`, `shifted` × $N = 15$ seeds × 2 delays)
- **SNDlib Abilene External Validity** (`abilene_regime_switching` × $N = 20$ seeds × 2 delays)"""
        ),
        code_cell(
            """HELDOUT_PRIMARY_SEEDS = list(range(2001, 2001 + CFG["heldout_primary_n"]))
HELDOUT_CONTROL_SEEDS = list(range(3001, 3001 + CFG["heldout_control_n"]))
HELDOUT_ABILENE_SEEDS = list(range(4001, 4001 + CFG["heldout_abilene_n"]))

base_csv = os.path.join(OUT_DIR, "heldout_baselines_and_oracle.csv")
fixed_csv = os.path.join(OUT_DIR, "heldout_dense_fixed_frontier.csv")
base_logs_jsonl = os.path.join(OUT_DIR, "heldout_baselines_review_logs.jsonl")

t0 = time.perf_counter()
base_rows, fixed_rows, base_rlogs = [], [], []

workloads = [
    ("8link", "regime_switching", HELDOUT_PRIMARY_SEEDS),
    ("8link", "stable", HELDOUT_CONTROL_SEEDS),
    ("8link", "bursty", HELDOUT_CONTROL_SEEDS),
    ("8link", "alternating", HELDOUT_CONTROL_SEEDS),
    ("8link", "shifted", HELDOUT_CONTROL_SEEDS),
    ("abilene", "abilene_regime_switching", HELDOUT_ABILENE_SEEDS),
]

for topo, reg, seed_list in workloads:
    print(f"  Running CPU baselines + dense frontier + oracle: {topo}/{reg} ({len(seed_list)} seeds x 2 delays)...")
    for dl in ("low_delay", "elevated_delay"):
        for s in seed_list:
            tr = generate_trace_v2(reg, dl, s, proto, topology=topo)
            # Dense 15-point static fixed frontier for unbiased log-KB interpolation
            for fi in DENSE_FIXED_INTERVALS_S:
                rf, _ = simulate_trace(tr, "fixed", proto, fixed_interval_override=fi)
                fixed_rows.append(rf)
            # Frozen reference fixed@2.5s
            rf25, _ = simulate_trace(tr, "fixed", proto, fixed_interval_override=proto.fixed_interval_s)
            base_rows.append(rf25)
            # Static CA, Rules, Random Floor, Learnability Ceiling, Hindsight Oracle
            for ctrl in ("change_aware", "rule_simple", "rule_ewma_cusum", "random", "learnability_ceiling", "oracle"):
                log_oracle = (ctrl in ("rule_simple", "rule_ewma_cusum", "learnability_ceiling")) and (
                    ("switching" in reg) and (s in seed_list[: min(15, len(seed_list))])
                )
                r, lg = simulate_trace(
                    tr, ctrl, proto, pct_ref=pct_ref, trained_clf=learnability_clf,
                    random_p_ca=pilot_report["pilot_oracle_ca_rate"],
                    compute_oracle_labels=log_oracle, record_review_logs=log_oracle,
                )
                base_rows.append(r)
                if log_oracle:
                    for rec in lg:
                        rec.pop("features", None)  # Keep JSONL compact
                        base_rlogs.append(rec)

df_base = pd.DataFrame(base_rows)
df_fixed_dense = pd.DataFrame(fixed_rows)
df_base.to_csv(base_csv, index=False)
df_fixed_dense.to_csv(fixed_csv, index=False)
with open(base_logs_jsonl, "w") as f:
    for rec in base_rlogs:
        f.write(json.dumps(rec) + "\\n")

print(f"✓ Completed {len(df_base)} baseline/oracle runs + {len(df_fixed_dense)} dense frontier runs in {time.perf_counter()-t0:.1f}s")"""
        ),
        md_cell(
            """## ⚡ 8. Stage 4 — Live GPU Held-Out Evaluation: `laya` (Raw vs. Zhao-Calibrated + Repeat Stability)
Runs `convaiinnovations/laya` across:
- Full $N = 50$ primary `regime_switching` seeds (Zhao-calibrated) + $N = 15$ raw uncalibrated ablation (directly quantifying class collapse vs. contextual calibration!)
- All 4 homogeneous control regimes (`stable`, `bursty`, `alternating`, `shifted`)
- External-validity **SNDlib Abilene** (12 nodes, 15 links, OSPF failover)
- 3× repeat stability on identical held-out traces (§2.4)"""
        ),
        code_cell(
            """laya_rows, laya_rlogs, repeat_rows = [], [], []
t0 = time.perf_counter()

try:
    proc, base_url, api_field = start_http_model_server("laya", OUT_DIR)
    l_scr = pilot_model_screens["laya"]
    l_para = PARAPHRASES[l_scr["best_paraphrase_idx"]]
    l_prior = l_scr["selected_prior_p_ca"]

    client_cal = SystemOneV2Client("laya", base_url, api_field, l_para, use_calibration=True, prior_p_ca=l_prior)
    client_raw = SystemOneV2Client("laya", base_url, api_field, l_para, use_calibration=False, prior_p_ca=0.5)

    # Warmup
    for _ in range(3):
        client_cal.query_text(pilot_screen_items[0][0])

    # 1. Primary 8-Link regime_switching (N=50 calibrated + N=15 raw ablation)
    print(f"  [laya_calibrated] Primary 8-link regime_switching ({len(HELDOUT_PRIMARY_SEEDS)} seeds x 2 delays)...")
    for dl in ("low_delay", "elevated_delay"):
        for idx_s, s in enumerate(HELDOUT_PRIMARY_SEEDS):
            tr = generate_trace_v2("regime_switching", dl, s, proto, topology="8link")
            need_o = idx_s < 15
            r, lg = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=client_cal,
                                   compute_oracle_labels=need_o, record_review_logs=need_o)
            r["controller"] = "laya_calibrated"
            laya_rows.append(r)
            for rec in lg:
                rec["controller"] = "laya_calibrated"
                rec.pop("features", None)
                laya_rlogs.append(rec)

            # Raw uncalibrated ablation on first N_raw seeds
            if idx_s < CFG["heldout_raw_ablation_n"]:
                r_raw, lg_raw = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=client_raw,
                                               compute_oracle_labels=True, record_review_logs=True)
                r_raw["controller"] = "laya_raw"
                laya_rows.append(r_raw)
                for rec in lg_raw:
                    rec["controller"] = "laya_raw"
                    rec.pop("features", None)
                    laya_rlogs.append(rec)

    # 2. Homogeneous Controls (stable, bursty, alternating, shifted)
    print(f"  [laya_calibrated] Homogeneous controls ({len(HELDOUT_CONTROL_SEEDS)} seeds x 4 regimes x 2 delays)...")
    for reg in ("stable", "bursty", "alternating", "shifted"):
        for dl in ("low_delay", "elevated_delay"):
            for s in HELDOUT_CONTROL_SEEDS:
                tr = generate_trace_v2(reg, dl, s, proto, topology="8link")
                r, _ = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=client_cal)
                r["controller"] = "laya_calibrated"
                laya_rows.append(r)

    # 3. External Validity: SNDlib Abilene (12 nodes, 15 links, OSPF failover)
    print(f"  [laya_calibrated] SNDlib Abilene ({len(HELDOUT_ABILENE_SEEDS)} seeds x 2 delays)...")
    for dl in ("low_delay", "elevated_delay"):
        for idx_s, s in enumerate(HELDOUT_ABILENE_SEEDS):
            tr = generate_trace_v2("abilene_regime_switching", dl, s, proto, topology="abilene")
            need_o = idx_s < 10
            r, lg = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=client_cal,
                                   compute_oracle_labels=need_o, record_review_logs=need_o)
            r["controller"] = "laya_calibrated"
            laya_rows.append(r)
            for rec in lg:
                rec["controller"] = "laya_calibrated"
                rec.pop("features", None)
                laya_rlogs.append(rec)

    # 4. Repeat-Stability Variance (3 identical runs on fixed seeds per §2.4)
    for s in CFG["repeat_stability_seeds"]:
        tr = generate_trace_v2("regime_switching", "low_delay", s, proto, topology="8link")
        for rep_i in range(3):
            r, _ = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=client_cal)
            r["controller"] = "laya_calibrated"
            r["repeat_idx"] = rep_i
            repeat_rows.append(r)
finally:
    kill_all_model_servers()

df_laya = pd.DataFrame(laya_rows)
df_laya.to_csv(os.path.join(OUT_DIR, "heldout_laya.csv"), index=False)
with open(os.path.join(OUT_DIR, "heldout_laya_review_logs.jsonl"), "w") as f:
    for rec in laya_rlogs:
        f.write(json.dumps(rec) + "\\n")
print(f"✓ Laya held-out complete ({len(df_laya)} runs) in {time.perf_counter()-t0:.1f}s")"""
        ),
        md_cell(
            """## 🧠 9. Stage 5 — Secondary Model-Sensitivity Backbones (`clm_8b`, `qwen25_7b_zeroshot`, `qwen25_7b_fewshot`)
Evaluates larger 7B/8B architectures on paired held-out seeds so our findings cannot be dismissed as a small-model (421M) artifact:
1. **`Contrastive-LM/CLM-v0.1-8B`** (4-bit Qwen3-8B + InfoNCE projection heads, Zhao-calibrated)
2. **`Qwen/Qwen2.5-7B-Instruct` (Zero-Shot)** (4-bit NF4, single-pass exact A/B logit scoring, Zhao-calibrated)
3. **`Qwen/Qwen2.5-7B-Instruct` (Pilot 4-Shot)** (training-free 4 oracle-labeled pilot examples in-context, mirroring rule calibration without any weight updates!)"""
        ),
        code_cell(
            """sec_rows, sec_rlogs = [], []
SEC_PRIMARY_SEEDS = HELDOUT_PRIMARY_SEEDS[: CFG["heldout_secondary_8b_n"]]
SEC_ABILENE_SEEDS = HELDOUT_ABILENE_SEEDS[: CFG["heldout_abilene_7b_n"]]

# --- A. Contrastive-LM/CLM-v0.1-8B ---
if CFG["run_clm_8b"] and "clm_8b" in pilot_model_screens:
    t0 = time.perf_counter()
    print(f"\\n🚀 Running Secondary Model 1: clm_8b ({len(SEC_PRIMARY_SEEDS)} primary seeds x 2 delays)...")
    try:
        proc, base_url, api_field = start_http_model_server("clm_8b", OUT_DIR)
        c_scr = pilot_model_screens["clm_8b"]
        c_client = SystemOneV2Client(
            "clm_8b", base_url, api_field,
            PARAPHRASES[c_scr["best_paraphrase_idx"]],
            use_calibration=True, prior_p_ca=c_scr["selected_prior_p_ca"],
        )
        c_client.query_text(pilot_screen_items[0][0])  # warmup
        for dl in ("low_delay", "elevated_delay"):
            for s in SEC_PRIMARY_SEEDS:
                tr = generate_trace_v2("regime_switching", dl, s, proto, topology="8link")
                r, lg = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=c_client,
                                       compute_oracle_labels=True, record_review_logs=True)
                r["controller"] = "clm_8b_calibrated"
                sec_rows.append(r)
                for rec in lg:
                    rec["controller"] = "clm_8b_calibrated"
                    rec.pop("features", None)
                    sec_rlogs.append(rec)
        print(f"✓ clm_8b complete in {time.perf_counter()-t0:.1f}s")
    except Exception as ex:
        print(f"⚠ clm_8b held-out error: {ex}")
    finally:
        kill_all_model_servers()

# --- B. Qwen2.5-7B-Instruct (Zero-Shot & Pilot 4-Shot) ---
if CFG["run_qwen25_7b"] and "qwen25_7b_zeroshot" in pilot_model_screens:
    t0 = time.perf_counter()
    print(f"\\n🚀 Running Secondary Model 2: Qwen2.5-7B-Instruct Zero-Shot & Pilot 4-Shot...")
    qwen_eng = None
    try:
        qwen_eng = Qwen25LogitSelector("Qwen/Qwen2.5-7B-Instruct")
        zs_scr = pilot_model_screens["qwen25_7b_zeroshot"]
        fs_scr = pilot_model_screens["qwen25_7b_fewshot"]
        q_zs = qwen_eng.make_callable(
            PARAPHRASES[zs_scr["best_paraphrase_idx"]],
            few_shot_examples=None,
            use_calibration=True, prior_p_ca=zs_scr["selected_prior_p_ca"],
        )
        q_fs = qwen_eng.make_callable(
            PARAPHRASES[fs_scr["best_paraphrase_idx"]],
            few_shot_examples=PILOT_FEW_SHOT_EXAMPLES,
            use_calibration=True, prior_p_ca=fs_scr["selected_prior_p_ca"],
        )
        q_zs.query_text(pilot_screen_items[0][0])  # warmup

        for cname, qfn in [("qwen25_7b_zeroshot", q_zs), ("qwen25_7b_fewshot", q_fs)]:
            print(f"  [{cname}] Primary 8-link regime_switching ({len(SEC_PRIMARY_SEEDS)} seeds x 2 delays)...")
            for dl in ("low_delay", "elevated_delay"):
                for s in SEC_PRIMARY_SEEDS:
                    tr = generate_trace_v2("regime_switching", dl, s, proto, topology="8link")
                    r, lg = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=qfn,
                                           compute_oracle_labels=True, record_review_logs=True)
                    r["controller"] = cname
                    sec_rows.append(r)
                    for rec in lg:
                        rec["controller"] = cname
                        rec.pop("features", None)
                        sec_rlogs.append(rec)

        # Also evaluate Pilot 4-Shot on SNDlib Abilene external validity + 3x repeat stability
        print(f"  [qwen25_7b_fewshot] SNDlib Abilene ({len(SEC_ABILENE_SEEDS)} seeds x 2 delays)...")
        for dl in ("low_delay", "elevated_delay"):
            for s in SEC_ABILENE_SEEDS:
                tr = generate_trace_v2("abilene_regime_switching", dl, s, proto, topology="abilene")
                r, lg = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=q_fs,
                                       compute_oracle_labels=True, record_review_logs=True)
                r["controller"] = "qwen25_7b_fewshot"
                sec_rows.append(r)
                for rec in lg:
                    rec["controller"] = "qwen25_7b_fewshot"
                    rec.pop("features", None)
                    sec_rlogs.append(rec)

        for s in CFG["repeat_stability_seeds"]:
            tr = generate_trace_v2("regime_switching", "low_delay", s, proto, topology="8link")
            for rep_i in range(3):
                r, _ = simulate_trace(tr, "model", proto, pct_ref=pct_ref, model_selector_fn=q_fs)
                r["controller"] = "qwen25_7b_fewshot"
                r["repeat_idx"] = rep_i
                repeat_rows.append(r)
        print(f"✓ Qwen2.5-7B-Instruct complete in {time.perf_counter()-t0:.1f}s")
    except Exception as ex:
        print(f"⚠ Qwen2.5-7B-Instruct held-out error: {ex}")
    finally:
        if qwen_eng is not None:
            qwen_eng.unload()
        kill_all_model_servers()

df_sec = pd.DataFrame(sec_rows)
df_sec.to_csv(os.path.join(OUT_DIR, "heldout_secondary_models.csv"), index=False)
with open(os.path.join(OUT_DIR, "heldout_secondary_review_logs.jsonl"), "w") as f:
    for rec in sec_rlogs:
        f.write(json.dumps(rec) + "\\n")
pd.DataFrame(repeat_rows).to_csv(os.path.join(OUT_DIR, "heldout_repeat_stability.csv"), index=False)
print(f"✓ Saved {len(df_sec)} secondary model runs & {len(repeat_rows)} repeat-stability runs.")"""
        ),
        md_cell(
            """## 🌐 10. Stage 6 — Real Public Network Dataset External Validity (SNDlib Abilene Traffic Matrices)
Attempts to fetch genuine public **SNDlib Abilene 5-minute traffic matrices** (`sndlib.put.poznan.pl`), routes them via Dijkstra OSPF over the 12-node 15-link Abilene topology, and evaluates our frozen controllers on disjoint real-world time windows (**explicitly documented as time-rescaled**: 1 simulator second $\\leftrightarrow$ real 5-min measurement step). Gracefully logs and continues if the external archive is unreachable."""
        ),
        code_cell(
            """import tarfile, io, xml.etree.ElementTree as ET
from ndt_v2_engine import ABILENE_NODES, ABILENE_LINK_NAMES, ABILENE_OSPF_NORMAL

real_tm_status = {"attempted": CFG["fetch_real_sndlib_tm"], "success": False, "note": ""}
real_tm_rows = []

if CFG["fetch_real_sndlib_tm"]:
    SNDLIB_URL = "https://sndlib.put.poznan.pl/download/directed-abilene-zhang-5min-over-6months-ALL-native.tgz"
    try:
        print(f"Fetching public SNDlib Abilene traffic matrix archive from {SNDLIB_URL} ...")
        r_http = requests.get(SNDLIB_URL, timeout=25, stream=True)
        if r_http.status_code == 200:
            # Read archive (typically ~25 MB compressed; extract first 1600 consecutive 5-min matrices = ~5.5 days)
            raw_bytes = r_http.content
            node_idx = {n: i for i, n in enumerate(ABILENE_NODES)}
            link_series = []
            with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:gz") as tf:
                members = sorted([m for m in tf.getmembers() if m.isfile() and "demand" in m.name.lower() or m.name.endswith(".txt")],
                                 key=lambda x: x.name)[:1800]
                for mem in members:
                    fobj = tf.extractfile(mem)
                    if fobj is None:
                        continue
                    txt = fobj.read().decode("utf-8", errors="ignore")
                    u_links = np.zeros(len(ABILENE_LINK_NAMES), dtype=np.float64)
                    in_demands = False
                    for line in txt.splitlines():
                        ls = line.strip()
                        if ls.startswith("DEMANDS"):
                            in_demands = True
                            continue
                        if in_demands and ls.startswith(")"):
                            break
                        if in_demands and "(" in ls and ")" in ls:
                            try:
                                inner = ls.split("(")[1].split(")")[0].split()
                                val = float(ls.split(")")[1].strip().split()[1])
                                s_i, d_i = node_idx.get(inner[0]), node_idx.get(inner[1])
                                if s_i is not None and d_i is not None:
                                    for e_id in ABILENE_OSPF_NORMAL.get((s_i, d_i), []):
                                        u_links[e_id] += val
                            except Exception:
                                pass
                    if np.sum(u_links) > 0:
                        link_series.append(u_links)

            if len(link_series) >= 600:
                mat = np.vstack(link_series)  # [T_real, 15]
                # Scale to realistic link utilization [5%, 92%] preserving exact relative peaks & diurnal structure
                p98 = np.percentile(mat, 98, axis=0) + 1e-6
                util_mat = np.clip(10.0 + 75.0 * (mat / p98), 1.0, 98.0)
                # Interpolate smoothly to 100ms simulator ticks (1 real 5-min step -> 1.0s rescaled simulator time)
                n_win = min(5, len(util_mat) // 240)
                for w_i in range(n_win):
                    seg = util_mat[w_i * 240 : (w_i + 1) * 240]
                    t_orig = np.arange(len(seg), dtype=np.float64)
                    t_sim = np.round(np.arange(0.0, len(seg) - 1 + 1e-9, proto.tick_dt_s), 4)
                    u_interp = np.column_stack([np.interp(t_sim, t_orig, seg[:, j]) for j in range(seg.shape[1])])
                    rng_w = np.random.default_rng(9000 + w_i)
                    dly = np.clip(rng_w.normal(50.0, 10.0, size=len(t_sim)), 10.0, 150.0) / 1000.0
                    tr_real = {
                        "topology": "abilene_real_sndlib",
                        "regime": "real_tm_rescaled",
                        "delay_condition": "low_delay",
                        "seed": 9000 + w_i,
                        "link_names": ABILENE_LINK_NAMES,
                        "num_links": len(ABILENE_LINK_NAMES),
                        "timestamps": t_sim,
                        "true_u": u_interp,
                        "delays_s": dly,
                        "is_change_event": np.zeros(len(t_sim), dtype=np.int32),
                        "phase_labels": ["real_tm"] * len(t_sim),
                    }
                    for ctrl in ("fixed", "change_aware", "rule_simple", "rule_ewma_cusum", "oracle"):
                        r, _ = simulate_trace(tr_real, ctrl, proto, pct_ref=pct_ref)
                        real_tm_rows.append(r)
                real_tm_status = {
                    "attempted": True, "success": True,
                    "source_url": SNDLIB_URL,
                    "num_real_matrices_parsed": len(link_series),
                    "num_disjoint_windows": n_win,
                    "time_rescaling_note": "1 real 5-min SNDlib TM step mapped to 1.0s simulated time (explicit time-rescaled validation).",
                }
                df_real = pd.DataFrame(real_tm_rows)
                df_real.to_csv(os.path.join(OUT_DIR, "heldout_real_sndlib_abilene_tm.csv"), index=False)
                print("✓ Real SNDlib Abilene TM validation complete!")
                print(tabulate(df_real.groupby("controller")[["mae_pct", "monitoring_kb", "cost_j", "ca_frac"]].mean().round(4).reset_index(),
                               headers="keys", tablefmt="github", showindex=False))
            else:
                real_tm_status["note"] = f"Archive downloaded but parsed only {len(link_series)} matrices; skipped."
        else:
            real_tm_status["note"] = f"HTTP {r_http.status_code} from SNDlib; synthetic Abilene OSPF validation used."
    except Exception as ex:
        real_tm_status["note"] = f"SNDlib fetch unavailable ({type(ex).__name__}: {ex}); synthetic Abilene OSPF validation used."

with open(os.path.join(OUT_DIR, "real_sndlib_status.json"), "w") as f:
    json.dump(real_tm_status, f, indent=2)
print("Real SNDlib TM Status:", json.dumps(real_tm_status, indent=2))"""
        ),
        md_cell(
            """## 📈 11. Stage 7 — Full PhD Statistical Analysis (Iso-Bytes Pareto, Headroom, ROC-AUC, Wilcoxon+Holm & TOST)"""
        ),
        code_cell(
            """# Merge all Held-Out results & review logs
dfs = [df_base]
if not df_laya.empty:
    dfs.append(df_laya)
if not df_sec.empty:
    dfs.append(df_sec)

df_all_raw = pd.concat(dfs, ignore_index=True)
df_all = compute_iso_bytes_delta(df_all_raw, df_fixed_dense)
df_all.to_csv(os.path.join(OUT_DIR, "heldout_MASTER_all_runs.csv"), index=False)

cond_df, contrast_df = compute_headroom_and_stats_table(df_all, proto, reference_baseline="rule_ewma_cusum")
cond_df.to_csv(os.path.join(OUT_DIR, "table1_condition_summary_and_headroom.csv"), index=False)
contrast_df.to_csv(os.path.join(OUT_DIR, "table2_paired_stats_wilcoxon_holm_tost.csv"), index=False)

all_rlogs = base_rlogs + laya_rlogs + sec_rlogs
sig_df = compute_signal_diagnostics(pd.DataFrame(all_rlogs))
sig_df.to_csv(os.path.join(OUT_DIR, "table3_threshold_free_signal_roc_auc.csv"), index=False)

print("\\n" + "=" * 80)
print("TABLE 1: PRIMARY 8-LINK REGIME-SWITCHING & SNDLIB ABILENE SUMMARY (LOW DELAY)")
print("=" * 80)
t1_view = cond_df[(cond_df["regime"].isin(["regime_switching", "abilene_regime_switching"])) & (cond_df["delay"] == "low_delay")]
print(tabulate(t1_view[["regime", "controller", "n_seeds", "mae_mean", "kb_mean", "cost_j_mean", "delta_mae_iso_mean", "headroom_j", "headroom_captured_pct", "congestion_f1", "ca_frac", "latency_p95_ms", "fallback_rate"]],
               headers="keys", tablefmt="github", showindex=False))

print("\\n" + "=" * 80)
print("TABLE 2: PAIRED STATISTICAL CONTRASTS vs STRONGEST RULE (rule_ewma_cusum) — WILCOXON+HOLM & TOST (SESOI=0.10)")
print("=" * 80)
if not contrast_df.empty:
    t2_view = contrast_df[contrast_df["regime"].isin(["regime_switching", "abilene_regime_switching"])]
    print(tabulate(t2_view[["regime", "delay", "controller", "n_paired", "mean_diff_j", "ci95_lo", "ci95_hi", "p_wilcoxon_holm", "p_tost_sesoi", "equivalent_within_sesoi"]],
                   headers="keys", tablefmt="github", showindex=False))

print("\\n" + "=" * 80)
print("TABLE 3: THRESHOLD-FREE SIGNAL DIAGNOSTICS (ROC-AUC & AP vs PER-REVIEW ORACLE LABELS)")
print("=" * 80)
if not sig_df.empty:
    print(tabulate(sig_df, headers="keys", tablefmt="github", showindex=False))

# Repeat stability summary (§2.4)
rep_df = pd.DataFrame(repeat_rows)
if not rep_df.empty:
    rep_sum = rep_df.groupby(["controller", "seed"])[["mae_pct", "monitoring_kb", "cost_j", "change_aware_frac"]].agg(["mean", "std"]).round(4)
    print("\\n=== REPEAT-STABILITY VARIANCE ON IDENTICAL TRACES (3x REPEATS) ===")
    print(rep_sum.to_string())"""
        ),
        md_cell(
            """## 🎨 12. Stage 8 — Generate 6 Publication-Grade 300-DPI Figures & Export Complete `.zip` Archive"""
        ),
        code_cell(
            """import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from IPython.display import Image, display

plt.rcParams.update({
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.35,
})

STYLE = {
    "fixed_2.500s":        ("Fixed (2.5s)",          "#3b6ea5", "o", 90),
    "change_aware":        ("Change-Aware (Tuned)",  "#e07b39", "s", 95),
    "rule_simple":         ("Rule: Simple (Δ+AoI)",  "#6ab187", "^", 100),
    "rule_ewma_cusum":     ("Rule: EWMA+CUSUM",      "#1b7837", "P", 125),
    "random":              ("Random Floor",          "#999999", "x", 85),
    "learnability_ceiling":("Learnability Ceiling",  "#762a83", "v", 105),
    "oracle":              ("Hindsight Oracle",      "#111111", "*", 175),
    "laya_raw":            ("Laya 421M (Raw)",       "#f4a582", "d", 95),
    "laya_calibrated":     ("Laya 421M (Calib.)",    "#ca0020", "D", 115),
    "clm_8b_calibrated":   ("CLM-v0.1-8B (Calib.)",  "#b2182b", "h", 120),
    "qwen25_7b_zeroshot":  ("Qwen2.5-7B (0-Shot)",   "#2166ac", "p", 120),
    "qwen25_7b_fewshot":   ("Qwen2.5-7B (4-Shot)",   "#053061", "H", 130),
}

# ------------------------------------------------------------------------------
# FIGURE 1: Dense Pareto Frontier (Primary 8-Link Low/Elev + SNDlib Abilene)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
panels = [
    ("8link", "regime_switching", "low_delay", "Primary 8-Link: Regime-Switching (Low Delay 50ms)"),
    ("8link", "regime_switching", "elevated_delay", "Primary 8-Link: Regime-Switching (Elev. Delay 500ms)"),
    ("abilene", "abilene_regime_switching", "low_delay", "External Validity: SNDlib Abilene 15-Link OSPF (50ms)"),
]
for ax, (topo, reg, dl, title) in zip(axes, panels):
    fsub = df_fixed_dense[(df_fixed_dense["topology"] == topo) & (df_fixed_dense["regime"] == reg) & (df_fixed_dense["delay"] == dl)]
    fcurve = fsub.groupby("fixed_interval_s", as_index=False)[["monitoring_kb", "mae_pct"]].mean().sort_values("monitoring_kb")
    ax.plot(fcurve["monitoring_kb"], fcurve["mae_pct"], "--", color="#3b6ea5", lw=2.0, alpha=0.85, label="Dense Fixed Frontier (0.5s–10s)")
    sub = df_all[(df_all["topology"] == topo) & (df_all["regime"] == reg) & (df_all["delay"] == dl)].groupby("controller", as_index=False)[["monitoring_kb", "mae_pct"]].mean()
    for ctrl, (lbl, clr, mk, sz) in STYLE.items():
        pt = sub[sub["controller"] == ctrl]
        if not pt.empty:
            ax.scatter(pt["monitoring_kb"], pt["mae_pct"], color=clr, marker=mk, s=sz,
                       edgecolor="black" if mk != "x" else clr, linewidth=0.8, zorder=6, label=lbl)
    ax.set_title(title)
    ax.set_xlabel("Application-Level Monitoring Traffic (KB)")
    ax.set_ylabel("Mean Absolute Twin-State Error (% pts)")
axes[0].legend(fontsize=7.3, loc="upper right", ncol=1)
plt.suptitle("Figure 1: True Pareto Frontier (Lower-Left Is Strictly Better)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig1_path = os.path.join(OUT_DIR, "fig1_pareto_frontiers.png")
plt.savefig(fig1_path, dpi=220)
plt.close(fig)

# ------------------------------------------------------------------------------
# FIGURE 2: Fraction of Oracle Headroom Captured (%) with 95% Bootstrap CIs
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
for ax, (topo, reg, dl, title) in zip(axes, [
    ("8link", "regime_switching", "low_delay", "Primary 8-Link Regime-Switching (Low Delay)"),
    ("abilene", "abilene_regime_switching", "low_delay", "SNDlib Abilene 15-Link OSPF (Low Delay)"),
]):
    piv = df_all[(df_all["topology"] == topo) & (df_all["regime"] == reg) & (df_all["delay"] == dl)].pivot_table(
        index="seed", columns="controller", values="cost_j", aggfunc="first"
    )
    if "fixed_2.500s" in piv and "change_aware" in piv and "oracle" in piv:
        best_st = np.minimum(piv["fixed_2.500s"], piv["change_aware"])
        mean_hr = float(np.maximum(1e-6, best_st - piv["oracle"]).mean())
        ctrls_show = [c for c in ["random", "laya_raw", "laya_calibrated", "clm_8b_calibrated",
                                  "qwen25_7b_zeroshot", "qwen25_7b_fewshot", "rule_simple",
                                  "rule_ewma_cusum", "learnability_ceiling"] if c in piv.columns]
        means, lo_e, hi_e, clrs, lbls = [], [], [], [], []
        for c in ctrls_show:
            pair = pd.DataFrame({"bs": best_st, "c": piv[c]}).dropna()
            pct_vals = 100.0 * (pair["bs"] - pair["c"]).to_numpy() / mean_hr
            m = float(np.mean(pct_vals))
            lo, hi = paired_bootstrap_ci(pct_vals)
            means.append(m); lo_e.append(max(0.0, m - lo)); hi_e.append(max(0.0, hi - m))
            lbl, clr, _, _ = STYLE.get(c, (c, "#555555", "o", 80))
            lbls.append(lbl); clrs.append(clr)
        y_pos = np.arange(len(lbls))
        ax.barh(y_pos, means, xerr=[lo_e, hi_e], color=clrs, edgecolor="black", alpha=0.88, capsize=3)
        ax.axvline(0.0, color="black", lw=1.2)
        ax.axvline(100.0, color="#111111", lw=1.2, ls=":", label="Oracle Ceiling (100%)")
        ax.set_yticks(y_pos); ax.set_yticklabels(lbls)
        ax.set_xlabel(f"Fraction of Oracle Headroom Captured (%)  [Mean Headroom H = {mean_hr:.3f} J]")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
plt.suptitle("Figure 2: Headroom Captured Relative to Best Static Policy (0%) and Oracle (100%)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig2_path = os.path.join(OUT_DIR, "fig2_headroom_captured.png")
plt.savefig(fig2_path, dpi=220)
plt.close(fig)

# ------------------------------------------------------------------------------
# FIGURE 3: Threshold-Free Signal Diagnostics (ROC Curves vs Per-Review Oracle)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))
rlog_df = pd.DataFrame(all_rlogs)
for ax, (reg, title) in zip(axes, [
    ("regime_switching", "Primary 8-Link: ROC vs. Per-Review Oracle Labels"),
    ("abilene_regime_switching", "SNDlib Abilene: ROC vs. Per-Review Oracle Labels"),
]):
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Chance (AUC = 0.50)")
    if not rlog_df.empty and "oracle_label" in rlog_df.columns:
        sub = rlog_df[(rlog_df["regime"] == reg) & (rlog_df["oracle_label"].isin(["fixed", "change-aware"]))]
        for ctrl, g in sub.groupby("controller"):
            y = (g["oracle_label"] == "change-aware").astype(int).to_numpy()
            s = g["score_ca"].to_numpy(dtype=np.float64)
            if len(np.unique(y)) == 2:
                fpr, tpr, _ = roc_curve(y, s)
                from sklearn.metrics import auc as sk_auc
                a_val = sk_auc(fpr, tpr)
                lbl, clr, _, _ = STYLE.get(ctrl, (ctrl, "#555555", "o", 80))
                ax.plot(fpr, tpr, lw=2.2, color=clr, label=f"{lbl} (AUC={a_val:.3f})")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate (Selecting Change-Aware When Fixed Optimal)")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8, loc="lower right")
plt.suptitle("Figure 3: Threshold-Free Decision Signal (Separating No Signal from Miscalibration)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig3_path = os.path.join(OUT_DIR, "fig3_roc_signal_diagnostics.png")
plt.savefig(fig3_path, dpi=220)
plt.close(fig)

# ------------------------------------------------------------------------------
# FIGURE 4: Forest Plot of Paired Delta-J vs rule_ewma_cusum with TOST SESOI Band
# ------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11.5, 5.5))
if not contrast_df.empty:
    fsub = contrast_df[(contrast_df["regime"] == "regime_switching") & (contrast_df["delay"] == "low_delay")].sort_values("mean_diff_j")
    y_pos = np.arange(len(fsub))
    ax.axvspan(-proto.sesoi_cost, proto.sesoi_cost, color="#d9f0d3", alpha=0.65, label=f"TOST Equivalence Band (±SESOI = ±{proto.sesoi_cost:.2f})")
    ax.axvline(0.0, color="black", lw=1.2)
    for i, (_, row) in enumerate(fsub.iterrows()):
        lbl, clr, _, _ = STYLE.get(row["controller"], (row["controller"], "#333333", "o", 80))
        m = row["mean_diff_j"]
        ax.errorbar(m, i, xerr=[[max(0.0, m - row["ci95_lo"])], [max(0.0, row["ci95_hi"] - m)]],
                    fmt="o", color=clr, ecolor="black", elinewidth=1.6, capsize=4, markersize=8)
        tag = "EQUIV" if row["equivalent_within_sesoi"] else (f"p={row['p_wilcoxon_holm']:.3f}")
        ax.text(row["ci95_hi"] + 0.015, i, f"{m:+.3f} [{tag}]", va="center", fontsize=8.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([STYLE.get(c, (c,))[0] for c in fsub["controller"]])
    ax.set_xlabel("Paired Cost Difference ΔJ = J(controller) − J(rule_ewma_cusum)  [Negative = Better]")
    ax.set_title("Figure 4: Paired Superiority & TOST Equivalence vs. EWMA+CUSUM Baseline (95% Bootstrap CI)")
    ax.legend(loc="lower right", fontsize=8.5)
plt.tight_layout()
fig4_path = os.path.join(OUT_DIR, "fig4_tost_forest_plot.png")
plt.savefig(fig4_path, dpi=220)
plt.close(fig)

# ------------------------------------------------------------------------------
# FIGURE 5: Homogeneous Control Regimes (Approved Research Plan §2.2)
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, reg in zip(axes, ["stable", "bursty", "alternating", "shifted"]):
    fsub = df_fixed_dense[(df_fixed_dense["regime"] == reg) & (df_fixed_dense["delay"] == "low_delay")]
    fc = fsub.groupby("fixed_interval_s", as_index=False)[["monitoring_kb", "mae_pct"]].mean().sort_values("monitoring_kb")
    ax.plot(fc["monitoring_kb"], fc["mae_pct"], "--", color="#3b6ea5", lw=1.8, label="Fixed Frontier")
    sub = df_all[(df_all["regime"] == reg) & (df_all["delay"] == "low_delay")].groupby("controller", as_index=False)[["monitoring_kb", "mae_pct"]].mean()
    for ctrl in ["fixed_2.500s", "change_aware", "rule_simple", "rule_ewma_cusum", "laya_calibrated", "oracle"]:
        pt = sub[sub["controller"] == ctrl]
        if not pt.empty:
            lbl, clr, mk, sz = STYLE[ctrl]
            ax.scatter(pt["monitoring_kb"], pt["mae_pct"], color=clr, marker=mk, s=sz, edgecolor="black", zorder=5, label=lbl)
    ax.set_title(f"Control: {reg}")
    ax.set_xlabel("Monitoring Traffic (KB)")
axes[0].set_ylabel("Mean Absolute Error (% pts)")
axes[0].legend(fontsize=7.2, loc="upper right")
plt.suptitle("Figure 5: Approved Plan §2.2 Homogeneous Control Regimes (Verifying Zero-Harm)", fontsize=13, fontweight="bold")
plt.tight_layout()
fig5_path = os.path.join(OUT_DIR, "fig5_homogeneous_controls.png")
plt.savefig(fig5_path, dpi=220)
plt.close(fig)

# ------------------------------------------------------------------------------
# FIGURE 6: Operational Overhead & Paraphrase Robustness (§2.4 Separate Reporting)
# ------------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 4.8))
model_ctrls = [c for c in ["laya_calibrated", "clm_8b_calibrated", "qwen25_7b_zeroshot", "qwen25_7b_fewshot"]
               if c in df_all["controller"].unique()]
if model_ctrls:
    lat_sub = df_all[df_all["controller"].isin(model_ctrls)].groupby("controller")[["latency_p50_ms", "latency_p95_ms"]].mean()
    x = np.arange(len(lat_sub))
    ax1.bar(x - 0.18, lat_sub["latency_p50_ms"], width=0.35, color="#4c72b0", edgecolor="black", label="P50 Latency (ms)")
    ax1.bar(x + 0.18, lat_sub["latency_p95_ms"], width=0.35, color="#c44e52", edgecolor="black", label="P95 Latency (ms)")
    ax1.axhline(proto.decision_deadline_ms, color="black", ls="--", lw=1.3, label="Deadline (2000 ms)")
    ax1.set_xticks(x); ax1.set_xticklabels([STYLE.get(c, (c,))[0] for c in lat_sub.index], rotation=12)
    ax1.set_ylabel("Inference Latency (ms)")
    ax1.set_title("Live GPU Decision Latency vs. 2000 ms Deadline")
    ax1.legend(fontsize=8)

# Paraphrase ROC-AUC spread across all screened models
p_names, p_aucs, p_clrs = [], [], []
for mk, scr in pilot_model_screens.items():
    for row in scr["all_paraphrases"]:
        p_names.append(f"{mk}\\nP{row['paraphrase_idx']}")
        p_aucs.append(row["pilot_roc_auc"])
        p_clrs.append("#2166ac" if "qwen" in mk else ("#b2182b" if "clm" in mk else "#ca0020"))
if p_names:
    ax2.bar(np.arange(len(p_names)), p_aucs, color=p_clrs, edgecolor="black", alpha=0.85)
    ax2.axhline(0.5, color="black", ls="--", lw=1.2, label="Chance (0.50)")
    ax2.set_xticks(np.arange(len(p_names))); ax2.set_xticklabels(p_names, fontsize=7.5, rotation=35)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Pilot ROC-AUC vs. Oracle")
    ax2.set_title("Prompt Paraphrase Robustness (4 Neutral Paraphrases + Order Swap)")
    ax2.legend(fontsize=8)
plt.suptitle("Figure 6: Operational Inference Latency & Prompt-Paraphrase Sensitivity", fontsize=13, fontweight="bold")
plt.tight_layout()
fig6_path = os.path.join(OUT_DIR, "fig6_operational_and_paraphrase.png")
plt.savefig(fig6_path, dpi=220)
plt.close(fig)

for fp in (fig1_path, fig2_path, fig3_path, fig4_path, fig5_path, fig6_path):
    display(Image(filename=fp))

zip_base = "/kaggle/working/ndt_phd_v2_complete_outputs" if os.path.exists("/kaggle/working") else "ndt_phd_v2_complete_outputs"
zip_path = shutil.make_archive(zip_base, "zip", root_dir=".", base_dir=OUT_DIR)
print(f"\\n🎉 ALL PhD v2 EXPERIMENTS & FIGURES COMPLETE! Download archive: {zip_path}")"""
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    nb = build_v2_notebook()
    out_path = os.path.join(HERE, "ndt_phd_v2_kaggle_experiments.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✓ Generated {out_path} ({len(nb['cells'])} cells)")
