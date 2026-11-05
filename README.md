# Jev-like Decision Models for Network Digital Twin Synchronization (II2202)

**Authors:** Gabriele Bandino (`bandino@kth.se`) & Jago Revrenna (`revrenna@kth.se`)  
**Course:** KTH II2202 — Research Methodology and Scientific Writing (Autumn 2026)

---

## 1. Overview & Experimental Design

This repository implements the quantitative trace-driven simulation, polling policies, policy selectors, quality assurance (QA) verification suite, and Kaggle T4 GPU benchmark for **Decision-Model Policy Selection for Network Digital Twin Synchronization**.

At scheduled review points, a policy selector observes a leak-free summary of measurements already received by the Network Digital Twin (NDT) across an **8-link shared-path network topology** and selects which polling policy should control the next observation period:
- **`fixed` (Fixed-Interval Polling):** Polls all 8 links at a constant preset interval (`2.5s` default; `1.0s`, `2.5s`, and `5.0s` evaluated for the static trade-off curve).
- **`change-aware` (OpenNetMon-Inspired Adaptive Polling):** Shortens its polling interval after observing large link-utilization changes and lengthens it after consecutive quiet periods within `[0.5s, 6.0s]`.

### Four Compared Controllers
1. **Static Fixed-Interval (`fixed`)**
2. **Static Change-Aware (`change_aware`)**
3. **Deterministic Threshold-Rule Switcher (`rule_switcher`)**
4. **System-1 Decision-Model Switcher (`model_switcher`)**

### Three Open-Weight System-1 Candidate Decision Models Evaluated
- **`convaiinnovations/laya`** (~421M ModernBERT-large non-autoregressive System-1 model)
- **`jaredpalmer/kev-4b`** (`Qwen/Qwen3.5-4B-Base` + Rank-16 LoRA + calibrated Pointer Head)
- **`Contrastive-LM/CLM-v0.1-8B`** (4-bit NF4 `Qwen/Qwen3-8B` last-token pooling encoder + `CLM_v0.1-8B.pt` InfoNCE state/action projection heads)

---

## 2. Repository Structure

- `ndt_sync/`: Core Python simulation and evaluation package
  - `config.py`: 8-link shared-path topology, 4 traffic regimes (`stable`, `bursty`, `alternating`, `shifted`), 2 delivery-delay conditions (`low_delay`, `elevated_delay`), seed splits (`PILOT_SEEDS` vs. `HELDOUT_SEEDS`), and frozen protocol parameters.
  - `trace_generator.py`: Deterministic synthetic traffic generator saving and loading CSV traces.
  - `twin.py`: Network Digital Twin state store, delayed delivery queue, out-of-order rejection, and `FixedIntervalPolicy` / `ChangeAwarePolicy`.
  - `selectors.py`: Leak-free review summary builder (`ReviewSummary`), frozen `/v1/systemone` `Choice` schema, and `DeterministicRuleSelector`.
  - `simulator.py`: Discrete-time trace-driven simulator enforcing asynchronous decision latency, single in-flight request (no backlog), deadline timeouts/fallbacks, and switching without extra polls.
  - `models.py`: Unified HTTP `/v1/systemone` client and standalone 4-bit Transformers server for `CLM-v0.1-8B`.
  - `qa_checks.py`: The 6 mandatory verification checks from Research Plan §2.4.
  - `runner.py`: Experiment runner, paired confidence-interval calculator, and plot generator.
- `traces/`: Pre-generated deterministic CSV traces (`24` Pilot traces in `traces/pilot/` and `96` Held-Out traces in `traces/heldout/`) so all controllers and models are tested on the exact same data.
- `logs/`: Verified QA reports (`logs/qa_verification_report.json`), environment manifest (`logs/environment_manifest.json`), baseline run logs (`logs/pilot_runs/`, `logs/heldout_runs/`), and preview figures (`logs/mentor_preview_plots/`).
- `ndt_kaggle_experiments.ipynb`: Self-contained Kaggle 2× T4 GPU notebook that clones this repository, benchmarks all 3 open-weight decision models sequentially on the pre-generated CSV traces, and exports `ndt_complete_experiment_logs.zip`.

---

## 3. Running Locally vs. on Kaggle GPUs

### Local Execution (QA Suite + Trace Generation + Deterministic Baselines)
```bash
python3 run_local_setup_and_baselines.py
```

### Kaggle 2× T4 GPU Execution (All 3 Decision Models)
1. Upload `ndt_kaggle_experiments.ipynb` to Kaggle.
2. Set **Accelerator** to **GPU T4 x2** and **Internet** to **ON**.
3. Click **Run All** and download `/kaggle/working/ndt_complete_experiment_logs.zip`.
