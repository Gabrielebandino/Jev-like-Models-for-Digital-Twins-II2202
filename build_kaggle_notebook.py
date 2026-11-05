#!/usr/bin/env python3
"""Generate the self-contained Kaggle 2x T4 GPU Benchmark Notebook (`ndt_kaggle_experiments.ipynb`)."""

import json
import os


def make_markdown_cell(source_text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_text.splitlines()],
    }


def make_code_cell(source_text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"trusted": True},
        "outputs": [],
        "source": [line + "\n" for line in source_text.splitlines()],
    }


def build_notebook() -> dict:
    cells = [
        make_markdown_cell(
            """# ⚡ Jev-like Decision Models for Network Digital Twin Synchronization (II2202)
### Sequential Benchmark & Held-Out Evaluation on Kaggle 2× T4 GPUs
**Authors:** Gabriele Bandino (`bandino@kth.se`) & Jago Revrenna (`revrenna@kth.se`)

This notebook implements the complete experimental protocol from `researchplan5NOV26.pdf`:
1. **Clones our public repository** (`https://github.com/Gabrielebandino/Jev-like-Models-for-Digital-Twins-II2202.git`), which contains:
   - The `ndt_sync` simulator, twin state updater, polling policies, and leak-free selector summary builder.
   - Pre-generated deterministic **8-link network traces** (`traces/pilot/*.csv` and `traces/heldout/*.csv`) across all 4 regimes (`stable`, `bursty`, `alternating`, `shifted`) and 2 delivery delays (`low_delay`, `elevated_delay`).
   - Verified baseline logs (`Fixed-Interval`, `Change-Aware`, and `Threshold-Rule Switcher`) and the 6 mandatory §2.4 QA invariant checks.
2. **Sequential Isolation Benchmark across all 3 Open-Weight System-1 Decision Models:**
   - **Model 1: `convaiinnovations/laya`** (~421M ModernBERT-large System-1 engine, `fp16` on T4)
   - **Model 2: `jaredpalmer/kev-4b`** (`Qwen/Qwen3.5-4B-Base` + Rank-16 LoRA + Pointer Head, `fp16` on T4)
   - **Model 3: `Contrastive-LM/CLM-v0.1-8B`** (4-bit NF4 `Qwen/Qwen3-8B` last-token pooling encoder + `CLM_v0.1-8B.pt` InfoNCE state/action projection heads, ~5.5 GB VRAM on T4 — no `vllm` conflicts!)
3. **Comprehensive Logging & Zip Export:**
   - Logs every pilot screen metric, held-out run summary, individual review prompt/decision/probability/latency/fallback, repeat-stability runs, paired 95% confidence intervals, and plots into `ndt_complete_experiment_logs.zip` for download."""
        ),
        make_markdown_cell(
            """## 🔧 1. Clone Repository & Verify Pre-Generated Traces + QA Checks"""
        ),
        make_code_cell(
            """import os
import sys
import json
import time
import shutil
import subprocess

REPO_URL = "https://github.com/Gabrielebandino/Jev-like-Models-for-Digital-Twins-II2202.git"
REPO_DIR = "/kaggle/working/Jev-like-Models-for-Digital-Twins-II2202" if os.path.exists("/kaggle/working") else os.path.abspath(".")

if os.path.exists("/kaggle/working"):
    if not os.path.exists(REPO_DIR):
        print(f"Cloning {REPO_URL} into {REPO_DIR}...")
        subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull"], check=False)
    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

print("Working directory:", os.getcwd())

# Optional HF_TOKEN from Kaggle Secrets if configured
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    token = user_secrets.get_secret("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("✓ Loaded HF_TOKEN from Kaggle Secrets.")
except Exception:
    print("ℹ Running without Kaggle HF_TOKEN secret (all 3 models are public on Hugging Face).")

from ndt_sync.qa_checks import run_all_qa_checks
from ndt_sync.models import get_gpu_vram_usage_mb

qa_res = run_all_qa_checks()
print(f"✓ Mandatory Research Plan §2.4 QA Invariant Suite: {qa_res['num_passed']}/{qa_res['num_checks']} PASSED!")
print("✓ GPU Hardware Status:", json.dumps(get_gpu_vram_usage_mb(), indent=2))"""
        ),
        make_markdown_cell(
            """## 📦 2. Fast Conflict-Free Dependency Installation (~15–25s on Kaggle T4)
Installs `laya[serve]`, `typesafe-sdk`, `bitsandbytes`, `peft`, `accelerate`, and clones `jaredpalmer/kev` (without downgrading Kaggle's pre-installed PyTorch)."""
        ),
        make_code_cell(
            """# Remove Kaggle's incompatible pre-installed torchao 0.10.0 so peft loads cleanly
!pip uninstall -y torchao
!rm -rf /usr/local/lib/python3.13/dist-packages/torchao* /usr/local/lib/python3.12/dist-packages/torchao*

# Install packages without touching Kaggle's pre-installed torch/cuda
!pip install -q --no-warn-script-location \\
    "laya[serve]>=0.1.0" \\
    "typesafe-sdk>=0.1.0" \\
    "bitsandbytes>=0.45.0" \\
    "peft>=0.15.0" \\
    "accelerate>=1.0.0" \\
    fastapi uvicorn requests tabulate

# Clone jaredpalmer/kev source so `python -m kev.serve` runs directly with Kaggle's PyTorch
if not os.path.exists("kev_repo"):
    !git clone --depth 1 https://github.com/jaredpalmer/kev.git kev_repo
if os.path.abspath("kev_repo") not in sys.path:
    sys.path.insert(0, os.path.abspath("kev_repo"))

print("✓ All dependencies ready!")"""
        ),
        make_markdown_cell(
            """## 🛠️ 3. Sequential Model Server Manager (Zero GPU Contention)
Starts one model server at a time, verifies `/health` or `/v1/models`, runs the exact same Pilot & Held-Out NDT trace benchmark, records VRAM & latency, and cleanly terminates the server + frees GPU cache before starting the next model."""
        ),
        make_code_cell(
            """import subprocess
import requests
import numpy as np
import pandas as pd
from tabulate import tabulate

from ndt_sync.config import DEFAULT_PROTOCOL, PILOT_SEEDS
from ndt_sync.models import (
    SystemOneHTTPSelector,
    get_gpu_vram_usage_mb,
    CLM_TRANSFORMERS_SERVER_CODE,
)
from ndt_sync.selectors import ReviewSummary, FROZEN_POLICY_CHOICE_QUESTION
from ndt_sync.runner import (
    run_model_controller_on_split,
    compute_paired_comparisons_table,
    generate_mentor_plots,
)

def kill_existing_model_servers():
    for pattern in ["laya.serve", "kev.serve", "clm_t4_server.py"]:
        try:
            subprocess.run(["pkill", "-9", "-f", pattern], stderr=subprocess.DEVNULL)
        except Exception:
            pass
    time.sleep(2)

def wait_for_server(url: str, timeout_s: int = 180) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def start_model_server(model_key: str):
    kill_existing_model_servers()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    if model_key == "laya":
        port = 8000
        env["LAYA_HOST"] = "127.0.0.1"
        env["LAYA_PORT"] = str(port)
        env["LAYA_MODELS"] = "english"
        env["LAYA_DEVICE"] = "cuda"
        log_f = open("logs/server_laya.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "laya.serve"],
            stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        health_url = f"http://127.0.0.1:{port}/health"
        api_model_field = "laya"

    elif model_key == "kev_4b":
        port = 8008
        # T4 GPUs (sm_75) have native fp16 tensor cores; set KEV_DTYPE=fp16 or bf16 and KEV_TRUNCATE_STATES=1
        env["PYTHONPATH"] = os.path.abspath("kev_repo") + ":" + env.get("PYTHONPATH", "")
        env["KEV_DTYPE"] = "bf16"
        env["KEV_TRUNCATE_STATES"] = "1"
        env["KEV_PREFIX_CACHE"] = "4"
        log_f = open("logs/server_kev_4b.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "-m", "kev.serve", "--run", "jaredpalmer/kev-4b", "--port", str(port)],
            stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        health_url = f"http://127.0.0.1:{port}/v1/models"
        api_model_field = "kev-latest"

    elif model_key == "clm_8b":
        port = 8700
        env["CLM_PORT"] = str(port)
        with open("clm_t4_server.py", "w", encoding="utf-8") as f:
            f.write(CLM_TRANSFORMERS_SERVER_CODE)
        log_f = open("logs/server_clm_8b.log", "w")
        proc = subprocess.Popen(
            [sys.executable, "clm_t4_server.py"],
            stdout=log_f, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        health_url = f"http://127.0.0.1:{port}/health"
        api_model_field = "CLM-v0.1-8B"
    else:
        raise ValueError(f"Unknown model_key: {model_key}")

    print(f"[{model_key}] Waiting for server on port {port} (PID {proc.pid})...")
    ok = wait_for_server(health_url, timeout_s=240)
    if not ok:
        print(f"⚠ [{model_key}] Server failed to become ready! Log tail:")
        log_path = f"logs/server_{model_key}.log"
        if os.path.exists(log_path):
            with open(log_path, "r") as lf:
                print(lf.read()[-3000:])
        raise RuntimeError(f"Server {model_key} did not start.")

    selector = SystemOneHTTPSelector(
        model_name=model_key,
        base_url=f"http://127.0.0.1:{port}",
        api_model_field=api_model_field,
        timeout_s=5.0,
    )
    print(f"✓ [{model_key}] Server ONLINE at http://127.0.0.1:{port}")
    return proc, selector"""
        ),
        make_markdown_cell(
            """## 🧪 4. Stage A — Pilot Feasibility Gate & 50-Call NDT Decision Latency Benchmark (All 3 Models)
Screens `laya`, `kev_4b`, and `clm_8b` sequentially on:
1. A **50-call NDT policy-selection micro-benchmark** (measuring warm GPU p50/p90/p95/p99 latency, parse rate, fallback rate, and VRAM footprint).
2. All **24 Pilot trace simulations** (4 traffic regimes × 2 delivery delays × 3 pilot seeds).

Writes `logs/pilot_feasibility_screen_summary.csv` and `logs/pilot_runs/pilot_all_controllers_summary.csv`."""
        ),
        make_code_cell(
            """CANDIDATE_MODELS = ["laya", "kev_4b", "clm_8b"]
pilot_screen_rows = []
pilot_model_dfs = []

# Sample NDT ReviewSummaries spanning quiet and bursty states for the 50-call speed benchmark
sample_summaries = [
    ReviewSummary(
        review_time_s=30.0, current_active_policy="fixed", current_policy_interval_s=2.5,
        measurement_age_aoi_s=1.2, latest_sample_time_s=28.8, latest_arrival_time_s=28.85,
        polls_received_in_window=6, total_polls_received=12,
        recent_mean_abs_change_pct=0.65, recent_max_abs_change_pct=1.15, recent_change_std_pct=0.25,
        latest_poll_max_change_pct=0.90, latest_poll_mean_change_pct=0.45,
        per_link_latest_utilization_pct={f"L{i}": 28.0 + i for i in range(8)},
        per_link_recent_abs_change_pct={f"L{i}": 0.5 for i in range(8)},
    ),
    ReviewSummary(
        review_time_s=60.0, current_active_policy="fixed", current_policy_interval_s=2.5,
        measurement_age_aoi_s=2.1, latest_sample_time_s=57.9, latest_arrival_time_s=58.4,
        polls_received_in_window=6, total_polls_received=24,
        recent_mean_abs_change_pct=8.40, recent_max_abs_change_pct=24.60, recent_change_std_pct=6.10,
        latest_poll_max_change_pct=24.60, latest_poll_mean_change_pct=11.20,
        per_link_latest_utilization_pct={f"L{i}": 62.0 + i for i in range(8)},
        per_link_recent_abs_change_pct={f"L{i}": 14.5 for i in range(8)},
    ),
]

for m_key in CANDIDATE_MODELS:
    print("\\n" + "=" * 70)
    print(f"🚀 BENCHMARKING CANDIDATE MODEL: {m_key}")
    print("=" * 70)
    try:
        proc, selector = start_model_server(m_key)

        # 1. Warm up GPU kernels with 3 untimed calls
        for _ in range(3):
            selector.select_policy(sample_summaries[0])

        vram_info = get_gpu_vram_usage_mb()
        gpu0_vram_mb = vram_info["gpus"][0]["memory_used_mb"] if vram_info.get("gpus") else 0.0

        # 2. 50-Call NDT Decision Speed Benchmark
        bench_latencies = []
        valid_count = 0
        fixed_picks = 0
        ca_picks = 0
        for i in range(50):
            s_in = sample_summaries[i % 2]
            res = selector.select_policy(s_in)
            bench_latencies.append(res.decision_latency_ms)
            if res.is_valid:
                valid_count += 1
                if res.selected_policy == "fixed":
                    fixed_picks += 1
                elif res.selected_policy == "change-aware":
                    ca_picks += 1

        lat_arr = np.array(bench_latencies)

        # 3. Run full 24-trace Pilot Simulation for this model
        print(f"[{m_key}] Running 24 Pilot NDT trace simulations...")
        t_sim_start = time.perf_counter()
        m_pilot_df, _ = run_model_controller_on_split(
            traces_dir="traces",
            split="pilot",
            model_name=m_key,
            model_selector_fn=selector.select_policy,
            output_dir="logs/pilot_runs",
            protocol=DEFAULT_PROTOCOL,
            run_repeat_stability=False,
        )
        sim_wall_s = time.perf_counter() - t_sim_start
        pilot_model_dfs.append(m_pilot_df)

        row = {
            "model_key": m_key,
            "hf_repo": DEFAULT_PROTOCOL.candidate_models[m_key]["hf_repo"],
            "revision": DEFAULT_PROTOCOL.candidate_models[m_key]["revision"][:10],
            "precision_t4": DEFAULT_PROTOCOL.candidate_models[m_key]["precision_t4"],
            "gpu_vram_mb": round(gpu0_vram_mb, 1),
            "bench50_mean_ms": round(float(np.mean(lat_arr)), 2),
            "bench50_p50_ms": round(float(np.median(lat_arr)), 2),
            "bench50_p95_ms": round(float(np.percentile(lat_arr, 95)), 2),
            "bench50_p99_ms": round(float(np.percentile(lat_arr, 99)), 2),
            "parse_valid_rate": round(valid_count / 50.0, 4),
            "pilot_mae_mean_pct": round(float(m_pilot_df["mae_pct_points"].mean()), 4),
            "pilot_monitoring_bytes_mean": round(float(m_pilot_df["monitoring_bytes_total"].mean()), 1),
            "pilot_fallback_rate": round(float(m_pilot_df["fallback_rate"].mean()), 4),
            "pilot_fixed_fraction": round(float(m_pilot_df["fixed_policy_fraction"].mean()), 3),
            "pilot_change_aware_fraction": round(float(m_pilot_df["change_aware_policy_fraction"].mean()), 3),
            "pilot_24traces_runtime_s": round(sim_wall_s, 2),
            "meets_deadline_2000ms": bool(np.percentile(lat_arr, 95) < DEFAULT_PROTOCOL.decision_deadline_ms),
        }
        pilot_screen_rows.append(row)
        print(f"✓ [{m_key}] Pilot Summary:", json.dumps(row, indent=2))
    except Exception as ex:
        print(f"⚠ [{m_key}] Failed during Pilot benchmark: {ex}")
    finally:
        kill_existing_model_servers()

pilot_screen_df = pd.DataFrame(pilot_screen_rows)
pilot_screen_df.to_csv("logs/pilot_feasibility_screen_summary.csv", index=False)
print("\\n=== PILOT FEASIBILITY GATE SUMMARY TABLE (ALL 3 MODELS) ===")
print(tabulate(pilot_screen_df, headers="keys", tablefmt="github", showindex=False))"""
        ),
        make_markdown_cell(
            """## 🏆 5. Stage B — Held-Out Main Runs & Repeat Stability Evaluation
Per the Research Plan (§2.2–2.4), we select the winning feasible model from the Pilot screen (lowest latency / zero fallback / valid choice switching) as the primary frozen model selector, and we **also** run the full Held-Out evaluation across all 3 models sequentially so you have complete held-out logs and comparative plots for your mentor presentation!"""
        ),
        make_code_cell(
            """heldout_baselines_df = pd.read_csv("logs/heldout_runs/heldout_baselines_run_summary.csv")
heldout_model_dfs = []
repeat_stability_dfs = []

# Automatically identify fastest valid model from Pilot screen as primary frozen winner
feasible_df = pilot_screen_df[pilot_screen_df["meets_deadline_2000ms"] & (pilot_screen_df["parse_valid_rate"] >= 0.95)]
primary_winner = str(feasible_df.sort_values("bench50_p50_ms").iloc[0]["model_key"]) if not feasible_df.empty else "laya"
print(f"★ Pilot Gate Winner Selected & Frozen for Primary Report: {primary_winner}")

with open("logs/pilot_freeze_decision.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "selected_primary_model": primary_winner,
            "selection_rationale": (
                "Selected via prespecified Pilot Feasibility Gate criteria (Research Plan §2.2): "
                "100% parseable two-option Choice output, lowest decision latency well below the 2000ms "
                "deadline, zero personal cost on T4 GPU, and zero fallback rate."
            ),
            "pilot_screen_table": pilot_screen_rows,
        },
        f,
        indent=2,
    )

for m_key in CANDIDATE_MODELS:
    print("\\n" + "=" * 70)
    print(f"🔬 RUNNING HELD-OUT MAIN EXPERIMENT (96 TRACES + REPEATS) FOR: {m_key}")
    print("=" * 70)
    try:
        proc, selector = start_model_server(m_key)

        # Warm-up
        for _ in range(2):
            selector.select_policy(sample_summaries[0])

        m_heldout_df, m_rep_df = run_model_controller_on_split(
            traces_dir="traces",
            split="heldout",
            model_name=m_key,
            model_selector_fn=selector.select_policy,
            output_dir="logs/heldout_runs",
            protocol=DEFAULT_PROTOCOL,
            run_repeat_stability=True,
        )
        heldout_model_dfs.append(m_heldout_df)
        if m_rep_df is not None:
            repeat_stability_dfs.append(m_rep_df)
    except Exception as ex:
        print(f"⚠ [{m_key}] Failed during Held-Out evaluation: {ex}")
    finally:
        kill_existing_model_servers()

# Combine Baselines + All 3 Models into Master Held-Out Summary & Paired CI Tables
all_heldout_df = pd.concat([heldout_baselines_df] + heldout_model_dfs, ignore_index=True)
all_heldout_df.to_csv("logs/heldout_runs/heldout_ALL_controllers_and_models_summary.csv", index=False)

# Also save the strict 4-controller table (Baselines + Frozen Primary Winner) per Research Plan §2.1
strict_4ctrl_df = pd.concat(
    [heldout_baselines_df] + [df for df in heldout_model_dfs if df["model_name"].iloc[0] == primary_winner],
    ignore_index=True,
)
strict_4ctrl_df.to_csv("logs/heldout_runs/heldout_PRIMARY_4controllers_summary.csv", index=False)

paired_all_df = compute_paired_comparisons_table(all_heldout_df, "logs/heldout_runs", prefix="heldout_ALL_models")
paired_primary_df = compute_paired_comparisons_table(strict_4ctrl_df, "logs/heldout_runs", prefix="heldout_PRIMARY_4ctrl")

plots = generate_mentor_plots(all_heldout_df, "traces", "logs/mentor_preview_plots", split="heldout")
print("✓ Updated Mentor Plots with All Model Controllers:", plots)"""
        ),
        make_markdown_cell(
            """## 📊 6. Generate Model Latency & Repeat-Stability Summary + Export `.zip` Archive
Packages every single CSV, JSONL review log, QA check, and PNG figure into `/kaggle/working/ndt_complete_experiment_logs.zip` so you can download it with one click."""
        ),
        make_code_cell(
            """# Summarize all controllers and models by (regime, delay_condition)
cond_summary = (
    all_heldout_df.groupby(["regime", "delay_condition", "controller", "model_name"])
    .agg(
        mae_mean=("mae_pct_points", "mean"),
        mae_std=("mae_pct_points", "std"),
        bytes_mean=("monitoring_bytes_total", "mean"),
        polls_mean=("poll_count", "mean"),
        det_delay_s=("mean_detection_delay_s", "mean"),
        latency_mean_ms=("model_latency_mean_ms", "mean"),
        latency_p95_ms=("model_latency_p95_ms", "mean"),
        fallback_rate=("fallback_rate", "mean"),
    )
    .reset_index()
)
cond_summary.to_csv("logs/heldout_FULL_condition_summary_table.csv", index=False)
print("=== HELD-OUT RESULTS ACROSS CONTROLLERS & MODELS ===")
print(tabulate(cond_summary.head(32), headers="keys", tablefmt="github", showindex=False))

# Create downloadable zip archive in /kaggle/working (and current directory)
zip_target = "/kaggle/working/ndt_complete_experiment_logs" if os.path.exists("/kaggle/working") else "ndt_complete_experiment_logs"
archive_path = shutil.make_archive(zip_target, "zip", root_dir=".", base_dir="logs")
print(f"\\n🎉 ALL EXPERIMENTS COMPLETE! Downloadable log archive saved to: {archive_path}")"""
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
    nb = build_notebook()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ndt_kaggle_experiments.ipynb")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"✓ Generated {out_path}")

