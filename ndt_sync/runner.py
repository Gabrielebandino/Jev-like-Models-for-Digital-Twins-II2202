"""Experiment Runner, Comprehensive Logger, and Mentor Preview Plotter.

Executes:
1. Pre-generation of all Pilot and Held-Out CSV traces (`traces/`).
2. Mandatory 6-test Quality Assurance verification suite (`logs/qa_verification_report.json`).
3. Pilot and Held-Out baseline controller evaluations (`fixed` across 3 intervals,
   `change_aware`, and `rule_switcher`).
4. Sequential GPU Model Benchmark (Pilot Feasibility Gate + Held-Out + Repeat Stability Runs)
   for `laya`, `kev_4b`, and `clm_8b`.
5. Automated statistical analysis (paired seed differences, 95% confidence intervals,
   and publication-quality figures for the mentor review).
"""

from __future__ import annotations
import os
import sys
import json
import time
import platform
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import (
    DEFAULT_PROTOCOL,
    ExperimentProtocol,
    TRAFFIC_REGIMES,
    DELAY_CONDITIONS,
    PILOT_SEEDS,
    HELDOUT_SEEDS,
    REPEAT_STABILITY_SEEDS,
    TOPOLOGY_DEFINITION,
)
from .trace_generator import generate_and_save_all_traces, load_trace_csv
from .qa_checks import run_all_qa_checks
from .simulator import run_trace_simulation, SimulationRunResult, ReviewLogRecord
from .selectors import ReviewSummary, SelectorDecisionResult


def write_environment_manifest(logs_dir: str, protocol: ExperimentProtocol = DEFAULT_PROTOCOL) -> Dict[str, Any]:
    """Save complete reproducibility manifest (versions, seeds, protocol, topology) with zero credentials."""
    os.makedirs(logs_dir, exist_ok=True)
    manifest = {
        "project": "Decision-Model Policy Selection for Network Digital Twin Synchronization (II2202)",
        "authors": ["Gabriele Bandino <bandino@kth.se>", "Jago Revrenna <revrenna@kth.se>"],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "pilot_seeds": PILOT_SEEDS,
        "heldout_seeds": HELDOUT_SEEDS,
        "repeat_stability_seeds": REPEAT_STABILITY_SEEDS,
        "traffic_regimes": list(TRAFFIC_REGIMES),
        "delay_conditions": list(DELAY_CONDITIONS),
        "topology_summary": {
            "name": TOPOLOGY_DEFINITION["name"],
            "num_links": TOPOLOGY_DEFINITION["num_links"],
            "num_od_flows": len(TOPOLOGY_DEFINITION["od_flows"]),
        },
        "frozen_protocol": protocol.to_dict(),
    }
    try:
        import torch
        manifest["torch_version"] = torch.__version__
        manifest["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            manifest["cuda_device_count"] = int(torch.cuda.device_count())
            manifest["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception:
        manifest["torch_version"] = "not_installed_locally"

    manifest_path = os.path.join(logs_dir, "environment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    frozen_path = os.path.join(logs_dir, "frozen_protocol_config.json")
    protocol.save_json(frozen_path)
    return manifest


def save_simulation_logs(
    results: List[SimulationRunResult],
    output_dir: str,
    prefix: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Persist run summaries (.csv) and detailed review logs (.jsonl and .csv)."""
    os.makedirs(output_dir, exist_ok=True)
    summary_rows = [r.to_summary_dict() for r in results]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(output_dir, f"{prefix}_run_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    all_reviews: List[Dict[str, Any]] = []
    for r in results:
        for rev in r.review_logs:
            all_reviews.append(asdict(rev))

    reviews_df = pd.DataFrame(all_reviews)
    if not reviews_df.empty:
        reviews_csv_path = os.path.join(output_dir, f"{prefix}_reviews_log.csv")
        reviews_df.to_csv(reviews_csv_path, index=False)
        reviews_jsonl_path = os.path.join(output_dir, f"{prefix}_reviews_log.jsonl")
        with open(reviews_jsonl_path, "w", encoding="utf-8") as f:
            for item in all_reviews:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Save sample trajectory timeseries if present
    ts_list = []
    for r in results:
        if r.timeseries_df is not None:
            df_ts = r.timeseries_df.copy()
            df_ts.insert(0, "run_id", r.run_id)
            df_ts.insert(1, "controller", r.controller)
            df_ts.insert(2, "model_name", r.model_name)
            df_ts.insert(3, "regime", r.regime)
            df_ts.insert(4, "delay_condition", r.delay_condition)
            df_ts.insert(5, "seed", r.seed)
            ts_list.append(df_ts)
    if ts_list:
        full_ts = pd.concat(ts_list, ignore_index=True)
        full_ts.to_csv(os.path.join(output_dir, f"{prefix}_sample_trajectories.csv"), index=False)

    return summary_df, reviews_df


def run_deterministic_controllers_on_split(
    traces_dir: str,
    split: str,
    output_dir: str,
    protocol: ExperimentProtocol = DEFAULT_PROTOCOL,
) -> pd.DataFrame:
    """Run Fixed-Interval (all 3 candidate intervals), Change-Aware, and Rule-Switcher on all traces in `split`."""
    split_dir = os.path.join(traces_dir, split)
    csv_files = sorted(
        os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".csv")
    )
    results: List[SimulationRunResult] = []
    first_seed = PILOT_SEEDS[0] if split == "pilot" else HELDOUT_SEEDS[0]

    for fpath in csv_files:
        df = load_trace_csv(fpath)
        seed = int(df["seed"].iloc[0])
        rec_ts = (seed == first_seed)

        # 1. Fixed-interval across the 3 candidate intervals (§2.2: short, medium, long)
        for fi in protocol.candidate_fixed_intervals_s:
            is_default_fi = abs(fi - protocol.default_fixed_interval_s) < 1e-6
            res_f = run_trace_simulation(
                trace_df=df,
                controller_type="fixed",
                protocol=protocol,
                fixed_interval_s=fi,
                model_name=f"fixed_{fi:.1f}s",
                run_id_prefix=split,
                record_timeseries=(rec_ts and is_default_fi),
            )
            results.append(res_f)

        # 2. Static Change-Aware policy
        res_ca = run_trace_simulation(
            trace_df=df,
            controller_type="change_aware",
            protocol=protocol,
            fixed_interval_s=protocol.default_fixed_interval_s,
            model_name="change_aware",
            run_id_prefix=split,
            record_timeseries=rec_ts,
        )
        results.append(res_ca)

        # 3. Deterministic Threshold-Rule Switcher
        res_rule = run_trace_simulation(
            trace_df=df,
            controller_type="rule_switcher",
            protocol=protocol,
            fixed_interval_s=protocol.default_fixed_interval_s,
            model_name="threshold_rule",
            run_id_prefix=split,
            record_timeseries=rec_ts,
        )
        results.append(res_rule)

    summary_df, _ = save_simulation_logs(results, output_dir, prefix=f"{split}_baselines")
    return summary_df


def run_model_controller_on_split(
    traces_dir: str,
    split: str,
    model_name: str,
    model_selector_fn: Callable[[ReviewSummary], SelectorDecisionResult],
    output_dir: str,
    protocol: ExperimentProtocol = DEFAULT_PROTOCOL,
    run_repeat_stability: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Run a single Decision-Model selector across all pre-generated traces in `split`."""
    split_dir = os.path.join(traces_dir, split)
    csv_files = sorted(
        os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith(".csv")
    )
    results: List[SimulationRunResult] = []
    first_seed = PILOT_SEEDS[0] if split == "pilot" else HELDOUT_SEEDS[0]

    for fpath in csv_files:
        df = load_trace_csv(fpath)
        seed = int(df["seed"].iloc[0])
        rec_ts = (seed == first_seed)

        res_m = run_trace_simulation(
            trace_df=df,
            controller_type="model_switcher",
            protocol=protocol,
            fixed_interval_s=protocol.default_fixed_interval_s,
            model_selector_fn=model_selector_fn,
            model_name=model_name,
            run_id_prefix=split,
            record_timeseries=rec_ts,
        )
        results.append(res_m)

    summary_df, _ = save_simulation_logs(results, output_dir, prefix=f"{split}_model_{model_name}")

    repeat_df: Optional[pd.DataFrame] = None
    if run_repeat_stability and split == "heldout":
        # Research Plan §2.4 & Table 2: "repeat a predefined subset of model-selector runs"
        # "distinguish model variation on an identical trace from variation among independently generated traffic traces"
        repeat_results: List[SimulationRunResult] = []
        for fpath in csv_files:
            df = load_trace_csv(fpath)
            seed = int(df["seed"].iloc[0])
            if seed in REPEAT_STABILITY_SEEDS:
                for rep_idx in range(1, 4):
                    res_rep = run_trace_simulation(
                        trace_df=df,
                        controller_type="model_switcher",
                        protocol=protocol,
                        fixed_interval_s=protocol.default_fixed_interval_s,
                        model_selector_fn=model_selector_fn,
                        model_name=f"{model_name}_rep{rep_idx}",
                        run_id_prefix=f"repeat{rep_idx}",
                        record_timeseries=False,
                    )
                    repeat_results.append(res_rep)
        repeat_df, _ = save_simulation_logs(
            repeat_results, output_dir, prefix=f"heldout_repeat_stability_{model_name}"
        )

    return summary_df, repeat_df


def compute_paired_comparisons_table(
    combined_df: pd.DataFrame,
    output_dir: str,
    prefix: str = "heldout",
) -> pd.DataFrame:
    """Compute paired seed differences and 95% confidence intervals (§2.4).

    Pairs runs strictly by `(regime, delay_condition, seed)` across controllers.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Filter Fixed-Interval to the default 2.5s setting for 1-to-1 paired comparisons
    df_main = combined_df[
        (combined_df["controller"] != "fixed")
        | (np.isclose(combined_df["fixed_interval_setting_s"], DEFAULT_PROTOCOL.default_fixed_interval_s))
    ].copy()

    # Create unique controller label
    df_main["controller_label"] = np.where(
        df_main["controller"] == "model_switcher",
        "model:" + df_main["model_name"].astype(str),
        df_main["controller"],
    )

    controllers = sorted(df_main["controller_label"].unique())
    baselines = [c for c in ("fixed", "change_aware", "rule_switcher") if c in controllers]
    candidates = [c for c in controllers if c not in ("fixed", "change_aware")]

    paired_rows: List[Dict[str, Any]] = []
    for regime in TRAFFIC_REGIMES:
        for delay_cond in DELAY_CONDITIONS:
            sub = df_main[(df_main["regime"] == regime) & (df_main["delay_condition"] == delay_cond)]
            if sub.empty:
                continue
            for cand in candidates:
                cand_sub = sub[sub["controller_label"] == cand].set_index("seed")
                for base in baselines:
                    if cand == base:
                        continue
                    base_sub = sub[sub["controller_label"] == base].set_index("seed")
                    common_seeds = sorted(set(cand_sub.index).intersection(set(base_sub.index)))
                    if not common_seeds:
                        continue

                    mae_diff = (cand_sub.loc[common_seeds, "mae_pct_points"] - base_sub.loc[common_seeds, "mae_pct_points"]).to_numpy()
                    bytes_diff = (cand_sub.loc[common_seeds, "monitoring_bytes_total"] - base_sub.loc[common_seeds, "monitoring_bytes_total"]).to_numpy()
                    n = len(common_seeds)
                    se_mae = float(np.std(mae_diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    se_bytes = float(np.std(bytes_diff, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
                    t_crit = 2.201 if n == 12 else 1.96  # 95% two-sided t-interval for df=11 (n=12 seeds)

                    paired_rows.append(
                        {
                            "regime": regime,
                            "delay_condition": delay_cond,
                            "candidate_controller": cand,
                            "reference_baseline": base,
                            "num_paired_seeds": n,
                            "candidate_mean_mae_pct": round(float(cand_sub.loc[common_seeds, "mae_pct_points"].mean()), 4),
                            "baseline_mean_mae_pct": round(float(base_sub.loc[common_seeds, "mae_pct_points"].mean()), 4),
                            "paired_mae_diff_mean_pct": round(float(np.mean(mae_diff)), 4),
                            "paired_mae_diff_ci95_low": round(float(np.mean(mae_diff) - t_crit * se_mae), 4),
                            "paired_mae_diff_ci95_high": round(float(np.mean(mae_diff) + t_crit * se_mae), 4),
                            "candidate_mean_monitoring_bytes": round(float(cand_sub.loc[common_seeds, "monitoring_bytes_total"].mean()), 1),
                            "baseline_mean_monitoring_bytes": round(float(base_sub.loc[common_seeds, "monitoring_bytes_total"].mean()), 1),
                            "paired_bytes_diff_mean": round(float(np.mean(bytes_diff)), 1),
                            "paired_bytes_diff_ci95_low": round(float(np.mean(bytes_diff) - t_crit * se_bytes), 1),
                            "paired_bytes_diff_ci95_high": round(float(np.mean(bytes_diff) + t_crit * se_bytes), 1),
                        }
                    )

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(os.path.join(output_dir, f"{prefix}_paired_comparisons_ci95.csv"), index=False)
    return paired_df


def generate_mentor_plots(
    combined_df: pd.DataFrame,
    traces_dir: str,
    plots_dir: str,
    split: str = "heldout",
) -> List[str]:
    """Generate publication-quality plots for the mentor review & first report draft (§1.3, §2.4)."""
    os.makedirs(plots_dir, exist_ok=True)
    generated_files: List[str] = []

    # 1. Plot Sample Synthetic Network Traces across all 4 Regimes
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    seed_to_plot = HELDOUT_SEEDS[0] if split == "heldout" else PILOT_SEEDS[0]
    for idx, regime in enumerate(TRAFFIC_REGIMES):
        ax = axes[idx // 2, idx % 2]
        fpath = os.path.join(traces_dir, split, f"trace_{regime}_low_delay_seed{seed_to_plot}.csv")
        if os.path.exists(fpath):
            tdf = load_trace_csv(fpath)
            for l_idx in [0, 2, 4, 6]:
                ax.plot(tdf["timestamp_s"], tdf[f"u_L{l_idx}"], label=f"Link L{l_idx}", alpha=0.85, linewidth=1.2)
            ax.axvline(DEFAULT_PROTOCOL.warmup_duration_s, color="black", linestyle="--", alpha=0.6, label="Warm-up end (20s)")
            ax.set_title(f"Regime: {regime.capitalize()} (Seed {seed_to_plot})", fontsize=11, fontweight="bold")
            ax.set_ylabel("Link Utilization (%)")
            ax.grid(True, alpha=0.3)
            if idx >= 2:
                ax.set_xlabel("Simulation Time (s)")
            if idx == 0:
                ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Synthetic 8-Link Network Digital Twin Workload Regimes (Shared Multi-Hop Paths)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(plots_dir, f"{split}_fig1_synthetic_traffic_regimes.png")
    fig.savefig(p1, dpi=180)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Trade-off Plot: Mean Absolute Twin-State Error vs. Application-Level Monitoring Bytes (§1.3, §2.4)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), sharey=False)
    markers = {
        "fixed": "o",
        "change_aware": "s",
        "rule_switcher": "D",
        "model_switcher": "*",
    }
    colors = {
        "fixed": "#1f77b4",
        "change_aware": "#2ca02c",
        "rule_switcher": "#ff7f0e",
        "laya": "#d62728",
        "kev_4b": "#9467bd",
        "clm_8b": "#8c564b",
    }

    for r_idx, delay_cond in enumerate(DELAY_CONDITIONS):
        for c_idx, regime in enumerate(TRAFFIC_REGIMES):
            ax = axes[r_idx, c_idx]
            sub = combined_df[(combined_df["regime"] == regime) & (combined_df["delay_condition"] == delay_cond)]
            if sub.empty:
                continue

            # Plot Fixed-Interval curve across the 3 intervals (1.0s, 2.5s, 5.0s)
            f_sub = sub[sub["controller"] == "fixed"].groupby("fixed_interval_setting_s").agg(
                mae_mean=("mae_pct_points", "mean"),
                mae_std=("mae_pct_points", "std"),
                bytes_mean=("monitoring_bytes_total", "mean"),
            ).reset_index().sort_values("bytes_mean")
            if not f_sub.empty:
                ax.plot(
                    f_sub["bytes_mean"] / 1000.0,
                    f_sub["mae_mean"],
                    "--o",
                    color=colors["fixed"],
                    label="Fixed (1.0s, 2.5s, 5.0s)",
                    alpha=0.85,
                )
                for _, row in f_sub.iterrows():
                    ax.annotate(
                        f"{row['fixed_interval_setting_s']:.1f}s",
                        (row["bytes_mean"] / 1000.0, row["mae_mean"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7.5,
                    )

            # Plot Change-Aware and Rule-Switcher
            for ctrl, label_str in [("change_aware", "Change-Aware"), ("rule_switcher", "Rule Switcher")]:
                c_sub = sub[sub["controller"] == ctrl]
                if not c_sub.empty:
                    bx = c_sub["monitoring_bytes_total"].mean() / 1000.0
                    by = c_sub["mae_pct_points"].mean()
                    yerr = c_sub["mae_pct_points"].std() / np.sqrt(max(1, len(c_sub)))
                    ax.errorbar(
                        bx, by, yerr=yerr, fmt=markers[ctrl], color=colors[ctrl],
                        markersize=8, capsize=3, label=label_str
                    )

            # Plot any Decision-Model Switchers present
            m_sub = sub[sub["controller"] == "model_switcher"]
            for m_name, m_grp in m_sub.groupby("model_name"):
                bx = m_grp["monitoring_bytes_total"].mean() / 1000.0
                by = m_grp["mae_pct_points"].mean()
                yerr = m_grp["mae_pct_points"].std() / np.sqrt(max(1, len(m_grp)))
                col = colors.get(str(m_name), "#e377c2")
                ax.errorbar(
                    bx, by, yerr=yerr, fmt="*", color=col,
                    markersize=11, capsize=3, label=f"Model ({m_name})"
                )

            ax.set_title(f"{regime.capitalize()} | {delay_cond.replace('_', ' ')}", fontsize=10, fontweight="bold")
            if r_idx == 1:
                ax.set_xlabel("Monitoring Traffic (KB)")
            if c_idx == 0:
                ax.set_ylabel("Twin-State MAE (% pts)")
            ax.grid(True, alpha=0.3)
            if r_idx == 0 and c_idx == 0:
                ax.legend(loc="best", fontsize=7.5)

    fig.suptitle("Twin-State Error vs. Application-Level Monitoring Traffic Across Regimes & Delays", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p2 = os.path.join(plots_dir, f"{split}_fig2_error_vs_traffic_tradeoff.png")
    fig.savefig(p2, dpi=180)
    plt.close(fig)
    generated_files.append(p2)

    return generated_files
