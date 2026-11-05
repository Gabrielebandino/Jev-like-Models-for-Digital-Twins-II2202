#!/usr/bin/env python3
"""Local Execution Script: Generates all CSV Traces, Runs QA Suite, and Evaluates Baselines."""

import os
import json
import time
from ndt_sync.config import DEFAULT_PROTOCOL
from ndt_sync.trace_generator import generate_and_save_all_traces
from ndt_sync.qa_checks import run_all_qa_checks
from ndt_sync.runner import (
    write_environment_manifest,
    run_deterministic_controllers_on_split,
    compute_paired_comparisons_table,
    generate_mentor_plots,
)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    traces_dir = os.path.join(base_dir, "traces")
    logs_dir = os.path.join(base_dir, "logs")
    plots_dir = os.path.join(logs_dir, "mentor_preview_plots")
    os.makedirs(logs_dir, exist_ok=True)

    print("=== 1. Writing Reproducibility Environment Manifest & Frozen Protocol ===")
    write_environment_manifest(logs_dir, DEFAULT_PROTOCOL)

    print("=== 2. Running All 6 Mandatory Research Plan §2.4 QA Invariant Checks ===")
    qa_report_path = os.path.join(logs_dir, "qa_verification_report.json")
    qa_report = run_all_qa_checks(save_report_path=qa_report_path)
    for c in qa_report["checks"]:
        status = "PASS" if c["passed"] else "FAIL"
        print(f"  [Check #{c['check_id']}] {c['name']}: {status} -> {c['details']}")
    if not qa_report["all_passed"]:
        raise RuntimeError("One or more mandatory QA invariant checks failed!")
    print(f"✓ All {qa_report['num_passed']}/{qa_report['num_checks']} QA checks PASSED! Saved to {qa_report_path}\n")

    print("=== 3. Pre-Generating Synthetic 8-Link Network Traces (Pilot & Held-Out CSVs) ===")
    t0 = time.perf_counter()
    trace_manifest = generate_and_save_all_traces(traces_dir, DEFAULT_PROTOCOL)
    print(
        f"✓ Generated {len(trace_manifest['pilot'])} Pilot traces and "
        f"{len(trace_manifest['heldout'])} Held-Out traces in {time.perf_counter() - t0:.2f}s\n"
    )

    print("=== 4. Running Deterministic Controllers on Pilot Split ===")
    pilot_out = os.path.join(logs_dir, "pilot_runs")
    pilot_df = run_deterministic_controllers_on_split(traces_dir, "pilot", pilot_out, DEFAULT_PROTOCOL)
    compute_paired_comparisons_table(pilot_df, pilot_out, prefix="pilot_baselines")
    generate_mentor_plots(pilot_df, traces_dir, plots_dir, split="pilot")
    print(f"✓ Completed {len(pilot_df)} Pilot baseline runs.\n")

    print("=== 5. Running Deterministic Controllers on Held-Out Split ===")
    heldout_out = os.path.join(logs_dir, "heldout_runs")
    heldout_df = run_deterministic_controllers_on_split(traces_dir, "heldout", heldout_out, DEFAULT_PROTOCOL)
    paired_df = compute_paired_comparisons_table(heldout_df, heldout_out, prefix="heldout_baselines")
    plot_files = generate_mentor_plots(heldout_df, traces_dir, plots_dir, split="heldout")
    print(f"✓ Completed {len(heldout_df)} Held-Out baseline runs.")
    print(f"✓ Generated Mentor Preview Plots: {plot_files}\n")

    # Print quick summary table of Held-Out baselines across regimes
    summary_pivot = (
        heldout_df.groupby(["regime", "delay_condition", "controller", "model_name"])
        .agg(
            mae_mean=("mae_pct_points", "mean"),
            mae_std=("mae_pct_points", "std"),
            bytes_mean=("monitoring_bytes_total", "mean"),
            polls_mean=("poll_count", "mean"),
            det_delay_s=("mean_detection_delay_s", "mean"),
        )
        .reset_index()
    )
    summary_pivot.to_csv(os.path.join(logs_dir, "heldout_baselines_Condition_Summary.csv"), index=False)
    print("=== Held-Out Baselines Condition Summary (First 15 rows) ===")
    print(summary_pivot.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
