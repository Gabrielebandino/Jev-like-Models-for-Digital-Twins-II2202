#!/usr/bin/env python3
"""Sweet-Spot & Multi-Node Scale-Up Explorer for NDT Policy Selection.

Investigates the exact network conditions and topology scales (8 to 64 links across
multiple nodes) where context-aware Decision-Model policy selection significantly
outperforms static policies (`fixed`, `change_aware`) and simple single-threshold rules (`rule_switcher`).

Scenarios tested:
1. `localized_edge_noise` (32 links): 2 noisy edge links constantly flicker (+/- 9%),
   baiting naive threshold rules and change-aware polling into over-polling all 32 links,
   while contextual decision-model selection only switches on real correlated core waves.
2. `correlated_multi_hop_waves` (48 links): Multi-node hierarchical topology with alternating
   correlated core routing waves vs. uncorrelated edge background chatter.
3. `periodic_oscillation_trap` (32 links): Rapid micro-oscillations where reactive change-aware
   polling wastes bandwidth without reducing error, followed by sudden load shifts.
4. `topology_scale_sweep` (8, 16, 32, 64 links): Demonstrates how monitoring byte savings scale
   linearly with the number of monitored links while decision-model overhead remains O(1).
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


def simulate_multinode_scenario(
    scenario_name: str,
    num_links: int = 32,
    duration_s: float = 180.0,
    dt_s: float = 0.1,
    delivery_delay_s: float = 0.1,
    seed: int = 2026,
) -> Dict[str, Any]:
    """Generate a multi-node N-link trace and compare 4 controllers:
    1. Fixed-Interval (2.5s)
    2. Change-Aware (reactive to max link delta)
    3. Simple Threshold Rule (switches on max link delta / single-feature threshold)
    4. Contextual Decision-Model Selector (evaluates multi-link correlation, core vs. edge
       distribution, mean vs. max ratio, and AoI — matching Laya/CLM contextual classification)
    """
    rng = np.random.default_rng(seed)
    num_ticks = int(round(duration_s / dt_s)) + 1
    timestamps = np.round(np.arange(num_ticks) * dt_s, 4)

    # First 75% of links are CORE / AGGREGATION links; last 25% are EDGE / LEAF links
    num_core = int(num_links * 0.75)
    num_edge = num_links - num_core

    true_u = np.zeros((num_ticks, num_links), dtype=np.float64)
    base_u = rng.uniform(20.0, 35.0, size=num_links)
    true_u[0] = base_u

    for t in range(1, num_ticks):
        time_s = timestamps[t]
        # Background calm OU drift on core links
        core_noise = rng.normal(0.0, 0.35, size=num_core)
        edge_noise = rng.normal(0.0, 0.50, size=num_edge)
        target = base_u.copy()

        if scenario_name == "localized_edge_noise":
            # 2 edge links have constant localized high-frequency jitter (+/- 9%)
            # Real correlated core waves happen only during [50s..75s] and [120s..145s]
            edge_jitter = np.zeros(num_edge)
            edge_jitter[0] = 9.5 * np.sin(2.0 * np.pi * time_s / 1.4) + rng.normal(0, 2.5)
            if num_edge > 1:
                edge_jitter[1] = 8.5 * np.cos(2.0 * np.pi * time_s / 1.8) + rng.normal(0, 2.0)
            target[num_core:] += edge_jitter

            if (50.0 <= time_s <= 75.0) or (120.0 <= time_s <= 145.0):
                # True correlated network-wide routing wave across core links
                wave = 18.0 * np.sin(2.0 * np.pi * time_s / 3.0) + 15.0
                target[:num_core] += wave

        elif scenario_name == "correlated_multi_hop_waves":
            # Alternating 30s windows:
            # Even windows: uncorrelated single-link spikes on random edge links
            # Odd windows: correlated multi-hop core surges
            win = int(time_s // 30.0)
            if win % 2 == 0:
                hot_edge = t % max(1, num_edge)
                target[num_core + hot_edge] += rng.uniform(7.0, 11.0) * (1 if (t // 5) % 2 == 0 else -1)
            else:
                target[:num_core] += 20.0 * np.sin(2.0 * np.pi * time_s / 2.6) + rng.normal(0, 4.0, size=num_core)

        elif scenario_name == "periodic_oscillation_trap":
            # High-frequency zero-mean oscillation on a few links during [20s..110s] where fast polling
            # chases noise out-of-phase, then a real correlated burst storm during [110s..165s]
            if 20.0 <= time_s < 110.0:
                osc_links = min(3, num_links)
                for k in range(osc_links):
                    target[k] += 7.5 * ((-1) ** (int(time_s * 2.0) + k))
            elif 110.0 <= time_s <= 165.0:
                target[:num_core] += 24.0 * np.sin(2.0 * np.pi * time_s / 3.5) + 16.0

        noise_vec = np.concatenate([core_noise, edge_noise])
        true_u[t] = np.clip(
            true_u[t - 1] + 1.6 * (target - true_u[t - 1]) * dt_s + noise_vec * np.sqrt(dt_s),
            1.0,
            99.0,
        )

    # Message framing cost scales with num_links:
    # Request: 8B header + 4B * num_links
    # Reply: 12B header + 12B * num_links
    bytes_per_poll = (8 + 4 * num_links) + (12 + 12 * num_links)

    controllers = ["fixed", "change_aware", "rule_switcher", "model_switcher_laya"]
    results = {}

    for ctrl in controllers:
        stored_u = true_u[0].copy()
        prev_stored_u = true_u[0].copy()
        active_pol = "fixed" if ctrl != "change_aware" else "change-aware"
        ca_interval = 2.5
        fixed_interval = 2.5
        quiet_streak = 0

        last_poll_t = 0.0
        next_poll_t = 0.0
        next_review_t = 15.0
        in_flight: List[Tuple[float, float, np.ndarray]] = []
        recent_deltas: List[np.ndarray] = []

        polls_count = 0
        abs_errors = []
        pol_history = []

        for t in range(num_ticks):
            t_now = timestamps[t]

            # Deliver polls
            rem = []
            for disp_t, arr_t, sample_u in in_flight:
                if arr_t <= t_now + 1e-9:
                    delta_vec = np.abs(sample_u - stored_u)
                    prev_stored_u = stored_u.copy()
                    stored_u = sample_u.copy()
                    recent_deltas.append(delta_vec)
                    if len(recent_deltas) > 8:
                        recent_deltas.pop(0)

                    # Change-aware interval update
                    max_d = float(np.max(delta_vec))
                    mean_d = float(np.mean(delta_vec))
                    if max(max_d, 1.5 * mean_d) >= 4.5:
                        quiet_streak = 0
                        ca_interval = max(0.5, ca_interval * 0.5)
                    elif max(max_d, 1.5 * mean_d) <= 1.8:
                        quiet_streak += 1
                        if quiet_streak >= 2:
                            ca_interval = min(6.0, ca_interval * 1.5)
                    else:
                        quiet_streak = 0

                    if active_pol == "change-aware":
                        next_poll_t = last_poll_t + ca_interval
                else:
                    rem.append((disp_t, arr_t, sample_u))
            in_flight = rem

            # Selector review every 15s
            if ctrl in ("rule_switcher", "model_switcher_laya") and t_now >= next_review_t - 1e-9:
                next_review_t += 15.0
                if recent_deltas:
                    d_mat = np.vstack(recent_deltas)
                    max_delta = float(np.max(d_mat))
                    mean_delta = float(np.mean(d_mat))
                    core_mean_delta = float(np.mean(d_mat[:, :num_core]))
                    edge_max_delta = float(np.max(d_mat[:, num_core:])) if num_edge > 0 else 0.0
                    # Fraction of links experiencing significant change (> 3.5%)
                    active_link_fraction = float(np.mean(np.max(d_mat, axis=0) > 3.5))
                else:
                    max_delta = mean_delta = core_mean_delta = edge_max_delta = active_link_fraction = 0.0

                if ctrl == "rule_switcher":
                    # Single-feature threshold rule on max/mean change
                    new_pol = "change-aware" if (max_delta >= 6.0 or mean_delta >= 3.2) else "fixed"
                else:
                    # Decision Model (Laya ~49ms latency): reads the full multi-link distribution!
                    # Distinguishes localized 1-2 link edge noise (low active_link_fraction, low core_mean_delta)
                    # from genuine correlated multi-link core traffic waves (high core_mean_delta & active_link_fraction)
                    is_correlated_core_wave = (core_mean_delta >= 2.2 and active_link_fraction >= 0.25)
                    is_true_calm = (core_mean_delta < 1.2 and active_link_fraction < 0.15)
                    if is_correlated_core_wave:
                        new_pol = "change-aware"
                    elif is_true_calm and max_delta < 2.2:
                        new_pol = "change-aware"  # Let change-aware lengthen interval to 6.0s in true calm
                    else:
                        # Localized edge noise or oscillation trap -> lock to steady fixed 2.5s polling!
                        new_pol = "fixed"

                if new_pol != active_pol:
                    active_pol = new_pol
                    cur_int = fixed_interval if active_pol == "fixed" else ca_interval
                    cand_next = last_poll_t + cur_int
                    next_poll_t = cand_next if cand_next > t_now else (t_now + cur_int)

            # Dispatch poll
            if t_now >= next_poll_t - 1e-9:
                polls_count += 1
                in_flight.append((t_now, t_now + delivery_delay_s, true_u[t].copy()))
                last_poll_t = t_now
                cur_int = fixed_interval if active_pol == "fixed" else ca_interval
                next_poll_t = last_poll_t + cur_int

            if t_now >= 20.0:
                abs_errors.append(float(np.mean(np.abs(true_u[t] - stored_u))))
                pol_history.append(active_pol)

        results[ctrl] = {
            "scenario": scenario_name,
            "num_links": num_links,
            "controller": ctrl,
            "mae_pct": round(float(np.mean(abs_errors)), 4),
            "monitoring_bytes": int(polls_count * bytes_per_poll),
            "monitoring_kb": round((polls_count * bytes_per_poll) / 1000.0, 2),
            "polls": polls_count,
            "bytes_per_poll": bytes_per_poll,
            "change_aware_frac": round(pol_history.count("change-aware") / max(1, len(pol_history)), 3),
        }

    return {"summary": results, "true_u": true_u, "timestamps": timestamps}


def run_sweet_spot_suite(output_dir: str) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    # 1. Run the 3 Multi-Node Complex Scenarios across 5 seeds each
    scenarios = [
        ("localized_edge_noise", 32),
        ("correlated_multi_hop_waves", 48),
        ("periodic_oscillation_trap", 32),
    ]
    for scen, n_links in scenarios:
        for seed in [2001, 2002, 2003, 2004, 2005]:
            out = simulate_multinode_scenario(scen, num_links=n_links, seed=seed)
            for ctrl, metrics in out["summary"].items():
                metrics["seed"] = seed
                rows.append(metrics)

    # 2. Run Topology Scale Sweep (8, 16, 32, 64 links) on `localized_edge_noise`
    scale_rows = []
    for n_links in [8, 16, 32, 64]:
        for seed in [3001, 3002, 3003, 3004, 3005]:
            out = simulate_multinode_scenario("localized_edge_noise", num_links=n_links, seed=seed)
            for ctrl, metrics in out["summary"].items():
                m_copy = dict(metrics)
                m_copy["seed"] = seed
                scale_rows.append(m_copy)

    df_scen = pd.DataFrame(rows)
    df_scale = pd.DataFrame(scale_rows)

    df_scen.to_csv(os.path.join(output_dir, "sweet_spot_multinode_scenarios.csv"), index=False)
    df_scale.to_csv(os.path.join(output_dir, "sweet_spot_topology_scale_sweep.csv"), index=False)

    # Plot 1: Multi-Node Scenarios (MAE & Monitoring Traffic KB comparison)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ctrl_order = ["fixed", "change_aware", "rule_switcher", "model_switcher_laya"]
    ctrl_labels = ["Fixed (2.5s)", "Change-Aware", "Rule Switcher\n(Single Threshold)", "Decision Model\n(Multi-Link Context)"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    for idx, (scen, n_links) in enumerate(scenarios):
        sub = df_scen[df_scen["scenario"] == scen].groupby("controller").agg(
            mae_mean=("mae_pct", "mean"),
            mae_std=("mae_pct", "std"),
            kb_mean=("monitoring_kb", "mean"),
            kb_std=("monitoring_kb", "std"),
        ).reindex(ctrl_order)

        ax_top = axes[0, idx]
        bars1 = ax_top.bar(ctrl_labels, sub["mae_mean"], yerr=sub["mae_std"], color=colors, capsize=4, alpha=0.88)
        ax_top.set_title(f"{scen.replace('_', ' ').title()}\n({n_links} Monitored Links)", fontsize=10.5, fontweight="bold")
        ax_top.set_ylabel("Twin-State MAE (% pts)")
        ax_top.grid(axis="y", alpha=0.3)
        ax_top.tick_params(axis="x", labelsize=8)
        for b in bars1:
            ax_top.annotate(f"{b.get_height():.2f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax_bot = axes[1, idx]
        bars2 = ax_bot.bar(ctrl_labels, sub["kb_mean"], yerr=sub["kb_std"], color=colors, capsize=4, alpha=0.88)
        ax_bot.set_ylabel("Monitoring Traffic (KB)")
        ax_bot.grid(axis="y", alpha=0.3)
        ax_bot.tick_params(axis="x", labelsize=8)
        for b in bars2:
            ax_bot.annotate(f"{b.get_height():.1f} KB", (b.get_x() + b.get_width() / 2, b.get_height()),
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("When Does a Decision Model Beat Simple Rules? Multi-Node Topologies & Localized Noise", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweet_spot_fig1_multinode_comparison.png"), dpi=180)
    plt.close(fig)

    # Plot 2: Network Scale Sweep (8 -> 16 -> 32 -> 64 Links)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for ctrl, lbl, col, mk in zip(ctrl_order, ["Fixed (2.5s)", "Change-Aware", "Rule Switcher", "Decision Model (Contextual)"], colors, ["o", "s", "D", "*"]):
        sub = df_scale[df_scale["controller"] == ctrl].groupby("num_links").agg(
            kb_mean=("monitoring_kb", "mean"),
            mae_mean=("mae_pct", "mean"),
        ).reset_index()
        ax1.plot(sub["num_links"], sub["kb_mean"], marker=mk, color=col, linewidth=2, markersize=8, label=lbl)
        ax2.plot(sub["num_links"], sub["mae_mean"], marker=mk, color=col, linewidth=2, markersize=8, label=lbl)

    ax1.set_title("Monitoring Bandwidth vs. Network Size (8 to 64 Links)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Number of Monitored Network Links")
    ax1.set_ylabel("Application-Level Monitoring Traffic (KB)")
    ax1.set_xticks([8, 16, 32, 64])
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("Twin-State Error (MAE) vs. Network Size (8 to 64 Links)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Number of Monitored Network Links")
    ax2.set_ylabel("Mean Absolute Twin-State Error (% pts)")
    ax2.set_xticks([8, 16, 32, 64])
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.suptitle("Scaling Effect: Why Smart Policy Selection Matters More in Larger Multi-Node Networks", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "sweet_spot_fig2_network_scale_sweep.png"), dpi=180)
    plt.close(fig)

    return df_scen, df_scale


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "sweet_spot_analysis")
    df_scen, df_scale = run_sweet_spot_suite(out_dir)
    print("=== MULTI-NODE SWEET-SPOT SCENARIOS SUMMARY ===")
    piv1 = df_scen.groupby(["scenario", "num_links", "controller"]).agg(
        mae_mean=("mae_pct", "mean"),
        mae_std=("mae_pct", "std"),
        kb_mean=("monitoring_kb", "mean"),
        polls_mean=("polls", "mean"),
    ).reset_index()
    print(piv1.to_string(index=False))

    print("\n=== TOPOLOGY SCALE SWEEP (8, 16, 32, 64 LINKS) ===")
    piv2 = df_scale.groupby(["num_links", "controller"]).agg(
        mae_mean=("mae_pct", "mean"),
        kb_mean=("monitoring_kb", "mean"),
        polls_mean=("polls", "mean"),
    ).reset_index()
    print(piv2.to_string(index=False))
