#!/usr/bin/env python3
"""Live GPU Sweet-Spot & Multi-Node Scale-Up Runner (Real `convaiinnovations/laya` on Kaggle T4).

Replaces the Phase 2 analytical proxy with REAL HTTP `/v1/systemone` inference calls to
`convaiinnovations/laya` running on CUDA, measuring:
- Real twin MAE (% pts) & application-level monitoring bytes (20 + 16*N bytes/poll)
- Real wall-clock GPU decision latency (mean & P95 ms) applied causally in the simulator
- Real parse validity, fallback rates, and request/reply byte overhead
- Complete audit log of every model review decision (`sweet_spot_LIVE_LAYA_review_logs.jsonl`)
"""

from __future__ import annotations
import os
import json
import time
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

LAYA_BASE_URL = os.environ.get("TYPESAFE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LAYA_ENDPOINT = f"{LAYA_BASE_URL}/v1/systemone"
LAYA_API_KEY = os.environ.get("UNSLOTH_API_KEY", "sk-local-kaggle-token")

MULTINODE_CHOICE_QUESTION = {
    "selected_policy": {
        "type": "choice",
        "instructions": (
            "Select the network digital twin polling policy ('fixed' or 'change-aware') from the "
            "multi-link spatial telemetry summary. "
            "Pick 'change-aware' ONLY when there is a WIDESPREAD_CORE_WAVE (multiple core links "
            "changing together) OR TRUE_NETWORK_CALM (all core and edge links completely flat). "
            "Pick 'fixed' when there is LOCALIZED_EDGE_NOISE_OR_OSCILLATION (only a few edge links "
            "flickering while core links remain calm) to avoid false-alarm over-polling."
        ),
        "criteria": {
            "fixed": (
                "LOCALIZED_EDGE_NOISE_OR_OSCILLATION: Only a small subset of edge links are "
                "flickering or oscillating while the majority of core links remain calm. Use steady "
                "fixed polling to prevent wasting bandwidth on false alarms."
            ),
            "change-aware": (
                "WIDESPREAD_CORE_WAVE or TRUE_NETWORK_CALM: Either a correlated surge is affecting "
                "many core links simultaneously (requiring fast adaptive polling), OR every link in "
                "the network is completely flat and quiet (allowing adaptive polling to slow down)."
            ),
        },
    }
}


class LiveLayaMultiNodeClient:
    """Queries the live local `laya.serve` CUDA server via POST /v1/systemone."""

    def __init__(self, endpoint: str = LAYA_ENDPOINT, api_key: str = LAYA_API_KEY, timeout_s: float = 2.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def query(self, state_text: str) -> Dict[str, Any]:
        payload = {
            "model": "laya",
            "state": state_text,
            "questions": MULTINODE_CHOICE_QUESTION,
        }
        payload_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        t0 = time.perf_counter()
        try:
            resp = self.session.post(self.endpoint, data=payload_bytes, timeout=self.timeout_s)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                return {
                    "choice": None, "is_valid": False, "latency_ms": latency_ms,
                    "req_bytes": len(payload_bytes), "rep_bytes": len(resp.content),
                    "p_fixed": 0.0, "p_ca": 0.0, "error": f"http_{resp.status_code}",
                }
            data = resp.json()
            ans = (data.get("answers") or {}).get("selected_policy") or {}
            choice = ans.get("choice")
            if choice == "change_aware":
                choice = "change-aware"
            probs = ans.get("probabilities") or {}
            p_fixed = float(probs.get("fixed", 0.0))
            p_ca = float(probs.get("change-aware", probs.get("change_aware", 0.0)))
            is_valid = choice in ("fixed", "change-aware")
            return {
                "choice": choice if is_valid else None,
                "is_valid": is_valid,
                "latency_ms": latency_ms,
                "req_bytes": len(payload_bytes),
                "rep_bytes": len(resp.content),
                "p_fixed": p_fixed,
                "p_ca": p_ca,
                "error": None if is_valid else f"invalid_choice:{choice!r}",
            }
        except Exception as ex:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return {
                "choice": None, "is_valid": False, "latency_ms": latency_ms,
                "req_bytes": len(payload_bytes), "rep_bytes": 0,
                "p_fixed": 0.0, "p_ca": 0.0, "error": f"{type(ex).__name__}:{ex}",
            }


def build_multinode_state_text(
    t_now: float,
    num_links: int,
    num_core: int,
    num_edge: int,
    active_pol: str,
    aoi_s: float,
    core_mean_delta: float,
    edge_max_delta: float,
    mean_delta: float,
    max_delta: float,
    active_link_count: int,
    active_link_fraction: float,
) -> str:
    if core_mean_delta >= 2.2 and active_link_fraction >= 0.25:
        spatial_regime = (
            "WIDESPREAD_CORE_WAVE (correlated multi-link surge across core backbone -> "
            "requires change-aware adaptive polling)"
        )
    elif core_mean_delta < 1.2 and active_link_fraction < 0.15 and max_delta < 2.2:
        spatial_regime = (
            "TRUE_NETWORK_CALM (all core and edge links completely flat -> "
            "change-aware can relax interval to save traffic)"
        )
    else:
        spatial_regime = (
            "LOCALIZED_EDGE_NOISE_OR_OSCILLATION (core links are calm while only isolated "
            "edge links flicker -> use fixed polling to avoid false-alarm over-polling)"
        )

    return (
        f"Multi-Node Network Digital Twin Review at t={t_now:.1f}s\n"
        f"Topology Scale: {num_links} monitored links ({num_core} core backbone, {num_edge} edge access)\n"
        f"Spatial Regime Assessment: {spatial_regime}\n"
        f"Current Active Policy: {active_pol} | Measurement Age (AoI): {aoi_s:.2f}s\n"
        f"Spatial Telemetry Distribution (received polls only):\n"
        f"- Active links with delta > 3.5%: {active_link_count}/{num_links} ({active_link_fraction*100:.1f}%)\n"
        f"- Core backbone mean delta: {core_mean_delta:.2f}%\n"
        f"- Edge access max delta: {edge_max_delta:.2f}%\n"
        f"- Global mean delta: {mean_delta:.2f}% | Global max delta: {max_delta:.2f}%"
    )


def simulate_multinode_scenario_live(
    scenario_name: str,
    client: LiveLayaMultiNodeClient,
    num_links: int = 32,
    duration_s: float = 180.0,
    dt_s: float = 0.1,
    delivery_delay_s: float = 0.1,
    seed: int = 2026,
    review_logs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    num_ticks = int(round(duration_s / dt_s)) + 1
    timestamps = np.round(np.arange(num_ticks) * dt_s, 4)

    num_core = int(num_links * 0.75)
    num_edge = num_links - num_core

    true_u = np.zeros((num_ticks, num_links), dtype=np.float64)
    base_u = rng.uniform(20.0, 35.0, size=num_links)
    true_u[0] = base_u

    for t in range(1, num_ticks):
        time_s = timestamps[t]
        core_noise = rng.normal(0.0, 0.35, size=num_core)
        edge_noise = rng.normal(0.0, 0.50, size=num_edge)
        target = base_u.copy()

        if scenario_name == "localized_edge_noise":
            edge_jitter = np.zeros(num_edge)
            edge_jitter[0] = 9.5 * np.sin(2.0 * np.pi * time_s / 1.4) + rng.normal(0, 2.5)
            if num_edge > 1:
                edge_jitter[1] = 8.5 * np.cos(2.0 * np.pi * time_s / 1.8) + rng.normal(0, 2.0)
            target[num_core:] += edge_jitter
            if (50.0 <= time_s <= 75.0) or (120.0 <= time_s <= 145.0):
                wave = 18.0 * np.sin(2.0 * np.pi * time_s / 3.0) + 15.0
                target[:num_core] += wave

        elif scenario_name == "correlated_multi_hop_waves":
            win = int(time_s // 30.0)
            if win % 2 == 0:
                hot_edge = t % max(1, num_edge)
                target[num_core + hot_edge] += rng.uniform(7.0, 11.0) * (1 if (t // 5) % 2 == 0 else -1)
            else:
                target[:num_core] += 20.0 * np.sin(2.0 * np.pi * time_s / 2.6) + rng.normal(0, 4.0, size=num_core)

        elif scenario_name == "periodic_oscillation_trap":
            if 20.0 <= time_s < 110.0:
                osc_links = min(3, num_links)
                for k in range(osc_links):
                    target[k] += 7.5 * ((-1) ** (int(time_s * 2.0) + k))
            elif 110.0 <= time_s <= 165.0:
                target[:num_core] += 24.0 * np.sin(2.0 * np.pi * time_s / 3.5) + 16.0

        noise_vec = np.concatenate([core_noise, edge_noise])
        true_u[t] = np.clip(
            true_u[t - 1] + 1.6 * (target - true_u[t - 1]) * dt_s + noise_vec * np.sqrt(dt_s),
            1.0, 99.0,
        )

    bytes_per_poll = (8 + 4 * num_links) + (12 + 12 * num_links)
    controllers = ["fixed", "change_aware", "rule_switcher", "model_switcher_laya_live"]
    results = {}

    for ctrl in controllers:
        stored_u = true_u[0].copy()
        latest_sample_t = 0.0
        active_pol = "fixed" if ctrl != "change_aware" else "change-aware"
        ca_interval = 2.5
        fixed_interval = 2.5
        quiet_streak = 0

        last_poll_t = 0.0
        next_poll_t = 0.0
        next_review_t = 15.0
        in_flight: List[Tuple[float, float, np.ndarray]] = []
        recent_deltas: List[np.ndarray] = []
        pending_decision: Tuple[float, str] | None = None

        polls_count = 0
        abs_errors = []
        pol_history = []
        latencies_ms: List[float] = []
        fallbacks = 0
        reviews_total = 0
        api_req_bytes = 0
        api_rep_bytes = 0

        for t in range(num_ticks):
            t_now = timestamps[t]

            # 1. Deliver polls whose arrival time <= t_now
            rem = []
            for disp_t, arr_t, sample_u in in_flight:
                if arr_t <= t_now + 1e-9:
                    if disp_t >= latest_sample_t:
                        delta_vec = np.abs(sample_u - stored_u)
                        stored_u = sample_u.copy()
                        latest_sample_t = disp_t
                        recent_deltas.append(delta_vec)
                        if len(recent_deltas) > 8:
                            recent_deltas.pop(0)

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

            # 2. Apply pending decision once real GPU latency has elapsed
            if pending_decision is not None and t_now >= pending_decision[0] - 1e-9:
                new_pol = pending_decision[1]
                pending_decision = None
                if new_pol != active_pol:
                    active_pol = new_pol
                    cur_int = fixed_interval if active_pol == "fixed" else ca_interval
                    cand_next = last_poll_t + cur_int
                    next_poll_t = cand_next if cand_next > t_now else (t_now + cur_int)

            # 3. Selector review every 15s (only if no request currently outstanding)
            if ctrl in ("rule_switcher", "model_switcher_laya_live") and t_now >= next_review_t - 1e-9:
                next_review_t += 15.0
                if pending_decision is None:
                    reviews_total += 1
                    if recent_deltas:
                        d_mat = np.vstack(recent_deltas)
                        max_delta = float(np.max(d_mat))
                        mean_delta = float(np.mean(d_mat))
                        core_mean_delta = float(np.mean(d_mat[:, :num_core]))
                        edge_max_delta = float(np.max(d_mat[:, num_core:])) if num_edge > 0 else 0.0
                        active_mask = np.max(d_mat, axis=0) > 3.5
                        active_link_count = int(np.sum(active_mask))
                        active_link_fraction = float(np.mean(active_mask))
                    else:
                        max_delta = mean_delta = core_mean_delta = edge_max_delta = 0.0
                        active_link_count = 0
                        active_link_fraction = 0.0

                    if ctrl == "rule_switcher":
                        new_pol = "change-aware" if (max_delta >= 6.0 or mean_delta >= 3.2) else "fixed"
                        pending_decision = (t_now, new_pol)
                    else:
                        state_txt = build_multinode_state_text(
                            t_now=t_now, num_links=num_links, num_core=num_core, num_edge=num_edge,
                            active_pol=active_pol, aoi_s=t_now - latest_sample_t,
                            core_mean_delta=core_mean_delta, edge_max_delta=edge_max_delta,
                            mean_delta=mean_delta, max_delta=max_delta,
                            active_link_count=active_link_count, active_link_fraction=active_link_fraction,
                        )
                        res = client.query(state_txt)
                        latencies_ms.append(res["latency_ms"])
                        api_req_bytes += res["req_bytes"]
                        api_rep_bytes += res["rep_bytes"]
                        timed_out = (not res["is_valid"]) or (res["latency_ms"] > 2000.0)
                        if timed_out:
                            fallbacks += 1
                            chosen = active_pol
                        else:
                            chosen = res["choice"]
                            t_eff = t_now + (res["latency_ms"] / 1000.0)
                            pending_decision = (t_eff, chosen)

                        if review_logs is not None:
                            review_logs.append({
                                "scenario": scenario_name, "num_links": num_links, "seed": seed,
                                "t_review_s": t_now, "chosen_policy": chosen,
                                "is_valid": res["is_valid"], "fallback": timed_out,
                                "latency_ms": round(res["latency_ms"], 2),
                                "p_fixed": round(res["p_fixed"], 4), "p_ca": round(res["p_ca"], 4),
                                "core_mean_delta": round(core_mean_delta, 3),
                                "edge_max_delta": round(edge_max_delta, 3),
                                "active_link_fraction": round(active_link_fraction, 3),
                            })

            # 4. Dispatch poll
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
            "mean_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
            "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
            "fallback_rate": round(fallbacks / max(1, reviews_total), 4) if ctrl == "model_switcher_laya_live" else 0.0,
            "api_total_bytes": api_req_bytes + api_rep_bytes,
            "seed": seed,
        }

    return results


def run_live_sweet_spot_suite(output_dir: str = "logs/sweet_spot_live_laya") -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    client = LiveLayaMultiNodeClient()
    review_logs: List[Dict[str, Any]] = []

    # 1. Run the 3 Multi-Node Complex Scenarios (5 seeds each)
    scen_rows = []
    scenarios = [
        ("localized_edge_noise", 32),
        ("correlated_multi_hop_waves", 48),
        ("periodic_oscillation_trap", 32),
    ]
    for scen, n_links in scenarios:
        print(f"Running LIVE Laya on {scen} ({n_links} links, 5 seeds)...")
        for seed in [2001, 2002, 2003, 2004, 2005]:
            res = simulate_multinode_scenario_live(scen, client, num_links=n_links, seed=seed, review_logs=review_logs)
            for row in res.values():
                scen_rows.append(row)

    # 2. Run Topology Scale Sweep (8, 16, 32, 64 links) on `localized_edge_noise`
    scale_rows = []
    for n_links in [8, 16, 32, 64]:
        print(f"Running LIVE Laya scale sweep: {n_links} links (5 seeds)...")
        for seed in [3001, 3002, 3003, 3004, 3005]:
            res = simulate_multinode_scenario_live("localized_edge_noise", client, num_links=n_links, seed=seed, review_logs=review_logs)
            for row in res.values():
                scale_rows.append(row)

    df_scen = pd.DataFrame(scen_rows)
    df_scale = pd.DataFrame(scale_rows)
    df_scen.to_csv(os.path.join(output_dir, "sweet_spot_LIVE_LAYA_multinode_scenarios.csv"), index=False)
    df_scale.to_csv(os.path.join(output_dir, "sweet_spot_LIVE_LAYA_scale_sweep.csv"), index=False)
    with open(os.path.join(output_dir, "sweet_spot_LIVE_LAYA_review_logs.jsonl"), "w", encoding="utf-8") as f:
        for entry in review_logs:
            f.write(json.dumps(entry) + "\n")

    # Plot 1: Multi-Node Scenarios
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ctrl_order = ["fixed", "change_aware", "rule_switcher", "model_switcher_laya_live"]
    ctrl_labels = ["Fixed\n(2.5s)", "Change-\nAware", "Rule\nSwitcher", "Live Laya\n(GPU)"]
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    for ax, (scen, n_links) in zip(axes, scenarios):
        sub = df_scen[df_scen["scenario"] == scen].groupby("controller", as_index=False).mean(numeric_only=True)
        sub = sub.set_index("controller").reindex(ctrl_order).reset_index()
        x = np.arange(len(ctrl_order))
        bars = ax.bar(x, sub["monitoring_kb"], color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(ctrl_labels, fontsize=9)
        ax.set_ylabel("Monitoring Traffic (KB)", fontweight="bold")
        ax.set_title(f"{scen}\n({n_links} Links, Live Laya GPU)", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        for bar, mae in zip(bars, sub["mae_pct"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{bar.get_height():.1f} KB\n({mae:.2f}%)", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sweet_spot_fig1_LIVE_LAYA_multinode.png"), dpi=180)
    plt.close(fig)

    # Plot 2: Scale Sweep (8 -> 64 links) + O(1) GPU Latency
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ctrl, lbl, clr, mk in zip(ctrl_order, ["Fixed (2.5s)", "Change-Aware", "Rule Switcher", "Live Laya (GPU)"], colors, ["o", "s", "^", "D"]):
        sub = df_scale[df_scale["controller"] == ctrl].groupby("num_links", as_index=False).mean(numeric_only=True)
        ax1.plot(sub["num_links"], sub["monitoring_kb"], marker=mk, linewidth=2.2, label=lbl, color=clr)
    ax1.set_xlabel("Number of Monitored Links (N)", fontweight="bold")
    ax1.set_ylabel("Monitoring Traffic per 3-Min Run (KB)", fontweight="bold")
    ax1.set_title("O(N) Telemetry Scaling: Live Laya vs. Baselines", fontweight="bold")
    ax1.set_xticks([8, 16, 32, 64])
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend()

    laya_sub = df_scale[df_scale["controller"] == "model_switcher_laya_live"].groupby("num_links", as_index=False).mean(numeric_only=True)
    ax2.bar([str(n) for n in laya_sub["num_links"]], laya_sub["mean_latency_ms"], color="#c44e52", edgecolor="black", width=0.5, label="Mean RTT (ms)")
    ax2.plot([str(n) for n in laya_sub["num_links"]], laya_sub["p95_latency_ms"], color="#1e1e2e", marker="o", linewidth=2, label="P95 RTT (ms)")
    ax2.set_xlabel("Number of Monitored Links (N)", fontweight="bold")
    ax2.set_ylabel("Measured Laya GPU Decision Latency (ms)", fontweight="bold")
    ax2.set_title("O(1) Live GPU Inference Latency Across Topology Scales", fontweight="bold")
    ax2.set_ylim(0, max(100.0, float(laya_sub["p95_latency_ms"].max()) * 1.4))
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sweet_spot_fig2_LIVE_LAYA_scale_sweep.png"), dpi=180)
    plt.close(fig)

    print(f"\n✓ Finished {len(review_logs)} live GPU reviews! Saved outputs to {output_dir}/")
    return df_scen, df_scale


if __name__ == "__main__":
    df_scen, df_scale = run_live_sweet_spot_suite()
    print("\n=== MULTI-NODE SCENARIOS SUMMARY (LIVE LAYA GPU) ===")
    print(df_scen.groupby(["scenario", "num_links", "controller"])[["mae_pct", "monitoring_kb", "polls", "change_aware_frac", "mean_latency_ms", "p95_latency_ms", "fallback_rate"]].mean().round(3))
    print("\n=== TOPOLOGY SCALE SWEEP SUMMARY (LIVE LAYA GPU) ===")
    print(df_scale.groupby(["num_links", "controller"])[["mae_pct", "monitoring_kb", "mean_latency_ms", "p95_latency_ms", "fallback_rate"]].mean().round(3))
