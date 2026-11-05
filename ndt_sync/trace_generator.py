"""Synthetic trace generator for the 8-link shared-path Network Digital Twin.

Implements standard networking stochastic demand models routed across shared multi-hop paths:
1. `stable`: Mean-reverting Ornstein-Uhlenbeck (AR(1)) background flows.
2. `bursty`: OU background flows superposed with compound Poisson / Pareto micro-bursts.
3. `alternating`: 2-state Markov-modulated regime switching (quiet vs. volatile periods).
4. `shifted`: Persistent step-changes (traffic engineering / failover load shifts).

All generated traces can be saved to deterministic CSV files and reloaded identically
across all controllers (Fixed, Change-Aware, Deterministic Rule, and Decision Models).
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from .config import (
    NUM_LINKS,
    LINK_NAMES,
    OD_FLOW_PATHS,
    TOPOLOGY_DEFINITION,
    TRAFFIC_REGIMES,
    DELAY_CONDITIONS,
    DELAY_PARAMS_MS,
    PILOT_SEEDS,
    HELDOUT_SEEDS,
    DEFAULT_PROTOCOL,
    ExperimentProtocol,
)


def generate_trace_dataframe(
    regime: str,
    delay_condition: str,
    seed: int,
    protocol: ExperimentProtocol = DEFAULT_PROTOCOL,
) -> pd.DataFrame:
    """Generate a single synthetic link-utilization trace for all 8 links."""
    if regime not in TRAFFIC_REGIMES:
        raise ValueError(f"Unknown traffic regime {regime!r}. Expected one of {TRAFFIC_REGIMES}")
    if delay_condition not in DELAY_CONDITIONS:
        raise ValueError(f"Unknown delay condition {delay_condition!r}. Expected one of {DELAY_CONDITIONS}")

    rng = np.random.default_rng(seed)
    dt = protocol.tick_dt_s
    num_ticks = int(round(protocol.trace_duration_s / dt)) + 1
    timestamps = np.round(np.arange(num_ticks) * dt, 4)

    flow_names = list(OD_FLOW_PATHS.keys())
    num_flows = len(flow_names)

    # Base flow demand levels (in utilization percentage points before path routing)
    base_demands = rng.uniform(12.0, 24.0, size=num_flows)
    flow_demands = np.zeros((num_ticks, num_flows), dtype=np.float64)
    flow_demands[0] = base_demands

    # Track scheduled significant change events for detection-delay evaluation (§2.4)
    is_change_event = np.zeros(num_ticks, dtype=np.int32)
    active_phase = ["baseline"] * num_ticks

    if regime == "stable":
        # Mean-reverting Ornstein-Uhlenbeck process with low volatility
        theta = 0.35
        sigma = 0.85
        for t in range(1, num_ticks):
            noise = rng.normal(0.0, sigma, size=num_flows)
            flow_demands[t] = (
                flow_demands[t - 1]
                + theta * (base_demands - flow_demands[t - 1]) * dt
                + noise * np.sqrt(dt)
            )
            active_phase[t] = "stable"

    elif regime == "bursty":
        # Quiet OU background + scheduled & random short intense traffic bursts
        theta = 0.45
        sigma = 0.70
        burst_envelope = np.zeros((num_ticks, num_flows), dtype=np.float64)

        # Schedule 8 distinct bursts across the trace (after warmup and during warmup)
        burst_starts = np.linspace(25.0, protocol.trace_duration_s - 15.0, 8)
        burst_starts += rng.uniform(-3.0, 3.0, size=len(burst_starts))
        for b_start in burst_starts:
            start_idx = max(1, min(num_ticks - 20, int(round(b_start / dt))))
            duration_ticks = int(rng.integers(15, 38))  # 1.5s to 3.8s burst duration
            end_idx = min(num_ticks, start_idx + duration_ticks)
            affected_flows = rng.choice(num_flows, size=3, replace=False)
            burst_amp = rng.uniform(22.0, 38.0, size=len(affected_flows))
            for idx_f, f_id in enumerate(affected_flows):
                burst_envelope[start_idx:end_idx, f_id] += burst_amp[idx_f]
            is_change_event[start_idx] = 1
            for k in range(start_idx, end_idx):
                active_phase[k] = "burst"

        for t in range(1, num_ticks):
            noise = rng.normal(0.0, sigma, size=num_flows)
            target = base_demands + burst_envelope[t]
            flow_demands[t] = (
                flow_demands[t - 1]
                + (1.8 if active_phase[t] == "burst" else theta) * (target - flow_demands[t - 1]) * dt
                + noise * np.sqrt(dt)
            )

    elif regime == "alternating":
        # Alternates every ~25-30 seconds between quiet steady periods and rapidly changing volatile periods
        window_s = 25.0
        for t in range(1, num_ticks):
            time_s = timestamps[t]
            cycle_idx = int(time_s // window_s)
            is_volatile = (cycle_idx % 2) == 1
            prev_volatile = (int(timestamps[t - 1] // window_s) % 2) == 1
            if is_volatile != prev_volatile and time_s >= protocol.warmup_duration_s:
                is_change_event[t] = 1

            if is_volatile:
                active_phase[t] = "volatile"
                # Fast oscillatory + high-variance demand shocks
                osc = 14.0 * np.sin(2.0 * np.pi * time_s / 3.2 + np.arange(num_flows))
                noise = rng.normal(0.0, 8.5, size=num_flows)
                target = base_demands + 12.0 + osc
                flow_demands[t] = (
                    flow_demands[t - 1]
                    + 1.6 * (target - flow_demands[t - 1]) * dt
                    + noise * np.sqrt(dt)
                )
            else:
                active_phase[t] = "quiet"
                noise = rng.normal(0.0, 0.45, size=num_flows)
                flow_demands[t] = (
                    flow_demands[t - 1]
                    + 0.8 * (base_demands - flow_demands[t - 1]) * dt
                    + noise * np.sqrt(dt)
                )

    elif regime == "shifted":
        # Persistent load shifts (step changes at 3 regime boundaries)
        shift_times = [45.0, 95.0, 140.0]
        shift_vectors = [
            rng.uniform(14.0, 26.0, size=num_flows) * np.array([1, 1, -0.4, 1, 0.8, -0.3]),
            rng.uniform(15.0, 28.0, size=num_flows) * np.array([-0.6, 0.9, 1.2, -0.5, 1.0, 0.8]),
            rng.uniform(12.0, 24.0, size=num_flows) * np.array([0.8, -0.7, 0.9, 1.1, -0.6, 1.0]),
        ]
        current_target = base_demands.copy()
        shift_idx = 0
        for t in range(1, num_ticks):
            time_s = timestamps[t]
            if shift_idx < len(shift_times) and time_s >= shift_times[shift_idx]:
                current_target = np.clip(base_demands + shift_vectors[shift_idx], 5.0, 60.0)
                is_change_event[t] = 1
                shift_idx += 1
            active_phase[t] = f"shift_stage_{shift_idx}"
            noise = rng.normal(0.0, 1.1, size=num_flows)
            flow_demands[t] = (
                flow_demands[t - 1]
                + 1.5 * (current_target - flow_demands[t - 1]) * dt
                + noise * np.sqrt(dt)
            )

    # Route OD flow demands onto the 8 physical links via shared paths
    link_utils = np.zeros((num_ticks, NUM_LINKS), dtype=np.float64)
    for f_idx, f_name in enumerate(flow_names):
        for link_id, weight in OD_FLOW_PATHS[f_name]:
            link_utils[:, link_id] += flow_demands[:, f_idx] * weight

    # Add small link-local uncorrelated jitter and clamp to [1.0, 99.0] percentage points
    link_local_noise = rng.normal(0.0, 0.25, size=(num_ticks, NUM_LINKS))
    link_utils = np.clip(link_utils + link_local_noise, 1.0, 99.0)

    # Pre-generate deterministic per-tick delivery delays (ms) so every controller experiences
    # the exact same network channel delay if it dispatches a poll at tick `t`
    d_cfg = DELAY_PARAMS_MS[delay_condition]
    delivery_delays_ms = rng.normal(d_cfg["base_ms"], d_cfg["jitter_std_ms"], size=num_ticks)
    delivery_delays_ms = np.clip(delivery_delays_ms, 10.0, d_cfg["base_ms"] * 3.0)

    # Inject occasional delay spikes so out-of-order protection (§2.3) is realistically exercised
    spike_mask = rng.random(num_ticks) < d_cfg["out_of_order_prob"]
    delivery_delays_ms[spike_mask] += rng.uniform(1200.0, 2800.0, size=int(np.sum(spike_mask)))

    # Assemble DataFrame
    data: Dict[str, object] = {
        "tick_idx": np.arange(num_ticks, dtype=np.int32),
        "timestamp_s": timestamps,
        "regime": regime,
        "delay_condition": delay_condition,
        "seed": seed,
        "active_phase": active_phase,
        "is_change_event": is_change_event,
        "delivery_delay_ms": np.round(delivery_delays_ms, 2),
    }
    for i, lname in enumerate(LINK_NAMES):
        data[f"u_{lname}"] = np.round(link_utils[:, i], 4)

    return pd.DataFrame(data)


def generate_and_save_all_traces(
    output_dir: str,
    protocol: ExperimentProtocol = DEFAULT_PROTOCOL,
) -> Dict[str, List[str]]:
    """Generate and persist all Pilot and Held-Out CSV traces plus topology metadata."""
    os.makedirs(output_dir, exist_ok=True)
    pilot_dir = os.path.join(output_dir, "pilot")
    heldout_dir = os.path.join(output_dir, "heldout")
    os.makedirs(pilot_dir, exist_ok=True)
    os.makedirs(heldout_dir, exist_ok=True)

    # Save topology metadata
    topo_path = os.path.join(output_dir, "topology_8links.json")
    with open(topo_path, "w", encoding="utf-8") as f:
        json.dump(TOPOLOGY_DEFINITION, f, indent=2)

    manifest: Dict[str, List[str]] = {"pilot": [], "heldout": []}

    for split_name, seeds, target_dir in [
        ("pilot", PILOT_SEEDS, pilot_dir),
        ("heldout", HELDOUT_SEEDS, heldout_dir),
    ]:
        for regime in TRAFFIC_REGIMES:
            for delay_cond in DELAY_CONDITIONS:
                for seed in seeds:
                    df = generate_trace_dataframe(regime, delay_cond, seed, protocol)
                    fname = f"trace_{regime}_{delay_cond}_seed{seed}.csv"
                    fpath = os.path.join(target_dir, fname)
                    df.to_csv(fpath, index=False)
                    manifest[split_name].append(fpath)

    manifest_path = os.path.join(output_dir, "traces_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_pilot_traces": len(manifest["pilot"]),
                "num_heldout_traces": len(manifest["heldout"]),
                "pilot_seeds": PILOT_SEEDS,
                "heldout_seeds": HELDOUT_SEEDS,
                "regimes": list(TRAFFIC_REGIMES),
                "delay_conditions": list(DELAY_CONDITIONS),
                "protocol": protocol.to_dict(),
                "files": manifest,
            },
            f,
            indent=2,
        )
    return manifest


def load_trace_csv(csv_path: str) -> pd.DataFrame:
    """Load a pre-generated trace CSV from disk."""
    return pd.read_csv(csv_path)
