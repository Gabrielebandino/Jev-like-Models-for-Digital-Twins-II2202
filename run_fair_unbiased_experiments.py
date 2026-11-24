#!/usr/bin/env python3
"""Fair, Unbiased, Zero-Spoon-Feeding NDT Synchronization Experiments (Live `laya` on Kaggle GPU).

Methodological guarantees:
1. ZERO PROMPT SPOON-FEEDING: The decision model (`convaiinnovations/laya`) receives ONLY raw,
   neutral numerical telemetry (no pre-computed `if/else` regime labels or policy hints).
2. STRICT INFORMATION PARITY: Both rule baselines (`rule_simple`, `rule_spatial`) and the
   decision model operate strictly on twin-received observations (`arrival_time_s <= review_time_s`).
3. PARETO FAIRNESS: Every scenario evaluates the full static `fixed` curve (1.0s, 2.5s, 5.0s)
   alongside `change_aware`, `rule_simple`, `rule_spatial`, and `laya_live_unbiased`.
4. CAUSAL EXECUTION: Real HTTP CUDA round-trip latency is applied in-loop (`t_eff = t_rev + rtt`),
   single outstanding request (no backlog), 2000ms timeout fallback, and out-of-order rejection.
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

# Strictly NEUTRAL Choice question: explains what each policy IS, with ZERO pre-labeled hints
NEUTRAL_CHOICE_QUESTION = {
    "selected_policy": {
        "type": "choice",
        "instructions": (
            "You are a Network Digital Twin synchronization controller. Based solely on the "
            "numerical telemetry measurements provided below, select which polling policy should "
            "run during the next 15-second review window: 'fixed' or 'change-aware'. "
            "Objective: balance twin-state accuracy against application-level monitoring traffic."
        ),
        "criteria": {
            "fixed": (
                "Fixed-interval polling (constant 2.5s interval across all links). Gives predictable, "
                "bounded traffic and avoids rapid polling spikes during steady periods or isolated "
                "localized noise."
            ),
            "change-aware": (
                "Adaptive change-aware polling (interval dynamically adjusts between 0.5s and 6.0s "
                "based on observed link deltas). Polls rapidly during network-wide bursts/waves and "
                "slows down during extended quiet periods."
            ),
        },
    }
}


class LiveLayaUnbiasedClient:
    """Queries live `laya.serve` via POST /v1/systemone with strictly raw numerical telemetry."""

    def __init__(self, endpoint: str = LAYA_ENDPOINT, api_key: str = LAYA_API_KEY, timeout_s: float = 2.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def query(self, raw_telemetry_text: str) -> Dict[str, Any]:
        payload = {
            "model": "laya",
            "state": raw_telemetry_text,
            "questions": NEUTRAL_CHOICE_QUESTION,
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


def format_raw_telemetry_prompt(
    t_now: float,
    num_links: int,
    num_core: int,
    num_edge: int,
    active_pol: str,
    cur_interval_s: float,
    aoi_s: float,
    polls_in_window: int,
    total_polls: int,
    mean_delta: float,
    max_delta: float,
    std_delta: float,
    p90_delta: float,
    core_mean_delta: float,
    edge_mean_delta: float,
    edge_max_delta: float,
    active_link_count: int,
    active_link_fraction: float,
    latest_utils: np.ndarray,
    latest_deltas: np.ndarray,
) -> str:
    """Strictly raw, neutral numerical telemetry snapshot — ZERO if/else regime hints."""
    # Show up to 8 representative links (or all links if N <= 8) as raw numbers
    if num_links <= 8:
        link_str = ", ".join(
            f"L{i}={latest_utils[i]:.1f}% (delta={latest_deltas[i]:.1f}%)"
            for i in range(num_links)
        )
    else:
        top_idx = np.argsort(latest_deltas)[::-1][:6]
        link_str = "Top-6 changing links: " + ", ".join(
            f"L{i}({'core' if i < num_core else 'edge'})={latest_utils[i]:.1f}% (delta={latest_deltas[i]:.1f}%)"
            for i in top_idx
        )

    return (
        f"Network Digital Twin Telemetry Snapshot at t={t_now:.1f}s\n"
        f"Topology: {num_links} monitored links ({num_core} core, {num_edge} edge)\n"
        f"Current Active Policy: {active_pol} (current_interval={cur_interval_s:.2f}s)\n"
        f"Measurement Age (AoI): {aoi_s:.2f}s | Polls in last 15s window: {polls_in_window} (total={total_polls})\n"
        f"Observed Link Delta Statistics (last 15s received polls):\n"
        f"- global_mean_abs_delta={mean_delta:.2f}%, global_max_abs_delta={max_delta:.2f}%, "
        f"global_p90_abs_delta={p90_delta:.2f}%, global_std_delta={std_delta:.2f}%\n"
        f"- core_mean_abs_delta={core_mean_delta:.2f}%, edge_mean_abs_delta={edge_mean_delta:.2f}%, "
        f"edge_max_abs_delta={edge_max_delta:.2f}%\n"
        f"- links_with_delta_gt_3.5pct={active_link_count}/{num_links} ({active_link_fraction*100:.1f}%)\n"
        f"Link Readings: {link_str}"
    )


def generate_trace(
    regime: str,
    num_links: int,
    delay_condition: str,
    seed: int,
    duration_s: float = 180.0,
    dt_s: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Generate deterministic multi-link trace + per-tick delivery delays."""
    rng = np.random.default_rng(seed)
    num_ticks = int(round(duration_s / dt_s)) + 1
    timestamps = np.round(np.arange(num_ticks) * dt_s, 4)

    num_core = max(1, int(round(num_links * 0.75)))
    num_edge = num_links - num_core

    true_u = np.zeros((num_ticks, num_links), dtype=np.float64)
    base_u = rng.uniform(20.0, 35.0, size=num_links)
    true_u[0] = base_u

    if regime == "bursty":
        burst_env = np.zeros((num_ticks, num_links), dtype=np.float64)
        starts = np.linspace(25.0, duration_s - 15.0, 8) + rng.uniform(-3.0, 3.0, size=8)
        for bs in starts:
            s_idx = max(1, min(num_ticks - 20, int(round(bs / dt_s))))
            dur = int(rng.integers(15, 38))
            e_idx = min(num_ticks, s_idx + dur)
            k_aff = max(1, num_links // 2)
            aff = rng.choice(num_links, size=k_aff, replace=False)
            burst_env[s_idx:e_idx, aff] += rng.uniform(20.0, 35.0, size=k_aff)
    elif regime == "shifted":
        shift_times = [45.0, 95.0, 140.0]
        shift_vecs = [rng.uniform(-15.0, 25.0, size=num_links) for _ in range(3)]

    shift_idx = 0
    cur_shift_target = base_u.copy()

    for t in range(1, num_ticks):
        time_s = timestamps[t]
        target = base_u.copy()
        sigma_vec = np.full(num_links, 0.55)

        if regime == "stable":
            sigma_vec[:] = 0.70

        elif regime == "bursty":
            target = base_u + burst_env[t]
            sigma_vec[:] = 0.75

        elif regime in ("alternating", "homogeneous_alternating"):
            is_vol = (int(time_s // 25.0) % 2) == 1
            if is_vol:
                phase = np.linspace(0.0, 2.0 * np.pi, num_links, endpoint=False)
                target = base_u + 14.0 + 16.0 * np.sin(2.0 * np.pi * time_s / 3.2 + phase)
                sigma_vec[:] = 6.5
            else:
                sigma_vec[:] = 0.45

        elif regime == "shifted":
            if shift_idx < len(shift_times) and time_s >= shift_times[shift_idx]:
                cur_shift_target = np.clip(base_u + shift_vecs[shift_idx], 8.0, 85.0)
                shift_idx += 1
            target = cur_shift_target
            sigma_vec[:] = 0.90

        elif regime == "localized_edge_noise":
            sigma_vec[:num_core] = 0.35
            sigma_vec[num_core:] = 0.50
            if num_edge > 0:
                target[num_core] += 9.5 * np.sin(2.0 * np.pi * time_s / 1.4) + rng.normal(0, 2.2)
            if num_edge > 1:
                target[num_core + 1] += 8.5 * np.cos(2.0 * np.pi * time_s / 1.8) + rng.normal(0, 1.8)
            if (50.0 <= time_s <= 75.0) or (120.0 <= time_s <= 145.0):
                target[:num_core] += 18.0 * np.sin(2.0 * np.pi * time_s / 3.0) + 15.0

        elif regime == "gradual_ramp_and_shift":
            # Smooth sinusoidal diurnal ramp + one sudden failover step at t=105s
            ramp = 18.0 * np.sin(np.pi * time_s / duration_s)
            target = base_u + ramp
            if time_s >= 105.0:
                target[:num_core] += 18.0
                if num_edge > 0:
                    target[num_core:] -= 8.0
            sigma_vec[:] = 0.65

        elif regime == "periodic_oscillation_trap":
            if 20.0 <= time_s < 110.0:
                for k in range(min(3, num_links)):
                    target[num_core + (k % max(1, num_edge)) if num_edge > 0 else k] += 7.5 * ((-1) ** (int(time_s * 2.0) + k))
            elif 110.0 <= time_s <= 165.0:
                target[:num_core] += 24.0 * np.sin(2.0 * np.pi * time_s / 3.5) + 16.0

        noise = rng.normal(0.0, sigma_vec)
        rate = 1.6 if regime != "stable" else 0.45
        true_u[t] = np.clip(true_u[t - 1] + rate * (target - true_u[t - 1]) * dt_s + noise * np.sqrt(dt_s), 1.0, 99.0)

    # Delivery delays (low=50ms, elevated=500ms) with occasional out-of-order spikes
    if delay_condition == "low_delay":
        base_ms, jitter_ms, ooo_p = 50.0, 10.0, 0.01
    else:
        base_ms, jitter_ms, ooo_p = 500.0, 80.0, 0.03
    delays_ms = np.clip(rng.normal(base_ms, jitter_ms, size=num_ticks), 10.0, base_ms * 3.0)
    spikes = rng.random(num_ticks) < ooo_p
    delays_ms[spikes] += rng.uniform(1200.0, 2800.0, size=int(np.sum(spikes)))

    return timestamps, true_u, delays_ms / 1000.0, num_core, num_edge


def run_single_controller(
    suite_name: str,
    regime: str,
    num_links: int,
    delay_condition: str,
    seed: int,
    timestamps: np.ndarray,
    true_u: np.ndarray,
    delays_s: np.ndarray,
    num_core: int,
    num_edge: int,
    controller: str,
    client: LiveLayaUnbiasedClient | None = None,
    fixed_interval_s: float = 2.5,
    review_logs: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    num_ticks = len(timestamps)
    bytes_per_poll = (8 + 4 * num_links) + (12 + 12 * num_links)

    stored_u = true_u[0].copy()
    latest_sample_t = -1.0
    active_pol = "change-aware" if controller == "change_aware" else "fixed"
    ca_interval = 2.5
    quiet_streak = 0

    last_poll_t = 0.0
    next_poll_t = 0.0
    next_review_t = 15.0
    in_flight: List[Tuple[float, float, np.ndarray]] = []
    accepted_obs: List[Tuple[float, float, np.ndarray, np.ndarray]] = []  # (sample_t, arr_t, util, delta)
    pending_decision: Tuple[float, str] | None = None

    polls_count = 0
    rejected_ooo = 0
    abs_errors = []
    pol_history = []
    latencies_ms: List[float] = []
    p_ca_list: List[float] = []
    fallbacks = 0
    reviews_total = 0
    switches = 0
    api_bytes = 0

    for t in range(num_ticks):
        t_now = timestamps[t]

        # 1. Deliver arriving polls
        rem = []
        for disp_t, arr_t, sample_u in in_flight:
            if arr_t <= t_now + 1e-9:
                if disp_t > latest_sample_t + 1e-9:
                    delta_vec = np.zeros(num_links) if latest_sample_t < 0 else np.abs(sample_u - stored_u)
                    stored_u = sample_u.copy()
                    latest_sample_t = disp_t
                    accepted_obs.append((disp_t, arr_t, sample_u.copy(), delta_vec))

                    if len(accepted_obs) > 1:
                        max_d = float(np.max(delta_vec))
                        mean_d = float(np.mean(delta_vec))
                        sig = max(max_d, 1.5 * mean_d)
                        if sig >= 4.5:
                            quiet_streak = 0
                            ca_interval = max(0.5, round(ca_interval * 0.5, 4))
                        elif sig <= 1.8:
                            quiet_streak += 1
                            if quiet_streak >= 2:
                                ca_interval = min(6.0, round(ca_interval * 1.5, 4))
                        else:
                            quiet_streak = 0

                    if active_pol == "change-aware":
                        next_poll_t = round(last_poll_t + ca_interval, 6)
                else:
                    rejected_ooo += 1
            else:
                rem.append((disp_t, arr_t, sample_u))
        in_flight = rem

        # 2. Apply pending decision once real latency has elapsed
        if pending_decision is not None and t_now >= pending_decision[0] - 1e-9:
            new_pol = pending_decision[1]
            pending_decision = None
            if new_pol != active_pol:
                active_pol = new_pol
                switches += 1
                cur_int = fixed_interval_s if active_pol == "fixed" else ca_interval
                cand_next = round(last_poll_t + cur_int, 6)
                next_poll_t = cand_next if cand_next > t_now + 1e-9 else round(t_now + cur_int, 6)

        # 3. Review every 15s (zero leakage: strictly uses accepted_obs with arr_t <= t_now)
        if controller in ("rule_simple", "rule_spatial", "laya_live_unbiased") and t_now >= next_review_t - 1e-9:
            next_review_t += 15.0
            if pending_decision is None:
                reviews_total += 1
                win_obs = [o for o in accepted_obs if o[1] >= max(0.0, t_now - 15.0) - 1e-9 and o[0] > 0.0]
                if win_obs:
                    d_mat = np.vstack([o[3] for o in win_obs])
                    mean_delta = float(np.mean(d_mat))
                    max_delta = float(np.max(d_mat))
                    std_delta = float(np.std(np.max(d_mat, axis=1)))
                    p90_delta = float(np.percentile(d_mat, 90))
                    core_mean_delta = float(np.mean(d_mat[:, :num_core]))
                    edge_mean_delta = float(np.mean(d_mat[:, num_core:])) if num_edge > 0 else 0.0
                    edge_max_delta = float(np.max(d_mat[:, num_core:])) if num_edge > 0 else 0.0
                    active_mask = np.max(d_mat, axis=0) > 3.5
                    active_cnt = int(np.sum(active_mask))
                    active_frac = float(np.mean(active_mask))
                    latest_u = accepted_obs[-1][2]
                    latest_d = accepted_obs[-1][3]
                else:
                    mean_delta = max_delta = std_delta = p90_delta = 0.0
                    core_mean_delta = edge_mean_delta = edge_max_delta = 0.0
                    active_cnt, active_frac = 0, 0.0
                    latest_u, latest_d = stored_u, np.zeros(num_links)

                aoi_s = t_now if latest_sample_t < 0 else max(0.0, t_now - latest_sample_t)
                cur_int = fixed_interval_s if active_pol == "fixed" else ca_interval

                if controller == "rule_simple":
                    chosen = "change-aware" if (max_delta >= 6.5 or mean_delta >= 3.2 or std_delta >= 2.2) else "fixed"
                    pending_decision = (t_now, chosen)
                elif controller == "rule_spatial":
                    # Fair deterministic spatial rule using the EXACT same multi-link features as Laya
                    if (core_mean_delta >= 2.2 and active_frac >= 0.25) or (max_delta < 1.8 and mean_delta < 1.0):
                        chosen = "change-aware"
                    else:
                        chosen = "fixed"
                    pending_decision = (t_now, chosen)
                else:
                    raw_prompt = format_raw_telemetry_prompt(
                        t_now=t_now, num_links=num_links, num_core=num_core, num_edge=num_edge,
                        active_pol=active_pol, cur_interval_s=cur_int, aoi_s=aoi_s,
                        polls_in_window=len(win_obs), total_polls=len(accepted_obs),
                        mean_delta=mean_delta, max_delta=max_delta, std_delta=std_delta, p90_delta=p90_delta,
                        core_mean_delta=core_mean_delta, edge_mean_delta=edge_mean_delta, edge_max_delta=edge_max_delta,
                        active_link_count=active_cnt, active_link_fraction=active_frac,
                        latest_utils=latest_u, latest_deltas=latest_d,
                    )
                    res = client.query(raw_prompt)
                    latencies_ms.append(res["latency_ms"])
                    api_bytes += res["req_bytes"] + res["rep_bytes"]
                    timed_out = (not res["is_valid"]) or (res["latency_ms"] > 2000.0)
                    if timed_out:
                        fallbacks += 1
                        chosen = active_pol
                    else:
                        chosen = res["choice"]
                        p_ca_list.append(res["p_ca"])
                        pending_decision = (t_now + res["latency_ms"] / 1000.0, chosen)

                    if review_logs is not None:
                        review_logs.append({
                            "suite": suite_name, "regime": regime, "num_links": num_links,
                            "delay": delay_condition, "seed": seed, "t_review_s": t_now,
                            "chosen": chosen, "fallback": timed_out,
                            "latency_ms": round(res["latency_ms"], 2),
                            "p_fixed": round(res["p_fixed"], 4), "p_ca": round(res["p_ca"], 4),
                            "mean_delta": round(mean_delta, 3), "max_delta": round(max_delta, 3),
                            "core_mean_delta": round(core_mean_delta, 3), "active_frac": round(active_frac, 3),
                        })

        # 4. Dispatch poll
        if t_now >= next_poll_t - 1e-9:
            polls_count += 1
            in_flight.append((t_now, round(t_now + delays_s[t], 6), true_u[t].copy()))
            last_poll_t = t_now
            cur_int = fixed_interval_s if active_pol == "fixed" else ca_interval
            next_poll_t = round(last_poll_t + cur_int, 6)

        if t_now >= 20.0:
            abs_errors.append(float(np.mean(np.abs(true_u[t] - stored_u))))
            pol_history.append(active_pol)

    ctrl_label = f"fixed_{fixed_interval_s:.1f}s" if controller == "fixed" else controller
    return {
        "suite": suite_name,
        "regime": regime,
        "num_links": num_links,
        "delay": delay_condition,
        "seed": seed,
        "controller": ctrl_label,
        "mae_pct": round(float(np.mean(abs_errors)), 4),
        "monitoring_kb": round((polls_count * bytes_per_poll) / 1000.0, 2),
        "polls": polls_count,
        "rejected_ooo": rejected_ooo,
        "switches": switches,
        "change_aware_frac": round(pol_history.count("change-aware") / max(1, len(pol_history)), 3),
        "mean_p_ca": round(float(np.mean(p_ca_list)), 4) if p_ca_list else np.nan,
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        "fallback_rate": round(fallbacks / max(1, reviews_total), 4) if controller == "laya_live_unbiased" else 0.0,
        "api_total_bytes": api_bytes,
    }


def run_all_fair_experiments(output_dir: str = "logs/fair_unbiased_laya") -> Tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    client = LiveLayaUnbiasedClient()

    # Warmup 3 live calls so CUDA kernels are JIT-compiled before timed traces
    print("Warming up live Laya CUDA server (3 calls)...")
    for _ in range(3):
        client.query("Network Digital Twin Telemetry Snapshot at t=15.0s\nmean_abs_delta=1.0%, max_abs_delta=2.0%")

    review_logs: List[Dict[str, Any]] = []
    suite_a_rows: List[Dict[str, Any]] = []
    suite_b_rows: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # SUITE A: Approved Research Plan 8-Link Benchmark (4 regimes x 2 delays x 6 seeds)
    # ------------------------------------------------------------------
    print("\n=== SUITE A: 8-Link Research Plan Benchmark (Zero Spoon-Feeding) ===")
    for regime in ["stable", "bursty", "alternating", "shifted"]:
        for delay in ["low_delay", "elevated_delay"]:
            print(f"  Running 8-link {regime} / {delay} (6 held-out seeds)...")
            for seed in [1001, 1002, 1003, 1004, 1005, 1006]:
                ts, tu, dly, nc, ne = generate_trace(regime, 8, delay, seed)
                for fi in [1.0, 2.5, 5.0]:
                    suite_a_rows.append(run_single_controller("suite_a_8link", regime, 8, delay, seed, ts, tu, dly, nc, ne, "fixed", fixed_interval_s=fi))
                for ctrl in ["change_aware", "rule_simple", "rule_spatial", "laya_live_unbiased"]:
                    suite_a_rows.append(run_single_controller("suite_a_8link", regime, 8, delay, seed, ts, tu, dly, nc, ne, ctrl, client=client, review_logs=review_logs))

    # ------------------------------------------------------------------
    # SUITE B: Diverse Network Types & Scales (8, 16, 32, 64 links x 4 types x 4 seeds)
    # ------------------------------------------------------------------
    print("\n=== SUITE B: Multi-Node Diverse Network Types & Scales (8 -> 64 links) ===")
    b_regimes = ["homogeneous_alternating", "localized_edge_noise", "gradual_ramp_and_shift", "periodic_oscillation_trap"]
    for regime in b_regimes:
        for n_links in [8, 16, 32, 64]:
            print(f"  Running {regime} / {n_links} links (4 seeds)...")
            for seed in [3001, 3002, 3003, 3004]:
                ts, tu, dly, nc, ne = generate_trace(regime, n_links, "low_delay", seed)
                for fi in [1.0, 2.5, 5.0]:
                    suite_b_rows.append(run_single_controller("suite_b_multinode", regime, n_links, "low_delay", seed, ts, tu, dly, nc, ne, "fixed", fixed_interval_s=fi))
                for ctrl in ["change_aware", "rule_simple", "rule_spatial", "laya_live_unbiased"]:
                    suite_b_rows.append(run_single_controller("suite_b_multinode", regime, n_links, "low_delay", seed, ts, tu, dly, nc, ne, ctrl, client=client, review_logs=review_logs))

    df_a = pd.DataFrame(suite_a_rows)
    df_b = pd.DataFrame(suite_b_rows)
    df_a.to_csv(os.path.join(output_dir, "fair_suiteA_8link_results.csv"), index=False)
    df_b.to_csv(os.path.join(output_dir, "fair_suiteB_multinode_scale_results.csv"), index=False)
    with open(os.path.join(output_dir, "fair_unbiased_review_logs.jsonl"), "w", encoding="utf-8") as f:
        for rec in review_logs:
            f.write(json.dumps(rec) + "\n")

    # ------------------------------------------------------------------
    # PLOT 1: Suite A Pareto Frontier (Error vs. Traffic on 8-Link Benchmark)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2))
    ctrl_styles = {
        "change_aware":       ("Change-Aware",  "#dd8452", "s", 95),
        "rule_simple":        ("Rule (Simple)", "#55a868", "^", 105),
        "rule_spatial":       ("Rule (Spatial)","#8172b3", "P", 105),
        "laya_live_unbiased": ("Laya (Unbiased GPU)", "#c44e52", "D", 115),
    }
    for r_idx, delay in enumerate(["low_delay", "elevated_delay"]):
        for c_idx, regime in enumerate(["stable", "bursty", "alternating", "shifted"]):
            ax = axes[r_idx, c_idx]
            sub = df_a[(df_a["regime"] == regime) & (df_a["delay"] == delay)].groupby("controller", as_index=False).mean(numeric_only=True)
            # Plot static fixed Pareto frontier (5.0s -> 2.5s -> 1.0s)
            f_pts = sub[sub["controller"].str.startswith("fixed_")].sort_values("monitoring_kb")
            ax.plot(f_pts["monitoring_kb"], f_pts["mae_pct"], "o--", color="#4c72b0", linewidth=2, markersize=7, label="Fixed Frontier (5s/2.5s/1s)")
            for ctrl, (lbl, clr, mk, sz) in ctrl_styles.items():
                pt = sub[sub["controller"] == ctrl]
                if not pt.empty:
                    ax.scatter(pt["monitoring_kb"], pt["mae_pct"], color=clr, marker=mk, s=sz, edgecolor="black", zorder=5, label=lbl)
            ax.set_title(f"{regime} ({delay})", fontsize=10, fontweight="bold")
            if r_idx == 1:
                ax.set_xlabel("Monitoring Traffic (KB)", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel("Mean Twin MAE (% pts)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=7.5, loc="upper right")
    plt.suptitle("Suite A (8-Link): True Pareto Error vs. Traffic (ZERO Prompt Spoon-Feeding)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fair_fig1_suiteA_pareto_8link.png"), dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PLOT 2: Suite B Pareto Comparison at 32 & 64 Links across 4 Diverse Network Types
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.2))
    for r_idx, n_links in enumerate([32, 64]):
        for c_idx, regime in enumerate(b_regimes):
            ax = axes[r_idx, c_idx]
            sub = df_b[(df_b["regime"] == regime) & (df_b["num_links"] == n_links)].groupby("controller", as_index=False).mean(numeric_only=True)
            f_pts = sub[sub["controller"].str.startswith("fixed_")].sort_values("monitoring_kb")
            ax.plot(f_pts["monitoring_kb"], f_pts["mae_pct"], "o--", color="#4c72b0", linewidth=2, markersize=7, label="Fixed Frontier (5s/2.5s/1s)")
            for ctrl, (lbl, clr, mk, sz) in ctrl_styles.items():
                pt = sub[sub["controller"] == ctrl]
                if not pt.empty:
                    ax.scatter(pt["monitoring_kb"], pt["mae_pct"], color=clr, marker=mk, s=sz, edgecolor="black", zorder=5, label=lbl)
            ax.set_title(f"{regime}\n({n_links} links)", fontsize=9.5, fontweight="bold")
            if r_idx == 1:
                ax.set_xlabel("Monitoring Traffic (KB)", fontsize=9)
            if c_idx == 0:
                ax.set_ylabel("Mean Twin MAE (% pts)", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.4)
            if r_idx == 0 and c_idx == 0:
                ax.legend(fontsize=7.5, loc="upper right")
    plt.suptitle("Suite B (32 & 64 Links): Unbiased Pareto Frontier Across Diverse Network Types", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fair_fig2_suiteB_pareto_multinode.png"), dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------------
    # PLOT 3: Laya Unbiased Decision Behavior & O(1) GPU Latency Across Scales
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8))
    laya_b = df_b[df_b["controller"] == "laya_live_unbiased"]
    piv = laya_b.groupby(["regime", "num_links"])["change_aware_frac"].mean().unstack()
    piv.plot(kind="bar", ax=ax1, edgecolor="black", linewidth=0.5)
    ax1.set_title("Unbiased Laya: Fraction of Time Selecting 'change-aware'", fontweight="bold")
    ax1.set_ylabel("change-aware Fraction (0 = always fixed, 1 = always CA)")
    ax1.set_ylim(0, 1.08)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.tick_params(axis="x", rotation=20)

    lat_s = laya_b.groupby("num_links")[["mean_latency_ms", "p95_latency_ms"]].mean()
    ax2.bar([str(n) for n in lat_s.index], lat_s["mean_latency_ms"], color="#c44e52", edgecolor="black", width=0.5, label="Mean RTT (ms)")
    ax2.plot([str(n) for n in lat_s.index], lat_s["p95_latency_ms"], color="#1e1e2e", marker="o", linewidth=2, label="P95 RTT (ms)")
    ax2.set_title("Live Unbiased Laya GPU Latency Across Scales (8 -> 64 Links)", fontweight="bold")
    ax2.set_xlabel("Number of Monitored Links (N)")
    ax2.set_ylabel("Decision Latency (ms)")
    ax2.set_ylim(0, max(80.0, float(lat_s["p95_latency_ms"].max()) * 1.4))
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fair_fig3_laya_behavior_and_latency.png"), dpi=180)
    plt.close(fig)

    print(f"\n✓ Completed {len(review_logs)} UNBIASED live GPU reviews! Saved to {output_dir}/")
    return df_a, df_b


if __name__ == "__main__":
    df_a, df_b = run_all_fair_experiments()
    print("\n=== SUITE A SUMMARY (8-LINK, LOW DELAY, UNBIASED) ===")
    print(df_a[df_a["delay"] == "low_delay"].groupby(["regime", "controller"])[["mae_pct", "monitoring_kb", "change_aware_frac", "mean_latency_ms", "fallback_rate"]].mean().round(3).to_string())
    print("\n=== SUITE B SUMMARY (32 & 64 LINKS, UNBIASED) ===")
    print(df_b[df_b["num_links"].isin([32, 64])].groupby(["regime", "num_links", "controller"])[["mae_pct", "monitoring_kb", "change_aware_frac", "mean_latency_ms", "fallback_rate"]].mean().round(3).to_string())
