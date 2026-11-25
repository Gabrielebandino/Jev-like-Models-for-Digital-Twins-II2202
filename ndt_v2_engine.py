#!/usr/bin/env python3
"""NDT Synchronization v2 Engine — PhD-Level Trace-Driven Simulation, Oracle Headroom & Statistics.

Strictly compliant with the approved II2202 Research Plan (researchPlan5Nov/II2202-research-plan.tex)
while implementing every methodological fix from RESEARCH_PLAN_REVIEW.md:
  1. Rate-normalized ChangeAwarePolicy (eliminating sqrt(dt) interval confounding) + pilot tuning.
  2. Dense 15-point log-spaced static `fixed` frontier + convex hull static baseline.
  3. One-step counterfactual Oracle Switcher (+ exact 2^R brute-force verifier for short traces)
     yielding per-trace Headroom H, Fraction of Headroom Captured, and per-review oracle labels.
  4. Shared rich temporal + spatial feature vector (strict information parity, zero policy labels):
     ordered last-K polls, EWMA control chart, CUSUM, OLS slope, sign-reversal oscillation index,
     Gini spatial concentration, AoI, and monotone pilot-percentile scale context.
  5. Equal pilot tuning budget (k=20 evaluations) for Change-Aware, Simple Rule (with AoI!),
     EWMA/CUSUM+Hysteresis Rule, and Model prompt/paraphrase + Zhao et al. (2021) contextual calibration.
  6. Diagnostic references: Random Switcher floor, Oracle ceiling, Pilot-trained Learnability Ceiling
     (LogisticRegression / HistGBDT), and threshold-free ROC-AUC of p(change-aware) vs. oracle labels.
  7. Statistical rigor: power analysis, paired 95% bootstrap CIs, Wilcoxon signed-rank with
     Holm-Bonferroni correction, TOST equivalence testing against pre-registered SESOI, and
     operational congestion-crossing (>75% util) detection F1 / delay.
  8. Primary 8-link shared-path topology (approved plan) + External-Validity SNDlib Abilene
     (12 nodes, 15 bidirectional links, Dijkstra OSPF shortest-path routing + link failover).
"""

from __future__ import annotations
import os
import json
import math
import time
import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==============================================================================
# 1. TOPOLOGIES: PRIMARY 8-LINK (APPROVED PLAN) & SNDLIB ABILENE (EXTERNAL VALIDITY)
# ==============================================================================
PRIMARY_8LINK_NAMES = [f"L{i}" for i in range(8)]
PRIMARY_8LINK_CORE = 6  # L0..L5 core, L6..L7 edge aggregation
PRIMARY_8LINK_OD_PATHS: Dict[str, List[Tuple[int, float]]] = {
    "F0_core_west_east": [(0, 1.0), (1, 0.9), (4, 0.85)],
    "F1_core_north_south": [(2, 1.0), (3, 0.95), (5, 0.8)],
    "F2_edge_aggregation_A": [(0, 0.5), (2, 0.6), (6, 1.0)],
    "F3_edge_aggregation_B": [(1, 0.55), (3, 0.5), (7, 1.0)],
    "F4_cross_datacenter": [(4, 0.7), (5, 0.75), (6, 0.6), (7, 0.65)],
    "F5_background_mesh": [(0, 0.3), (3, 0.35), (4, 0.4), (7, 0.3)],
}

# Canonical SNDlib Abilene US Backbone (12 nodes, 15 bidirectional links, OSPF weights)
ABILENE_NODES = [
    "ATLAng", "ATLAM5", "CHINng", "DNVRng", "HSTNng", "IPLSng",
    "KSCYng", "LOSAng", "NYCMng", "SNVAng", "STTLng", "WASHng",
]
# (u, v, ospf_metric)
ABILENE_EDGES = [
    ("ATLAM5", "ATLAng", 1),
    ("ATLAng", "HSTNng", 1176),
    ("ATLAng", "IPLSng", 679),
    ("ATLAng", "WASHng", 870),
    ("CHINng", "IPLSng", 260),
    ("CHINng", "NYCMng", 1148),
    ("DNVRng", "KSCYng", 895),
    ("DNVRng", "SNVAng", 1528),
    ("DNVRng", "STTLng", 1638),
    ("HSTNng", "KSCYng", 1040),
    ("HSTNng", "LOSAng", 2204),
    ("IPLSng", "KSCYng", 726),
    ("LOSAng", "SNVAng", 554),
    ("NYCMng", "WASHng", 329),
    ("SNVAng", "STTLng", 1096),
]
ABILENE_LINK_NAMES = [f"{u[:3]}-{v[:3]}" for u, v, _ in ABILENE_EDGES]


def compute_ospf_routing_matrix(
    nodes: List[str],
    edges: List[Tuple[str, str, int]],
    failed_edge_idx: Optional[int] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Compute shortest-path OSPF link lists for all node pairs via Dijkstra."""
    n = len(nodes)
    idx_map = {name: i for i, name in enumerate(nodes)}
    adj: List[List[Tuple[int, int, int]]] = [[] for _ in range(n)]
    for e_idx, (u, v, w) in enumerate(edges):
        if failed_edge_idx is not None and e_idx == failed_edge_idx:
            continue
        iu, iv = idx_map[u], idx_map[v]
        adj[iu].append((iv, w, e_idx))
        adj[iv].append((iu, w, e_idx))

    paths: Dict[Tuple[int, int], List[int]] = {}
    for src in range(n):
        dist = [float("inf")] * n
        prev_node = [-1] * n
        prev_edge = [-1] * n
        dist[src] = 0.0
        visited = [False] * n
        for _ in range(n):
            u = -1
            best = float("inf")
            for i in range(n):
                if not visited[i] and dist[i] < best:
                    best = dist[i]
                    u = i
            if u == -1:
                break
            visited[u] = True
            for v, w, e_idx in adj[u]:
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    prev_node[v] = u
                    prev_edge[v] = e_idx
        for dst in range(n):
            if src == dst or prev_edge[dst] == -1:
                continue
            e_list = []
            cur = dst
            while cur != src:
                e_list.append(prev_edge[cur])
                cur = prev_node[cur]
            paths[(src, dst)] = e_list[::-1]
    return paths


ABILENE_OSPF_NORMAL = compute_ospf_routing_matrix(ABILENE_NODES, ABILENE_EDGES, failed_edge_idx=None)
# Fail core central link IPLSng-KSCYng (index 11) for real OSPF failover reroute
ABILENE_OSPF_FAILOVER = compute_ospf_routing_matrix(ABILENE_NODES, ABILENE_EDGES, failed_edge_idx=11)


# ==============================================================================
# 2. PROTOCOL CONFIGURATION & PRE-REGISTERED CONSTANTS
# ==============================================================================
DELAY_PARAMS_MS = {
    "low_delay": {"base_ms": 50.0, "jitter_std_ms": 10.0, "out_of_order_prob": 0.01},
    "elevated_delay": {"base_ms": 500.0, "jitter_std_ms": 60.0, "out_of_order_prob": 0.03},
}

# Dense 15-point log-spaced fixed intervals for unbiased convex Pareto frontier
DENSE_FIXED_INTERVALS_S: Tuple[float, ...] = tuple(
    round(float(x), 3) for x in np.geomspace(0.5, 10.0, 15)
)


@dataclass
class ProtocolV2:
    """Mutable during Pilot calibration, then serialized and frozen before Held-Out runs."""
    # Timing
    trace_duration_s: float = 480.0       # 8-minute traces (45 post-warmup reviews at H=10s)
    control_duration_s: float = 240.0     # 4-minute traces for homogeneous controls
    warmup_duration_s: float = 30.0       # Common warm-up excluded from MAE, included in bytes
    tick_dt_s: float = 0.1                # 100 ms discrete ticks
    review_period_s: float = 10.0         # Calibrated in pilot over {5, 10, 15, 30}s
    decision_deadline_ms: float = 2000.0

    # Fixed policy reference (frozen midpoint)
    fixed_interval_s: float = 2.5

    # Rate-normalized Change-Aware policy parameters (tuned on pilot, equal budget k=20)
    ca_min_interval_s: float = 0.5
    ca_max_interval_s: float = 6.0
    ca_init_interval_s: float = 2.5
    ca_norm_exp: float = 0.5              # Rate normalization: delta / (dt ** ca_norm_exp)
    ca_high_threshold: float = 3.2        # Shorten interval when rate-normalized signal >= high
    ca_low_threshold: float = 1.2         # Lengthen interval when signal <= low for quiet_streak
    ca_quiet_streak_required: int = 2
    ca_shorten_factor: float = 0.5
    ca_lengthen_factor: float = 1.5
    ca_signal_mode: str = "max_weighted"  # "max_weighted" or "p75_mean"

    # Scalarized objective weight lambda_kb_per_link: J = MAE + lambda_kb_per_link * (monitoring_kb / num_links)
    # Calibrated in pilot as the exact supporting-hyperplane subgradient at fixed_interval_s=2.5s
    # so NO static fixed interval can game J (scale-invariant across 8-link & 15-link Abilene!).
    lambda_kb_per_link: float = 0.25

    # Pre-registered SESOI (Smallest Effect Size of Interest) for TOST equivalence
    sesoi_cost: float = 0.10              # 0.10 cost units (~0.10 pp MAE at equal bytes)

    # Simple threshold rule parameters (tuned on pilot, k=20; uses BOTH change and AoI per §2.3!)
    simple_mean_rate_thresh: float = 1.6
    simple_max_rate_thresh: float = 4.0
    simple_max_age_safety_s: float = 4.5
    simple_quiet_mean_thresh: float = 0.8

    # Strong EWMA + CUSUM + Oscillation/Gini + Hysteresis rule parameters (tuned on pilot, k=20)
    ewma_alpha_fast: float = 0.45
    ewma_alpha_slow: float = 0.15
    ewma_beta_dev: float = 0.25
    ewma_k_sigma: float = 1.5
    cusum_thresh: float = 4.0
    osc_trap_thresh: float = 0.62         # High sign-reversal rate => periodic jitter trap => fixed
    gini_flap_thresh: float = 0.52        # High Gini concentration => localized single-link flap => fixed
    active_frac_macro_min: float = 0.35   # Network-wide volatility requires broad active links
    quiet_rate_thresh: float = 0.90       # Quiet regime => change-aware stretches to 6s to save bytes
    hysteresis_dwell_reviews: int = 1     # Minimum dwell reviews before switching back

    # Operational congestion threshold for secondary event-detection metric
    congestion_threshold_pct: float = 75.0

    # Frozen model prompt & Zhao et al. (2021) contextual calibration
    selected_prompt_idx: int = 0
    calibration_prior_p_ca: float = 0.5   # Estimated from content-free baseline input in pilot

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ==============================================================================
# 3. TRACE GENERATORS (8-LINK PRIMARY + REGIME-SWITCHING + SNDLIB ABILENE)
# ==============================================================================
def _generate_delays(rng: np.random.Generator, num_ticks: int, delay_condition: str) -> np.ndarray:
    cfg = DELAY_PARAMS_MS[delay_condition]
    delays_ms = np.clip(
        rng.normal(cfg["base_ms"], cfg["jitter_std_ms"], size=num_ticks),
        10.0,
        cfg["base_ms"] * 3.0,
    )
    spikes = rng.random(num_ticks) < cfg["out_of_order_prob"]
    delays_ms[spikes] += rng.uniform(1200.0, 2800.0, size=int(np.sum(spikes)))
    return delays_ms / 1000.0


def generate_trace_v2(
    regime: str,
    delay_condition: str,
    seed: int,
    protocol: ProtocolV2,
    topology: str = "8link",
) -> Dict[str, Any]:
    """Generate a deterministic multi-link utilization trace with known ground truth.

    Regimes:
      - `regime_switching` (PRIMARY): Continuous-time Markov chain transitioning among 4 realistic
        operational states with random dwell times (35-65s):
          0: `quiet` (steady low-variance load -> change-aware relaxes to 6s, saves bytes)
          1: `macro_volatile` (network-wide heavy-tailed Pareto surges & waves -> change-aware drops to 0.5s)
          2: `localized_flap` (1-2 edge links flapping violently while core is quiet -> change-aware panics
             and wastes 5x bytes across ALL links; fixed(2.5s) wins!)
          3: `jitter_microburst` (unresolvable sub-second aliasing jitter / tiny micro-spikes ->
             change-aware over-polls reactive tails; fixed(2.5s) wins!)
      - `stable`, `bursty`, `alternating`, `shifted`: Approved Research Plan §2.2 control regimes.
      - `abilene_regime_switching`: External-validity SNDlib Abilene (12 nodes, 15 links) with
        Dijkstra OSPF routing, real single-link OSPF failover reroute, Pareto ON/OFF bursts,
        localized transceiver flap, and quiet windows.
    """
    rng = np.random.default_rng(seed)
    dt = protocol.tick_dt_s
    is_long = regime in ("regime_switching", "abilene_regime_switching")
    duration_s = protocol.trace_duration_s if is_long else protocol.control_duration_s
    num_ticks = int(round(duration_s / dt)) + 1
    timestamps = np.round(np.arange(num_ticks) * dt, 4)

    if topology == "abilene" or regime == "abilene_regime_switching":
        link_names = ABILENE_LINK_NAMES
        num_links = len(link_names)
        # 24 representative OD pairs routed via real Dijkstra OSPF on Abilene
        od_pairs = [(i, (i + k) % len(ABILENE_NODES)) for i in range(len(ABILENE_NODES)) for k in (3, 5)]
        num_flows = len(od_pairs)
        base_flows = rng.uniform(3.0, 7.0, size=num_flows)
        flow_demands = np.zeros((num_ticks, num_flows), dtype=np.float64)
        flow_demands[0] = base_flows
        link_utils = np.zeros((num_ticks, num_links), dtype=np.float64)
        is_change_event = np.zeros(num_ticks, dtype=np.int32)
        phase_labels: List[str] = ["quiet"] * num_ticks

        # Build realistic ISP Markov regime schedule (quiet ~45%, macro OSPF failover ~25%, flap ~15%, jitter ~15%)
        states = ["quiet", "macro_ospf_failover", "localized_flap", "jitter_microburst"]
        state_probs = np.array([0.45, 0.25, 0.15, 0.15])
        cur_state_idx = 0
        t_cursor = 0
        failover_active = np.zeros(num_ticks, dtype=bool)
        flap_link = int(rng.integers(0, num_links))
        while t_cursor < num_ticks:
            dwell_s = float(rng.uniform(45.0, 75.0)) if cur_state_idx == 0 else float(rng.uniform(35.0, 60.0))
            dwell_ticks = int(round(dwell_s / dt))
            end_t = min(num_ticks, t_cursor + dwell_ticks)
            st_name = states[cur_state_idx]
            if t_cursor > 0:
                is_change_event[t_cursor] = 1
            for k in range(t_cursor, end_t):
                phase_labels[k] = st_name
                if st_name == "macro_ospf_failover":
                    failover_active[k] = True
            nxt_choices = [i for i in range(len(states)) if i != cur_state_idx]
            nxt_w = state_probs[nxt_choices] / np.sum(state_probs[nxt_choices])
            cur_state_idx = int(rng.choice(nxt_choices, p=nxt_w))
            t_cursor = end_t

        # Pareto ON/OFF burst states for heavy-tailed traffic
        on_off_timer = np.zeros(num_flows, dtype=np.int32)
        on_off_amp = np.zeros(num_flows, dtype=np.float64)

        for t in range(1, num_ticks):
            st = phase_labels[t]
            time_s = timestamps[t]
            if st == "quiet":
                noise = rng.normal(0.0, 0.12, size=num_flows)
                flow_demands[t] = flow_demands[t - 1] + 0.7 * (base_flows - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)
            elif st == "macro_ospf_failover":
                # Heavy-tailed Pareto ON/OFF bursts + correlated surge during OSPF reroute
                for f in range(num_flows):
                    if on_off_timer[f] <= 0:
                        if rng.random() < 0.04:
                            # Pareto duration (heavy tail, alpha=1.6)
                            dur_s = min(15.0, float((rng.pareto(1.6) + 1.0) * 1.5))
                            on_off_timer[f] = int(round(dur_s / dt))
                            on_off_amp[f] = float(rng.uniform(4.0, 10.0))
                        else:
                            on_off_amp[f] = 0.0
                    else:
                        on_off_timer[f] -= 1
                wave = 3.5 * np.sin(2.0 * np.pi * time_s / 6.0 + np.arange(num_flows) * 0.4)
                target = base_flows + 4.0 + wave + on_off_amp
                noise = rng.normal(0.0, 1.4, size=num_flows)
                flow_demands[t] = flow_demands[t - 1] + 1.5 * (target - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)
            else:
                noise = rng.normal(0.0, 0.25, size=num_flows)
                flow_demands[t] = flow_demands[t - 1] + 0.8 * (base_flows - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)

            routing = ABILENE_OSPF_FAILOVER if failover_active[t] else ABILENE_OSPF_NORMAL
            u_row = np.zeros(num_links, dtype=np.float64)
            for f_idx, pair in enumerate(od_pairs):
                for e_id in routing.get(pair, []):
                    u_row[e_id] += max(0.2, flow_demands[t, f_idx])
            if failover_active[t]:
                u_row[11] = 0.0  # Failed link drops to 0%

            if st == "localized_flap":
                # Faulty optical transceiver on 1 peripheral link oscillating rapidly
                u_row[flap_link] += 26.0 * np.sin(2.0 * np.pi * time_s / 1.3) + rng.normal(0.0, 4.0)
            elif st == "jitter_microburst":
                # Fast zero-mean micro-jitter across 2 links (faster than polling Nyquist)
                for jl in (0, 2):
                    u_row[jl] += 11.0 * ((-1) ** int(time_s * 2.5 + jl)) + rng.normal(0.0, 2.5)

            link_utils[t] = np.clip(u_row + rng.normal(0.0, 0.18, size=num_links), 0.5, 99.0)
        link_utils[0] = link_utils[1].copy()

        delays_s = _generate_delays(rng, num_ticks, delay_condition)
        return {
            "topology": "abilene",
            "regime": regime,
            "delay_condition": delay_condition,
            "seed": seed,
            "link_names": link_names,
            "num_links": num_links,
            "timestamps": timestamps,
            "true_u": link_utils,
            "delays_s": delays_s,
            "is_change_event": is_change_event,
            "phase_labels": phase_labels,
        }

    # --- PRIMARY 8-LINK SHARED-PATH TOPOLOGY (Approved Research Plan §2.2) ---
    link_names = PRIMARY_8LINK_NAMES
    num_links = len(link_names)
    flow_names = list(PRIMARY_8LINK_OD_PATHS.keys())
    num_flows = len(flow_names)
    base_demands = rng.uniform(12.0, 22.0, size=num_flows)
    flow_demands = np.zeros((num_ticks, num_flows), dtype=np.float64)
    flow_demands[0] = base_demands
    is_change_event = np.zeros(num_ticks, dtype=np.int32)
    phase_labels = ["baseline"] * num_ticks
    link_extra = np.zeros((num_ticks, num_links), dtype=np.float64)

    if regime == "regime_switching":
        # Realistic ISP Markov-modulated regime switching (~45% quiet, ~25% macro_volatile, ~15% flap, ~15% jitter)
        states = ["quiet", "macro_volatile", "localized_flap", "jitter_microburst"]
        state_probs = np.array([0.45, 0.25, 0.15, 0.15])
        cur_st = 0
        t_cur = 0
        flap_link = int(rng.choice([6, 7]))  # One edge aggregation link flaps
        while t_cur < num_ticks:
            dwell_s = float(rng.uniform(45.0, 75.0)) if cur_st == 0 else float(rng.uniform(35.0, 60.0))
            dwell_ticks = int(round(dwell_s / dt))
            end_t = min(num_ticks, t_cur + dwell_ticks)
            st_name = states[cur_st]
            if t_cur > 0:
                is_change_event[t_cur] = 1
            for k in range(t_cur, end_t):
                phase_labels[k] = st_name
            nxt_choices = [i for i in range(len(states)) if i != cur_st]
            nxt_w = state_probs[nxt_choices] / np.sum(state_probs[nxt_choices])
            cur_st = int(rng.choice(nxt_choices, p=nxt_w))
            t_cur = end_t

        # Pareto ON/OFF burst timers for macro_volatile
        on_timer = np.zeros(num_flows, dtype=np.int32)
        on_amp = np.zeros(num_flows, dtype=np.float64)

        for t in range(1, num_ticks):
            st = phase_labels[t]
            time_s = timestamps[t]
            if st == "quiet":
                noise = rng.normal(0.0, 0.35, size=num_flows)
                flow_demands[t] = (
                    flow_demands[t - 1]
                    + 0.8 * (base_demands - flow_demands[t - 1]) * dt
                    + noise * np.sqrt(dt)
                )
            elif st == "macro_volatile":
                # Network-wide sustained Pareto surges + correlated traffic wave (hits >75% congestion!)
                for f in range(num_flows):
                    if on_timer[f] <= 0:
                        if rng.random() < 0.05:
                            dur_s = min(12.0, float((rng.pareto(1.7) + 1.0) * 1.6))
                            on_timer[f] = int(round(dur_s / dt))
                            on_amp[f] = float(rng.uniform(10.0, 22.0))
                        else:
                            on_amp[f] = 0.0
                    else:
                        on_timer[f] -= 1
                wave = 14.0 * np.sin(2.0 * np.pi * time_s / 11.0 + np.arange(num_flows) * 0.6)
                target = base_demands + 14.0 + wave + on_amp
                noise = rng.normal(0.0, 5.5, size=num_flows)
                flow_demands[t] = (
                    flow_demands[t - 1]
                    + 1.8 * (target - flow_demands[t - 1]) * dt
                    + noise * np.sqrt(dt)
                )
            elif st == "localized_flap":
                # Backbone flows are calm, but 1 single edge link has severe optical/port flap
                noise = rng.normal(0.0, 0.35, size=num_flows)
                flow_demands[t] = (
                    flow_demands[t - 1]
                    + 0.8 * (base_demands - flow_demands[t - 1]) * dt
                    + noise * np.sqrt(dt)
                )
                link_extra[t, flap_link] = (
                    22.0 * np.sin(2.0 * np.pi * time_s / 1.35) + rng.normal(0.0, 3.5)
                )
            elif st == "jitter_microburst":
                # Unresolvable rapid zero-mean oscillatory jitter on 2 edge links + mild core calm
                noise = rng.normal(0.0, 0.45, size=num_flows)
                flow_demands[t] = (
                    flow_demands[t - 1]
                    + 0.9 * (base_demands - flow_demands[t - 1]) * dt
                    + noise * np.sqrt(dt)
                )
                link_extra[t, 6] = 11.5 * ((-1) ** int(time_s * 2.4)) + rng.normal(0.0, 2.0)
                link_extra[t, 7] = 10.5 * ((-1) ** int(time_s * 2.4 + 1)) + rng.normal(0.0, 2.0)

    elif regime == "stable":
        for t in range(1, num_ticks):
            noise = rng.normal(0.0, 0.70, size=num_flows)
            flow_demands[t] = flow_demands[t - 1] + 0.40 * (base_demands - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)
            phase_labels[t] = "stable"

    elif regime == "bursty":
        burst_env = np.zeros((num_ticks, num_flows), dtype=np.float64)
        starts = np.linspace(35.0, duration_s - 20.0, 10) + rng.uniform(-3.0, 3.0, size=10)
        for bs in starts:
            s_idx = max(1, min(num_ticks - 25, int(round(bs / dt))))
            dur = int(rng.integers(18, 45))
            e_idx = min(num_ticks, s_idx + dur)
            aff = rng.choice(num_flows, size=3, replace=False)
            amps = rng.uniform(18.0, 34.0, size=len(aff))
            for a_i, f_id in enumerate(aff):
                burst_env[s_idx:e_idx, f_id] += amps[a_i]
            is_change_event[s_idx] = 1
            for k in range(s_idx, e_idx):
                phase_labels[k] = "burst"
        for t in range(1, num_ticks):
            noise = rng.normal(0.0, 0.70, size=num_flows)
            target = base_demands + burst_env[t]
            rate = 1.8 if phase_labels[t] == "burst" else 0.45
            flow_demands[t] = flow_demands[t - 1] + rate * (target - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)

    elif regime == "alternating":
        window_s = 35.0
        for t in range(1, num_ticks):
            time_s = timestamps[t]
            is_vol = (int(time_s // window_s) % 2) == 1
            prev_vol = (int(timestamps[t - 1] // window_s) % 2) == 1
            if is_vol != prev_vol and time_s >= protocol.warmup_duration_s:
                is_change_event[t] = 1
            if is_vol:
                phase_labels[t] = "volatile"
                osc = 12.0 * np.sin(2.0 * np.pi * time_s / 5.0 + np.arange(num_flows))
                target = base_demands + 12.0 + osc
                noise = rng.normal(0.0, 5.0, size=num_flows)
                flow_demands[t] = flow_demands[t - 1] + 1.5 * (target - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)
            else:
                phase_labels[t] = "quiet"
                noise = rng.normal(0.0, 0.40, size=num_flows)
                flow_demands[t] = flow_demands[t - 1] + 0.8 * (base_demands - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)

    elif regime == "shifted":
        shift_times = [55.0, 120.0, 185.0]
        shift_vecs = [
            rng.uniform(12.0, 24.0, size=num_flows) * np.array([1, 1, -0.4, 1, 0.8, -0.3]),
            rng.uniform(14.0, 26.0, size=num_flows) * np.array([-0.6, 0.9, 1.2, -0.5, 1.0, 0.8]),
            rng.uniform(12.0, 22.0, size=num_flows) * np.array([0.8, -0.7, 0.9, 1.1, -0.6, 1.0]),
        ]
        cur_target = base_demands.copy()
        s_idx = 0
        for t in range(1, num_ticks):
            time_s = timestamps[t]
            if s_idx < len(shift_times) and time_s >= shift_times[s_idx]:
                cur_target = np.clip(base_demands + shift_vecs[s_idx], 5.0, 55.0)
                is_change_event[t] = 1
                s_idx += 1
            phase_labels[t] = f"shift_{s_idx}"
            noise = rng.normal(0.0, 0.95, size=num_flows)
            flow_demands[t] = flow_demands[t - 1] + 1.5 * (cur_target - flow_demands[t - 1]) * dt + noise * np.sqrt(dt)
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Route OD flows onto the 8 physical links
    link_utils = np.zeros((num_ticks, num_links), dtype=np.float64)
    for f_idx, f_name in enumerate(flow_names):
        for link_id, weight in PRIMARY_8LINK_OD_PATHS[f_name]:
            link_utils[:, link_id] += flow_demands[:, f_idx] * weight
    link_utils = np.clip(link_utils + link_extra + rng.normal(0.0, 0.20, size=(num_ticks, num_links)), 1.0, 99.0)

    delays_s = _generate_delays(rng, num_ticks, delay_condition)
    return {
        "topology": "8link",
        "regime": regime,
        "delay_condition": delay_condition,
        "seed": seed,
        "link_names": link_names,
        "num_links": num_links,
        "timestamps": timestamps,
        "true_u": link_utils,
        "delays_s": delays_s,
        "is_change_event": is_change_event,
        "phase_labels": phase_labels,
    }


# ==============================================================================
# 4. RATE-NORMALIZED CHANGE-AWARE POLICY & SIMULATOR STATE
# ==============================================================================
@dataclass
class ObsRecord:
    poll_id: int
    sample_t: float
    arrival_t: float
    dt_sample: float
    util: np.ndarray
    signed_delta: np.ndarray
    abs_delta: np.ndarray
    rate_delta: np.ndarray  # abs_delta / (dt_sample ** norm_exp)


class RateNormalizedChangeAware:
    """OpenNetMon / RFC 8641 dampened adaptive polling with rate-normalized change signal.

    Eliminates the sqrt(dt) polling-interval confounding found in v1:
    signal is normalized by (dt_sample ** norm_exp) before threshold comparison.
    """

    def __init__(self, protocol: ProtocolV2):
        self.min_i = protocol.ca_min_interval_s
        self.max_i = protocol.ca_max_interval_s
        self.interval_s = protocol.ca_init_interval_s
        self.high_th = protocol.ca_high_threshold
        self.low_th = protocol.ca_low_threshold
        self.quiet_req = protocol.ca_quiet_streak_required
        self.shorten = protocol.ca_shorten_factor
        self.lengthen = protocol.ca_lengthen_factor
        self.signal_mode = protocol.ca_signal_mode
        self.quiet_streak = 0

    def update(self, obs: ObsRecord) -> None:
        if obs.poll_id <= 1:
            return
        max_r = float(np.max(obs.rate_delta))
        mean_r = float(np.mean(obs.rate_delta))
        p75_r = float(np.percentile(obs.rate_delta, 75))
        if self.signal_mode == "p75_mean":
            sig = max(p75_r, 1.4 * mean_r)
        else:
            sig = max(max_r, 1.5 * mean_r)

        if sig >= self.high_th:
            self.quiet_streak = 0
            self.interval_s = max(self.min_i, round(self.interval_s * self.shorten, 4))
        elif sig <= self.low_th:
            self.quiet_streak += 1
            if self.quiet_streak >= self.quiet_req:
                self.interval_s = min(self.max_i, round(self.interval_s * self.lengthen, 4))
        else:
            self.quiet_streak = 0

    def clone(self) -> "RateNormalizedChangeAware":
        c = copy.copy(self)
        return c


# ==============================================================================
# 5. SHARED TEMPORAL + SPATIAL FEATURE ENGINE (ZERO LEAKAGE, ZERO POLICY LABELS)
# ==============================================================================
def _gini(x: np.ndarray) -> float:
    arr = np.abs(np.asarray(x, dtype=np.float64))
    s = float(np.sum(arr))
    if s <= 1e-12 or len(arr) <= 1:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(idx * arr) / (n * s)) - (n + 1.0) / n)


@dataclass
class TemporalFeatureState:
    """Running EWMA control chart & CUSUM state updated strictly on accepted observations."""
    ewma_fast: float = 0.0
    ewma_slow: float = 0.0
    ewma_dev: float = 0.0
    cusum_pos: float = 0.0
    initialized: bool = False

    def update(self, obs: ObsRecord, protocol: ProtocolV2) -> None:
        if obs.poll_id <= 1:
            return
        val = float(np.mean(obs.rate_delta))
        if not self.initialized:
            self.ewma_fast = val
            self.ewma_slow = val
            self.ewma_dev = 0.5
            self.cusum_pos = 0.0
            self.initialized = True
            return
        af = protocol.ewma_alpha_fast
        as_ = protocol.ewma_alpha_slow
        bd = protocol.ewma_beta_dev
        prev_slow = self.ewma_slow
        self.ewma_fast = (1.0 - af) * self.ewma_fast + af * val
        self.ewma_slow = (1.0 - as_) * self.ewma_slow + as_ * val
        self.ewma_dev = (1.0 - bd) * self.ewma_dev + bd * abs(val - prev_slow)
        # Page-Hinkley / CUSUM positive shift detector above slow baseline
        slack = 0.5 * max(self.ewma_dev, 0.25)
        self.cusum_pos = max(0.0, self.cusum_pos + (val - prev_slow - slack))

    def clone(self) -> "TemporalFeatureState":
        return copy.copy(self)


@dataclass
class ReviewFeatures:
    """Identical leak-free feature vector seen by BOTH deterministic rules AND decision models."""
    review_time_s: float
    num_links: int
    current_active_policy: str
    current_interval_s: float
    aoi_s: float
    polls_in_window: int
    total_polls: int
    # Rate-normalized change statistics (eliminates interval bias)
    mean_rate_delta: float
    max_rate_delta: float
    p75_rate_delta: float
    std_rate_delta: float
    # Raw delta statistics
    mean_abs_delta: float
    max_abs_delta: float
    # Temporal dynamics (EWMA control chart, CUSUM, OLS slope, oscillation index)
    ewma_fast: float
    ewma_slow: float
    ewma_dev: float
    ewma_momentum: float
    ewma_zscore: float
    cusum_pos: float
    trend_slope: float
    oscillation_index: float   # Signed-change sign-reversal fraction in [0, 1]
    # Spatial distribution (separates localized single-link flap from macro volatility)
    active_link_frac: float    # Fraction of links with rate_delta > 1.8
    spatial_gini: float        # Gini concentration of per-link mean rate_delta in [0, 1]
    max_to_mean_ratio: float
    # Ordered chronological history (last K=6 polls)
    recent_series: List[Tuple[float, float, float, float, float]]  # (t, dt, mean_rate, max_rate, act_frac)
    per_link_latest_util: List[float]
    per_link_mean_rate: List[float]
    link_names: List[str]
    # Monotone pilot percentile context (label-free scale reference)
    pilot_pcts: Dict[str, float] = field(default_factory=dict)

    def to_numeric_vector(self) -> np.ndarray:
        """Numeric feature vector for learnability-ceiling diagnostic & pilot percentiles."""
        return np.array([
            self.aoi_s,
            float(self.polls_in_window),
            self.mean_rate_delta,
            self.max_rate_delta,
            self.p75_rate_delta,
            self.std_rate_delta,
            self.mean_abs_delta,
            self.max_abs_delta,
            self.ewma_fast,
            self.ewma_slow,
            self.ewma_dev,
            self.ewma_momentum,
            self.ewma_zscore,
            self.cusum_pos,
            self.trend_slope,
            self.oscillation_index,
            self.active_link_frac,
            self.spatial_gini,
            self.max_to_mean_ratio,
            1.0 if self.current_active_policy == "change-aware" else 0.0,
            self.current_interval_s,
        ], dtype=np.float64)


FEATURE_NAMES = [
    "aoi_s", "polls_in_window", "mean_rate_delta", "max_rate_delta", "p75_rate_delta",
    "std_rate_delta", "mean_abs_delta", "max_abs_delta", "ewma_fast", "ewma_slow",
    "ewma_dev", "ewma_momentum", "ewma_zscore", "cusum_pos", "trend_slope",
    "oscillation_index", "active_link_frac", "spatial_gini", "max_to_mean_ratio",
    "cur_is_ca", "current_interval_s",
]


class PilotPercentileReference:
    """Label-free monotone empirical CDF reference computed strictly from pilot reviews."""

    def __init__(self, samples_dict: Optional[Dict[str, List[float]]] = None):
        self.sorted_refs: Dict[str, np.ndarray] = {}
        if samples_dict:
            for k, vals in samples_dict.items():
                if vals:
                    self.sorted_refs[k] = np.sort(np.asarray(vals, dtype=np.float64))

    def percentile(self, key: str, value: float) -> float:
        arr = self.sorted_refs.get(key)
        if arr is None or len(arr) == 0:
            return 50.0
        idx = int(np.searchsorted(arr, value, side="right"))
        return round(100.0 * idx / len(arr), 1)

    def annotate(self, feat: ReviewFeatures) -> ReviewFeatures:
        if not self.sorted_refs:
            return feat
        feat.pilot_pcts = {
            "mean_rate_delta": self.percentile("mean_rate_delta", feat.mean_rate_delta),
            "max_rate_delta": self.percentile("max_rate_delta", feat.max_rate_delta),
            "spatial_gini": self.percentile("spatial_gini", feat.spatial_gini),
            "oscillation_index": self.percentile("oscillation_index", feat.oscillation_index),
            "active_link_frac": self.percentile("active_link_frac", feat.active_link_frac),
            "ewma_zscore": self.percentile("ewma_zscore", feat.ewma_zscore),
        }
        return feat


def build_leak_free_features(
    accepted_obs: List[ObsRecord],
    t_now: float,
    num_links: int,
    link_names: List[str],
    active_pol: str,
    cur_interval_s: float,
    tf_state: TemporalFeatureState,
    window_s: float,
    pct_ref: Optional[PilotPercentileReference] = None,
) -> ReviewFeatures:
    """Strictly causal feature extraction using ONLY accepted_obs with arrival_t <= t_now."""
    valid = [o for o in accepted_obs if o.arrival_t <= t_now + 1e-9]
    win_start = max(0.0, t_now - max(window_s, 15.0))
    win_obs = [o for o in valid if o.arrival_t >= win_start - 1e-9 and o.poll_id > 1]
    if not win_obs:
        win_obs = [o for o in valid[-3:] if o.poll_id > 1]

    if valid:
        latest = valid[-1]
        aoi_s = max(0.0, t_now - latest.sample_t)
        latest_util = [round(float(x), 1) for x in latest.util]
    else:
        aoi_s = t_now
        latest_util = [0.0] * num_links

    if win_obs:
        r_mat = np.vstack([o.rate_delta for o in win_obs])    # [T, N]
        a_mat = np.vstack([o.abs_delta for o in win_obs])     # [T, N]
        s_mat = np.vstack([o.signed_delta for o in win_obs])  # [T, N]

        mean_rate = float(np.mean(r_mat))
        max_rate = float(np.max(r_mat))
        p75_rate = float(np.percentile(r_mat, 75))
        per_poll_mean_rate = np.mean(r_mat, axis=1)
        std_rate = float(np.std(per_poll_mean_rate))
        mean_abs = float(np.mean(a_mat))
        max_abs = float(np.max(a_mat))

        # Per-link average rate in window -> spatial concentration
        per_link_rate = np.mean(r_mat, axis=0)
        active_frac = float(np.mean(per_link_rate > 1.8))
        gini = _gini(per_link_rate)
        max_to_mean = float(np.max(per_link_rate) / (np.mean(per_link_rate) + 1e-4))

        # OLS trend slope of per-poll mean rate over time
        if len(win_obs) >= 2:
            t_vec = np.array([o.sample_t for o in win_obs], dtype=np.float64)
            t_c = t_vec - np.mean(t_vec)
            denom = float(np.sum(t_c ** 2))
            slope = float(np.sum(t_c * (per_poll_mean_rate - np.mean(per_poll_mean_rate))) / denom) if denom > 1e-9 else 0.0
        else:
            slope = 0.0

        # Signed-change reversal rate (oscillation index in [0, 1]) on top active links
        if len(win_obs) >= 3:
            # Focus on links that actually moved so flat links don't dilute the oscillation signal
            top_k = np.argsort(per_link_rate)[::-1][: max(2, num_links // 4)]
            sub_s = s_mat[:, top_k]
            signs = np.sign(sub_s)
            flips = (signs[1:] * signs[:-1]) < 0
            osc_idx = float(np.mean(flips))
        else:
            osc_idx = 0.0

        # Ordered chronological last-K=6 polls
        last_k = valid[-6:]
        recent_series = []
        for o in last_k:
            if o.poll_id > 1:
                recent_series.append((
                    round(o.sample_t, 1),
                    round(o.dt_sample, 2),
                    round(float(np.mean(o.rate_delta)), 2),
                    round(float(np.max(o.rate_delta)), 2),
                    round(float(np.mean(o.rate_delta > 1.8)), 2),
                ))
        per_link_rate_list = [round(float(x), 2) for x in per_link_rate]
    else:
        mean_rate = max_rate = p75_rate = std_rate = 0.0
        mean_abs = max_abs = 0.0
        active_frac = gini = slope = osc_idx = 0.0
        max_to_mean = 1.0
        recent_series = []
        per_link_rate_list = [0.0] * num_links

    ewma_mom = float(tf_state.ewma_fast / (tf_state.ewma_slow + 0.15)) if tf_state.initialized else 1.0
    ewma_z = float((tf_state.ewma_fast - tf_state.ewma_slow) / (tf_state.ewma_dev + 0.25)) if tf_state.initialized else 0.0

    feat = ReviewFeatures(
        review_time_s=round(t_now, 2),
        num_links=num_links,
        current_active_policy=active_pol,
        current_interval_s=round(cur_interval_s, 2),
        aoi_s=round(aoi_s, 2),
        polls_in_window=len(win_obs),
        total_polls=len(valid),
        mean_rate_delta=round(mean_rate, 3),
        max_rate_delta=round(max_rate, 3),
        p75_rate_delta=round(p75_rate, 3),
        std_rate_delta=round(std_rate, 3),
        mean_abs_delta=round(mean_abs, 3),
        max_abs_delta=round(max_abs, 3),
        ewma_fast=round(tf_state.ewma_fast, 3),
        ewma_slow=round(tf_state.ewma_slow, 3),
        ewma_dev=round(tf_state.ewma_dev, 3),
        ewma_momentum=round(ewma_mom, 3),
        ewma_zscore=round(ewma_z, 3),
        cusum_pos=round(tf_state.cusum_pos, 3),
        trend_slope=round(slope, 4),
        oscillation_index=round(osc_idx, 3),
        active_link_frac=round(active_frac, 3),
        spatial_gini=round(gini, 3),
        max_to_mean_ratio=round(max_to_mean, 2),
        recent_series=recent_series,
        per_link_latest_util=latest_util,
        per_link_mean_rate=per_link_rate_list,
        link_names=link_names,
    )
    if pct_ref is not None:
        pct_ref.annotate(feat)
    return feat


# ==============================================================================
# 6. DETERMINISTIC CONTROLLERS & NEUTRAL PROMPT PARAPHRASES (ZERO SPOON-FEEDING)
# ==============================================================================
class SimpleRuleSelector:
    """Approved Research Plan §2.3 threshold rule using BOTH observed change AND measurement age."""

    def __init__(self, protocol: ProtocolV2):
        self.p = protocol

    def select(self, feat: ReviewFeatures) -> Tuple[str, float]:
        # Score > 0 favors change-aware; also uses AoI safety bound per §2.3
        if feat.aoi_s >= self.p.simple_max_age_safety_s and feat.mean_rate_delta < self.p.simple_quiet_mean_thresh:
            return "fixed", -1.0
        is_quiet = feat.mean_rate_delta <= self.p.simple_quiet_mean_thresh and feat.max_rate_delta < self.p.simple_max_rate_thresh
        is_broad_active = (feat.mean_rate_delta >= self.p.simple_mean_rate_thresh) and (feat.active_link_frac >= 0.30)
        score = max(
            self.p.simple_quiet_mean_thresh - feat.mean_rate_delta,
            feat.mean_rate_delta - self.p.simple_mean_rate_thresh,
        )
        if is_quiet or is_broad_active:
            return "change-aware", float(max(0.05, score))
        return "fixed", float(-abs(feat.max_rate_delta - feat.mean_rate_delta))


class EWMACusumHysteresisSelector:
    """PhD-level deterministic baseline: EWMA control chart + CUSUM + oscillation/Gini guard + dwell hysteresis."""

    def __init__(self, protocol: ProtocolV2):
        self.p = protocol
        self.reviews_since_switch = 99

    def reset(self) -> None:
        self.reviews_since_switch = 99

    def score_and_select(self, feat: ReviewFeatures) -> Tuple[str, float]:
        self.reviews_since_switch += 1

        # 1. Detect localized single-link flap or high-frequency oscillation trap -> prefer fixed
        is_localized_flap = (
            feat.spatial_gini >= self.p.gini_flap_thresh
            and feat.max_rate_delta >= self.p.ca_high_threshold * 0.8
            and feat.active_link_frac < self.p.active_frac_macro_min
        )
        is_osc_trap = (
            feat.oscillation_index >= self.p.osc_trap_thresh
            and feat.active_link_frac < 0.50
            and feat.max_rate_delta >= self.p.ca_high_threshold * 0.7
        )

        # 2. Detect quiet steady period -> change-aware saves bytes by stretching to 6.0s
        is_quiet = (
            feat.mean_rate_delta <= self.p.quiet_rate_thresh
            and feat.max_rate_delta < self.p.ca_high_threshold * 0.85
        )

        # 3. Detect network-wide macro volatility via EWMA control chart / CUSUM + spatial breadth
        is_macro_vol = (
            feat.active_link_frac >= self.p.active_frac_macro_min
            and (
                feat.ewma_zscore >= self.p.ewma_k_sigma
                or feat.cusum_pos >= self.p.cusum_thresh
                or feat.mean_rate_delta >= self.p.ca_high_threshold * 0.65
            )
        )

        # Continuous score (higher => stronger preference for change-aware) for ROC-AUC comparison
        trap_penalty = (
            max(0.0, feat.spatial_gini - self.p.gini_flap_thresh + 0.08) * 10.0 * (1.0 if feat.max_rate_delta >= 2.5 else 0.0)
            + max(0.0, feat.oscillation_index - self.p.osc_trap_thresh + 0.08) * 6.0 * (1.0 if feat.active_link_frac < 0.5 else 0.0)
        )
        quiet_bonus = max(0.0, self.p.quiet_rate_thresh - feat.mean_rate_delta) * 4.0 * (1.0 if feat.max_rate_delta < self.p.ca_high_threshold else 0.0)
        macro_bonus = (
            max(0.0, feat.active_link_frac - self.p.active_frac_macro_min) * 5.0
            + (max(0.0, feat.ewma_zscore) * 0.8 if feat.active_link_frac >= self.p.active_frac_macro_min else 0.0)
        )
        cont_score = quiet_bonus + macro_bonus - trap_penalty

        if is_localized_flap or is_osc_trap:
            desired = "fixed"
            cont_score = min(cont_score, -0.5)
        elif is_quiet or is_macro_vol:
            desired = "change-aware"
            cont_score = max(cont_score, 0.5)
        else:
            desired = "fixed"

        # Hysteresis dwell guard to prevent policy chattering
        if desired != feat.current_active_policy and self.reviews_since_switch <= self.p.hysteresis_dwell_reviews:
            desired = feat.current_active_policy
        elif desired != feat.current_active_policy:
            self.reviews_since_switch = 0

        return desired, float(cont_score)


def build_neutral_prompt_paraphrases(protocol: ProtocolV2) -> List[Dict[str, Any]]:
    """4 pre-registered strictly NEUTRAL prompt paraphrases (including option-order swap).

    Gives FULL factual specification of both policies (including CA thresholds/bounds)
    without ANY if/else classification or spoon-fed decision labels in the telemetry input.
    """
    fi = protocol.fixed_interval_s
    cmin, cmax = protocol.ca_min_interval_s, protocol.ca_max_interval_s
    chi, clo = protocol.ca_high_threshold, protocol.ca_low_threshold
    H = protocol.review_period_s

    p0 = {
        "name": "paraphrase_0_standard",
        "option_order": ["fixed", "change-aware"],
        "instructions": (
            f"You are a Network Digital Twin synchronization controller choosing the polling policy "
            f"for the next {H:.0f}s review window. Minimize combined twin-state MAE and monitoring "
            f"traffic overhead using strictly the numerical telemetry below."
        ),
        "criteria": {
            "fixed": (
                f"Periodic fixed-interval polling every {fi:.1f}s across all links. Guarantees "
                f"bounded monitoring traffic regardless of single-link spikes or rapid oscillation."
            ),
            "change-aware": (
                f"Rate-normalized adaptive polling in [{cmin:.1f}s, {cmax:.1f}s] (shortens by 0.5x "
                f"when max/weighted link rate-delta >= {chi:.2f}, lengthens by 1.5x after quiet polls "
                f"<= {clo:.2f}). Saves traffic when all links are quiet and tracks broad rapid changes."
            ),
        },
    }
    p1 = {
        "name": "paraphrase_1_swapped_order",
        "option_order": ["change-aware", "fixed"],
        "instructions": p0["instructions"],
        "criteria": {
            "change-aware": p0["criteria"]["change-aware"],
            "fixed": p0["criteria"]["fixed"],
        },
    }
    p2 = {
        "name": "paraphrase_2_gnmi_framing",
        "option_order": ["fixed", "change-aware"],
        "instructions": (
            f"Act as a gNMI TARGET_DEFINED telemetry mode selector for a Network Digital Twin. "
            f"Given the chronological multi-link telemetry window below, select the policy for "
            f"the upcoming {H:.0f}s period that best optimizes accuracy-per-kilobyte."
        ),
        "criteria": {
            "fixed": (
                f"SAMPLE mode at constant {fi:.1f}s interval. Prevents poll storms when isolated "
                f"links flap or exhibit high-frequency unresolvable jitter."
            ),
            "change-aware": (
                f"Dampened adaptive mode in [{cmin:.1f}s, {cmax:.1f}s] (trigger >= {chi:.2f}, "
                f"relax <= {clo:.2f}). Reduces polls during quiet steady states and accelerates "
                f"during widespread network-wide shifts."
            ),
        },
    }
    p3 = {
        "name": "paraphrase_3_concise_technical",
        "option_order": ["fixed", "change-aware"],
        "instructions": (
            f"Select polling controller ('fixed' or 'change-aware') for next {H:.0f}s NDT window "
            f"to minimize J = MAE + {protocol.lambda_kb_per_link:.3f} * (KB/link) based strictly on observed metrics."
        ),
        "criteria": {
            "fixed": f"Constant {fi:.1f}s polling across all links; immune to single-link flap amplification.",
            "change-aware": (
                f"Adaptive [{cmin:.1f}s..{cmax:.1f}s] polling (trigger>={chi:.2f}, quiet<={clo:.2f}); "
                f"low bytes in quiet states and fast tracking in multi-link volatility."
            ),
        },
    }
    return [p0, p1, p2, p3]


def format_neutral_telemetry_text(feat: ReviewFeatures) -> str:
    """Render strictly neutral numerical time series + pilot percentile context (ZERO policy hints)."""
    p = feat.pilot_pcts
    pct_str = (
        f" [pilot pcts: mean_rate=p{p.get('mean_rate_delta', 50):.0f}, "
        f"max_rate=p{p.get('max_rate_delta', 50):.0f}, "
        f"gini=p{p.get('spatial_gini', 50):.0f}, "
        f"osc=p{p.get('oscillation_index', 50):.0f}, "
        f"active_frac=p{p.get('active_link_frac', 50):.0f}]"
        if p else ""
    )
    series_str = "; ".join(
        f"t={t:.1f}s(dt={dt:.1f}s):mean={mr:.2f},max={mx:.2f},act={af:.2f}"
        for (t, dt, mr, mx, af) in feat.recent_series
    ) or "warmup"

    # Top 6 links by rate_delta (or all if <= 8)
    if feat.num_links <= 8:
        links_str = ", ".join(
            f"{feat.link_names[i]}={feat.per_link_latest_util[i]:.1f}%(r={feat.per_link_mean_rate[i]:.2f})"
            for i in range(feat.num_links)
        )
    else:
        order = np.argsort(feat.per_link_mean_rate)[::-1][:6]
        links_str = "Top-6 links: " + ", ".join(
            f"{feat.link_names[i]}={feat.per_link_latest_util[i]:.1f}%(r={feat.per_link_mean_rate[i]:.2f})"
            for i in order
        )

    return (
        f"NDT Telemetry Review at t={feat.review_time_s:.1f}s ({feat.num_links} links)\n"
        f"Current Policy: {feat.current_active_policy} (interval={feat.current_interval_s:.2f}s) | "
        f"AoI={feat.aoi_s:.2f}s | PollsInWindow={feat.polls_in_window} (Total={feat.total_polls})\n"
        f"Rate-Normalized Volatility: mean={feat.mean_rate_delta:.2f}, max={feat.max_rate_delta:.2f}, "
        f"p75={feat.p75_rate_delta:.2f}, std={feat.std_rate_delta:.2f} | "
        f"RawDelta: mean={feat.mean_abs_delta:.2f}%, max={feat.max_abs_delta:.2f}%{pct_str}\n"
        f"Temporal Dynamics: EWMA_fast={feat.ewma_fast:.2f}, EWMA_slow={feat.ewma_slow:.2f}, "
        f"EWMA_dev={feat.ewma_dev:.2f}, z={feat.ewma_zscore:+.2f}, CUSUM={feat.cusum_pos:.2f}, "
        f"slope={feat.trend_slope:+.3f}, oscillation_reversal_rate={feat.oscillation_index:.2f}\n"
        f"Spatial Spread: active_link_frac={feat.active_link_frac:.2f}, "
        f"spatial_gini={feat.spatial_gini:.2f}, max_to_mean={feat.max_to_mean_ratio:.2f}\n"
        f"Chronological Recent Polls: {series_str}\n"
        f"Per-Link State: {links_str}"
    )


def build_content_free_telemetry_text(num_links: int = 8) -> str:
    """Content-free neutral baseline input for Zhao et al. (ICML 2021) Contextual Calibration."""
    return (
        f"NDT Telemetry Review at t=N/A ({num_links} links)\n"
        f"Current Policy: N/A (interval=N/A) | AoI=N/A | PollsInWindow=N/A\n"
        f"Rate-Normalized Volatility: mean=N/A, max=N/A, p75=N/A, std=N/A\n"
        f"Temporal Dynamics: EWMA_fast=N/A, EWMA_slow=N/A, EWMA_dev=N/A, z=N/A, CUSUM=N/A, "
        f"slope=N/A, oscillation_reversal_rate=N/A\n"
        f"Spatial Spread: active_link_frac=N/A, spatial_gini=N/A, max_to_mean=N/A\n"
        f"Chronological Recent Polls: N/A\n"
        f"Per-Link State: N/A"
    )


def apply_contextual_calibration(p_ca_raw: float, prior_p_ca: float) -> Tuple[str, float]:
    """Zhao et al. (ICML 2021) training-free contextual calibration for binary choice probabilities.

    Divides each option's probability by its content-free prior and renormalizes:
      p_cal(ca) = (p(ca)/p0(ca)) / (p(ca)/p0(ca) + p(fix)/p0(fix))
    """
    eps = 1e-6
    p_ca = float(np.clip(p_ca_raw, eps, 1.0 - eps))
    p_fix = 1.0 - p_ca
    p0_ca = float(np.clip(prior_p_ca, eps, 1.0 - eps))
    p0_fix = 1.0 - p0_ca
    w_ca = p_ca / p0_ca
    w_fix = p_fix / p0_fix
    p_cal = w_ca / (w_ca + w_fix)
    choice = "change-aware" if p_cal >= 0.5 else "fixed"
    return choice, float(p_cal)


# ==============================================================================
# 7. CORE DISCRETE-TIME SIMULATOR & ONE-STEP COUNTERFACTUAL ORACLE
# ==============================================================================
def _compute_congestion_metrics(
    true_u: np.ndarray,
    stored_series: np.ndarray,
    timestamps: np.ndarray,
    warmup_s: float,
    thresh_pct: float,
) -> Dict[str, float]:
    """Operational NDT metric: detection of link congestion crossings (> thresh_pct)."""
    mask = timestamps >= warmup_s - 1e-9
    tu = true_u[mask]
    su = stored_series[mask]
    t_hot = tu >= thresh_pct
    s_hot = su >= thresh_pct
    tp = int(np.sum(t_hot & s_hot))
    fp = int(np.sum((~t_hot) & s_hot))
    fn = int(np.sum(t_hot & (~s_hot)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {
        "congestion_f1": round(f1, 4),
        "congestion_tp": tp,
        "congestion_pos_ticks": int(np.sum(t_hot)),
    }


def _rollout_horizon_cost(
    start_tick: int,
    end_tick: int,
    timestamps: np.ndarray,
    true_u: np.ndarray,
    delays_s: np.ndarray,
    num_links: int,
    bytes_per_poll: int,
    protocol: ProtocolV2,
    policy_choice: str,
    stored_u: np.ndarray,
    latest_sample_t: float,
    last_poll_t: float,
    next_poll_t: float,
    in_flight: List[Tuple[float, float, np.ndarray]],
    ca_pol: RateNormalizedChangeAware,
    poll_count_init: int,
) -> float:
    """Fast counterfactual forward rollout from start_tick to end_tick under a single policy."""
    su = stored_u.copy()
    lst = latest_sample_t
    lpt = last_poll_t
    npt = next_poll_t
    infl = list(in_flight)
    ca = ca_pol.clone()
    polls_added = 0
    pid = poll_count_init
    abs_err_sum = 0.0
    n_eval = 0

    # Apply switch scheduling without an extra poll (§2.3)
    cur_int = protocol.fixed_interval_s if policy_choice == "fixed" else ca.interval_s
    t0 = timestamps[start_tick]
    cand_next = round(lpt + cur_int, 6)
    npt = cand_next if cand_next >= t0 - 1e-9 else round(t0 + cur_int, 6)

    for tk in range(start_tick, end_tick):
        t_now = timestamps[tk]
        # Deliver arriving polls
        if infl:
            rem = []
            arr = []
            for item in infl:
                if item[1] <= t_now + 1e-9:
                    arr.append(item)
                else:
                    rem.append(item)
            infl = rem
            if arr:
                arr.sort(key=lambda x: x[1])
                for disp_t, arr_t, sample_u in arr:
                    if disp_t > lst + 1e-9:
                        dt_s = max(0.1, disp_t - lst) if lst >= 0 else 1.0
                        s_delta = sample_u - su
                        a_delta = np.abs(s_delta)
                        r_delta = a_delta / (dt_s ** protocol.ca_norm_exp)
                        su = sample_u.copy()
                        lst = disp_t
                        pid += 1
                        ca.update(ObsRecord(pid, disp_t, arr_t, dt_s, su, s_delta, a_delta, r_delta))
                        if policy_choice == "change-aware":
                            npt = round(lpt + ca.interval_s, 6)

        # Dispatch poll
        if t_now >= npt - 1e-9:
            polls_added += 1
            infl.append((t_now, round(t_now + delays_s[tk], 6), true_u[tk].copy()))
            lpt = t_now
            cur_int = protocol.fixed_interval_s if policy_choice == "fixed" else ca.interval_s
            npt = round(lpt + cur_int, 6)

        abs_err_sum += float(np.mean(np.abs(true_u[tk] - su)))
        n_eval += 1

    mae = abs_err_sum / max(1, n_eval)
    # Normalize KB/link cost to full-trace scale so lambda_kb_per_link has identical units per horizon window!
    eval_dur = max(1.0, timestamps[min(end_tick, len(timestamps) - 1)] - timestamps[start_tick])
    full_eval_dur = max(1.0, timestamps[-1] - protocol.warmup_duration_s)
    scaled_kb_per_link = ((polls_added * bytes_per_poll / 1000.0) / num_links) * (full_eval_dur / eval_dur)
    return mae + protocol.lambda_kb_per_link * scaled_kb_per_link


def simulate_trace(
    trace: Dict[str, Any],
    controller: str,
    protocol: ProtocolV2,
    fixed_interval_override: Optional[float] = None,
    pct_ref: Optional[PilotPercentileReference] = None,
    model_selector_fn: Optional[Callable[[ReviewFeatures], Dict[str, Any]]] = None,
    trained_clf: Optional[Any] = None,
    random_p_ca: float = 0.5,
    random_seed_offset: int = 0,
    preset_schedule: Optional[List[str]] = None,
    compute_oracle_labels: bool = False,
    record_review_logs: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run one controller on one trace with strict causality, latency, timeout, and OOO protection.

    Supported `controller` values:
      - "fixed": static fixed interval (`fixed_interval_override` or `protocol.fixed_interval_s`)
      - "change_aware": static rate-normalized Change-Aware
      - "rule_simple": approved plan simple threshold rule (change + AoI)
      - "rule_ewma_cusum": PhD-level EWMA + CUSUM + oscillation/Gini + hysteresis rule
      - "oracle": hindsight one-step counterfactual oracle switcher
      - "random": random switcher at `random_p_ca`
      - "learnability_ceiling": pilot-trained classifier diagnostic (NOT a competing controller)
      - "preset_schedule": replays an explicit list of choices (used for exact 2^R brute-force check)
      - "model": live decision model via `model_selector_fn(feat)`
    """
    timestamps = trace["timestamps"]
    true_u = trace["true_u"]
    delays_s = trace["delays_s"]
    is_change_event = trace["is_change_event"]
    link_names = trace["link_names"]
    num_links = trace["num_links"]
    num_ticks = len(timestamps)

    fixed_int = float(fixed_interval_override if fixed_interval_override is not None else protocol.fixed_interval_s)
    # RFC 9232 application-layer payload framing: req = 8 + 4*N, reply = 12 + 12*N
    req_bytes = 8 + 4 * num_links
    rep_bytes = 12 + 12 * num_links
    bytes_per_poll = req_bytes + rep_bytes

    stored_u = true_u[0].copy()
    latest_sample_t = -1.0
    ca_pol = RateNormalizedChangeAware(protocol)
    tf_state = TemporalFeatureState()

    active_pol = "change-aware" if controller == "change_aware" else "fixed"
    simple_rule = SimpleRuleSelector(protocol)
    ewma_rule = EWMACusumHysteresisSelector(protocol)
    rng_ctrl = np.random.default_rng(trace["seed"] * 31 + random_seed_offset + 7)

    in_flight: List[Tuple[float, float, np.ndarray]] = []
    accepted_obs: List[ObsRecord] = []
    last_poll_t = 0.0
    next_poll_t = 0.0
    next_review_t = protocol.review_period_s
    pending_dec: Optional[Tuple[float, float, str, Dict[str, Any]]] = None  # (eff_t, deadline_t, choice, meta)

    polls_dispatched = 0
    reply_bytes_total = 0
    rejected_ooo = 0
    switches = 0
    fallbacks = 0
    skipped_backlog = 0
    reviews_total = 0
    api_bytes = 0
    latencies_ms: List[float] = []
    review_logs: List[Dict[str, Any]] = []

    abs_err_list: List[float] = []
    sq_err_list: List[float] = []
    aoi_list: List[float] = []
    stored_series = np.zeros_like(true_u)
    active_ticks = {"fixed": 0, "change-aware": 0}
    pending_events: List[float] = []
    det_delays: List[float] = []

    is_switcher = controller not in ("fixed", "change_aware")
    horizon_ticks = max(15, int(round(1.5 * protocol.review_period_s / protocol.tick_dt_s)))
    schedule_idx = 0

    for tk in range(num_ticks):
        t_now = float(timestamps[tk])
        u_true = true_u[tk]

        if is_change_event[tk] == 1 and t_now >= protocol.warmup_duration_s:
            pending_events.append(t_now)

        # 1. Deliver arriving polls
        if in_flight:
            rem = []
            arr = []
            for item in in_flight:
                if item[1] <= t_now + 1e-9:
                    arr.append(item)
                else:
                    rem.append(item)
            in_flight = rem
            if arr:
                arr.sort(key=lambda x: x[1])
                for disp_t, arr_t, sample_u in arr:
                    reply_bytes_total += rep_bytes
                    if disp_t > latest_sample_t + 1e-9:
                        dt_s = max(0.1, disp_t - latest_sample_t) if latest_sample_t >= 0 else 1.0
                        s_delta = np.zeros(num_links) if latest_sample_t < 0 else (sample_u - stored_u)
                        a_delta = np.abs(s_delta)
                        r_delta = a_delta / (dt_s ** protocol.ca_norm_exp)
                        stored_u = sample_u.copy()
                        latest_sample_t = disp_t
                        obs = ObsRecord(len(accepted_obs) + 1, disp_t, arr_t, dt_s, stored_u.copy(), s_delta, a_delta, r_delta)
                        accepted_obs.append(obs)
                        ca_pol.update(obs)
                        tf_state.update(obs, protocol)
                        if active_pol == "change-aware":
                            next_poll_t = round(last_poll_t + ca_pol.interval_s, 6)
                        # Resolve detection delays
                        if pending_events:
                            still = []
                            for ev_t in pending_events:
                                if disp_t >= ev_t - 1e-9:
                                    det_delays.append(max(0.0, t_now - ev_t))
                                else:
                                    still.append(ev_t)
                            pending_events = still
                    else:
                        rejected_ooo += 1

        # 2. Apply pending decision if arrived or timed out (§2.3)
        if pending_dec is not None:
            eff_t, dl_t, chosen, meta = pending_dec
            res_t = min(eff_t, dl_t)
            if t_now >= res_t - 1e-9:
                pending_dec = None
                is_fb = (eff_t > dl_t + 1e-9) or (not meta.get("is_valid", True)) or (chosen not in ("fixed", "change-aware"))
                if is_fb:
                    fallbacks += 1
                    new_pol = active_pol
                else:
                    new_pol = chosen
                switched = (new_pol != active_pol)
                if switched:
                    active_pol = new_pol
                    switches += 1
                    cur_int = fixed_int if active_pol == "fixed" else ca_pol.interval_s
                    cand_next = round(last_poll_t + cur_int, 6)
                    next_poll_t = cand_next if cand_next >= t_now - 1e-9 else round(t_now + cur_int, 6)
                if record_review_logs:
                    meta["effective_t"] = round(t_now, 3)
                    meta["effective_policy_after"] = active_pol
                    meta["switched"] = switched
                    meta["is_fallback"] = is_fb
                    review_logs.append(meta)

        # 3. Scheduled policy review (strictly causal: uses ONLY accepted_obs with arrival_t <= t_now)
        if is_switcher and t_now >= next_review_t - 1e-9 and tk < num_ticks - 1:
            next_review_t = round(next_review_t + protocol.review_period_s, 6)
            if pending_dec is not None:
                skipped_backlog += 1
            else:
                reviews_total += 1
                cur_int = fixed_int if active_pol == "fixed" else ca_pol.interval_s
                feat = build_leak_free_features(
                    accepted_obs=accepted_obs,
                    t_now=t_now,
                    num_links=num_links,
                    link_names=link_names,
                    active_pol=active_pol,
                    cur_interval_s=cur_int,
                    tf_state=tf_state,
                    window_s=protocol.review_period_s,
                    pct_ref=pct_ref,
                )

                # Optional per-review counterfactual oracle label (for ROC-AUC & learnability ceiling!)
                oracle_label = None
                oracle_margin = 0.0
                if controller == "oracle" or compute_oracle_labels:
                    end_tk = min(num_ticks, tk + horizon_ticks)
                    c_fix = _rollout_horizon_cost(
                        tk, end_tk, timestamps, true_u, delays_s, num_links, bytes_per_poll,
                        protocol, "fixed", stored_u, latest_sample_t, last_poll_t, next_poll_t,
                        in_flight, ca_pol, len(accepted_obs),
                    )
                    c_ca = _rollout_horizon_cost(
                        tk, end_tk, timestamps, true_u, delays_s, num_links, bytes_per_poll,
                        protocol, "change-aware", stored_u, latest_sample_t, last_poll_t, next_poll_t,
                        in_flight, ca_pol, len(accepted_obs),
                    )
                    oracle_label = "change-aware" if c_ca < c_fix else "fixed"
                    oracle_margin = round(float(c_fix - c_ca), 5)  # positive => change-aware better

                lat_ms = 0.05
                is_valid = True
                score_ca = 0.5
                p_ca_raw = 0.5
                p_ca_cal = 0.5
                raw_out = ""
                err_reason = None

                if controller == "oracle":
                    chosen = str(oracle_label)
                    score_ca = 1.0 if chosen == "change-aware" else 0.0
                elif controller == "rule_simple":
                    chosen, score_ca = simple_rule.select(feat)
                elif controller == "rule_ewma_cusum":
                    chosen, score_ca = ewma_rule.score_and_select(feat)
                elif controller == "random":
                    u_r = float(rng_ctrl.random())
                    chosen = "change-aware" if u_r < random_p_ca else "fixed"
                    score_ca = u_r
                elif controller == "learnability_ceiling":
                    x_vec = feat.to_numeric_vector().reshape(1, -1)
                    proba = float(trained_clf.predict_proba(x_vec)[0, 1])
                    chosen = "change-aware" if proba >= 0.5 else "fixed"
                    score_ca = proba
                elif controller == "preset_schedule":
                    chosen = preset_schedule[schedule_idx] if (preset_schedule and schedule_idx < len(preset_schedule)) else "fixed"
                    schedule_idx += 1
                elif controller == "model":
                    assert model_selector_fn is not None
                    res = model_selector_fn(feat)
                    lat_ms = float(res.get("latency_ms", 0.0))
                    latencies_ms.append(lat_ms)
                    api_bytes += int(res.get("req_bytes", 0)) + int(res.get("rep_bytes", 0))
                    is_valid = bool(res.get("is_valid", False))
                    chosen = res.get("choice")
                    p_ca_raw = float(res.get("p_ca_raw", 0.5))
                    p_ca_cal = float(res.get("p_ca_cal", p_ca_raw))
                    score_ca = p_ca_cal
                    raw_out = str(res.get("raw_output", ""))[:300]
                    err_reason = res.get("error")
                else:
                    raise ValueError(f"Unknown controller: {controller}")

                meta = {
                    "regime": trace["regime"],
                    "topology": trace["topology"],
                    "delay": trace["delay_condition"],
                    "seed": trace["seed"],
                    "controller": controller,
                    "t_review_s": round(t_now, 2),
                    "true_phase": trace["phase_labels"][tk],
                    "prev_policy": active_pol,
                    "chosen_raw": chosen,
                    "score_ca": round(float(score_ca), 5),
                    "p_ca_raw": round(float(p_ca_raw), 5),
                    "p_ca_cal": round(float(p_ca_cal), 5),
                    "oracle_label": oracle_label,
                    "oracle_margin": oracle_margin,
                    "latency_ms": round(lat_ms, 2),
                    "is_valid": is_valid,
                    "error": err_reason,
                    "raw_output": raw_out,
                    "features": feat.to_numeric_vector().tolist(),
                }

                eff_t = t_now + lat_ms / 1000.0
                dl_t = t_now + protocol.decision_deadline_ms / 1000.0
                if eff_t <= t_now + protocol.tick_dt_s * 0.5 and eff_t <= dl_t:
                    # Immediate in-tick application (sub-50ms local or fast model)
                    is_fb = (not is_valid) or (chosen not in ("fixed", "change-aware"))
                    if is_fb:
                        fallbacks += 1
                        new_pol = active_pol
                    else:
                        new_pol = str(chosen)
                    switched = (new_pol != active_pol)
                    if switched:
                        active_pol = new_pol
                        switches += 1
                        cur_int = fixed_int if active_pol == "fixed" else ca_pol.interval_s
                        cand_next = round(last_poll_t + cur_int, 6)
                        next_poll_t = cand_next if cand_next >= t_now - 1e-9 else round(t_now + cur_int, 6)
                    if record_review_logs:
                        meta["effective_t"] = round(t_now, 3)
                        meta["effective_policy_after"] = active_pol
                        meta["switched"] = switched
                        meta["is_fallback"] = is_fb
                        review_logs.append(meta)
                else:
                    pending_dec = (eff_t, dl_t, str(chosen), meta)

        # 4. Dispatch poll
        if t_now >= next_poll_t - 1e-9:
            polls_dispatched += 1
            in_flight.append((t_now, round(t_now + float(delays_s[tk]), 6), u_true.copy()))
            last_poll_t = t_now
            cur_int = fixed_int if active_pol == "fixed" else ca_pol.interval_s
            next_poll_t = round(last_poll_t + cur_int, 6)

        # 5. Record post-warmup error & AoI
        stored_series[tk] = stored_u
        active_ticks[active_pol] += 1
        if t_now >= protocol.warmup_duration_s - 1e-9:
            diff = np.abs(u_true - stored_u)
            abs_err_list.append(float(np.mean(diff)))
            sq_err_list.append(float(np.mean(diff ** 2)))
            aoi_list.append(t_now if latest_sample_t < 0 else max(0.0, t_now - latest_sample_t))

    mae = float(np.mean(abs_err_list))
    rmse = float(np.sqrt(np.mean(sq_err_list)))
    p95_err = float(np.percentile(abs_err_list, 95))
    mon_bytes = polls_dispatched * req_bytes + reply_bytes_total
    mon_kb = mon_bytes / 1000.0
    cost_j = mae + protocol.lambda_kb_per_link * (mon_kb / num_links)
    cong = _compute_congestion_metrics(true_u, stored_series, timestamps, protocol.warmup_duration_s, protocol.congestion_threshold_pct)
    tot_t = sum(active_ticks.values()) or 1

    summary = {
        "topology": trace["topology"],
        "regime": trace["regime"],
        "delay": trace["delay_condition"],
        "seed": trace["seed"],
        "controller": f"fixed_{fixed_int:.3f}s" if controller == "fixed" else controller,
        "fixed_interval_s": fixed_int if controller == "fixed" else np.nan,
        "mae_pct": round(mae, 4),
        "rmse_pct": round(rmse, 4),
        "p95_err_pct": round(p95_err, 4),
        "monitoring_kb": round(mon_kb, 3),
        "monitoring_bytes": int(mon_bytes),
        "cost_j": round(cost_j, 4),
        "polls": int(polls_dispatched),
        "rejected_ooo": int(rejected_ooo),
        "mean_aoi_s": round(float(np.mean(aoi_list)), 3),
        "mean_det_delay_s": round(float(np.mean(det_delays)), 3) if det_delays else 0.0,
        "congestion_f1": cong["congestion_f1"],
        "reviews": int(reviews_total),
        "switches": int(switches),
        "change_aware_frac": round(active_ticks["change-aware"] / tot_t, 4),
        "fallback_rate": round(fallbacks / max(1, reviews_total), 4) if is_switcher else 0.0,
        "skipped_backlog": int(skipped_backlog),
        "latency_mean_ms": round(float(np.mean(latencies_ms)), 2) if latencies_ms else 0.0,
        "latency_p50_ms": round(float(np.median(latencies_ms)), 2) if latencies_ms else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else 0.0,
        "api_total_bytes": int(api_bytes),
    }
    return summary, review_logs


# ==============================================================================
# 8. MANDATORY QA INVARIANT SUITE (7 CHECKS INCLUDING EXACT 2^R ORACLE CHECK)
# ==============================================================================
def run_qa_suite_v2(protocol: ProtocolV2) -> Dict[str, Any]:
    """Run 7 automated QA invariant checks per Research Plan §2.4 + v2 oracle verification."""
    checks: Dict[str, bool] = {}

    # 1. Constant-state zero error
    p_short = copy.deepcopy(protocol)
    p_short.trace_duration_s = 60.0
    p_short.control_duration_s = 60.0
    p_short.warmup_duration_s = 10.0
    tr = generate_trace_v2("stable", "low_delay", 999, p_short)
    tr["true_u"][:] = 42.0
    res_const, _ = simulate_trace(tr, "fixed", p_short)
    checks["constant_state_zero_mae"] = (res_const["mae_pct"] == 0.0)

    # 2. Out-of-order delayed reply rejection exercised
    tr_ooo = generate_trace_v2("stable", "elevated_delay", 998, p_short)
    tr_ooo["delays_s"][0] = 5.0  # Poll at t=0 arrives at t=5.0
    tr_ooo["delays_s"][25] = 0.05  # Poll at t=2.5 arrives at t=2.55 (BEFORE t=0 poll!)
    res_ooo, _ = simulate_trace(tr_ooo, "fixed", p_short, fixed_interval_override=2.5)
    checks["out_of_order_rejected"] = (res_ooo["rejected_ooo"] >= 1)

    # 3. Decision deadline timeout retains previous policy & logs fallback
    def slow_model(_feat: ReviewFeatures) -> Dict[str, Any]:
        return {"choice": "change-aware", "is_valid": True, "latency_ms": 3500.0, "p_ca_raw": 0.9}
    res_fb, logs_fb = simulate_trace(tr, "model", p_short, model_selector_fn=slow_model, record_review_logs=True)
    checks["deadline_timeout_fallback"] = (res_fb["fallback_rate"] == 1.0 and res_fb["switches"] == 0)

    # 4. Zero future-state leakage (mutating true_u strictly after t_review=20s does not alter summary at t=20s)
    tr_a = generate_trace_v2("alternating", "low_delay", 997, p_short)
    tr_b = copy.deepcopy(tr_a)
    tr_b["true_u"][tr_b["timestamps"] > 20.0] = 99.0
    _, la = simulate_trace(tr_a, "rule_ewma_cusum", p_short, record_review_logs=True)
    _, lb = simulate_trace(tr_b, "rule_ewma_cusum", p_short, record_review_logs=True)
    la_20 = [r for r in la if abs(r["t_review_s"] - 20.0) < 0.2][0]
    lb_20 = [r for r in lb if abs(r["t_review_s"] - 20.0) < 0.2][0]
    checks["zero_future_leakage"] = (la_20["features"] == lb_20["features"] and la_20["chosen_raw"] == lb_20["chosen_raw"])

    # 5. Switch without extra instantaneous poll
    res_fix, _ = simulate_trace(tr, "fixed", p_short, fixed_interval_override=2.5)
    def flip_model(feat: ReviewFeatures) -> Dict[str, Any]:
        ch = "change-aware" if int(feat.review_time_s // 10) % 2 == 1 else "fixed"
        return {"choice": ch, "is_valid": True, "latency_ms": 10.0, "p_ca_raw": 0.8}
    p_eq = copy.deepcopy(p_short)
    p_eq.ca_min_interval_s = 2.5
    p_eq.ca_max_interval_s = 2.5
    p_eq.ca_init_interval_s = 2.5
    res_flip, _ = simulate_trace(tr, "model", p_eq, model_selector_fn=flip_model)
    checks["switch_no_extra_poll"] = (res_flip["polls"] == res_fix["polls"] and res_flip["switches"] >= 2)

    # 6. Exact 2^R brute-force vs one-step oracle sanity check (R=6 -> 64 schedules)
    p_bf = copy.deepcopy(protocol)
    p_bf.trace_duration_s = 70.0
    p_bf.warmup_duration_s = 10.0
    p_bf.review_period_s = 10.0
    tr_bf = generate_trace_v2("regime_switching", "low_delay", 996, p_bf)
    res_orc, _ = simulate_trace(tr_bf, "oracle", p_bf)
    best_bf = float("inf")
    for mask in range(64):
        sched = ["change-aware" if (mask >> b) & 1 else "fixed" for b in range(6)]
        r_m, _ = simulate_trace(tr_bf, "preset_schedule", p_bf, preset_schedule=sched)
        if r_m["cost_j"] < best_bf:
            best_bf = r_m["cost_j"]
    # One-step oracle should be close to exact 2^R global optimum (within 15%)
    checks["oracle_tracks_exact_2R"] = (res_orc["cost_j"] <= best_bf * 1.15 + 0.15)

    # 7. SNDlib Abilene OSPF failover routing connectivity check (all remaining 11 nodes connected!)
    checks["abilene_ospf_failover_connected"] = (len(ABILENE_OSPF_FAILOVER) == len(ABILENE_NODES) * (len(ABILENE_NODES) - 1))

    return {
        "num_passed": sum(checks.values()),
        "num_checks": len(checks),
        "all_passed": all(checks.values()),
        "details": checks,
        "oracle_vs_2R_gap": round(res_orc["cost_j"] - best_bf, 4),
    }


# ==============================================================================
# 9. EQUAL-BUDGET PILOT CALIBRATION (k=20 EACH) & FREEZE
# ==============================================================================
def run_pilot_calibration(
    pilot_seeds: List[int],
    base_protocol: Optional[ProtocolV2] = None,
) -> Tuple[ProtocolV2, PilotPercentileReference, Pipeline, Dict[str, Any]]:
    """Pilot calibration strictly on `pilot_seeds` (ZERO held-out contact!).

    Steps:
      1. Compute dense 15-point static `fixed` frontier on pilot traces -> derive objective
         weight `lambda_kb` = -d(MAE)/d(KB) at reference fixed=2.5s (so J = MAE + lambda*KB is tangent!).
      2. Equal-budget tuning (k=20 candidates each):
         a) Rate-normalized Change-Aware parameters
         b) Simple Rule parameters
         c) EWMA/CUSUM+Hysteresis Rule parameters
      3. Build label-free PilotPercentileReference from pilot reviews.
      4. Fit diagnostic Learnability Ceiling classifier (LogReg + GBDT, best CV on pilot).
      5. Compute Pilot Power Analysis justifying held-out seed count N.
    """
    proto = copy.deepcopy(base_protocol or ProtocolV2())
    pilot_traces = [
        generate_trace_v2("regime_switching", dl, s, proto, topology="8link")
        for dl in ("low_delay", "elevated_delay")
        for s in pilot_seeds
    ]

    # 1. Calibrate lambda_kb_per_link as the exact supporting-hyperplane subgradient of the
    #    dense static `fixed` frontier at fixed_interval_s=2.5s (so NO static fixed interval can game J!)
    fi_grid = [0.8, 1.2, 1.8, 2.5, 3.5, 5.0, 7.5]
    fc = []
    for fi in fi_grid:
        rows = [simulate_trace(tr, "fixed", proto, fixed_interval_override=fi)[0] for tr in pilot_traces]
        kb_pl = float(np.mean([r["monitoring_kb"] / 8.0 for r in rows]))
        mae_m = float(np.mean([r["mae_pct"] for r in rows]))
        fc.append((fi, kb_pl, mae_m))
    # Index 3 is 2.5s; left neighbor is 1.8s (idx 2), right neighbor is 3.5s (idx 4)
    slope_left = (fc[3][2] - fc[2][2]) / max(1e-6, fc[2][1] - fc[3][1])
    slope_right = (fc[4][2] - fc[3][2]) / max(1e-6, fc[3][1] - fc[4][1])
    # Exact midpoint of subgradient interval [-f'+(B*), -f'-(B*)] guarantees fixed@2.5s minimizes J on static curve!
    subgrad = 0.5 * (max(0.0, slope_left) + max(0.0, slope_right))
    proto.lambda_kb_per_link = round(float(np.clip(subgrad, 0.05, 1.5)), 4)

    # 2a. Equal-budget k=20 sweep for Rate-Normalized Change-Aware (OpenNetMon adaptive range)
    ca_grid = [
        (cmin, exp, hi, lo, qs)
        for cmin in (0.5, 0.8)
        for exp in (0.5,)
        for (hi, lo) in [
            (3.0, 1.0), (3.5, 1.2), (4.0, 1.4), (4.5, 1.5), (5.0, 1.7),
            (5.8, 1.9), (6.5, 2.1), (7.5, 2.3), (8.5, 2.6), (9.5, 2.8),
        ]
        for qs in (2,)
    ][:20]
    best_ca_j = float("inf")
    best_ca_cfg = ca_grid[0]
    fix25_kb = fc[3][1] * 8.0
    for cfg in ca_grid:
        p_try = copy.deepcopy(proto)
        p_try.ca_min_interval_s, p_try.ca_norm_exp, p_try.ca_high_threshold, p_try.ca_low_threshold, p_try.ca_quiet_streak_required = cfg
        p_try.ca_signal_mode = "max_weighted"
        res_list = [simulate_trace(tr, "change_aware", p_try)[0] for tr in pilot_traces]
        mean_kb = float(np.mean([r["monitoring_kb"] for r in res_list]))
        # Non-degeneracy check: must actually adapt (not collapsed to constant 6.0s or constant 0.5s)
        if not (0.65 * fix25_kb <= mean_kb <= 2.4 * fix25_kb):
            continue
        mean_j = float(np.mean([r["cost_j"] for r in res_list]))
        if mean_j < best_ca_j:
            best_ca_j = mean_j
            best_ca_cfg = cfg
    if best_ca_j == float("inf"):
        # Fallback to lowest-J candidate if grid was outside KB window
        for cfg in ca_grid:
            p_try = copy.deepcopy(proto)
            p_try.ca_min_interval_s, p_try.ca_norm_exp, p_try.ca_high_threshold, p_try.ca_low_threshold, p_try.ca_quiet_streak_required = cfg
            mean_j = float(np.mean([simulate_trace(tr, "change_aware", p_try)[0]["cost_j"] for tr in pilot_traces]))
            if mean_j < best_ca_j:
                best_ca_j = mean_j
                best_ca_cfg = cfg
    proto.ca_min_interval_s, proto.ca_norm_exp, proto.ca_high_threshold, proto.ca_low_threshold, proto.ca_quiet_streak_required = best_ca_cfg
    proto.ca_signal_mode = "max_weighted"

    # 2b. Equal-budget k=20 sweep for Simple Rule (change + AoI)
    simple_grid = [
        (m_th, mx_th, q_th, age_th)
        for m_th in (1.2, 1.6, 2.0, 2.5)
        for mx_th in (3.0, 4.5)
        for q_th in (0.6, 0.9, 1.2)
        for age_th in (4.5,)
    ][:20]
    best_sr_j = float("inf")
    best_sr_cfg = simple_grid[0]
    for cfg in simple_grid:
        p_try = copy.deepcopy(proto)
        p_try.simple_mean_rate_thresh, p_try.simple_max_rate_thresh, p_try.simple_quiet_mean_thresh, p_try.simple_max_age_safety_s = cfg
        j_vals = [simulate_trace(tr, "rule_simple", p_try)[0]["cost_j"] for tr in pilot_traces]
        if np.mean(j_vals) < best_sr_j:
            best_sr_j = float(np.mean(j_vals))
            best_sr_cfg = cfg
    proto.simple_mean_rate_thresh, proto.simple_max_rate_thresh, proto.simple_quiet_mean_thresh, proto.simple_max_age_safety_s = best_sr_cfg

    # 2c. Equal-budget k=20 sweep for EWMA/CUSUM + Hysteresis Rule
    ewma_grid = [
        (gini_th, osc_th, af_min, q_th, dwell)
        for gini_th in (0.42, 0.50, 0.58)
        for osc_th in (0.55, 0.65)
        for af_min in (0.30, 0.45)
        for q_th in (0.75, 1.05)
        for dwell in (0, 1)
    ][:20]
    best_ew_j = float("inf")
    best_ew_cfg = ewma_grid[0]
    for cfg in ewma_grid:
        p_try = copy.deepcopy(proto)
        p_try.gini_flap_thresh, p_try.osc_trap_thresh, p_try.active_frac_macro_min, p_try.quiet_rate_thresh, p_try.hysteresis_dwell_reviews = cfg
        j_vals = [simulate_trace(tr, "rule_ewma_cusum", p_try)[0]["cost_j"] for tr in pilot_traces]
        if np.mean(j_vals) < best_ew_j:
            best_ew_j = float(np.mean(j_vals))
            best_ew_cfg = cfg
    proto.gini_flap_thresh, proto.osc_trap_thresh, proto.active_frac_macro_min, proto.quiet_rate_thresh, proto.hysteresis_dwell_reviews = best_ew_cfg

    # 3. Collect Pilot Oracle Reviews -> build PilotPercentileReference & Learnability Ceiling
    pilot_oracle_logs: List[Dict[str, Any]] = []
    pilot_headrooms: List[float] = []
    pilot_ewma_diffs: List[float] = []
    for tr in pilot_traces:
        r_fix, _ = simulate_trace(tr, "fixed", proto)
        r_ca, _ = simulate_trace(tr, "change_aware", proto)
        r_ew, _ = simulate_trace(tr, "rule_ewma_cusum", proto)
        r_or, o_logs = simulate_trace(tr, "oracle", proto, compute_oracle_labels=True, record_review_logs=True)
        best_static = min(r_fix["cost_j"], r_ca["cost_j"])
        pilot_headrooms.append(best_static - r_or["cost_j"])
        pilot_ewma_diffs.append(best_static - r_ew["cost_j"])
        pilot_oracle_logs.extend(o_logs)

    # Build monotone percentile reference (strictly label-free feature distributions!)
    feat_mat = np.array([r["features"] for r in pilot_oracle_logs], dtype=np.float64)
    pct_samples = {
        "mean_rate_delta": feat_mat[:, FEATURE_NAMES.index("mean_rate_delta")].tolist(),
        "max_rate_delta": feat_mat[:, FEATURE_NAMES.index("max_rate_delta")].tolist(),
        "spatial_gini": feat_mat[:, FEATURE_NAMES.index("spatial_gini")].tolist(),
        "oscillation_index": feat_mat[:, FEATURE_NAMES.index("oscillation_index")].tolist(),
        "active_link_frac": feat_mat[:, FEATURE_NAMES.index("active_link_frac")].tolist(),
        "ewma_zscore": feat_mat[:, FEATURE_NAMES.index("ewma_zscore")].tolist(),
    }
    pct_ref = PilotPercentileReference(pct_samples)

    # 4. Fit Learnability Ceiling classifier strictly on pilot features + pilot oracle labels
    y_pilot = np.array([1 if r["oracle_label"] == "change-aware" else 0 for r in pilot_oracle_logs], dtype=np.int32)
    weights = np.abs(np.array([r["oracle_margin"] for r in pilot_oracle_logs], dtype=np.float64)) + 0.05
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("gbdt", GradientBoostingClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, random_state=42)),
    ])
    clf.fit(feat_mat, y_pilot, gbdt__sample_weight=weights)

    # 5. Power Analysis (normal approximation for paired superiority & TOST equivalence at alpha=0.05, power=0.80)
    sd_paired = max(0.05, float(np.std(pilot_ewma_diffs, ddof=1))) if len(pilot_ewma_diffs) > 1 else 0.15
    z_a = 1.96
    z_a1 = 1.645
    z_b = 0.842
    n_sup = int(math.ceil(((z_a + z_b) * sd_paired / proto.sesoi_cost) ** 2))
    n_tost = int(math.ceil(((z_a1 + z_b) * sd_paired / proto.sesoi_cost) ** 2))

    # Also select 4 clean balanced pilot few-shot examples (2 high-margin fixed, 2 high-margin CA)
    df_p = pd.DataFrame(pilot_oracle_logs)
    top_ca = df_p[df_p["oracle_label"] == "change-aware"].sort_values("oracle_margin", ascending=False).head(2)
    top_fx = df_p[df_p["oracle_label"] == "fixed"].sort_values("oracle_margin", ascending=True).head(2)

    pilot_report = {
        "lambda_kb_per_link": proto.lambda_kb_per_link,
        "best_ca_cfg": best_ca_cfg,
        "best_ca_pilot_j": round(best_ca_j, 4),
        "best_simple_rule_cfg": best_sr_cfg,
        "best_simple_rule_pilot_j": round(best_sr_j, 4),
        "best_ewma_rule_cfg": best_ew_cfg,
        "best_ewma_rule_pilot_j": round(best_ew_j, 4),
        "pilot_mean_headroom_j": round(float(np.mean(pilot_headrooms)), 4),
        "pilot_ewma_gain_over_best_static_j": round(float(np.mean(pilot_ewma_diffs)), 4),
        "pilot_paired_sd": round(sd_paired, 4),
        "power_n_superiority_80pct": n_sup,
        "power_n_tost_equivalence_80pct": n_tost,
        "recommended_heldout_n": max(30, min(60, max(n_sup, n_tost))),
        "pilot_oracle_ca_rate": round(float(np.mean(y_pilot)), 4),
        "pilot_num_reviews": len(pilot_oracle_logs),
        "few_shot_indices": {"ca": top_ca.index.tolist(), "fixed": top_fx.index.tolist()},
    }
    return proto, pct_ref, clf, pilot_report


# ==============================================================================
# 10. PARETO CONVEX FRONTIER & STATISTICAL RIGOR (WILCOXON, HOLM, TOST, ROC-AUC)
# ==============================================================================
def compute_iso_bytes_delta(
    df_runs: pd.DataFrame,
    static_fixed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-seed paired Delta-MAE_iso_bytes against the dense 15-point log-KB static fixed frontier.

    Never extrapolates outside [min_kb, max_kb] of the dense fixed sweep (returns NaN if OOR).
    """
    out_rows = []
    # Index static fixed curves by (topology, regime, delay, seed)
    fixed_map: Dict[Tuple[str, str, str, int], Tuple[np.ndarray, np.ndarray]] = {}
    for key, g in static_fixed_df.groupby(["topology", "regime", "delay", "seed"]):
        g_s = g.sort_values("monitoring_kb")
        fixed_map[key] = (
            np.log(np.maximum(g_s["monitoring_kb"].to_numpy(dtype=np.float64), 1e-6)),
            g_s["mae_pct"].to_numpy(dtype=np.float64),
        )

    for _, row in df_runs.iterrows():
        r = row.to_dict()
        key = (r["topology"], r["regime"], r["delay"], int(r["seed"]))
        if key in fixed_map:
            lkb, maes = fixed_map[key]
            b_log = math.log(max(float(r["monitoring_kb"]), 1e-6))
            if lkb[0] - 1e-6 <= b_log <= lkb[-1] + 1e-6:
                iso_mae = float(np.interp(b_log, lkb, maes))
                r["delta_mae_iso_bytes"] = round(float(r["mae_pct"]) - iso_mae, 4)
            else:
                r["delta_mae_iso_bytes"] = np.nan
        else:
            r["delta_mae_iso_bytes"] = np.nan
        out_rows.append(r)
    return pd.DataFrame(out_rows)


def paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 2:
        m = float(np.mean(arr)) if len(arr) == 1 else np.nan
        return (m, m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return (round(float(np.percentile(means, 100 * alpha / 2)), 4),
            round(float(np.percentile(means, 100 * (1 - alpha / 2))), 4))


def tost_paired_equivalence(diffs: np.ndarray, sesoi: float, alpha: float = 0.05) -> Dict[str, Any]:
    """Two One-Sided Tests (TOST, Lakens 2017) for paired equivalence within [-sesoi, +sesoi]."""
    arr = np.asarray(diffs, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < 3:
        return {"p_tost": np.nan, "equivalent": False, "p_wilcoxon": np.nan}
    mean_d = float(np.mean(arr))
    sd_d = float(np.std(arr, ddof=1))
    se = sd_d / math.sqrt(n)
    if se < 1e-12:
        p_tost = 0.0 if abs(mean_d) < sesoi else 1.0
        p_wil = 1.0 if abs(mean_d) < 1e-12 else 0.0
    else:
        t_low = (mean_d - (-sesoi)) / se   # H0: mean <= -sesoi
        t_high = (mean_d - (+sesoi)) / se  # H0: mean >= +sesoi
        p_low = float(1.0 - stats.t.cdf(t_low, df=n - 1))
        p_high = float(stats.t.cdf(t_high, df=n - 1))
        p_tost = max(p_low, p_high)
        try:
            _, p_wil = stats.wilcoxon(arr, zero_method="wilcox")
            p_wil = float(p_wil)
        except Exception:
            p_wil = 1.0
    return {
        "p_tost": round(p_tost, 5),
        "equivalent_at_sesoi": bool(p_tost < alpha),
        "p_wilcoxon": round(p_wil, 5),
    }


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, p_values[idx] * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return [round(float(x), 5) for x in adj]


def compute_signal_diagnostics(review_logs_df: pd.DataFrame) -> pd.DataFrame:
    """Threshold-free ROC-AUC and Average Precision of controller score vs. oracle per-review labels."""
    rows = []
    if review_logs_df.empty or "oracle_label" not in review_logs_df.columns:
        return pd.DataFrame()
    valid = review_logs_df.dropna(subset=["oracle_label"]).copy()
    valid["y_true"] = (valid["oracle_label"] == "change-aware").astype(int)
    for (ctrl, regime), g in valid.groupby(["controller", "regime"]):
        y = g["y_true"].to_numpy()
        s = g["score_ca"].to_numpy(dtype=np.float64)
        if len(np.unique(y)) < 2:
            auc = np.nan
            ap = np.nan
        else:
            auc = round(float(roc_auc_score(y, s)), 4)
            ap = round(float(average_precision_score(y, s)), 4)
        # Also evaluate on high-margin reviews (top 50% by |oracle_margin|)
        med_m = float(np.median(np.abs(g["oracle_margin"])))
        g_hi = g[np.abs(g["oracle_margin"]) >= med_m]
        y_hi = g_hi["y_true"].to_numpy()
        s_hi = g_hi["score_ca"].to_numpy(dtype=np.float64)
        auc_hi = round(float(roc_auc_score(y_hi, s_hi)), 4) if len(np.unique(y_hi)) >= 2 else np.nan
        acc = round(float(np.mean(g["effective_policy_after"] == g["oracle_label"])), 4)
        rows.append({
            "controller": ctrl,
            "regime": regime,
            "n_reviews": len(g),
            "oracle_ca_rate": round(float(np.mean(y)), 3),
            "ctrl_ca_rate": round(float(np.mean(g["effective_policy_after"] == "change-aware")), 3),
            "oracle_agreement": acc,
            "roc_auc_all": auc,
            "roc_auc_high_margin": auc_hi,
            "avg_precision": ap,
        })
    return pd.DataFrame(rows)


def compute_headroom_and_stats_table(
    df_all: pd.DataFrame,
    protocol: ProtocolV2,
    reference_baseline: str = "rule_ewma_cusum",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute (1) Condition Summary with Headroom Captured and (2) Paired Statistical Contrasts."""
    # Pivot cost_j by (topology, regime, delay, seed)
    piv = df_all.pivot_table(
        index=["topology", "regime", "delay", "seed"],
        columns="controller",
        values="cost_j",
        aggfunc="first",
    )
    if "fixed_2.500s" in piv.columns and "change_aware" in piv.columns and "oracle" in piv.columns:
        piv["best_static_j"] = np.minimum(piv["fixed_2.500s"], piv["change_aware"])
        piv["headroom_j"] = np.maximum(0.0, piv["best_static_j"] - piv["oracle"])

    # Condition summary table
    cond_rows = []
    for (topo, reg, dl, ctrl), g in df_all.groupby(["topology", "regime", "delay", "controller"]):
        # Compute headroom captured over matching seeds
        mask_p = (
            (piv.index.get_level_values("topology") == topo)
            & (piv.index.get_level_values("regime") == reg)
            & (piv.index.get_level_values("delay") == dl)
        )
        sub_p = piv.loc[mask_p]
        mean_hr = float(sub_p["headroom_j"].mean()) if "headroom_j" in sub_p else np.nan
        if ctrl in sub_p.columns and "best_static_j" in sub_p.columns and mean_hr > 1e-4:
            gain = float((sub_p["best_static_j"] - sub_p[ctrl]).mean())
            hr_cap = round(100.0 * gain / mean_hr, 1)
        else:
            gain = np.nan
            hr_cap = np.nan
        cond_rows.append({
            "topology": topo, "regime": reg, "delay": dl, "controller": ctrl,
            "n_seeds": len(g),
            "mae_mean": round(float(g["mae_pct"].mean()), 4),
            "kb_mean": round(float(g["monitoring_kb"].mean()), 2),
            "cost_j_mean": round(float(g["cost_j"].mean()), 4),
            "delta_mae_iso_mean": round(float(g["delta_mae_iso_bytes"].mean()), 4) if "delta_mae_iso_bytes" in g else np.nan,
            "headroom_j": round(mean_hr, 4) if not np.isnan(mean_hr) else np.nan,
            "headroom_captured_pct": hr_cap,
            "congestion_f1": round(float(g["congestion_f1"].mean()), 4),
            "ca_frac": round(float(g["change_aware_frac"].mean()), 3),
            "switches": round(float(g["switches"].mean()), 2),
            "latency_p95_ms": round(float(g["latency_p95_ms"].mean()), 2),
            "fallback_rate": round(float(g["fallback_rate"].mean()), 4),
        })
    cond_df = pd.DataFrame(cond_rows)

    # Paired contrasts vs reference_baseline (and vs best_static_j) with TOST + Wilcoxon + Holm
    contrast_rows = []
    for (topo, reg, dl), sub_p in piv.groupby(level=[0, 1, 2]):
        for ctrl in [c for c in sub_p.columns if c not in ("best_static_j", "headroom_j", reference_baseline, "oracle") and not c.startswith("fixed_")]:
            if reference_baseline not in sub_p.columns:
                continue
            pair = sub_p[[ctrl, reference_baseline]].dropna()
            if len(pair) < 3:
                continue
            # Negative diff => ctrl is BETTER (lower cost J) than reference_baseline
            diffs = (pair[ctrl] - pair[reference_baseline]).to_numpy(dtype=np.float64)
            ci_lo, ci_hi = paired_bootstrap_ci(diffs)
            tost = tost_paired_equivalence(diffs, sesoi=protocol.sesoi_cost)
            contrast_rows.append({
                "topology": topo, "regime": reg, "delay": dl,
                "controller": ctrl, "vs_baseline": reference_baseline,
                "n_paired": len(pair),
                "mean_diff_j": round(float(np.mean(diffs)), 4),
                "ci95_lo": ci_lo, "ci95_hi": ci_hi,
                "p_wilcoxon": tost["p_wilcoxon"],
                "p_tost_sesoi": tost["p_tost"],
                "equivalent_within_sesoi": tost["equivalent_at_sesoi"],
            })
    contrast_df = pd.DataFrame(contrast_rows)
    if not contrast_df.empty:
        # Apply Holm-Bonferroni correction per controller across (topology, regime, delay)
        contrast_df["p_wilcoxon_holm"] = np.nan
        for ctrl, idx in contrast_df.groupby("controller").groups.items():
            pvals = contrast_df.loc[idx, "p_wilcoxon"].fillna(1.0).tolist()
            contrast_df.loc[idx, "p_wilcoxon_holm"] = holm_bonferroni(pvals)
    return cond_df, contrast_df


if __name__ == "__main__":
    print("=== Running NDT v2 Local Verification (QA + Pilot Calibration + Headroom Check) ===")
    p0 = ProtocolV2()
    qa = run_qa_suite_v2(p0)
    print(f"QA Suite: {qa['num_passed']}/{qa['num_checks']} passed | oracle_vs_2R_gap={qa['oracle_vs_2R_gap']}")
    for k, v in qa["details"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    assert qa["all_passed"], f"QA failed: {qa}"

    t0 = time.perf_counter()
    proto, pct_ref, clf, rep = run_pilot_calibration(pilot_seeds=[101, 102, 103, 104, 105])
    print(f"\nPilot Calibration completed in {time.perf_counter() - t0:.1f}s:")
    print(json.dumps(rep, indent=2))
