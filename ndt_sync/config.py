"""Configuration, network topology, regime definitions, and frozen protocol parameters."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple
import json


# ==============================================================================
# 1. NETWORK TOPOLOGY (8 Monitored Links with Shared Multi-Hop Paths)
# ==============================================================================
# Matches Research Plan §2.2: "a fixed graph with around eight monitored links and shared paths"
NUM_LINKS = 8
LINK_NAMES = [f"L{i}" for i in range(NUM_LINKS)]

# 6 Origin-Destination (OD) Traffic Flows traversing shared paths across the 8 links
# Each entry maps Flow ID -> (List of Link Indices on path, Weight factor)
OD_FLOW_PATHS: Dict[str, List[Tuple[int, float]]] = {
    "F0_core_west_east": [(0, 1.0), (1, 0.9), (4, 0.85)],
    "F1_core_north_south": [(2, 1.0), (3, 0.95), (5, 0.8)],
    "F2_edge_aggregation_A": [(0, 0.5), (2, 0.6), (6, 1.0)],
    "F3_edge_aggregation_B": [(1, 0.55), (3, 0.5), (7, 1.0)],
    "F4_cross_datacenter": [(4, 0.7), (5, 0.75), (6, 0.6), (7, 0.65)],
    "F5_background_mesh": [(0, 0.3), (3, 0.35), (4, 0.4), (7, 0.3)],
}

TOPOLOGY_DEFINITION = {
    "name": "NDT_8Link_SharedPath_Backbone",
    "num_links": NUM_LINKS,
    "links": [
        {"id": 0, "name": "L0", "role": "core_ingress_west", "capacity_gbps": 10.0},
        {"id": 1, "name": "L1", "role": "core_transit_west", "capacity_gbps": 10.0},
        {"id": 2, "name": "L2", "role": "core_ingress_north", "capacity_gbps": 10.0},
        {"id": 3, "name": "L3", "role": "core_transit_north", "capacity_gbps": 10.0},
        {"id": 4, "name": "L4", "role": "core_backbone_east", "capacity_gbps": 10.0},
        {"id": 5, "name": "L5", "role": "core_backbone_south", "capacity_gbps": 10.0},
        {"id": 6, "name": "L6", "role": "aggregation_uplink_A", "capacity_gbps": 10.0},
        {"id": 7, "name": "L7", "role": "aggregation_uplink_B", "capacity_gbps": 10.0},
    ],
    "od_flows": {k: [{"link_id": lid, "share": w} for lid, w in v] for k, v in OD_FLOW_PATHS.items()},
}


# ==============================================================================
# 2. EXPERIMENTAL CONDITIONS & SEEDS (Strict Pilot vs. Held-Out Separation)
# ==============================================================================
TRAFFIC_REGIMES = ("stable", "bursty", "alternating", "shifted")
DELAY_CONDITIONS = ("low_delay", "elevated_delay")

# Delivery delay parameters in milliseconds (Research Plan §2.2, §2.3)
DELAY_PARAMS_MS = {
    "low_delay": {"base_ms": 50.0, "jitter_std_ms": 10.0, "out_of_order_prob": 0.01},
    "elevated_delay": {"base_ms": 500.0, "jitter_std_ms": 60.0, "out_of_order_prob": 0.03},
}

# Strict separation of Pilot and Held-Out seeds (Research Plan §2.2 & §2.4)
PILOT_SEEDS = [101, 102, 103]
HELDOUT_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012]
REPEAT_STABILITY_SEEDS = [1001, 1002]  # Predefined subset repeated 3x for model stability analysis (§2.4)


# ==============================================================================
# 3. FROZEN PROTOCOL HYPERPARAMETERS
# ==============================================================================
@dataclass(frozen=True)
class ExperimentProtocol:
    """Frozen simulation and controller parameters per Research Plan §2.2-2.4."""
    # Simulation timing
    trace_duration_s: float = 180.0       # 3-minute trace at 100ms resolution (1,800 ticks)
    warmup_duration_s: float = 20.0       # Common warm-up excluded from state error, included in byte cost (§2.4)
    tick_dt_s: float = 0.1                # 100 ms discrete simulation resolution

    # Application-level monitoring message framing bytes (§2.4: payload bytes, not wire/energy)
    # Poll request: 8-byte header + 4 bytes per queried link ID = 40 bytes
    # Poll reply: 12-byte header + 12 bytes per link (float32 util + float64 timestamp) = 108 bytes
    poll_request_bytes: int = 40
    poll_reply_bytes: int = 108

    # Candidate Fixed-Interval settings (§2.2: three candidate fixed intervals spanning short to long)
    candidate_fixed_intervals_s: Tuple[float, ...] = (1.0, 2.5, 5.0)
    default_fixed_interval_s: float = 2.5  # The fixed interval option available to the switchers

    # Change-Aware Policy parameters (OpenNetMon-style §1.2, §2.3)
    ca_min_interval_s: float = 0.5        # Minimum polling interval during high activity
    ca_max_interval_s: float = 6.0        # Maximum polling interval during quiet periods
    ca_init_interval_s: float = 2.5       # Starting interval
    ca_high_change_threshold: float = 4.5 # Mean or max link delta (pct points) to halve interval
    ca_low_change_threshold: float = 1.8  # Quiet threshold to lengthen interval
    ca_quiet_streak_required: int = 2     # Consecutive quiet polls before lengthening
    ca_shorten_factor: float = 0.5        # Multiplicative decrease on variation
    ca_lengthen_factor: float = 1.5       # Multiplicative increase on quiet

    # Selector Review Cadence & Deadline (§2.3)
    review_period_s: float = 15.0         # Preset review interval for Rule and Decision Model selectors
    decision_deadline_ms: float = 2000.0  # Max allowed decision latency before fallback retains previous policy

    # Deterministic Threshold-Rule Switcher parameters (§2.3)
    rule_volatility_threshold: float = 3.2   # Switch to change-aware if recent max change or burstiness is high
    rule_max_link_delta_threshold: float = 6.5
    rule_max_age_safety_s: float = 4.5       # If age grows large under quiet, prefer predictable fixed polling

    # Candidate Models Metadata
    candidate_models: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "laya": {
            "hf_repo": "convaiinnovations/laya",
            "revision": "7b928d828b7b0e022f929d9bd2e44165aa270148",
            "architecture": "ModernBERT-large non-autoregressive System-1 (~421M)",
            "precision_t4": "float16",
        },
        "kev_4b": {
            "hf_repo": "jaredpalmer/kev-4b",
            "revision": "6cfce5c2fa4b4bd64026336ab649c5ca78857d52",
            "base_model": "Qwen/Qwen3.5-4B-Base",
            "architecture": "Qwen3.5-4B + Rank-16 LoRA + Pointer Head",
            "precision_t4": "4bit-nf4-float16",
        },
        "clm_8b": {
            "hf_repo": "Contrastive-LM/CLM-v0.1-8B",
            "revision": "e939398d4556fcd9400c76fa8c5a513202f42b0a",
            "base_model": "Qwen/Qwen3-8B",
            "architecture": "Frozen Qwen3-8B last-token pooling + Dual InfoNCE Projection Heads (~40M)",
            "precision_t4": "4bit-nf4-float16",
        },
    })

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


DEFAULT_PROTOCOL = ExperimentProtocol()
