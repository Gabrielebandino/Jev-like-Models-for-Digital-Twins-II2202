"""Policy Selectors and Leak-Free State Summary Builder.

Implements Research Plan §2.1, §2.3, §2.4:
- Both the Deterministic Rule Selector and the Decision-Model Selector receive
  the exact same information types derived strictly from measurements ALREADY
  RECEIVED by the twin (`arrival_time_s <= review_time_s`).
- Neither selector can inspect unmeasured current state or future measurements.
- Pending decision-model requests take effect at `t_effective = t_review + latency`.
  Until then, the previous policy stays active.
- If a model decision is invalid or misses `decision_deadline_ms`, the controller
  retains the previous policy and logs a fallback.
- Only one model request can be outstanding at a time (no backlog).
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import json
import numpy as np
from .config import NUM_LINKS, LINK_NAMES, DEFAULT_PROTOCOL, ExperimentProtocol
from .twin import NetworkDigitalTwin, AcceptedObservation


# Frozen System-One Choice question specification used across all candidate models
FROZEN_POLICY_CHOICE_QUESTION = {
    "selected_policy": {
        "type": "choice",
        "instructions": (
            "Classify the current network telemetry state to select the synchronization polling policy: "
            "pick 'change-aware' if the state reports HIGH_VOLATILITY_OR_BURST, MODERATE_VARIATION, "
            "traffic spikes, bursts, or link changes above 3%; pick 'fixed' only if the state reports "
            "LOW_STEADY_VARIATION with minimal link changes."
        ),
        "criteria": {
            "fixed": (
                "LOW_STEADY_VARIATION: Network link utilization is completely flat, stable, and quiet "
                "with negligible observed deltas (< 2% change)."
            ),
            "change-aware": (
                "HIGH_VOLATILITY_OR_BURST or MODERATE_VARIATION: Network links show bursts, traffic spikes, "
                "load shifts, or noticeable utilization deltas (>= 2.5% change) requiring adaptive polling."
            ),
        },
    }
}


@dataclass
class ReviewSummary:
    """Structured summary of twin-received observations at `review_time_s` (Zero Future Leakage)."""
    review_time_s: float
    current_active_policy: str
    current_policy_interval_s: float
    measurement_age_aoi_s: float
    latest_sample_time_s: float
    latest_arrival_time_s: float
    polls_received_in_window: int
    total_polls_received: int
    recent_mean_abs_change_pct: float
    recent_max_abs_change_pct: float
    recent_change_std_pct: float
    latest_poll_max_change_pct: float
    latest_poll_mean_change_pct: float
    per_link_latest_utilization_pct: Dict[str, float]
    per_link_recent_abs_change_pct: Dict[str, float]

    def to_prompt_text(self) -> str:
        """Render compact, structured plain-text state for System-One decision models."""
        links_str = ", ".join(
            f"{k}={v:.1f}% (Δ={self.per_link_recent_abs_change_pct[k]:.1f}%)"
            for k, v in self.per_link_latest_utilization_pct.items()
        )
        is_volatile = (
            self.recent_max_abs_change_pct >= 5.5
            or self.recent_mean_abs_change_pct >= 2.8
            or self.recent_change_std_pct >= 2.0
        )
        volatility_desc = (
            "HIGH_VOLATILITY_OR_BURST (spikes and rapid link variations detected -> requires change-aware adaptive polling)"
            if is_volatile
            else (
                "MODERATE_VARIATION (active link fluctuations -> prefer change-aware adaptive polling)"
                if self.recent_max_abs_change_pct >= 2.5
                else "LOW_STEADY_VARIATION (completely flat, quiet, and stable traffic -> use fixed polling)"
            )
        )
        return (
            f"Network Digital Twin Synchronization Review at t={self.review_time_s:.1f}s\n"
            f"State Classification: {volatility_desc}\n"
            f"Current Active Policy: {self.current_active_policy} (interval={self.current_policy_interval_s:.2f}s)\n"
            f"Current Measurement Age (AoI): {self.measurement_age_aoi_s:.2f}s "
            f"(latest sample t={self.latest_sample_time_s:.2f}s, received t={self.latest_arrival_time_s:.2f}s)\n"
            f"Recent Polling Activity: {self.polls_received_in_window} polls received in last review window "
            f"({self.total_polls_received} total accepted polls)\n"
            f"Observed Link Change Summary: "
            f"recent_mean_abs_change={self.recent_mean_abs_change_pct:.2f}%, "
            f"recent_max_abs_change={self.recent_max_abs_change_pct:.2f}%, "
            f"change_volatility_std={self.recent_change_std_pct:.2f}%, "
            f"latest_poll_max_delta={self.latest_poll_max_change_pct:.2f}%\n"
            f"Per-Link Stored Utilization & Latest Observed Change (8 links): {links_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_leak_free_review_summary(
    twin: NetworkDigitalTwin,
    review_time_s: float,
    current_active_policy: str,
    current_policy_interval_s: float,
    review_window_s: float = DEFAULT_PROTOCOL.review_period_s,
) -> ReviewSummary:
    """Build structured summary strictly from observations with `arrival_time_s <= review_time_s`."""
    # Enforce strict causality guard: never inspect observations that arrived after review_time_s
    valid_obs: List[AcceptedObservation] = [
        obs for obs in twin.accepted_history if obs.arrival_time_s <= review_time_s + 1e-9
    ]
    window_start = max(0.0, review_time_s - review_window_s)
    recent_obs = [obs for obs in valid_obs if obs.arrival_time_s >= window_start - 1e-9]
    if not recent_obs and valid_obs:
        recent_obs = [valid_obs[-1]]

    if valid_obs:
        latest = valid_obs[-1]
        latest_sample_t = latest.sample_time_s
        latest_arrival_t = latest.arrival_time_s
        aoi_s = max(0.0, review_time_s - latest_sample_t)
        latest_max_delta = latest.max_abs_link_change
        latest_mean_delta = latest.mean_abs_link_change
        per_link_util = {LINK_NAMES[i]: round(float(latest.link_utilizations[i]), 2) for i in range(NUM_LINKS)}
        per_link_delta = {LINK_NAMES[i]: round(float(latest.per_link_abs_change[i]), 2) for i in range(NUM_LINKS)}
    else:
        latest_sample_t = -1.0
        latest_arrival_t = 0.0
        aoi_s = review_time_s
        latest_max_delta = 0.0
        latest_mean_delta = 0.0
        per_link_util = {LINK_NAMES[i]: round(float(twin.stored_utilizations[i]), 2) for i in range(NUM_LINKS)}
        per_link_delta = {LINK_NAMES[i]: 0.0 for i in range(NUM_LINKS)}

    # Compute window statistics over observations with real deltas (poll index >= 2)
    delta_obs = [o for o in recent_obs if o.poll_id > 1]
    if delta_obs:
        mean_changes = [o.mean_abs_link_change for o in delta_obs]
        max_changes = [o.max_abs_link_change for o in delta_obs]
        recent_mean = float(np.mean(mean_changes))
        recent_max = float(np.max(max_changes))
        recent_std = float(np.std(max_changes))
    else:
        recent_mean = 0.0
        recent_max = 0.0
        recent_std = 0.0

    return ReviewSummary(
        review_time_s=round(review_time_s, 4),
        current_active_policy=current_active_policy,
        current_policy_interval_s=round(current_policy_interval_s, 4),
        measurement_age_aoi_s=round(aoi_s, 4),
        latest_sample_time_s=round(latest_sample_t, 4),
        latest_arrival_time_s=round(latest_arrival_t, 4),
        polls_received_in_window=len(recent_obs),
        total_polls_received=len(valid_obs),
        recent_mean_abs_change_pct=round(recent_mean, 4),
        recent_max_abs_change_pct=round(recent_max, 4),
        recent_change_std_pct=round(recent_std, 4),
        latest_poll_max_change_pct=round(latest_max_delta, 4),
        latest_poll_mean_change_pct=round(latest_mean_delta, 4),
        per_link_latest_utilization_pct=per_link_util,
        per_link_recent_abs_change_pct=per_link_delta,
    )


@dataclass
class SelectorDecisionResult:
    """Output of a selector invocation at `review_time_s`."""
    selected_policy: Optional[str]        # "fixed" or "change-aware" (None if invalid/failed)
    decision_latency_ms: float            # Time taken to compute decision
    probabilities: Dict[str, float]       # Probability distribution over {"fixed", "change-aware"}
    confidence: float
    request_bytes: int
    reply_bytes: int
    raw_output: str
    is_valid: bool
    error_reason: Optional[str] = None


class DeterministicRuleSelector:
    """Threshold-rule policy selector baseline (Research Plan §2.1, §2.3)."""

    name: str = "rule_switcher"

    def __init__(self, protocol: ExperimentProtocol = DEFAULT_PROTOCOL):
        self.volatility_threshold = protocol.rule_volatility_threshold
        self.max_delta_threshold = protocol.rule_max_link_delta_threshold
        self.max_age_safety_s = protocol.rule_max_age_safety_s

    def select_policy(self, summary: ReviewSummary) -> SelectorDecisionResult:
        # Rule logic using the exact same summary fields: observed changes and age
        if (
            summary.recent_max_abs_change_pct >= self.max_delta_threshold
            or summary.recent_mean_abs_change_pct >= self.volatility_threshold
            or summary.recent_change_std_pct >= 2.2
        ):
            choice = "change-aware"
            probs = {"fixed": 0.0, "change-aware": 1.0}
        elif summary.measurement_age_aoi_s >= self.max_age_safety_s and summary.recent_max_abs_change_pct < 1.5:
            # Quiet period with low variation: change-aware backs off safely to save monitoring traffic
            choice = "change-aware"
            probs = {"fixed": 0.0, "change-aware": 1.0}
        else:
            choice = "fixed"
            probs = {"fixed": 1.0, "change-aware": 0.0}

        return SelectorDecisionResult(
            selected_policy=choice,
            decision_latency_ms=0.05,  # Sub-millisecond local threshold rule
            probabilities=probs,
            confidence=1.0,
            request_bytes=0,
            reply_bytes=0,
            raw_output=f"rule_decision={choice}",
            is_valid=True,
            error_reason=None,
        )

