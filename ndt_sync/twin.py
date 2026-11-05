"""Twin State Updater, Delayed Delivery Channel, and Polling Policies.

Implements all operational rules from Research Plan §2.1, §2.3, and §2.4:
- Every poll samples all 8 monitored links at dispatch time `t_dispatch`.
- The reply arrives at `t_arrival = t_dispatch + delivery_delay`.
- Out-of-order rejection: the twin rejects any arriving measurement whose
  `sample_timestamp <= latest_accepted_sample_timestamp`.
- Policy switching retains existing measurement history and schedules the next
  poll from `last_poll_dispatch_time` without triggering an extra instantaneous poll.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from .config import NUM_LINKS, LINK_NAMES, DEFAULT_PROTOCOL, ExperimentProtocol


@dataclass
class PollMeasurement:
    """Represents a single poll dispatched at `dispatch_time_s` and arriving at `arrival_time_s`."""
    poll_id: int
    dispatch_time_s: float
    arrival_time_s: float
    link_utilizations: np.ndarray  # shape (8,)
    active_policy_at_dispatch: str
    request_bytes: int
    reply_bytes: int


@dataclass
class AcceptedObservation:
    """Record stored in the Twin's observation history upon accepting a poll reply."""
    poll_id: int
    sample_time_s: float
    arrival_time_s: float
    link_utilizations: np.ndarray
    max_abs_link_change: float
    mean_abs_link_change: float
    per_link_abs_change: np.ndarray


class NetworkDigitalTwin:
    """Maintains stored link utilization state and strictly received observation history."""

    def __init__(self, initial_utilizations: Optional[np.ndarray] = None):
        if initial_utilizations is not None:
            self.stored_utilizations = np.array(initial_utilizations, dtype=np.float64).copy()
        else:
            self.stored_utilizations = np.zeros(NUM_LINKS, dtype=np.float64)
        self.latest_accepted_sample_time_s: float = -1.0
        self.latest_arrival_time_s: float = 0.0
        self.accepted_history: List[AcceptedObservation] = []
        self.rejected_out_of_order_count: int = 0
        self.total_polls_dispatched: int = 0
        self.total_monitoring_request_bytes: int = 0
        self.total_monitoring_reply_bytes: int = 0

    @property
    def total_monitoring_bytes(self) -> int:
        return self.total_monitoring_request_bytes + self.total_monitoring_reply_bytes

    def register_poll_dispatch(self, request_bytes: int) -> None:
        self.total_polls_dispatched += 1
        self.total_monitoring_request_bytes += request_bytes

    def receive_measurement(self, measurement: PollMeasurement) -> bool:
        """Process an arriving poll reply. Rejects stale/out-of-order measurements (§2.3)."""
        # Reply bytes arrive at application layer regardless of acceptance
        self.total_monitoring_reply_bytes += measurement.reply_bytes

        # Out-of-order protection (§2.3: "The twin will reject a measurement older than its latest accepted one")
        if measurement.dispatch_time_s <= self.latest_accepted_sample_time_s + 1e-9:
            self.rejected_out_of_order_count += 1
            return False

        if len(self.accepted_history) == 0:
            per_link_delta = np.zeros(NUM_LINKS, dtype=np.float64)
        else:
            per_link_delta = np.abs(measurement.link_utilizations - self.stored_utilizations)

        max_delta = float(np.max(per_link_delta))
        mean_delta = float(np.mean(per_link_delta))

        self.stored_utilizations = measurement.link_utilizations.copy()
        self.latest_accepted_sample_time_s = measurement.dispatch_time_s
        self.latest_arrival_time_s = measurement.arrival_time_s

        obs = AcceptedObservation(
            poll_id=measurement.poll_id,
            sample_time_s=measurement.dispatch_time_s,
            arrival_time_s=measurement.arrival_time_s,
            link_utilizations=measurement.link_utilizations.copy(),
            max_abs_link_change=max_delta,
            mean_abs_link_change=mean_delta,
            per_link_abs_change=per_link_delta,
        )
        self.accepted_history.append(obs)
        return True

    def current_measurement_age_s(self, current_time_s: float) -> float:
        """Age of Information (AoI): time since the freshest received update was generated (§1.2)."""
        if self.latest_accepted_sample_time_s < 0:
            return current_time_s
        return max(0.0, current_time_s - self.latest_accepted_sample_time_s)


class PollingPolicy:
    """Base interface for Fixed-Interval and Change-Aware polling policies."""

    name: str = "base"

    def current_interval_s(self) -> float:
        raise NotImplementedError

    def on_new_accepted_observation(self, obs: AcceptedObservation) -> None:
        pass

    def next_poll_time_s(self, last_poll_dispatch_time_s: float) -> float:
        """Schedules next poll strictly from `last_poll_dispatch_time_s` (§2.3: no extra poll on switch)."""
        return round(last_poll_dispatch_time_s + self.current_interval_s(), 6)


class FixedIntervalPolicy(PollingPolicy):
    """Polls all monitored links at a constant preset interval."""

    name: str = "fixed"

    def __init__(self, interval_s: float = DEFAULT_PROTOCOL.default_fixed_interval_s):
        self.interval_s = float(interval_s)

    def current_interval_s(self) -> float:
        return self.interval_s


class ChangeAwarePolicy(PollingPolicy):
    """OpenNetMon-inspired change-aware policy (§1.2, §2.3).

    Shortens polling interval after observing sufficiently large link changes,
    and lengthens it after consecutive quiet periods within preset bounds.
    """

    name: str = "change-aware"

    def __init__(self, protocol: ExperimentProtocol = DEFAULT_PROTOCOL):
        self.min_interval_s = protocol.ca_min_interval_s
        self.max_interval_s = protocol.ca_max_interval_s
        self.interval_s = protocol.ca_init_interval_s
        self.high_threshold = protocol.ca_high_change_threshold
        self.low_threshold = protocol.ca_low_change_threshold
        self.quiet_streak_required = protocol.ca_quiet_streak_required
        self.shorten_factor = protocol.ca_shorten_factor
        self.lengthen_factor = protocol.ca_lengthen_factor
        self.quiet_streak = 0

    def current_interval_s(self) -> float:
        return self.interval_s

    def on_new_accepted_observation(self, obs: AcceptedObservation) -> None:
        # Use observed change from received twin measurements only
        signal = max(obs.max_abs_link_change, 1.5 * obs.mean_abs_link_change)
        if signal >= self.high_threshold:
            self.quiet_streak = 0
            self.interval_s = max(self.min_interval_s, round(self.interval_s * self.shorten_factor, 4))
        elif signal <= self.low_threshold:
            self.quiet_streak += 1
            if self.quiet_streak >= self.quiet_streak_required:
                self.interval_s = min(self.max_interval_s, round(self.interval_s * self.lengthen_factor, 4))
        else:
            self.quiet_streak = 0
