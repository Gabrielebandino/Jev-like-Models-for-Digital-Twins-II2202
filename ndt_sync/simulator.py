"""Discrete-Time Trace-Driven Simulator & Ground-Truth Evaluator.

Implements Research Plan §2.1, §2.3, §2.4:
- Replays pre-generated synthetic link-utilization traces across any of the 4 controllers:
  1. `fixed` (static Fixed-Interval polling)
  2. `change_aware` (static Change-Aware polling)
  3. `rule_switcher` (Deterministic Threshold-Rule switching between `fixed` and `change-aware`)
  4. `model_switcher` (Decision-Model switching between `fixed` and `change-aware`)
- Enforces delayed poll delivery, out-of-order rejection, asynchronous decision latency,
  single outstanding model request (no backlog), deadline timeouts/fallbacks, and
  switching without an extra instantaneous poll.
- Calculates Mean Absolute Twin-State Error (MAE in percentage points after warm-up),
  application-level monitoring bytes (including warm-up), poll count, change-detection
  delay, and model overhead / fallback metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config import NUM_LINKS, LINK_NAMES, DEFAULT_PROTOCOL, ExperimentProtocol
from .twin import (
    NetworkDigitalTwin,
    PollMeasurement,
    PollingPolicy,
    FixedIntervalPolicy,
    ChangeAwarePolicy,
)
from .selectors import (
    ReviewSummary,
    SelectorDecisionResult,
    DeterministicRuleSelector,
    build_leak_free_review_summary,
)


@dataclass
class PendingModelDecision:
    """Tracks a single in-flight decision-model request formed at `review_time_s`."""
    review_idx: int
    review_time_s: float
    effective_time_s: float
    deadline_time_s: float
    summary_snapshot: ReviewSummary
    decision_result: SelectorDecisionResult


@dataclass
class ReviewLogRecord:
    """Full audit record for a single policy review event (§2.3, §2.4)."""
    run_id: str
    controller: str
    model_name: str
    regime: str
    delay_condition: str
    seed: int
    review_idx: int
    review_time_s: float
    effective_time_s: float
    previous_policy: str
    raw_selected_policy: Optional[str]
    effective_policy_after: str
    policy_switched: bool
    decision_latency_ms: float
    deadline_ms: float
    is_fallback: bool
    fallback_reason: Optional[str]
    prob_fixed: float
    prob_change_aware: float
    confidence: float
    request_bytes: int
    reply_bytes: int
    summary_aoi_s: float
    summary_recent_mean_change: float
    summary_recent_max_change: float
    summary_polls_in_window: int
    prompt_text: str
    raw_output: str


@dataclass
class SimulationRunResult:
    """Complete summary and detailed logs of one trace-driven simulation run."""
    run_id: str
    controller: str
    model_name: str
    fixed_interval_setting_s: float
    regime: str
    delay_condition: str
    seed: int
    trace_duration_s: float
    warmup_duration_s: float
    # Primary metrics (§2.4)
    mae_pct_points: float
    rmse_pct_points: float
    p95_abs_error_pct: float
    monitoring_bytes_total: int
    monitoring_request_bytes: int
    monitoring_reply_bytes: int
    poll_count: int
    accepted_poll_count: int
    rejected_out_of_order_count: int
    # Supporting metrics (§2.4)
    mean_detection_delay_s: float
    max_detection_delay_s: float
    mean_aoi_s: float
    review_count: int
    switch_count: int
    fallback_count: int
    fallback_rate: float
    skipped_reviews_backlog: int
    fixed_policy_fraction: float
    change_aware_policy_fraction: float
    # Model overhead metrics (separate from monitoring bytes per §2.4)
    model_latency_mean_ms: float
    model_latency_p50_ms: float
    model_latency_p95_ms: float
    model_latency_max_ms: float
    model_request_bytes_total: int
    model_reply_bytes_total: int
    # Detailed logs
    review_logs: List[ReviewLogRecord] = field(default_factory=list, repr=False)
    timeseries_df: Optional[pd.DataFrame] = field(default=None, repr=False)

    def to_summary_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("review_logs", None)
        d.pop("timeseries_df", None)
        return d


def run_trace_simulation(
    trace_df: pd.DataFrame,
    controller_type: str,
    protocol: ExperimentProtocol = DEFAULT_PROTOCOL,
    fixed_interval_s: Optional[float] = None,
    model_selector_fn: Optional[Callable[[ReviewSummary], SelectorDecisionResult]] = None,
    model_name: str = "none",
    run_id_prefix: str = "run",
    record_timeseries: bool = False,
) -> SimulationRunResult:
    """Execute a trace-driven simulation for one controller on one trace DataFrame.

    Parameters
    ----------
    trace_df : pd.DataFrame
        Pre-generated synthetic trace containing `tick_idx`, `timestamp_s`, `u_L0..u_L7`,
        `delivery_delay_ms`, `is_change_event`, `regime`, `delay_condition`, `seed`.
    controller_type : str
        One of `"fixed"`, `"change_aware"`, `"rule_switcher"`, `"model_switcher"`.
    """
    if controller_type not in ("fixed", "change_aware", "rule_switcher", "model_switcher"):
        raise ValueError(f"Invalid controller_type: {controller_type}")

    regime = str(trace_df["regime"].iloc[0])
    delay_condition = str(trace_df["delay_condition"].iloc[0])
    seed = int(trace_df["seed"].iloc[0])
    fixed_int = float(fixed_interval_s if fixed_interval_s is not None else protocol.default_fixed_interval_s)

    run_id = f"{run_id_prefix}_{controller_type}_{model_name}_{regime}_{delay_condition}_s{seed}_fi{fixed_int:.1f}"

    timestamps = trace_df["timestamp_s"].to_numpy(dtype=np.float64)
    delays_s = trace_df["delivery_delay_ms"].to_numpy(dtype=np.float64) / 1000.0
    change_events = trace_df["is_change_event"].to_numpy(dtype=np.int32)
    true_utils = trace_df[[f"u_{lname}" for lname in LINK_NAMES]].to_numpy(dtype=np.float64)
    num_ticks = len(timestamps)

    # Initialize policies (shared instances retained across switches per §2.3)
    policies: Dict[str, PollingPolicy] = {
        "fixed": FixedIntervalPolicy(interval_s=fixed_int),
        "change-aware": ChangeAwarePolicy(protocol=protocol),
    }

    if controller_type == "fixed":
        active_policy_name = "fixed"
    elif controller_type == "change_aware":
        active_policy_name = "change-aware"
    else:
        # Both switchers start in fixed-interval policy at t=0
        active_policy_name = "fixed"

    rule_selector = DeterministicRuleSelector(protocol=protocol)

    # Initialize Twin with t=0 state
    twin = NetworkDigitalTwin(initial_utilizations=true_utils[0])

    # In-flight poll queue: list of PollMeasurement sorted/checked by arrival_time_s
    in_flight_polls: List[PollMeasurement] = []
    last_poll_dispatch_time_s: float = -1e9
    next_scheduled_poll_time_s: float = 0.0  # First poll at t=0.0

    # Review schedule for switching controllers
    next_review_time_s: float = protocol.review_period_s
    review_idx: int = 0
    pending_decision: Optional[PendingModelDecision] = None
    skipped_reviews_backlog: int = 0
    switch_count: int = 0
    fallback_count: int = 0
    review_logs: List[ReviewLogRecord] = []

    # Evaluation accumulators
    post_warmup_abs_errors: List[np.ndarray] = []
    post_warmup_sq_errors: List[np.ndarray] = []
    aoi_samples: List[float] = []
    active_policy_ticks: Dict[str, int] = {"fixed": 0, "change-aware": 0}

    # Change-detection delay tracking (§2.4 supporting analysis: how long scheduled changes take to appear in twin)
    pending_change_event_times: List[Tuple[float, np.ndarray]] = []
    detection_delays_s: List[float] = []

    ts_records: List[Dict[str, Any]] = []

    for tick in range(num_ticks):
        t_now = round(float(timestamps[tick]), 6)
        u_true = true_utils[tick]

        # Record if a scheduled change event starts at this tick (post warm-up)
        if change_events[tick] == 1 and t_now >= protocol.warmup_duration_s:
            pending_change_event_times.append((t_now, u_true.copy()))

        # 1. Deliver any in-flight poll replies whose arrival_time_s <= t_now
        remaining_polls: List[PollMeasurement] = []
        arriving_now: List[PollMeasurement] = []
        for pm in in_flight_polls:
            if pm.arrival_time_s <= t_now + 1e-9:
                arriving_now.append(pm)
            else:
                remaining_polls.append(pm)
        in_flight_polls = remaining_polls

        # Process arrivals in arrival_time order
        arriving_now.sort(key=lambda x: x.arrival_time_s)
        for pm in arriving_now:
            accepted = twin.receive_measurement(pm)
            if accepted:
                # Update change-aware policy state whenever a new measurement is accepted
                policies["change-aware"].on_new_accepted_observation(twin.accepted_history[-1])
                # Reschedule next poll if change-aware is currently active (anchored to last_poll_dispatch_time_s)
                if active_policy_name == "change-aware":
                    next_scheduled_poll_time_s = policies["change-aware"].next_poll_time_s(
                        last_poll_dispatch_time_s
                    )
                # Resolve any pending change-detection events if this poll was dispatched at/after the change event
                still_pending_events: List[Tuple[float, np.ndarray]] = []
                for ev_time, ev_u in pending_change_event_times:
                    if pm.dispatch_time_s >= ev_time - 1e-9:
                        detection_delays_s.append(max(0.0, t_now - ev_time))
                    else:
                        still_pending_events.append((ev_time, ev_u))
                pending_change_event_times = still_pending_events

        # 2. Check if a pending selector decision arrives or times out by t_now (§2.3)
        if pending_decision is not None:
            res = pending_decision.decision_result
            # Determine when the controller learns the outcome (either arrival of reply or deadline timeout)
            resolution_time_s = min(pending_decision.effective_time_s, pending_decision.deadline_time_s)
            if t_now >= resolution_time_s - 1e-9:
                prev_pol = active_policy_name
                is_fallback = False
                fallback_reason: Optional[str] = None

                if pending_decision.effective_time_s > pending_decision.deadline_time_s + 1e-9:
                    is_fallback = True
                    fallback_reason = (
                        f"deadline_exceeded ({res.decision_latency_ms:.1f}ms > {protocol.decision_deadline_ms:.1f}ms)"
                    )
                elif not res.is_valid or res.selected_policy not in ("fixed", "change-aware"):
                    is_fallback = True
                    fallback_reason = res.error_reason or f"invalid_choice ({res.selected_policy!r})"

                if is_fallback:
                    fallback_count += 1
                    new_pol = prev_pol  # Retain previous policy on fallback (§2.3)
                else:
                    new_pol = str(res.selected_policy)

                switched = new_pol != prev_pol
                if switched:
                    switch_count += 1
                    active_policy_name = new_pol
                    # §2.3: "At a switch, the existing measurement history is retained, and the next poll is
                    # scheduled from the last poll time without an extra poll."
                    candidate_next = policies[active_policy_name].next_poll_time_s(
                        last_poll_dispatch_time_s
                    )
                    if candidate_next <= t_now + 1e-9:
                        candidate_next = round(t_now + policies[active_policy_name].current_interval_s(), 6)
                    next_scheduled_poll_time_s = candidate_next

                s_snap = pending_decision.summary_snapshot
                review_logs.append(
                    ReviewLogRecord(
                        run_id=run_id,
                        controller=controller_type,
                        model_name=model_name,
                        regime=regime,
                        delay_condition=delay_condition,
                        seed=seed,
                        review_idx=pending_decision.review_idx,
                        review_time_s=pending_decision.review_time_s,
                        effective_time_s=round(t_now, 4),
                        previous_policy=prev_pol,
                        raw_selected_policy=res.selected_policy,
                        effective_policy_after=active_policy_name,
                        policy_switched=switched,
                        decision_latency_ms=round(res.decision_latency_ms, 3),
                        deadline_ms=protocol.decision_deadline_ms,
                        is_fallback=is_fallback,
                        fallback_reason=fallback_reason,
                        prob_fixed=round(float(res.probabilities.get("fixed", 0.0)), 4),
                        prob_change_aware=round(float(res.probabilities.get("change-aware", 0.0)), 4),
                        confidence=round(float(res.confidence), 4),
                        request_bytes=res.request_bytes,
                        reply_bytes=res.reply_bytes,
                        summary_aoi_s=s_snap.measurement_age_aoi_s,
                        summary_recent_mean_change=s_snap.recent_mean_abs_change_pct,
                        summary_recent_max_change=s_snap.recent_max_abs_change_pct,
                        summary_polls_in_window=s_snap.polls_received_in_window,
                        prompt_text=s_snap.to_prompt_text(),
                        raw_output=res.raw_output,
                    )
                )
                pending_decision = None

        # 3. Check if a scheduled policy review occurs at t_now (for switching controllers)
        if controller_type in ("rule_switcher", "model_switcher") and t_now >= next_review_time_s - 1e-9:
            review_idx += 1
            next_review_time_s = round(next_review_time_s + protocol.review_period_s, 6)

            # §2.3: "Only one model request will be outstanding; a later review will not create a backlog."
            if pending_decision is not None:
                skipped_reviews_backlog += 1
            else:
                # Build leak-free summary strictly from twin observations received up to t_now
                summary = build_leak_free_review_summary(
                    twin=twin,
                    review_time_s=t_now,
                    current_active_policy=active_policy_name,
                    current_policy_interval_s=policies[active_policy_name].current_interval_s(),
                    review_window_s=protocol.review_period_s,
                )
                if controller_type == "rule_switcher":
                    dec_res = rule_selector.select_policy(summary)
                else:
                    if model_selector_fn is None:
                        raise ValueError("model_selector_fn must be provided when controller_type='model_switcher'")
                    dec_res = model_selector_fn(summary)

                eff_time_s = t_now + (dec_res.decision_latency_ms / 1000.0)
                deadline_time_s = t_now + (protocol.decision_deadline_ms / 1000.0)
                pending_decision = PendingModelDecision(
                    review_idx=review_idx,
                    review_time_s=t_now,
                    effective_time_s=eff_time_s,
                    deadline_time_s=deadline_time_s,
                    summary_snapshot=summary,
                    decision_result=dec_res,
                )

                # If decision latency is smaller than one simulation tick (e.g., rule_switcher or ultra-fast call),
                # apply immediately in this tick
                if eff_time_s <= t_now + (protocol.tick_dt_s * 0.5) and eff_time_s <= deadline_time_s:
                    prev_pol = active_policy_name
                    is_fb = not dec_res.is_valid or dec_res.selected_policy not in ("fixed", "change-aware")
                    fb_reason = dec_res.error_reason if is_fb else None
                    if is_fb:
                        fallback_count += 1
                        new_pol = prev_pol
                    else:
                        new_pol = str(dec_res.selected_policy)
                    switched = new_pol != prev_pol
                    if switched:
                        switch_count += 1
                        active_policy_name = new_pol
                        candidate_next = policies[active_policy_name].next_poll_time_s(
                            last_poll_dispatch_time_s
                        )
                        if candidate_next <= t_now + 1e-9:
                            candidate_next = round(t_now + policies[active_policy_name].current_interval_s(), 6)
                        next_scheduled_poll_time_s = candidate_next
                    review_logs.append(
                        ReviewLogRecord(
                            run_id=run_id,
                            controller=controller_type,
                            model_name=model_name,
                            regime=regime,
                            delay_condition=delay_condition,
                            seed=seed,
                            review_idx=review_idx,
                            review_time_s=t_now,
                            effective_time_s=round(t_now, 4),
                            previous_policy=prev_pol,
                            raw_selected_policy=dec_res.selected_policy,
                            effective_policy_after=active_policy_name,
                            policy_switched=switched,
                            decision_latency_ms=round(dec_res.decision_latency_ms, 3),
                            deadline_ms=protocol.decision_deadline_ms,
                            is_fallback=is_fb,
                            fallback_reason=fb_reason,
                            prob_fixed=round(float(dec_res.probabilities.get("fixed", 0.0)), 4),
                            prob_change_aware=round(float(dec_res.probabilities.get("change-aware", 0.0)), 4),
                            confidence=round(float(dec_res.confidence), 4),
                            request_bytes=dec_res.request_bytes,
                            reply_bytes=dec_res.reply_bytes,
                            summary_aoi_s=summary.measurement_age_aoi_s,
                            summary_recent_mean_change=summary.recent_mean_abs_change_pct,
                            summary_recent_max_change=summary.recent_max_abs_change_pct,
                            summary_polls_in_window=summary.polls_received_in_window,
                            prompt_text=summary.to_prompt_text(),
                            raw_output=dec_res.raw_output,
                        )
                    )
                    pending_decision = None

        # 4. Dispatch poll if t_now >= next_scheduled_poll_time_s
        dispatched_this_tick = False
        if t_now >= next_scheduled_poll_time_s - 1e-9:
            dispatched_this_tick = True
            twin.register_poll_dispatch(protocol.poll_request_bytes)
            poll_id = twin.total_polls_dispatched
            arrival_t = round(t_now + float(delays_s[tick]), 6)
            pm = PollMeasurement(
                poll_id=poll_id,
                dispatch_time_s=t_now,
                arrival_time_s=arrival_t,
                link_utilizations=u_true.copy(),
                active_policy_at_dispatch=active_policy_name,
                request_bytes=protocol.poll_request_bytes,
                reply_bytes=protocol.poll_reply_bytes,
            )
            in_flight_polls.append(pm)
            last_poll_dispatch_time_s = t_now
            next_scheduled_poll_time_s = policies[active_policy_name].next_poll_time_s(last_poll_dispatch_time_s)

        # 5. Evaluate Twin-State Error after common warm-up (§2.4)
        active_policy_ticks[active_policy_name] = active_policy_ticks.get(active_policy_name, 0) + 1
        abs_diff = np.abs(u_true - twin.stored_utilizations)
        aoi_now = twin.current_measurement_age_s(t_now)

        if t_now >= protocol.warmup_duration_s - 1e-9:
            post_warmup_abs_errors.append(abs_diff)
            post_warmup_sq_errors.append(abs_diff ** 2)
            aoi_samples.append(aoi_now)

        if record_timeseries:
            ts_records.append(
                {
                    "tick_idx": tick,
                    "timestamp_s": t_now,
                    "active_policy": active_policy_name,
                    "policy_interval_s": policies[active_policy_name].current_interval_s(),
                    "poll_dispatched": int(dispatched_this_tick),
                    "aoi_s": round(aoi_now, 4),
                    "mean_abs_error_pct": round(float(np.mean(abs_diff)), 4),
                    "max_abs_error_pct": round(float(np.max(abs_diff)), 4),
                    "true_u_L0": round(float(u_true[0]), 3),
                    "twin_u_L0": round(float(twin.stored_utilizations[0]), 3),
                }
            )

    # Aggregate evaluation metrics
    err_matrix = np.vstack(post_warmup_abs_errors)
    sq_matrix = np.vstack(post_warmup_sq_errors)
    mae_pct = float(np.mean(err_matrix))
    rmse_pct = float(np.sqrt(np.mean(sq_matrix)))
    p95_err = float(np.percentile(err_matrix, 95))

    mean_det_delay = float(np.mean(detection_delays_s)) if detection_delays_s else 0.0
    max_det_delay = float(np.max(detection_delays_s)) if detection_delays_s else 0.0
    mean_aoi = float(np.mean(aoi_samples)) if aoi_samples else 0.0

    total_pol_ticks = sum(active_policy_ticks.values()) or 1
    fixed_frac = active_policy_ticks["fixed"] / total_pol_ticks
    ca_frac = active_policy_ticks["change-aware"] / total_pol_ticks

    latencies = [r.decision_latency_ms for r in review_logs]
    req_bytes_model = sum(r.request_bytes for r in review_logs)
    rep_bytes_model = sum(r.reply_bytes for r in review_logs)
    num_reviews_completed = len(review_logs)
    fb_rate = (fallback_count / num_reviews_completed) if num_reviews_completed > 0 else 0.0

    return SimulationRunResult(
        run_id=run_id,
        controller=controller_type,
        model_name=model_name,
        fixed_interval_setting_s=fixed_int,
        regime=regime,
        delay_condition=delay_condition,
        seed=seed,
        trace_duration_s=protocol.trace_duration_s,
        warmup_duration_s=protocol.warmup_duration_s,
        mae_pct_points=round(mae_pct, 4),
        rmse_pct_points=round(rmse_pct, 4),
        p95_abs_error_pct=round(p95_err, 4),
        monitoring_bytes_total=twin.total_monitoring_bytes,
        monitoring_request_bytes=twin.total_monitoring_request_bytes,
        monitoring_reply_bytes=twin.total_monitoring_reply_bytes,
        poll_count=twin.total_polls_dispatched,
        accepted_poll_count=len(twin.accepted_history),
        rejected_out_of_order_count=twin.rejected_out_of_order_count,
        mean_detection_delay_s=round(mean_det_delay, 4),
        max_detection_delay_s=round(max_det_delay, 4),
        mean_aoi_s=round(mean_aoi, 4),
        review_count=num_reviews_completed,
        switch_count=switch_count,
        fallback_count=fallback_count,
        fallback_rate=round(fb_rate, 4),
        skipped_reviews_backlog=skipped_reviews_backlog,
        fixed_policy_fraction=round(fixed_frac, 4),
        change_aware_policy_fraction=round(ca_frac, 4),
        model_latency_mean_ms=round(float(np.mean(latencies)), 3) if latencies else 0.0,
        model_latency_p50_ms=round(float(np.median(latencies)), 3) if latencies else 0.0,
        model_latency_p95_ms=round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        model_latency_max_ms=round(float(np.max(latencies)), 3) if latencies else 0.0,
        model_request_bytes_total=req_bytes_model,
        model_reply_bytes_total=rep_bytes_model,
        review_logs=review_logs,
        timeseries_df=pd.DataFrame(ts_records) if record_timeseries else None,
    )
