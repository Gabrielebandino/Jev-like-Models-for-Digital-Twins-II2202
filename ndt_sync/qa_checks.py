"""Mandatory Quality Assurance (QA) & Methodological Invariant Checks.

Implements all six verification checks specified in Research Plan §2.4:
1. Constant-state input test (MAE = 0.0 after initial state synchronization)
2. Hand-worked error calculation test (exact analytical verification of MAE integration)
3. Policy switching without an extra poll test (schedules strictly from last_poll_dispatch_time)
4. Delayed and out-of-order reply rejection test (stale sample_time rejected by Twin)
5. Timeout and invalid-answer fallback test (retains previous policy and logs fallback)
6. Absence of future-state leakage test (selector summary sees zero unarrived or future data)
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .config import NUM_LINKS, LINK_NAMES, DEFAULT_PROTOCOL, ExperimentProtocol
from .twin import NetworkDigitalTwin, PollMeasurement
from .selectors import ReviewSummary, SelectorDecisionResult, build_leak_free_review_summary
from .simulator import run_trace_simulation


def _make_synthetic_test_trace(
    duration_s: float = 40.0,
    dt_s: float = 0.1,
    constant_val: float = 42.0,
    delay_ms: float = 100.0,
) -> pd.DataFrame:
    num_ticks = int(round(duration_s / dt_s)) + 1
    timestamps = np.round(np.arange(num_ticks) * dt_s, 4)
    data: Dict[str, Any] = {
        "tick_idx": np.arange(num_ticks, dtype=np.int32),
        "timestamp_s": timestamps,
        "regime": "stable",
        "delay_condition": "low_delay",
        "seed": 999,
        "active_phase": ["test"] * num_ticks,
        "is_change_event": np.zeros(num_ticks, dtype=np.int32),
        "delivery_delay_ms": np.full(num_ticks, delay_ms, dtype=np.float64),
    }
    for lname in LINK_NAMES:
        data[f"u_{lname}"] = np.full(num_ticks, constant_val, dtype=np.float64)
    return pd.DataFrame(data)


def test_1_constant_state_zero_error() -> Dict[str, Any]:
    """§2.4 Check 1: Under constant link utilization, twin-state error is identically 0.0."""
    df = _make_synthetic_test_trace(duration_s=50.0, constant_val=37.5, delay_ms=100.0)
    res_fixed = run_trace_simulation(df, controller_type="fixed")
    res_ca = run_trace_simulation(df, controller_type="change_aware")
    res_rule = run_trace_simulation(df, controller_type="rule_switcher")

    passed = (
        abs(res_fixed.mae_pct_points) < 1e-9
        and abs(res_ca.mae_pct_points) < 1e-9
        and abs(res_rule.mae_pct_points) < 1e-9
    )
    return {
        "check_id": 1,
        "name": "constant_state_zero_error",
        "passed": bool(passed),
        "details": {
            "fixed_mae": res_fixed.mae_pct_points,
            "change_aware_mae": res_ca.mae_pct_points,
            "rule_switcher_mae": res_rule.mae_pct_points,
        },
    }


def test_2_hand_worked_error_calculation() -> Dict[str, Any]:
    """§2.4 Check 2: Verify simulator MAE against an exact analytical step-function calculation."""
    # Construct a 30s trace with warmup=20s (so evaluation window is [20.0s, 30.0s] -> 101 ticks: 20.0..30.0)
    # Fixed interval = 5.0s, delivery_delay = 0.0ms (or 500ms exact).
    # Let link utilization be 20.0% for t < 22.0s, and jump to 50.0% (delta = +30.0%) at t = 22.0s.
    # Polls dispatch at t = 0, 5, 10, 15, 20, 25, 30s with delivery delay = 0.5s (5 ticks).
    # Therefore:
    # - From t=20.0s to t=21.9s (20 ticks): true=20, twin=20 -> error = 0.0
    # - From t=22.0s to t=25.4s (35 ticks, since poll at t=25.0s arrives at t=25.5s): true=50, twin=20 -> error = 30.0
    # - At t=25.5s to t=30.0s (46 ticks): poll dispatched at t=25.0s sampled true=50.0 and arrived at t=25.5s -> twin=50, error = 0.0
    # Total ticks in [20.0, 30.0] = 20 + 35 + 46 = 101 ticks.
    # Exact analytical MAE = (35 * 30.0) / 101 = 1050.0 / 101 = 10.3960396...
    df = _make_synthetic_test_trace(duration_s=30.0, dt_s=0.1, constant_val=20.0, delay_ms=500.0)
    for lname in LINK_NAMES:
        df.loc[df["timestamp_s"] >= 22.0 - 1e-9, f"u_{lname}"] = 50.0

    proto = ExperimentProtocol(
        trace_duration_s=30.0,
        warmup_duration_s=20.0,
        tick_dt_s=0.1,
        default_fixed_interval_s=5.0,
    )
    res = run_trace_simulation(df, controller_type="fixed", protocol=proto, fixed_interval_s=5.0)
    analytical_mae = round((35.0 * 30.0) / 101.0, 4)
    passed = abs(res.mae_pct_points - analytical_mae) < 1e-4
    return {
        "check_id": 2,
        "name": "hand_worked_analytical_mae",
        "passed": bool(passed),
        "details": {
            "simulator_mae": res.mae_pct_points,
            "analytical_expected_mae": analytical_mae,
            "abs_diff": round(abs(res.mae_pct_points - analytical_mae), 6),
        },
    }


def test_3_policy_switch_without_extra_poll() -> Dict[str, Any]:
    """§2.4 Check 3: Switching policies retains measurement history and does not trigger an extra poll."""
    df = _make_synthetic_test_trace(duration_s=45.0, dt_s=0.1, constant_val=25.0, delay_ms=100.0)

    # Custom selector that alternates between change-aware and fixed at t=15s and t=30s
    call_counter = {"n": 0}

    def alternating_selector(summary: ReviewSummary) -> SelectorDecisionResult:
        call_counter["n"] += 1
        choice = "change-aware" if call_counter["n"] % 2 == 1 else "fixed"
        return SelectorDecisionResult(
            selected_policy=choice,
            decision_latency_ms=200.0,  # Arrives at t=15.2s and t=30.2s
            probabilities={"fixed": 0.2, "change-aware": 0.8},
            confidence=0.8,
            request_bytes=120,
            reply_bytes=64,
            raw_output=choice,
            is_valid=True,
        )

    res = run_trace_simulation(
        df,
        controller_type="model_switcher",
        model_selector_fn=alternating_selector,
        model_name="test_switcher",
        record_timeseries=True,
    )
    ts = res.timeseries_df
    assert ts is not None
    # Decision 1 arrives at t=15.2s (last poll was at t=15.0s). No poll should fire at t=15.2s!
    poll_at_15_2 = int(ts.loc[np.isclose(ts["timestamp_s"], 15.2), "poll_dispatched"].iloc[0])
    poll_at_30_2 = int(ts.loc[np.isclose(ts["timestamp_s"], 30.2), "poll_dispatched"].iloc[0])
    passed = (res.switch_count == 2) and (poll_at_15_2 == 0) and (poll_at_30_2 == 0)
    return {
        "check_id": 3,
        "name": "policy_switching_without_extra_poll",
        "passed": bool(passed),
        "details": {
            "switch_count": res.switch_count,
            "spurious_poll_at_switch_1_t15_2s": poll_at_15_2,
            "spurious_poll_at_switch_2_t30_2s": poll_at_30_2,
        },
    }


def test_4_delayed_and_out_of_order_rejection() -> Dict[str, Any]:
    """§2.4 Check 4: Twin rejects a delayed reply whose sample timestamp is older than latest accepted."""
    twin = NetworkDigitalTwin(initial_utilizations=np.full(NUM_LINKS, 10.0))
    # Poll 1 dispatched at t=1.0s, severely delayed until t=4.0s (sampled 20.0%)
    pm1 = PollMeasurement(
        poll_id=1,
        dispatch_time_s=1.0,
        arrival_time_s=4.0,
        link_utilizations=np.full(NUM_LINKS, 20.0),
        active_policy_at_dispatch="fixed",
        request_bytes=40,
        reply_bytes=108,
    )
    # Poll 2 dispatched at t=2.0s, arrives early at t=2.2s (sampled 55.0%)
    pm2 = PollMeasurement(
        poll_id=2,
        dispatch_time_s=2.0,
        arrival_time_s=2.2,
        link_utilizations=np.full(NUM_LINKS, 55.0),
        active_policy_at_dispatch="fixed",
        request_bytes=40,
        reply_bytes=108,
    )
    accepted_2 = twin.receive_measurement(pm2)  # Arrives at t=2.2s -> accepted
    accepted_1 = twin.receive_measurement(pm1)  # Arrives late at t=4.0s -> must be rejected!

    passed = (
        accepted_2 is True
        and accepted_1 is False
        and twin.rejected_out_of_order_count == 1
        and float(twin.stored_utilizations[0]) == 55.0
        and twin.latest_accepted_sample_time_s == 2.0
    )
    return {
        "check_id": 4,
        "name": "delayed_and_out_of_order_reply_rejection",
        "passed": bool(passed),
        "details": {
            "poll2_accepted": accepted_2,
            "late_poll1_accepted": accepted_1,
            "rejected_out_of_order_count": twin.rejected_out_of_order_count,
            "stored_utilization_after_late_arrival": float(twin.stored_utilizations[0]),
        },
    }


def test_5_timeout_and_invalid_output_fallback() -> Dict[str, Any]:
    """§2.4 Check 5: Model decision exceeding deadline or returning invalid output triggers fallback and retains policy."""
    df = _make_synthetic_test_trace(duration_s=50.0, dt_s=0.1, constant_val=30.0, delay_ms=50.0)
    call_idx = {"n": 0}

    def faulty_selector(summary: ReviewSummary) -> SelectorDecisionResult:
        call_idx["n"] += 1
        if call_idx["n"] == 1:
            # Exceeds 2000ms deadline (3500ms latency)
            return SelectorDecisionResult(
                selected_policy="change-aware",
                decision_latency_ms=3500.0,
                probabilities={"fixed": 0.1, "change-aware": 0.9},
                confidence=0.9,
                request_bytes=100,
                reply_bytes=50,
                raw_output="late_response",
                is_valid=True,
            )
        else:
            # Invalid unparseable option
            return SelectorDecisionResult(
                selected_policy=None,
                decision_latency_ms=80.0,
                probabilities={},
                confidence=0.0,
                request_bytes=100,
                reply_bytes=20,
                raw_output="malformed_json",
                is_valid=False,
                error_reason="parse_error",
            )

    res = run_trace_simulation(
        df,
        controller_type="model_switcher",
        model_selector_fn=faulty_selector,
        model_name="faulty_test",
    )
    passed = (
        res.review_count == 3
        and res.fallback_count == 3
        and res.fallback_rate == 1.0
        and res.switch_count == 0
        and res.fixed_policy_fraction == 1.0
    )
    return {
        "check_id": 5,
        "name": "timeout_and_invalid_output_fallback",
        "passed": bool(passed),
        "details": {
            "review_count": res.review_count,
            "fallback_count": res.fallback_count,
            "fallback_rate": res.fallback_rate,
            "switch_count": res.switch_count,
            "retained_fixed_policy_fraction": res.fixed_policy_fraction,
        },
    }


def test_6_absence_of_future_state_leakage() -> Dict[str, Any]:
    """§2.4 Check 6: Selector summary at review time `t_r` cannot see unarrived polls or future trace state."""
    twin = NetworkDigitalTwin(initial_utilizations=np.full(NUM_LINKS, 15.0))
    # Accepted observation 1: sampled t=5.0s, arrived t=5.1s (value 25.0%)
    twin.receive_measurement(
        PollMeasurement(
            poll_id=1,
            dispatch_time_s=5.0,
            arrival_time_s=5.1,
            link_utilizations=np.full(NUM_LINKS, 25.0),
            active_policy_at_dispatch="fixed",
            request_bytes=40,
            reply_bytes=108,
        )
    )
    # Observation 2: sampled t=14.8s, arrives at t=15.3s (AFTER review at t=15.0s!) with spike value 95.0%
    twin.receive_measurement(
        PollMeasurement(
            poll_id=2,
            dispatch_time_s=14.8,
            arrival_time_s=15.3,
            link_utilizations=np.full(NUM_LINKS, 95.0),
            active_policy_at_dispatch="fixed",
            request_bytes=40,
            reply_bytes=108,
        )
    )
    # Build summary at t_review = 15.0s -> must only see Observation 1 (arrival 5.1s <= 15.0s), NOT Observation 2!
    summary_at_15 = build_leak_free_review_summary(
        twin=twin,
        review_time_s=15.0,
        current_active_policy="fixed",
        current_policy_interval_s=2.5,
    )
    passed = (
        summary_at_15.total_polls_received == 1
        and summary_at_15.latest_sample_time_s == 5.0
        and summary_at_15.per_link_latest_utilization_pct["L0"] == 25.0
    )
    return {
        "check_id": 6,
        "name": "absence_of_future_state_leakage",
        "passed": bool(passed),
        "details": {
            "polls_visible_at_t15": summary_at_15.total_polls_received,
            "latest_visible_sample_time_s": summary_at_15.latest_sample_time_s,
            "visible_L0_utilization_pct": summary_at_15.per_link_latest_utilization_pct["L0"],
            "unarrived_future_L0_utilization_pct": 95.0,
        },
    }


def run_all_qa_checks(save_report_path: str | None = None) -> Dict[str, Any]:
    """Execute all 6 mandatory Research Plan §2.4 verification checks and save report."""
    checks: List[Dict[str, Any]] = [
        test_1_constant_state_zero_error(),
        test_2_hand_worked_error_calculation(),
        test_3_policy_switch_without_extra_poll(),
        test_4_delayed_and_out_of_order_rejection(),
        test_5_timeout_and_invalid_output_fallback(),
        test_6_absence_of_future_state_leakage(),
    ]
    all_passed = all(c["passed"] for c in checks)
    report = {
        "suite": "II2202_Research_Plan_Section_2_4_QA_Verification",
        "all_passed": all_passed,
        "num_checks": len(checks),
        "num_passed": sum(1 for c in checks if c["passed"]),
        "checks": checks,
    }
    if save_report_path:
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return report
