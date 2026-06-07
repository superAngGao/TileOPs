"""Lightweight tests for the FA3-vs-TileLang gap contract."""

from __future__ import annotations

import json

import fa3_tilelang_gap_contract as contract


def test_fa3_fifo_order_is_valid() -> None:
    contract.validate_steady_state()


def test_reversed_qk_pv_order_is_rejected() -> None:
    bad_steps = []
    for step in contract.FA3_STEADY_STATE:
        if step.name == "issue_qk":
            bad_steps.append(
                contract.SteadyStateStep(
                    2, step.name, step.intent, step.fa3_contract
                )
            )
        elif step.name == "issue_pv":
            bad_steps.append(
                contract.SteadyStateStep(
                    1, step.name, step.intent, step.fa3_contract
                )
            )
        else:
            bad_steps.append(step)

    try:
        contract.validate_steady_state(bad_steps)
    except AssertionError as exc:
        assert "QK -> PV -> wait<1>" in str(exc)
    else:
        raise AssertionError("reversed QK/PV order should be rejected")


def test_no_capability_is_marked_resolved_without_a_correctness_kernel() -> None:
    resolved = {item.name for item in contract.resolved_capabilities()}
    assert resolved == set()


def test_structural_blockers_remain_missing() -> None:
    blocked = {item.name for item in contract.blocked_capabilities()}
    assert "qk_acc_to_pv_areg_fragment" in blocked
    assert "exact_cute_smem_layout" in blocked


def test_json_renderer_is_parseable() -> None:
    payload = json.loads(contract.as_json())
    assert payload["resolved"] == []
    assert "qk_acc_to_pv_areg_fragment" in payload["blocked"]
    assert "persistent_torp_fragment" in payload["uncertain"]
