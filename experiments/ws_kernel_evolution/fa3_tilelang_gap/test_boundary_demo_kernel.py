"""Correctness gate for the FP8 GQA TileLang boundary-demo kernel."""

from __future__ import annotations

import pytest

from boundary_demo_kernel import BoundaryDemoConfig, has_hopper_fp8, run_correctness


@pytest.mark.skipif(not has_hopper_fp8(), reason="requires Hopper CUDA + torch fp8")
def test_boundary_demo_candidate_matches_serial_tma_v_baseline() -> None:
    result = run_correctness(BoundaryDemoConfig())
    assert result.out_max_abs <= 1e-3
    assert result.lse_max_abs <= 1e-5
