"""Tests for the boundary-demo SASS scoreboard counter."""

from __future__ import annotations

import sass_scoreboard


def test_grouped_qgmma_runs_are_counted() -> None:
    text = """
        WARPGROUP.ARRIVE
        QGMMA.64x224x32 R0
        QGMMA.64x224x32 R1
        QGMMA.64x224x32 R2
        QGMMA.64x224x32 R3
        WARPGROUP.DEPBAR.LE gsb0, 0x0
        WARPGROUP.ARRIVE
        QGMMA.64x128x32 R4
        QGMMA.64x128x32 R5
        QGMMA.64x128x32 R6
        QGMMA.64x128x32 R7
        QGMMA.64x128x32 R8
        QGMMA.64x128x32 R9
        QGMMA.64x128x32 R10
        WARPGROUP.DEPBAR.LE gsb0, 0x1
        WARPGROUP.DEPBAR.LE gsb0, 0x0
    """

    summary = sass_scoreboard.summarize_sass(text)

    assert summary.qgmma == 11
    assert summary.qgmma_m64n224k32 == 4
    assert summary.qgmma_m64n128k32 == 7
    assert summary.warpgroup_arrive == 2
    assert summary.warpgroup_depbar == 3
    assert summary.qgmma_runs == (4, 7)
    assert not summary.looks_per_qgmma_scoreboard


def test_per_qgmma_scoreboard_is_flagged() -> None:
    text = "\n".join(
        [
            "WARPGROUP.ARRIVE",
            "QGMMA.64x128x32 R0",
            "WARPGROUP.DEPBAR.LE gsb0, 0x0",
            "WARPGROUP.ARRIVE",
            "QGMMA.64x128x32 R1",
            "WARPGROUP.DEPBAR.LE gsb0, 0x0",
        ]
    )

    summary = sass_scoreboard.summarize_sass(text)

    assert summary.qgmma == 2
    assert summary.warpgroup_depbar == 2
    assert summary.warpgroup_arrive == 2
    assert summary.qgmma_runs == (1, 1)
    assert summary.looks_per_qgmma_scoreboard
