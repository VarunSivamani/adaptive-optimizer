"""Smoke tests for the S11 assignment (fast, self-contained, no long training)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name: str):  # type: ignore[no-untyped-def]
    path = ROOT / name
    spec = importlib.util.spec_from_file_location(path.stem.replace("-", "_"), path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_adam_step1_matches_session_table() -> None:
    m1 = load("01_adam_by_hand.py")
    rows = m1.adam_by_hand([0.5], 1.0, 0.001, 0.9, 0.999, 1e-8)
    r = rows[0]
    assert abs(r["m"] - 0.05) < 1e-12
    assert abs(r["v"] - 0.00025) < 1e-12
    assert abs(r["m_hat"] - 0.5) < 1e-12
    assert abs(r["v_hat"] - 0.25) < 1e-12
    assert abs(r["step"] - 0.001) < 1e-9
    assert abs(r["w"] - 0.999) < 1e-9


def test_bias_correction_first_step_ratio() -> None:
    m2 = load("02_bias_correction.py")
    with_bc, without_bc = m2.steps(0.5, 1, 0.001, 0.9, 0.999, 1e-8)
    assert abs(with_bc[0] / 0.001 - 1.0) < 1e-6
    assert abs(without_bc[0] / 0.001 - 3.1622776) < 1e-4


def test_schedules_endpoints() -> None:
    m4 = load("04_cosine_vs_wsd.py")
    assert abs(m4.lr_cosine(1) - 3e-4 * 1 / 10) < 1e-12
    assert abs(m4.lr_cosine(300)) < 1e-12
    assert abs(m4.lr_wsd(100) - 3e-4) < 1e-12
    assert abs(m4.lr_wsd(300)) < 1e-12
