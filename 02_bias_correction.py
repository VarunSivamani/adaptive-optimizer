"""Task 2 — Disable bias correction; plot the first 20 steps both ways.

Constant gradient g=0.5, eta=0.001, beta1=0.9, beta2=0.999.
Compares the Adam step with bias correction against the raw m/sqrt(v) step,
then reports after how many steps the difference stops mattering at
10% / 5% / 1% relative-error thresholds.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LR: float = 0.001
BETA1: float = 0.9
BETA2: float = 0.999
EPS: float = 1e-8
G: float = 0.5
N_STEPS: int = 20


def steps(g: float, n: int, lr: float, b1: float, b2: float, eps: float) -> tuple[list[float], list[float]]:
    """Return (with_correction, without_correction) step sizes for n steps."""
    m: float = 0.0
    v: float = 0.0
    with_bc: list[float] = []
    without_bc: list[float] = []
    for t in range(1, n + 1):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        m_hat: float = m / (1.0 - b1**t)
        v_hat: float = v / (1.0 - b2**t)
        with_bc.append(lr * m_hat / (v_hat**0.5 + eps))
        without_bc.append(lr * m / (v**0.5 + eps))
    return with_bc, without_bc


def steps_to_agree(rel_tol: float, lr: float, b1: float, b2: float, eps: float, g: float = 0.5) -> int:
    """First step t where |without - with| / with <= rel_tol (cap 100k)."""
    m: float = 0.0
    v: float = 0.0
    for t in range(1, 100_001):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        with_bc: float = lr * (m / (1.0 - b1**t)) / (((v / (1.0 - b2**t)) ** 0.5) + eps)
        without_bc: float = lr * m / (v**0.5 + eps)
        if abs(without_bc - with_bc) / with_bc <= rel_tol:
            return t
    return -1


def main() -> None:
    with_bc, without_bc = steps(G, N_STEPS, LR, BETA1, BETA2, EPS)

    print(f"{'t':>3} {'with_bc':>10} {'without_bc':>12} {'ratio wo/w':>10}")
    for t, (a, b) in enumerate(zip(with_bc, without_bc), start=1):
        print(f"{t:3d} {a:10.6f} {b:12.6f} {b / a:10.3f}")
    print(f"\nAt t=1: without={without_bc[0] / LR:.2f}x eta, with={with_bc[0] / LR:.2f}x eta "
          f"(notes: 3.16x vs 1.00x).")

    for tol in (0.10, 0.05, 0.01):
        t = steps_to_agree(tol, LR, BETA1, BETA2, EPS)
        print(f"Difference < {tol:.0%} after step {t} (constant-gradient model).")

    ts = list(range(1, N_STEPS + 1))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(ts, [s / LR for s in with_bc], "o-", label="with bias correction")
    ax1.plot(ts, [s / LR for s in without_bc], "s-", label="without bias correction")
    ax1.axhline(1.0, color="k", ls=":", lw=1)
    ax1.set_ylabel("step size / eta")
    ax1.set_title("Adam step, first 20 steps (g=0.5 const, eta=0.001)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(ts, [b / a for a, b in zip(with_bc, without_bc)], "o-", color="C2")
    ax2.axhline(1.0, color="k", ls=":", lw=1)
    ax2.set_xlabel("step")
    ax2.set_ylabel("ratio without/with")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("outputs/02_bias_correction.png", dpi=150)
    print("Saved outputs/02_bias_correction.png")


if __name__ == "__main__":
    main()
