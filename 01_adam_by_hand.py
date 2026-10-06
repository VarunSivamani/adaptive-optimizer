"""Task 1 — Reproduce Adam by hand for one weight and five gradients.

Session 11, Section 6 table: eta=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
w0 = 1.0, grads = [0.50, 0.40, 0.60, 0.45, 0.55].

Computes m, v, m_hat, v_hat and the step by hand, then checks every value
against torch.optim.Adam. Agreement should be to ~1e-7 (float32).
"""

from __future__ import annotations

import torch

LR: float = 0.001
BETA1: float = 0.9
BETA2: float = 0.999
EPS: float = 1e-8
W0: float = 1.0
GRADS: list[float] = [0.50, 0.40, 0.60, 0.45, 0.55]


def adam_by_hand(grads: list[float], w0: float, lr: float, b1: float, b2: float, eps: float) -> list[dict[str, float]]:
    """Run the Adam recurrence by hand; return one row dict per step."""
    m: float = 0.0
    v: float = 0.0
    w: float = w0
    rows: list[dict[str, float]] = []
    for t, g in enumerate(grads, start=1):
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        m_hat: float = m / (1.0 - b1**t)
        v_hat: float = v / (1.0 - b2**t)
        step: float = lr * m_hat / (v_hat**0.5 + eps)
        w = w - step
        rows.append({"t": float(t), "g": g, "m": m, "v": v, "m_hat": m_hat, "v_hat": v_hat, "step": step, "w": w})
    return rows


def adam_via_torch(grads: list[float], w0: float, lr: float, b1: float, b2: float, eps: float) -> list[dict[str, float]]:
    """Run torch.optim.Adam one manual gradient at a time; record state."""
    w = torch.tensor([w0], dtype=torch.float32)
    opt = torch.optim.Adam([w], lr=lr, betas=(b1, b2), eps=eps, weight_decay=0.0)
    rows: list[dict[str, float]] = []
    for t, g in enumerate(grads, start=1):
        if w.grad is not None:
            w.grad.zero_()
        w.grad = torch.tensor([g], dtype=torch.float32)
        w_before: float = w.item()
        opt.step()
        st = opt.state[w]
        m: float = float(st["exp_avg"].item())
        v: float = float(st["exp_avg_sq"].item())
        m_hat: float = m / (1.0 - b1**t)
        v_hat: float = v / (1.0 - b2**t)
        step: float = w_before - float(w.item())
        rows.append({"t": float(t), "g": g, "m": m, "v": v, "m_hat": m_hat, "v_hat": v_hat, "step": step, "w": float(w.item())})
    return rows


def main() -> None:
    hand = adam_by_hand(GRADS, W0, LR, BETA1, BETA2, EPS)
    ref = adam_via_torch(GRADS, W0, LR, BETA1, BETA2, EPS)

    print(f"{'t':>2} {'g':>6} {'m':>8} {'v':>11} {'m_hat':>8} {'v_hat':>8} {'step':>10} {'w':>9}")
    max_err: float = 0.0
    for h, r in zip(hand, ref):
        for k in ("m", "v", "m_hat", "v_hat", "step", "w"):
            max_err = max(max_err, abs(h[k] - r[k]))
        print(
            f"{h['t']:2.0f} {h['g']:6.2f} {h['m']:8.4f} {h['v']:11.6f} "
            f"{h['m_hat']:8.4f} {h['v_hat']:8.4f} {h['step']:10.6f} {h['w']:9.6f}"
        )

    print(f"\nMax |hand - torch| over m, v, m_hat, v_hat, step, w: {max_err:.3e}")
    assert max_err < 1e-6, f"hand vs torch disagree: {max_err:.3e}"
    print("OK: hand computation agrees with PyTorch to ~1e-7 (float32 rounding only).")
    print("Note: every step is within 1.2% of eta=0.001 despite grads spanning 0.40-0.60:")
    for h in hand:
        print(f"  t={h['t']:.0f}: step/eta = {h['step'] / LR:.4f}")


if __name__ == "__main__":
    main()
