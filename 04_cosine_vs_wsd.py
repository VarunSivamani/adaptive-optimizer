"""Task 4 — Same model twice for 300 steps: cosine vs WSD, stopped at step 200.

Toy MLP regression, AdamW, identical init/batches; only the LR schedule differs.
Cosine: peak -> ~0 over exactly 300 steps (must know the budget up front).
WSD: flat peak, linear decay over the final 30 steps (last 10%).
Reports loss at step 200 (early stop), at step 300 (full run), and after a
20-step cooldown branch taken from each step-200 checkpoint.
Saves outputs/04_cosine_vs_wsd.png.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

SEED: int = 0
DIM_IN: int = 32
HIDDEN: int = 128
N_DATA: int = 4096
BATCH: int = 256
TOTAL: int = 300
EARLY: int = 200
WARMUP: int = 10
DECAY_STEPS: int = 30  # WSD decay = last 10% of the run
COOLDOWN: int = 20
PEAK_LR: float = 3e-4
WD: float = 0.01


def make_data(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N_DATA, DIM_IN, generator=g)
    w_true = torch.randn(DIM_IN, 1, generator=g) / DIM_IN**0.5
    y = (X @ w_true).squeeze(1) + 0.1 * torch.randn(N_DATA, generator=g)
    return X, y


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, HIDDEN)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x))).squeeze(1)


def lr_cosine(t: int) -> float:
    if t <= WARMUP:
        return PEAK_LR * t / WARMUP
    p = (t - WARMUP) / (TOTAL - WARMUP)
    return PEAK_LR * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))


def lr_wsd(t: int) -> float:
    if t <= WARMUP:
        return PEAK_LR * t / WARMUP
    if t > TOTAL - DECAY_STEPS:
        frac = (t - (TOTAL - DECAY_STEPS)) / DECAY_STEPS
        return PEAK_LR * (1.0 - frac)
    return PEAK_LR


def batches(seed: int, steps: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    out: list[torch.Tensor] = []
    idx = torch.randperm(N_DATA, generator=g)
    ptr = 0
    for _ in range(steps):
        if ptr + BATCH > N_DATA:
            idx = torch.randperm(N_DATA, generator=g)
            ptr = 0
        out.append(idx[ptr : ptr + BATCH])
        ptr += BATCH
    return out


def train(sched: str, batch_list: list[torch.Tensor], X: torch.Tensor, y: torch.Tensor) -> tuple[MLP, list[float], list[float]]:
    torch.manual_seed(SEED)
    model = MLP()
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, weight_decay=WD)
    fn = (lr_cosine if sched == "cosine" else lr_wsd)
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    lrs: list[float] = []
    for t in range(1, TOTAL + 1):
        lr = fn(t)
        for pg in opt.param_groups:
            pg["lr"] = lr
        lrs.append(lr)
        opt.zero_grad()
        loss = loss_fn(model(X[batch_list[t - 1]]), y[batch_list[t - 1]])
        loss.backward()
        opt.step()
        with torch.no_grad():
            losses.append(loss_fn(model(X), y).item())
    return model, losses, lrs


def cooldown(model: MLP, start_lr: float, X: torch.Tensor, y: torch.Tensor, seed: int) -> float:
    opt = torch.optim.AdamW(model.parameters(), lr=start_lr, weight_decay=WD)
    loss_fn = nn.MSELoss()
    bl = batches(seed, COOLDOWN)
    for k in range(1, COOLDOWN + 1):
        for pg in opt.param_groups:
            pg["lr"] = start_lr * (1.0 - k / COOLDOWN)
        opt.zero_grad()
        loss = loss_fn(model(X[bl[k - 1]]), y[bl[k - 1]])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return loss_fn(model(X), y).item()


def main() -> None:
    X, y = make_data(SEED)
    bl = batches(12345, TOTAL)
    _, cos_losses, cos_lrs = train("cosine", bl, X, y)
    _, wsd_losses, wsd_lrs = train("wsd", bl, X, y)

    print(f"Loss at step {EARLY}: cosine={cos_losses[EARLY - 1]:.4f}  WSD={wsd_losses[EARLY - 1]:.4f}")
    print(f"Loss at step {TOTAL}: cosine={cos_losses[-1]:.4f}  WSD={wsd_losses[-1]:.4f}")

    # Branch from each step-200 checkpoint with a 20-step cooldown.
    torch.manual_seed(SEED)
    m_cos = MLP()
    m_wsd = MLP()
    opt_c = torch.optim.AdamW(m_cos.parameters(), lr=PEAK_LR, weight_decay=WD)
    opt_w = torch.optim.AdamW(m_wsd.parameters(), lr=PEAK_LR, weight_decay=WD)
    loss_fn = nn.MSELoss()
    for t in range(1, EARLY + 1):
        for opt, fn in ((opt_c, lr_cosine), (opt_w, lr_wsd)):
            for pg in opt.param_groups:
                pg["lr"] = fn(t)
        for m, opt in ((m_cos, opt_c), (m_wsd, opt_w)):
            opt.zero_grad()
            loss = loss_fn(m(X[bl[t - 1]]), y[bl[t - 1]])
            loss.backward()
            opt.step()
    cos_branch = cooldown(m_cos, lr_cosine(EARLY), X, y, 999)
    wsd_branch = cooldown(m_wsd, lr_wsd(EARLY), X, y, 999)
    print(f"After 20-step cooldown from step-{EARLY} ckpt: cosine-branch={cos_branch:.4f}  WSD-branch={wsd_branch:.4f}")

    keep = "WSD" if wsd_branch <= cos_branch else "cosine"
    print(f"Keep the {keep} model: the WSD checkpoint is decayable to any budget, "
          f"while truncated cosine never finished its decay.")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.plot(cos_lrs, label="cosine (T=300)")
    ax1.plot(wsd_lrs, label="WSD (decay last 30)")
    ax1.axvline(EARLY, color="k", ls="--", lw=1, label="early stop (200)")
    ax1.set_ylabel("lr")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(cos_losses, label="cosine")
    ax2.plot(wsd_losses, label="WSD")
    ax2.axvline(EARLY, color="k", ls="--", lw=1)
    ax2.set_xlabel("step")
    ax2.set_ylabel("full-data MSE")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax1.set_title("Cosine vs WSD (same model, same batches, 300 steps)")
    fig.tight_layout()
    fig.savefig("outputs/04_cosine_vs_wsd.png", dpi=150)
    print("Saved outputs/04_cosine_vs_wsd.png")


if __name__ == "__main__":
    main()
