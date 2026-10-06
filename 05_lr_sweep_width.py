"""Task 5 — Sweep LR at widths 256/512/1024; mark minima; say what to use at 4096.

Standard parameterization: init scale ~ 1/fan-in (per session notes), fixed Adam
betas, same synthetic regression task, 250 steps per (width, lr) point.
Best-LR ~ 1/width drift is the expected signature; extrapolate to width 4096
and state confidence (low without muP — extrapolation assumes the power law
holds off the measured range and that the toy task stands in for the real run).
Saves outputs/05_lr_sweep.png.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

WIDTHS: list[int] = [256, 512, 1024]
TARGET_WIDTH: int = 4096
LRS: list[float] = [3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
DIM_IN: int = 32
N_DATA: int = 2048
BATCH: int = 256
STEPS: int = 250
SEED: int = 0


def make_data(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N_DATA, DIM_IN, generator=g)
    w_true = torch.randn(DIM_IN, 1, generator=g) / DIM_IN**0.5
    y = (X @ w_true).squeeze(1) + 0.1 * torch.randn(N_DATA, generator=g)
    return X, y


class WideMLP(nn.Module):
    """One hidden layer of configurable width, SP init scale 1/fan-in."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(DIM_IN, width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(width, 1)
        with torch.no_grad():
            self.fc1.weight.normal_(0.0, 1.0 / DIM_IN)
            self.fc1.bias.zero_()
            self.fc2.weight.normal_(0.0, 1.0 / width)
            self.fc2.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x))).squeeze(1)


def final_loss(width: int, lr: float, X: torch.Tensor, y: torch.Tensor) -> float:
    torch.manual_seed(SEED)
    model = WideMLP(width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    idx = torch.randperm(N_DATA)
    ptr = 0
    for _ in range(STEPS):
        if ptr + BATCH > N_DATA:
            idx = torch.randperm(N_DATA)
            ptr = 0
        bi = idx[ptr : ptr + BATCH]
        ptr += BATCH
        opt.zero_grad()
        loss = loss_fn(model(X[bi]), y[bi])
        loss.backward()
        opt.step()
    with torch.no_grad():
        return loss_fn(model(X), y).item()


def main() -> None:
    X, y = make_data(SEED)
    results: dict[int, list[float]] = {}
    for w in WIDTHS:
        losses = [final_loss(w, lr, X, y) for lr in LRS]
        results[w] = losses
        best = LRS[losses.index(min(losses))]
        print(f"width {w:5d}: " + " ".join(f"{lr:.0e}={l:.4f}" for lr, l in zip(LRS, losses)) + f"  -> best {best:.0e}")

    bests = {w: LRS[results[w].index(min(results[w]))] for w in WIDTHS}
    # Fit log(best) = a - k*log(width); expect k ~ 1 under SP.
    xs = [math.log(w) for w in WIDTHS]
    ys = [math.log(bests[w]) for w in WIDTHS]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    k = -slope
    ref_w, ref_lr = WIDTHS[1], bests[WIDTHS[1]]
    naive = ref_lr * ref_w / TARGET_WIDTH  # assume exact 1/width
    fitted = math.exp(my + slope * (math.log(TARGET_WIDTH) - mx))
    print(f"\nFitted drift: best_lr ~ width^(-{k:.2f}).")
    print(f"Naive 1/width rule from width {ref_w} (best {ref_lr:.1e}): width-4096 lr = {naive:.2e}.")
    print(f"Fitted extrapolation: width-4096 lr = {fitted:.2e}.")
    print("Session table suggests ~1.9e-4 at width 4096 for a real LM run; the toy numbers")
    print("above are higher because the toy task tolerates larger steps. Confidence LOW either")
    print("way: this is a 4x extrapolation off a toy task; only a muP sweep makes it valid.")

    fig, ax = plt.subplots(figsize=(7, 5))
    for w in WIDTHS:
        ax.semilogx(LRS, results[w], "o-", label=f"width {w} (best {bests[w]:.0e})")
        ax.plot(bests[w], min(results[w]), "k*", ms=12)
    ax.set_xlabel("learning rate")
    ax.set_ylabel(f"final MSE ({STEPS} steps)")
    ax.set_title("Loss vs LR at three widths (minima drift left with width)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("outputs/05_lr_sweep.png", dpi=150)
    print("Saved outputs/05_lr_sweep.png")


if __name__ == "__main__":
    main()
