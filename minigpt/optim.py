"""Muon: momentum SGD whose update is orthogonalised before it is applied.

Muon ("MomentUm Orthogonalized by Newton-schulz", Keller Jordan 2024) replaced
AdamW for the hidden matrices in the nanoGPT speedrun and reaches the same loss
in noticeably fewer tokens - which is exactly the budget a laptop is short of.

For every 2-D weight inside the transformer blocks it keeps one momentum buffer
(AdamW keeps two), and applies ``orthogonalize(momentum)``: the same singular
vectors, with every singular value pushed towards 1. A plain gradient step is
dominated by a few directions; the orthogonalised step moves all of them.

Embeddings, the output head and norm gains are not hidden matrices, so they
stay on AdamW. :class:`MuonAdamW` wraps both behind one optimizer, so the
training loop, the LR schedule and ``GradScaler`` see a single object.
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


@torch.no_grad()
def orthogonalize(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximately replace ``G`` by ``U V^T`` from its SVD.

    A quintic Newton-Schulz iteration with coefficients tuned to converge fast
    rather than exactly: singular values land roughly in [0.7, 1.2], which works
    as well as the exact polar factor at a fraction of the cost of an SVD.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tall = X.size(0) > X.size(1)
    if tall:  # iterate on the smaller Gram matrix
        X = X.mT
    X = X / (X.norm() + 1e-7)  # spectral norm <= Frobenius norm <= 1
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


class MuonAdamW(Optimizer):
    """Muon for groups flagged ``use_muon=True``, fused AdamW for the rest.

    Muon updates are scaled by ``0.2 * sqrt(max(rows, cols))`` so their RMS
    matches a typical AdamW update (Moonshot AI, "Muon is Scalable for LLM
    Training", 2025). That lets both halves share one learning rate and one
    schedule, and keeps ``--lr`` meaning the same thing for either optimizer.
    """

    def __init__(
        self,
        param_groups: list[dict],
        lr: float,
        betas: tuple[float, float] = (0.9, 0.95),
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, weight_decay=weight_decay, use_muon=False,
                        momentum=momentum, ns_steps=ns_steps)
        super().__init__(param_groups, defaults)
        for group in self.param_groups:
            if group["use_muon"] and any(p.dim() != 2 for p in group["params"]):
                raise ValueError("Muon only updates 2-D weight matrices")

        # The AdamW half is a real torch AdamW over the same tensors, so it keeps
        # the fused single-dispatch kernel on MPS/CUDA. Its groups mirror ours;
        # lr and weight decay are copied across on every step.
        self._adam_groups = [g for g in self.param_groups if not g["use_muon"]]
        adam_params = [{"params": g["params"]} for g in self._adam_groups]
        self._adam = None
        if adam_params:
            device = adam_params[0]["params"][0].device
            try:
                if device.type not in ("cuda", "mps"):
                    raise TypeError
                self._adam = torch.optim.AdamW(adam_params, lr=lr, betas=betas, fused=True)
            except (RuntimeError, TypeError):
                self._adam = torch.optim.AdamW(adam_params, lr=lr, betas=betas)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if not group["use_muon"]:
                continue
            lr, wd, mom = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.lerp_(p.grad, 1 - mom)
                update = p.grad.lerp(buf, mom)  # Nesterov look-ahead
                update = orthogonalize(update, group["ns_steps"]).to(p.dtype)
                scale = 0.2 * math.sqrt(max(p.size(0), p.size(1)))
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * scale)

        if self._adam is not None:
            for ours, theirs in zip(self._adam_groups, self._adam.param_groups):
                theirs["lr"] = ours["lr"]
                theirs["weight_decay"] = ours["weight_decay"]
            self._adam.step()
        return loss
