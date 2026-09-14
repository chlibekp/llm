"""LoRA: fine-tune a frozen pretrained model through small low-rank adapters.

Instead of updating a weight matrix ``W`` (``out x in``), LoRA (Hu et al., 2021)
learns ``W + (alpha / r) * B @ A`` with ``A`` of shape ``r x in`` and ``B`` of
shape ``out x r``, for a rank ``r`` far below either dimension. ``B`` starts at
zero, so training begins from exactly the pretrained model.

Why this fits a laptop: the frozen weights need no gradients and no optimizer
state, and they can stay in half precision. For SmolLM2-360M at rank 16 on every
projection, 8.7M adapter parameters train while 362M sit still - AdamW state
drops from ~2.9 GB to ~70 MB.

A :class:`LoRALinear` *is* an ``nn.Linear`` (same ``weight``, same state_dict
key), plus ``lora_A`` / ``lora_B``. Wrapping a model therefore changes nothing
about how its base weights load, and :func:`merge_lora` folds the adapters back
into ``weight`` for inference at no extra cost per token.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
ALL_TARGETS = ATTN_TARGETS + ("gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        # Build an empty shell, then adopt the base layer's tensors rather than
        # copying them: the frozen weight is shared, not duplicated.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Adapters are trainable, so they live in float32 whatever the base dtype.
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features, device=device))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # same as nn.Linear's init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        # The adapter runs in the activations' dtype: casting the input up to
        # float32 would save a full-width float32 copy of it for backward, in
        # every adapted layer. Gradients still land on the float32 parameters.
        a = self.lora_A.to(x.dtype)
        b = self.lora_B.to(x.dtype)
        return out + F.linear(F.linear(self.lora_dropout(x), a), b) * self.scale

    def merged_weight(self) -> torch.Tensor:
        delta = (self.lora_B @ self.lora_A) * self.scale
        return self.weight + delta.to(self.weight.dtype)


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    targets: tuple[str, ...] = ALL_TARGETS,
) -> int:
    """Freeze ``model`` and wrap every ``nn.Linear`` named in ``targets``.

    Returns the number of trainable parameters. The settings are recorded on
    ``model.lora_config`` so a checkpoint can rebuild the same adapters.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in targets and type(child) is nn.Linear:
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                wrapped += 1
    if wrapped == 0:
        raise ValueError(f"no nn.Linear layers named {targets} to adapt")
    model.lora_config = {"rank": rank, "alpha": alpha, "dropout": dropout, "targets": list(targets)}
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Only the adapter tensors - a few tens of MB instead of the whole model."""
    return {k: v for k, v in model.state_dict().items() if ".lora_" in k}


def merge_lora(model: nn.Module) -> None:
    """Fold every adapter into its base weight and restore plain ``nn.Linear``s."""
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                plain = nn.Linear(child.in_features, child.out_features,
                                  bias=child.bias is not None, device="meta")
                with torch.no_grad():
                    plain.weight = nn.Parameter(child.merged_weight(), requires_grad=False)
                plain.bias = child.bias
                setattr(parent, name, plain)
    if hasattr(model, "lora_config"):
        del model.lora_config  # nothing left to save as an adapter
