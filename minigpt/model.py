"""A decoder-only transformer (GPT) written from scratch in PyTorch.

Design choices, and why they are here:

* **Pre-norm blocks with RMSNorm** - cheaper than LayerNorm and numerically
  well behaved in float16/bfloat16, which matters on Apple Silicon (MPS).
* **Rotary position embeddings (RoPE)** - relative positions come for free and
  there is no learned position table to overfit on a tiny dataset.
* **Grouped-query attention (optional)** - fewer KV heads shrink the KV cache
  during generation; set ``n_kv_head`` to enable.
* **SwiGLU feed-forward** - consistently better than GELU-MLP at equal params.
* **``F.scaled_dot_product_attention``** - dispatches to the fastest kernel
  available on the device (including the MPS fused kernel).
* **Weight tying** - the embedding matrix doubles as the output projection,
  which removes ~1.5M parameters at ``n_embd=384, vocab=4096``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables of shape ``(seq_len, head_dim // 2)``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, n_head, T, head_dim)`` by the given angles."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.to(x.dtype)[None, None, :, :]
    sin = sin.to(x.dtype)[None, None, :, :]
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.stack((out1, out2), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = cfg.dropout

        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k.numel() > 0:
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        if self.n_rep > 1:  # grouped-query attention: broadcast KV heads
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.size(2)
        if T == S:
            # Training / first forward pass: plain causal mask.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=T > 1, dropout_p=self.dropout if self.training else 0.0
            )
        elif T == 1:
            # Single-token decode step: every cached position is visible.
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            # Prefill on top of an existing cache: build the offset causal mask.
            idx_q = torch.arange(S - T, S, device=x.device).unsqueeze(1)
            idx_k = torch.arange(S, device=x.device).unsqueeze(0)
            mask = idx_k <= idx_q
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = int(cfg.mlp_ratio * cfg.n_embd)
        hidden = 32 * ((hidden + 31) // 32)  # round up: nicer for GPU/MPS matmuls
        self.gate_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, kv_cache=None):
        h, new_cache = self.attn(self.attn_norm(x), cos, sin, kv_cache)
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE tables are deterministic, so they are recomputed per device and
        # kept in float32 - casting the model to fp16 must not blur the angles.
        self._rope: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}

        self.apply(self._init_weights)
        # Scaled init for the residual output projections (GPT-2 trick): keeps
        # the variance of the residual stream constant as depth grows.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def rope_tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._rope.get(device)
        if cached is None:
            cached = build_rope_cache(self.cfg.block_size, self.cfg.head_dim, self.cfg.rope_theta, device)
            self._rope[device] = cached
        return cached

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        pos_offset: int = 0,
        loss_only: bool = False,
    ):
        """Run the model.

        Args:
            idx: ``(B, T)`` int64 token ids.
            targets: ``(B, T)`` next-token labels; ``-100`` positions are ignored.
            kv_caches: per-layer ``(k, v)`` tensors for incremental decoding.
            pos_offset: index of the first token of ``idx`` in the full sequence
                (non-zero when continuing from a cache), used to slice RoPE.
            loss_only: with ``targets``, return ``logits=None`` and project only
                the supervised positions through ``lm_head``. During SFT most
                positions are prompt or padding and carry ``-100``, so this skips
                the majority of the widest matmul in the model.

        Returns:
            ``(logits, loss, new_kv_caches)``.
        """
        B, T = idx.shape
        end = pos_offset + T
        if end > self.cfg.block_size:
            raise ValueError(f"sequence length {end} exceeds block_size {self.cfg.block_size}")
        cos_all, sin_all = self.rope_tables(idx.device)
        cos = cos_all[pos_offset:end]
        sin = sin_all[pos_offset:end]

        x = self.drop(self.tok_emb(idx))
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            x, nc = block(x, cos, sin, cache)
            if nc is not None:
                new_caches.append(nc)
        x = self.norm(x)

        if targets is not None and loss_only:
            flat_x = x.reshape(-1, x.size(-1))
            flat_t = targets.reshape(-1)
            keep = (flat_t != -100).nonzero(as_tuple=True)[0]
            logits = None
            if keep.numel() == 0:
                # Nothing supervised in this batch; keep the graph connected so
                # ``backward`` still works and contributes a zero gradient.
                loss = (flat_x.sum() * 0.0).to(torch.float32)
            else:
                loss = F.cross_entropy(
                    self.lm_head(flat_x.index_select(0, keep)), flat_t.index_select(0, keep)
                )
        elif targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        else:
            # Inference: only the last position is needed for the next token.
            logits = self.lm_head(x[:, -1:, :])
            loss = None
        return logits, loss, (new_caches if kv_caches is not None else None)

    def empty_cache(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Allocate empty per-layer KV caches to start incremental decoding."""
        n_kv = self.cfg.n_kv_head or self.cfg.n_head
        shape = (batch_size, n_kv, 0, self.cfg.head_dim)
        return [
            (torch.empty(shape, device=device, dtype=dtype), torch.empty(shape, device=device, dtype=dtype))
            for _ in range(self.cfg.n_layer)
        ]

    def configure_optimizer(self, lr: float, weight_decay: float, betas: tuple[float, float]):
        """AdamW with weight decay applied only to matrices, not to norms/biases."""
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        fused = torch.cuda.is_available()
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
