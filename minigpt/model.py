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
from torch.utils.checkpoint import checkpoint

from .config import GPTConfig

# torch >= 2.4 ships a fused RMS norm; torch >= 2.5 lets scaled_dot_product_attention
# broadcast KV heads itself. Both are probed once, at import, so the hot path stays
# free of version checks.
_HAS_F_RMS_NORM = hasattr(F, "rms_norm")

try:
    _q = torch.zeros(1, 2, 1, 8)
    _kv = torch.zeros(1, 1, 1, 8)
    F.scaled_dot_product_attention(_q, _kv, _kv, enable_gqa=True)
    _HAS_ENABLE_GQA = True
except (TypeError, RuntimeError):
    _HAS_ENABLE_GQA = False
finally:
    del _q, _kv


def autocast_dtype(device_type: str, default: torch.dtype) -> torch.dtype:
    """The dtype ops will actually run in, so cached tables can match it."""
    try:
        if torch.is_autocast_enabled(device_type):
            return torch.get_autocast_dtype(device_type)
    except TypeError:  # very old torch: single-argument form
        pass
    return default


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _HAS_F_RMS_NORM:
            # One fused kernel. The fallback below is the same arithmetic spelled
            # out as ~9 elementwise ops, each allocating a full (B, T, C) temporary
            # - which at this model size costs more than the maths it performs.
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x32 * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute cos/sin tables of shape ``(seq_len, head_dim // 2)``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, n_head, T, head_dim)`` by the given angles.

    ``interleaved`` selects which pairs of channels form a rotation plane:

    * ``True`` (default, and what every existing checkpoint was trained with)
      pairs neighbouring channels ``(0,1), (2,3), ...``. Reading those needs a
      strided, non-coalesced gather.
    * ``False`` pairs channel ``i`` with ``i + head_dim/2`` (the GPT-NeoX layout).
      Both halves are contiguous slices, so the reads coalesce. Equivalent in
      expressiveness, but *not* interchangeable: a model trained with one layout
      produces garbage under the other.
    """
    if cos.dtype != x.dtype:  # no-op on the cached-table fast path
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    if interleaved:
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = cfg.dropout
        self.rope_interleaved = cfg.rope_interleaved

        self.q_proj = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        # QK-norm bounds the attention logits, so a spike in one projection cannot
        # saturate the softmax. Lets small models train at a higher learning rate.
        self.q_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else None
        self.k_norm = RMSNorm(self.head_dim, cfg.norm_eps) if cfg.qk_norm else None

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

        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)
        q = apply_rope(q, cos, sin, self.rope_interleaved)
        k = apply_rope(k, cos, sin, self.rope_interleaved)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k.numel() > 0:
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
            new_cache = (k, v)
        else:
            new_cache = None

        gqa: dict = {}
        if self.n_rep > 1:  # grouped-query attention: broadcast KV heads
            if _HAS_ENABLE_GQA:
                gqa = {"enable_gqa": True}  # the kernel broadcasts; nothing is allocated
            else:
                k = k.repeat_interleave(self.n_rep, dim=1)
                v = v.repeat_interleave(self.n_rep, dim=1)

        S = k.size(2)
        if T == S:
            # Training / first forward pass: plain causal mask.
            #
            # Right-padding needs no mask of its own: the mask is causal and the
            # padding is a strict suffix, so no real token ever attends to a pad.
            # The pads' own outputs are discarded by the -100 labels.
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=T > 1,
                dropout_p=self.dropout if self.training else 0.0, **gqa,
            )
        elif T == 1:
            # Single-token decode step: every cached position is visible.
            y = F.scaled_dot_product_attention(q, k, v, **gqa)
        else:
            # Prefill on top of an existing cache: build the offset causal mask.
            idx_q = torch.arange(S - T, S, device=x.device).unsqueeze(1)
            idx_k = torch.arange(S, device=x.device).unsqueeze(0)
            mask = idx_k <= idx_q
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, **gqa)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.head_dim)
        return self.resid_dropout(self.o_proj(y)), new_cache


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.intermediate_size
        if hidden is None:
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
        self.attn_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
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
        self.norm = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # RoPE tables are deterministic, so they are built once per (device, dtype).
        # They are always *computed* in float32 - casting the model to fp16 must not
        # blur the angles - and only then cast, so the hot path never re-casts them.
        self._rope: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        # When set, training forwards keep only each block's input and recompute
        # its internals during backward (activation checkpointing).
        self.grad_checkpointing = False

        self.apply(self._init_weights)
        # Scaled init for the residual output projections (GPT-2 trick): keeps
        # the variance of the residual stream constant as depth grows.
        # With zero_init_proj every block starts as the identity instead (nanoGPT
        # speedrun): the residual stream is just the embedding at step 0.
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
                if cfg.zero_init_proj:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def rope_tables(
        self, device: torch.device, dtype: torch.dtype = torch.float32
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (device, dtype)
        cached = self._rope.get(key)
        if cached is None:
            cos, sin = build_rope_cache(
                self.cfg.block_size, self.cfg.head_dim, self.cfg.rope_theta, device
            )
            if dtype is not torch.float32:
                cos, sin = cos.to(dtype), sin.to(dtype)
            cached = (cos, sin)
            self._rope[key] = cached
        return cached

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        """Project to vocabulary logits, soft-capped when configured.

        The cap keeps a small model from pushing a few logits to extreme values,
        which otherwise shows up as over-confident, repetitive samples.
        """
        logits = self.lm_head(x)
        cap = self.cfg.logit_softcap
        if cap > 0:
            logits = cap * torch.tanh(logits / cap)
        return logits

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
        keep_index: torch.Tensor | None = None,
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
            keep_index: flat indices of the supervised positions. Computing them
                here means calling ``nonzero``, whose output shape depends on the
                data and therefore forces a device synchronisation on every step.
                The data loader already knows them, so it passes them in. Callers
                that have no index (pretraining, where every position is
                supervised) should use the dense loss instead.

        Returns:
            ``(logits, loss, new_kv_caches)``.
        """
        B, T = idx.shape
        end = pos_offset + T
        if end > self.cfg.block_size:
            raise ValueError(f"sequence length {end} exceeds block_size {self.cfg.block_size}")
        rope_dtype = autocast_dtype(idx.device.type, self.tok_emb.weight.dtype)
        cos_all, sin_all = self.rope_tables(idx.device, rope_dtype)
        cos = cos_all[pos_offset:end]
        sin = sin_all[pos_offset:end]

        x = self.drop(self.tok_emb(idx))
        new_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches is not None else None
            if self.grad_checkpointing and self.training and cache is None and torch.is_grad_enabled():
                x = checkpoint(lambda h, b=block: b(h, cos, sin)[0], x, use_reentrant=False)
                continue
            x, nc = block(x, cos, sin, cache)
            if nc is not None:
                new_caches.append(nc)
        x = self.norm(x)

        if targets is not None and loss_only:
            flat_x = x.reshape(-1, x.size(-1))
            flat_t = targets.reshape(-1)
            logits = None
            if keep_index is None:
                # Fallback for callers that have no precomputed index. This is the
                # synchronising path; the training loop does not take it.
                keep_index = (flat_t != -100).nonzero(as_tuple=True)[0]
                if keep_index.numel() == 0:
                    # Keep the graph connected so backward still contributes zero.
                    return None, (flat_x.sum() * 0.0).to(torch.float32), None
            # .float(): with half-precision weights (a frozen pretrained base) the
            # log-softmax over the vocabulary must still run in float32.
            loss = F.cross_entropy(
                self._head(flat_x.index_select(0, keep_index)).float(),
                flat_t.index_select(0, keep_index),
            )
        elif targets is not None:
            logits = self._head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100
            )
        else:
            # Inference: only the last position is needed for the next token.
            logits = self._head(x[:, -1:, :])
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

    def configure_optimizer(
        self, lr: float, weight_decay: float, betas: tuple[float, float], name: str = "adamw"
    ):
        """AdamW with weight decay applied only to matrices, not to norms/biases.

        ``name="muon"`` moves the 2-D weights inside the blocks to Muon; the
        embedding (also the tied output head) and the norm gains stay on AdamW.
        """
        if name == "muon":
            from .optim import MuonAdamW

            hidden = [p for p in self.blocks.parameters() if p.requires_grad and p.dim() == 2]
            hidden_ids = {id(p) for p in hidden}
            rest = [p for p in self.parameters() if p.requires_grad and id(p) not in hidden_ids]
            groups = [
                {"params": hidden, "use_muon": True, "weight_decay": weight_decay},
                {"params": [p for p in rest if p.dim() >= 2], "weight_decay": weight_decay},
                {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0},
            ]
            return MuonAdamW([g for g in groups if g["params"]], lr=lr, betas=betas)
        if name != "adamw":
            raise ValueError(f"unknown optimizer {name!r}")
        decay, no_decay = [], []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append(p)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        # The fused kernel updates every tensor in a handful of dispatches instead
        # of ~10 per parameter. MPS has one since torch 2.4; older builds reject
        # fused=True at construction, so fall back rather than fail.
        device = next(self.parameters()).device
        if device.type in ("cuda", "mps"):
            try:
                return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=True)
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
