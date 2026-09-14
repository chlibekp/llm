"""Model + training configuration objects."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class GPTConfig:
    """Architecture hyper-parameters.

    The defaults describe a ~11M parameter model that trains in minutes on an
    M2 MacBook Air and still produces coherent answers on a small Q&A set.
    """

    vocab_size: int = 4096
    block_size: int = 256        # maximum context length in tokens
    n_layer: int = 6
    n_head: int = 6              # query heads
    n_kv_head: int | None = None # key/value heads; None => same as n_head (plain MHA)
    n_embd: int = 384
    mlp_ratio: float = 8 / 3     # SwiGLU keeps ~2/3 of the usual 4x to match param count
    dropout: float = 0.1
    bias: bool = False           # biases are not needed with RMSNorm pre-norm blocks
    rope_theta: float = 10000.0
    rope_interleaved: bool = True  # False = contiguous halves: faster, but a
                                   # different convention, so it breaks old checkpoints
    tie_weights: bool = True     # share input embedding with the output projection
    qk_norm: bool = False        # RMSNorm queries and keys per head: stable at higher LR
    logit_softcap: float = 0.0   # cap * tanh(logits / cap); 0 disables
    zero_init_proj: bool = False # start residual output projections at zero

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# Ready-made sizes, selectable with `--size` on the CLI.
PRESETS: dict[str, dict] = {
    "tiny":   dict(n_layer=4, n_head=4, n_embd=256, block_size=256, vocab_size=2048),
    "small":  dict(n_layer=6, n_head=6, n_embd=384, block_size=256, vocab_size=4096),
    "medium": dict(n_layer=8, n_head=8, n_embd=512, block_size=512, vocab_size=8192),
    "large":  dict(n_layer=12, n_head=12, n_embd=768, block_size=512, vocab_size=16384),
    # Deep and thin (MobileLLM: depth beats width at this scale), with the
    # small-model stabilisers from the nanoGPT speedrun. New, so it can use the
    # faster contiguous RoPE layout without breaking any existing checkpoint.
    "compact": dict(n_layer=12, n_head=8, n_kv_head=2, n_embd=320, block_size=256,
                    vocab_size=8192, dropout=0.0, rope_interleaved=False,
                    qk_norm=True, logit_softcap=30.0, zero_init_proj=True),
}
