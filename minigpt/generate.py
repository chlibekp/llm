"""Token sampling with a KV cache.

``generate_stream`` yields tokens one at a time so the CLI and the streaming API
endpoint can share exactly the same decoding code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
import torch.nn.functional as F

from .model import GPT
from .tokenizer import BPETokenizer


@dataclass
class SamplingParams:
    """Decoding knobs. ``temperature=0`` means greedy (argmax) decoding."""

    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 40
    top_p: float = 0.95
    repetition_penalty: float = 1.1   # >1 discourages repeating earlier tokens
    presence_penalty: float = 0.0     # OpenAI-style, additive on logits
    frequency_penalty: float = 0.0    # OpenAI-style, scaled by token count
    seed: int | None = None


def _apply_penalties(logits: torch.Tensor, history: torch.Tensor, p: SamplingParams) -> torch.Tensor:
    if p.repetition_penalty == 1.0 and p.presence_penalty == 0.0 and p.frequency_penalty == 0.0:
        return logits
    counts = torch.bincount(history, minlength=logits.size(-1)).to(logits.device)
    seen = counts > 0
    if p.repetition_penalty != 1.0:
        # CTRL-style: divide positive logits, multiply negative ones.
        penalised = torch.where(logits > 0, logits / p.repetition_penalty, logits * p.repetition_penalty)
        logits = torch.where(seen, penalised, logits)
    if p.presence_penalty:
        logits = logits - p.presence_penalty * seen.float()
    if p.frequency_penalty:
        logits = logits - p.frequency_penalty * counts.float()
    return logits


def _filter(logits: torch.Tensor, top_k: int | None, top_p: float) -> torch.Tensor:
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # Drop tokens whose *preceding* cumulative mass already exceeds top_p.
        remove = (probs.cumsum(dim=-1) - probs) > top_p
        remove[..., 0] = False  # always keep the most likely token
        remove = torch.zeros_like(remove).scatter(-1, sorted_idx, remove)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


@torch.inference_mode()
def generate_stream(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt_ids: Sequence[int],
    params: SamplingParams,
    device: torch.device,
    stop_ids: Sequence[int] | None = None,
) -> Iterator[int]:
    """Yield generated token ids, stopping at EOS, a stop id, or the context limit."""
    model.eval()
    block = model.cfg.block_size
    ids = list(prompt_ids)[-block:]
    if not ids:
        raise ValueError("empty prompt")
    stop = set(stop_ids or ()) | {tokenizer.eos_id}

    gen = None
    if params.seed is not None:
        gen = torch.Generator(device="cpu").manual_seed(params.seed)

    dtype = next(model.parameters()).dtype
    caches = model.empty_cache(1, device, dtype)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    pos = 0
    history = torch.tensor(ids, dtype=torch.long, device=device)

    budget = min(params.max_new_tokens, block - len(ids))
    for _ in range(max(0, budget)):
        logits, _, caches = model(x, kv_caches=caches, pos_offset=pos)
        pos += x.size(1)
        logits = logits[0, -1].float()
        logits = _apply_penalties(logits, history, params)

        if params.temperature <= 0:
            next_id = int(torch.argmax(logits))
        else:
            logits = _filter(logits / params.temperature, params.top_k, params.top_p)
            probs = F.softmax(logits, dim=-1)
            if gen is not None:
                next_id = int(torch.multinomial(probs.cpu(), 1, generator=gen))
            else:
                next_id = int(torch.multinomial(probs, 1))

        if next_id in stop:
            return
        yield next_id
        history = torch.cat((history, torch.tensor([next_id], device=device)))
        x = torch.tensor([[next_id]], dtype=torch.long, device=device)


def generate_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    params: SamplingParams,
    device: torch.device,
) -> str:
    ids = tokenizer.encode(prompt)
    out = list(generate_stream(model, tokenizer, ids, params, device))
    return tokenizer.decode(out)


def stream_text(
    model: GPT,
    tokenizer: BPETokenizer,
    prompt: str,
    params: SamplingParams,
    device: torch.device,
) -> Iterator[str]:
    """Stream decoded text, buffering partial UTF-8 sequences until they are valid.

    A BPE token can end mid-codepoint (byte-level vocabulary), so decoding each
    token in isolation would emit replacement characters.
    """
    ids = tokenizer.encode(prompt)
    pending: list[int] = []
    for tid in generate_stream(model, tokenizer, ids, params, device):
        pending.append(tid)
        text = tokenizer.decode(pending)
        if "�" in text:   # incomplete multi-byte character; wait for more
            continue
        pending.clear()
        yield text
    if pending:
        yield tokenizer.decode(pending)
