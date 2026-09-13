"""Saving and loading a model directory.

A checkpoint is a plain directory, easy to inspect and to copy::

    runs/my-model/
      config.json      model + training metadata
      model.pt         state_dict
      tokenizer.json   merges + control tokens
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPT
from .tokenizer import BPETokenizer


def resolve_device(name: str = "auto") -> torch.device:
    """Pick the best available backend. On a MacBook Air M2 this is ``mps``."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name != "auto":
        return {"float32": torch.float32, "fp32": torch.float32,
                "float16": torch.float16, "fp16": torch.float16,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[name]
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # MPS autocast is only reliable in float16, and for a model this small the
    # float32 path is fast enough while being noticeably more stable.
    return torch.float32


def save_checkpoint(
    out_dir: str | Path,
    model: GPT,
    tokenizer: BPETokenizer,
    meta: dict | None = None,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {k: v.to("cpu") for k, v in model.state_dict().items()}
    torch.save(state, out / "model.pt")
    tokenizer.save(out / "tokenizer.json")
    payload = {"model": model.cfg.to_dict(), "meta": meta or {}}
    (out / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_checkpoint(
    path: str | Path, device: str | torch.device = "auto", dtype: str = "auto"
) -> tuple[GPT, BPETokenizer, dict, torch.device]:
    """Load a checkpoint directory into an eval-mode model on ``device``."""
    d = Path(path)
    if not d.exists():
        raise SystemExit(f"model directory not found: {d}")
    for required in ("config.json", "model.pt", "tokenizer.json"):
        if not (d / required).exists():
            raise SystemExit(f"{d} is not a minigpt checkpoint (missing {required})")

    payload = json.loads((d / "config.json").read_text(encoding="utf-8"))
    cfg = GPTConfig.from_dict(payload["model"])
    dev = resolve_device(device) if isinstance(device, str) else device
    torch_dtype = resolve_dtype(dtype, dev)

    model = GPT(cfg)
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device=dev, dtype=torch_dtype)
    model.eval()
    tok = BPETokenizer.load(d / "tokenizer.json")
    return model, tok, payload.get("meta", {}), dev
