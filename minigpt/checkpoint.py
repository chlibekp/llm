"""Saving and loading a model directory.

A checkpoint is a plain directory, easy to inspect and to copy::

    runs/my-model/
      config.json      model + training metadata
      model.pt         state_dict
      tokenizer.json   merges + control tokens

A LoRA run on a pretrained base stores only what it trained::

    runs/my-lora/
      config.json      base model reference (repo + commit), LoRA settings, tokenizer ids
      adapter.pt       the LoRA matrices (tens of MB)
      tokenizer.json   the base model's tokenizer

Loading one reloads the pinned base from the Hugging Face cache, re-applies the
adapters and merges them into the weights, so decoding costs nothing extra.
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
    payload = {"model": model.cfg.to_dict(), "meta": meta or {}}
    lora = getattr(model, "lora_config", None)
    if lora is not None:
        from .lora import lora_state_dict

        torch.save({k: v.to("cpu") for k, v in lora_state_dict(model).items()}, out / "adapter.pt")
        payload.update(base=model.base_ref, lora=lora)
    else:
        state = {k: v.to("cpu") for k, v in model.state_dict().items()}
        torch.save(state, out / "model.pt")
    if hasattr(tokenizer, "meta"):
        payload["tokenizer"] = tokenizer.meta
    tokenizer.save(out / "tokenizer.json")
    (out / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def load_tokenizer(path: str | Path):
    """The tokenizer of a checkpoint directory, whichever kind it is."""
    from . import hf

    if hf.looks_like_pretrained(path):
        return hf.HFTokenizer.from_pretrained_dir(hf.resolve_files(str(path))[0])
    d = Path(path)
    payload = json.loads((d / "config.json").read_text(encoding="utf-8"))
    meta = payload.get("tokenizer")
    if meta and meta.get("type") == "hf":
        from .hf import HFTokenizer

        return HFTokenizer.load(d / "tokenizer.json", meta)
    return BPETokenizer.load(d / "tokenizer.json")


def load_checkpoint(
    path: str | Path, device: str | torch.device = "auto", dtype: str = "auto"
) -> tuple[GPT, BPETokenizer, dict, torch.device]:
    """Load a model into eval mode on ``device``.

    ``path`` is a minigpt checkpoint directory (from ``train``, ``pretrain`` or
    ``lora``), or a pretrained Hugging Face model: a hub id such as
    ``HuggingFaceTB/SmolLM2-360M-Instruct`` or a local directory holding one.
    """
    from . import hf

    dev = resolve_device(device) if isinstance(device, str) else device
    d = Path(path)
    if hf.looks_like_pretrained(path):
        torch_dtype = hf.default_dtype(dev) if dtype == "auto" else resolve_dtype(dtype, dev)
        model, tok, ref = hf.load_pretrained(str(path), torch_dtype)
        return model.to(dev).eval(), tok, {"task": "pretrained", "base": ref}, dev
    if not d.exists():
        raise SystemExit(f"model directory not found: {d}")
    if not (d / "config.json").exists():
        raise SystemExit(f"{d} is not a minigpt checkpoint (missing config.json)")
    payload = json.loads((d / "config.json").read_text(encoding="utf-8"))

    if "lora" in payload:
        from .lora import apply_lora, merge_lora

        if not (d / "adapter.pt").exists():
            raise SystemExit(f"{d} is not a minigpt checkpoint (missing adapter.pt)")
        base = payload["base"]
        torch_dtype = hf.default_dtype(dev) if dtype == "auto" else resolve_dtype(dtype, dev)
        model, _, _ = hf.load_pretrained(base["name"], torch_dtype, base.get("revision"))
        cfg = payload["lora"]
        apply_lora(model, cfg["rank"], cfg["alpha"], 0.0, tuple(cfg["targets"]))
        adapter = torch.load(d / "adapter.pt", map_location="cpu", weights_only=True)
        missing = set(k for k in model.state_dict() if ".lora_" in k) - set(adapter)
        if missing:
            raise SystemExit(f"adapter.pt is missing {len(missing)} LoRA tensors, e.g. {sorted(missing)[0]}")
        model.load_state_dict(adapter, strict=False)
        merge_lora(model)
        return model.to(dev).eval(), load_tokenizer(d), payload.get("meta", {}), dev

    for required in ("model.pt", "tokenizer.json"):
        if not (d / required).exists():
            raise SystemExit(f"{d} is not a minigpt checkpoint (missing {required})")
    cfg = GPTConfig.from_dict(payload["model"])
    torch_dtype = resolve_dtype(dtype, dev)

    model = GPT(cfg)
    state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device=dev, dtype=torch_dtype)
    model.eval()
    return model, load_tokenizer(d), payload.get("meta", {}), dev
