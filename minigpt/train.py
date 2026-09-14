"""Training loop: tokenizer training, supervised fine-tuning, optional pretraining.

Everything here is tuned for a single small device (Apple Silicon / CPU):

* AdamW with decoupled weight decay on matrices only.
* Linear warmup then cosine decay to ``min_lr``.
* Gradient accumulation so a large effective batch fits in unified memory.
* Gradient clipping at 1.0.
* Best-on-validation checkpointing, plus a final save if there is no val split.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import resolve_device, save_checkpoint
from .chat import iter_texts
from .config import PRESETS, GPTConfig
from .data import (
    ChatDataset,
    LengthGroupedSampler,
    PackedTextDataset,
    dynamic_collate,
    encode_corpus_to_file,
    iter_text_chunks,
    load_csv,
    train_val_split,
)
from .model import GPT
from .tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    epochs: int = 30
    batch_size: int = 64         # large enough to amortise the fixed per-step cost
    grad_accum: int = 1
    lr: float = 6e-4             # sqrt-scaled for the larger batch
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    eval_every: int = 0          # in steps; 0 => once per epoch
    log_every: int = 10
    seed: int = 1337
    device: str = "auto"
    compile: bool = False
    grad_checkpoint: bool = False  # recompute block activations in backward: less memory, ~30% slower
    amp: str = "auto"            # "auto" | "bf16" | "fp16" | "off"
    dynamic_padding: bool = True  # pad each batch to its own longest example
    pad_multiple: int = 8         # round that length up, to keep matmul shapes few
    num_workers: int = 0
    early_stop_patience: int = 0  # epochs without val improvement; 0 disables
    save: str = "last"            # "last" = final weights, "best" = lowest val loss
    extra: dict = field(default_factory=dict)


def lr_at(step: int, total: int, cfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay."""
    warmup = max(1, int(total * cfg.warmup_ratio))
    min_lr = cfg.lr * cfg.min_lr_ratio
    if step < warmup:
        return cfg.lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * progress))


def resolve_amp(name: str, device: torch.device) -> torch.dtype | None:
    """Pick the autocast dtype, or ``None`` to train in full float32.

    bfloat16 has the same exponent range as float32, so unlike float16 it needs
    no loss scaling - which is why it is the default wherever it is supported.
    """
    if name == "off":
        return None
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return None  # CPU autocast is usually slower than plain float32 here


def _build_loader(
    ds: Dataset, cfg: TrainConfig, device: torch.device, shuffle: bool
) -> tuple[DataLoader, LengthGroupedSampler | None]:
    """Wrap ``ds`` in a DataLoader, using dynamic padding where it applies."""
    kwargs = dict(
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = 2

    if cfg.dynamic_padding and isinstance(ds, ChatDataset):
        sampler = LengthGroupedSampler(
            ds.lengths, cfg.batch_size, shuffle=shuffle, seed=cfg.seed
        )
        loader = DataLoader(
            ds.raw_view(),
            batch_sampler=sampler,
            collate_fn=dynamic_collate(ds.pad_id, cfg.pad_multiple),
            **kwargs,
        )
        return loader, sampler

    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False, **kwargs
    )
    return loader, None


def _to_device(batch, device: torch.device):
    """Unpack a batch from either collate path onto ``device``.

    ``dynamic_collate`` yields ``(x, y, keep_index)``; the fixed-padding path
    (used by ``PackedTextDataset``) yields plain ``(x, y)``.
    """
    x, y, keep = batch if len(batch) == 3 else (batch[0], batch[1], None)
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if keep is not None:
        keep = keep.to(device, non_blocking=True)
    return x, y, keep


@torch.no_grad()
def evaluate(
    model: GPT,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    amp_dtype: torch.dtype | None = None,
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y, keep = _to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss, _ = model(x, targets=y, loss_only=True, keep_index=keep)
        total += loss.detach().float()
        n += 1
    model.train()
    return float(total) / max(1, n)


def train_model(
    model: GPT,
    tokenizer: BPETokenizer,
    train_ds: Dataset,
    val_ds: Dataset | None,
    cfg: TrainConfig,
    out_dir: str | Path,
    meta: dict | None = None,
) -> dict:
    """Run the optimisation loop and write checkpoints to ``out_dir``."""
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model.to(device)
    model.train()
    model.grad_checkpointing = cfg.grad_checkpoint

    train_loader, train_sampler = _build_loader(train_ds, cfg, device, shuffle=True)
    val_loader = (
        _build_loader(val_ds, cfg, device, shuffle=False)[0]
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = model.configure_optimizer(cfg.lr, cfg.weight_decay, cfg.betas)

    amp_dtype = resolve_amp(cfg.amp, device)
    # float16 silently underflows small gradients, so it needs a loss scaler.
    # bfloat16 and float32 do not.
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype is torch.float16)

    # Keep a handle on the uncompiled module: an OptimizedModule prefixes every
    # state_dict key with "_orig_mod.", which would write unloadable checkpoints.
    raw_model = model
    if cfg.compile and hasattr(torch, "compile") and device.type in ("cuda", "mps"):
        # dynamic=True: batches are padded to their own length, so the shapes vary
        # and a static compile would recompile for each one.
        model = torch.compile(model, dynamic=True)

    padding = "dynamic" if train_sampler is not None else "fixed"
    print(
        f"device={device.type} params={raw_model.num_parameters()/1e6:.2f}M "
        f"examples={len(train_ds)} steps/epoch={steps_per_epoch} total_steps={total_steps} "
        f"amp={amp_dtype and str(amp_dtype).removeprefix('torch.') or 'off'} padding={padding}"
    )

    history: list[dict] = []
    best_val = float("inf")
    best_step = 0
    stale_epochs = 0
    step = 0
    t0 = time.time()
    stop = False

    n_micro = len(train_loader)
    params = [p for p in raw_model.parameters() if p.requires_grad]
    # Loss is accumulated on-device and only read back when we actually log, so
    # the training step never blocks waiting for the accelerator to catch up.
    running = torch.zeros((), device=device)
    seen = 0

    for epoch in range(cfg.epochs):
        running.zero_()
        seen = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader):
            x, y, keep = _to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _, loss, _ = model(x, targets=y, loss_only=True, keep_index=keep)
            scaler.scale(loss / cfg.grad_accum).backward()
            running += loss.detach().float()
            seen += 1

            is_last = micro == n_micro - 1
            if (micro + 1) % cfg.grad_accum != 0 and not is_last:
                continue

            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                # foreach=None lets torch pick the fused path where it exists (CUDA) and
                # fall back elsewhere (MPS has no foreach kernels).
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip, foreach=None)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if cfg.log_every and step % cfg.log_every == 0:
                print(
                    f"epoch {epoch+1}/{cfg.epochs} step {step}/{total_steps} "
                    f"loss {float(running)/max(1,seen):.4f} lr {lr:.2e} "
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )
                running.zero_()
                seen = 0

            if cfg.eval_every and step % cfg.eval_every == 0 and val_loader is not None:
                val = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
                history.append({"step": step, "val_loss": val})
                print(f"  val loss {val:.4f} (ppl {math.exp(min(20, val)):.2f})", flush=True)
                if val < best_val:
                    best_val, best_step = val, step
                    if cfg.save == "best":
                        save_checkpoint(out_dir, raw_model, tokenizer,
                                        {**(meta or {}), "val_loss": val, "step": step})

        if val_loader is not None and not cfg.eval_every:
            val = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
            history.append({"epoch": epoch + 1, "step": step, "val_loss": val})
            print(
                f"epoch {epoch+1}/{cfg.epochs} done | val loss {val:.4f} "
                f"(ppl {math.exp(min(20, val)):.2f})",
                flush=True,
            )
            if val < best_val - 1e-4:
                best_val, best_step, stale_epochs = val, step, 0
                if cfg.save == "best":
                    save_checkpoint(out_dir, raw_model, tokenizer,
                                    {**(meta or {}), "val_loss": val, "step": step})
            else:
                stale_epochs += 1
                if cfg.early_stop_patience and stale_epochs >= cfg.early_stop_patience:
                    print(f"early stopping: no val improvement for {stale_epochs} epochs")
                    stop = True
        if stop:
            break

    if cfg.save == "best" and best_val < float("inf"):
        print(f"best val loss {best_val:.4f} at step {best_step}; checkpoint in {out_dir}")
    else:
        # "last" is the default: on a small Q&A set the lowest validation loss
        # usually lands on an underfit epoch, while what you actually want is a
        # model that has fully absorbed your answers.
        save_checkpoint(out_dir, raw_model, tokenizer,
                        {**(meta or {}), "step": step,
                         "val_loss": None if best_val == float("inf") else best_val})
        print(f"saved final checkpoint to {out_dir}")

    summary = {
        "save": cfg.save,
        "best_val_loss": None if best_val == float("inf") else best_val,
        "steps": step,
        "seconds": round(time.time() - t0, 1),
        "history": history,
    }
    Path(out_dir, "train_log.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_tokenizer(
    texts: Iterable[str], vocab_size: int, out_dir: Path, reuse: str | None = None
) -> BPETokenizer:
    if reuse:
        print(f"reusing tokenizer from {reuse}")
        return BPETokenizer.load(Path(reuse) / "tokenizer.json" if Path(reuse).is_dir() else reuse)
    print(f"training byte-level BPE tokenizer (target vocab {vocab_size}) ...")
    tok = BPETokenizer.train(texts, vocab_size=vocab_size, verbose=True)
    print(f"  learned {len(tok.merges)} merges -> vocab size {tok.vocab_size}")
    return tok


def load_pretrained_weights(model: GPT, src: Path) -> None:
    """Warm-start from another checkpoint, skipping tensors whose shape changed.

    This makes ``pretrain`` -> ``train`` work even when the two runs use a
    different depth or width: whatever lines up is copied, the rest stays at its
    freshly initialised value.
    """
    state = torch.load(src / "model.pt", map_location="cpu", weights_only=True)
    own = model.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(usable))
    model.load_state_dict(usable, strict=False)
    print(f"initialised {len(usable)}/{len(own)} tensors from {src}"
          + (f"; skipped {len(skipped)} with mismatched shapes" if skipped else ""))


def run_sft(args) -> None:
    """`minigpt train` - supervised fine-tuning on a question/answer CSV."""
    out_dir = Path(args.out)
    rows = load_csv(args.data, args.input_col, args.output_col, args.system_col, args.delimiter)
    print(f"loaded {len(rows)} rows from {args.data}")

    preset = dict(PRESETS[args.size])
    for key in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size", "dropout", "n_kv_head"):
        val = getattr(args, key, None)
        if val is not None:
            preset[key] = val
    if getattr(args, "rope_contiguous", False):
        preset["rope_interleaved"] = False

    tok = build_tokenizer(iter_texts(rows), preset["vocab_size"], out_dir, args.init_from)
    preset["vocab_size"] = tok.vocab_size
    cfg = GPTConfig(**preset)

    n_rows = len(rows)
    train_rows, val_rows = train_val_split(rows, args.val_ratio, args.seed)
    del rows
    val_ds = ChatDataset(val_rows, tok, cfg.block_size, mask_prompt=not args.train_on_prompt) if val_rows else None
    del val_rows
    train_ds = ChatDataset(train_rows, tok, cfg.block_size, mask_prompt=not args.train_on_prompt)
    # The text is fully tokenised into compact tensors now; free the strings.
    del train_rows
    if train_ds.n_truncated:
        print(f"warning: {train_ds.n_truncated} example(s) exceeded block_size={cfg.block_size} and were truncated")
    print(f"train {len(train_ds)} / val {len(val_ds) if val_ds else 0} examples, {train_ds.n_tokens} train tokens")

    model = GPT(cfg)
    if args.init_from:
        load_pretrained_weights(model, Path(args.init_from))

    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        val_ratio=args.val_ratio, seed=args.seed, device=args.device,
        num_workers=args.num_workers, early_stop_patience=args.patience,
        log_every=args.log_every, eval_every=args.eval_every, save=args.save,
        amp=args.amp, dynamic_padding=not args.no_dynamic_padding, compile=args.compile,
        grad_checkpoint=args.grad_checkpoint,
    )
    meta = {"task": "sft", "dataset": str(args.data), "rows": n_rows, "size": args.size}
    train_model(model, tok, train_ds, val_ds, tcfg, out_dir, meta)


def run_pretrain(args) -> None:
    """`minigpt pretrain` - optional next-token pretraining on a raw text file."""
    out_dir = Path(args.out)
    text_path = Path(args.text)
    if not text_path.exists():
        raise SystemExit(f"corpus not found: {text_path}")
    print(f"streaming {text_path.stat().st_size/1e6:.1f} MB from {text_path}")

    preset = dict(PRESETS[args.size])
    for key in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size", "dropout", "n_kv_head"):
        val = getattr(args, key, None)
        if val is not None:
            preset[key] = val
    if getattr(args, "rope_contiguous", False):
        preset["rope_interleaved"] = False

    # The corpus is never held in memory: the tokenizer counts pieces chunk by
    # chunk, and the ids are streamed to a compact on-disk file that is memmapped.
    tok = build_tokenizer(iter_text_chunks(text_path), preset["vocab_size"], out_dir, args.init_from)
    preset["vocab_size"] = tok.vocab_size
    cfg = GPTConfig(**preset)

    print("encoding corpus ...", flush=True)
    ids = encode_corpus_to_file(iter_text_chunks(text_path), tok, out_dir / "tokens.bin")
    tok._cache.clear()
    print(f"corpus = {len(ids)} tokens")
    n_val = int(len(ids) * args.val_ratio)
    train_ids, val_ids = (ids[:-n_val], ids[-n_val:]) if n_val > cfg.block_size else (ids, None)
    train_ds = PackedTextDataset(train_ids, cfg.block_size, args.stride)
    val_ds = PackedTextDataset(val_ids, cfg.block_size) if val_ids is not None else None

    model = GPT(cfg)
    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        seed=args.seed, device=args.device, num_workers=args.num_workers,
        log_every=args.log_every, eval_every=args.eval_every, save=args.save,
        amp=args.amp, dynamic_padding=not args.no_dynamic_padding, compile=args.compile,
        grad_checkpoint=args.grad_checkpoint,
    )
    train_model(model, tok, train_ds, val_ds, tcfg, out_dir, {"task": "pretrain", "corpus": str(args.text)})
