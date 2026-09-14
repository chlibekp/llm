"""Training loop: tokenizer training, supervised fine-tuning, optional pretraining.

Everything here is tuned for a single small device (Apple Silicon / CPU):

* AdamW with decoupled weight decay on matrices only.
* Linear warmup then cosine decay to ``min_lr``.
* Gradient accumulation so a large effective batch fits in unified memory.
* Gradient clipping at 1.0.
* Best-on-validation checkpointing, plus a final save if there is no val split.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import resolve_device, save_checkpoint
from .chat import iter_texts
from .config import PRESETS, GPTConfig
from .data import (
    ChatDataset,
    LengthGroupedSampler,
    PackedTextDataset,
    default_encode_workers,
    dynamic_collate,
    encode_corpus_to_file,
    iter_text_chunks,
    load_csv,
    sample_text_chunks,
    token_dtype,
    train_val_split,
)
from .model import GPT
from .tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    epochs: int = 30
    max_steps: int = 0           # cap on optimiser steps; 0 = no cap
    max_tokens: int = 0          # cap on training tokens; 0 = no cap
    # Effective batch 64. Throughput barely depends on the micro-batch on an M2
    # (compute-bound: 8.1k vs 8.4k tok/s at 16 vs 32) but activation memory is
    # proportional to it, so a small micro-batch with accumulation costs ~nothing
    # in speed and keeps a 64-sequence step from pushing an 8 GB machine into swap.
    batch_size: int = 16
    grad_accum: int = 4
    lr: float = 6e-4             # sqrt-scaled for the effective batch of 64
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    optimizer: str = "adamw"      # "adamw" | "muon" (Muon on hidden matrices, AdamW elsewhere)
    grad_clip: float = 1.0
    val_ratio: float = 0.1
    eval_every: int = 0          # in steps; 0 => once per epoch
    log_every: int = 10
    seed: int = 1337
    device: str = "auto"
    compile: bool = False
    grad_checkpoint: bool = False  # recompute block activations in backward: less memory, ~30% slower
    # Big models on MPS: wait for the device after every micro-batch and return cached
    # blocks after every step. MPS executes asynchronously, so without the wait the
    # CPU queues micro-batch after micro-batch and their memory piles up.
    release_cache: bool = False
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
    # MPS: measured on an M2, `small` at 16x255 tokens, float32 ran 378 ms/step
    # against 502 ms under bf16 autocast - the per-op casts cost more than the
    # half-width arithmetic saves. Half precision is still selectable explicitly.
    return None  # CPU autocast is usually slower than plain float32 too


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


def tokens_per_epoch(ds: Dataset) -> int:
    """Tokens the model is trained on in one pass over ``ds``."""
    if isinstance(ds, PackedTextDataset):
        return len(ds) * ds.block_size  # windows overlap when stride < block_size
    return getattr(ds, "n_tokens", len(ds))


def plan_steps(steps_per_epoch: int, epoch_tokens: int, cfg: TrainConfig) -> int:
    """Total optimiser steps: ``epochs`` passes, cut short by either budget."""
    total = steps_per_epoch * cfg.epochs
    if cfg.max_steps > 0:
        total = min(total, cfg.max_steps)
    if cfg.max_tokens > 0:
        tokens_per_step = max(1.0, epoch_tokens / steps_per_epoch)
        total = min(total, math.ceil(cfg.max_tokens / tokens_per_step))
    return max(1, total)


def auto_epochs(steps_per_epoch: int, target_steps: int = 3000, max_epochs: int = 30) -> int:
    """Epochs for SFT when ``--epochs`` is not given.

    A small Q&A set needs many passes to be absorbed; a large one reaches the
    same number of optimiser steps in a few. Aiming for a fixed step count keeps
    the 100-row demo at 30 epochs while an 80k-row CSV gets 3 instead of 30.
    """
    return max(1, min(max_epochs, math.ceil(target_steps / max(1, steps_per_epoch))))


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


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
    sync: bool = False,
) -> float:
    model.eval()
    total = torch.zeros((), device=device)
    n = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x, y, keep = _to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            _, loss, _ = model(x, targets=y, loss_only=keep is not None, keep_index=keep)
        total += loss.detach().float()
        n += 1
        if sync and device.type == "mps":
            torch.mps.synchronize()  # see TrainConfig.release_cache
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
    total_steps = plan_steps(steps_per_epoch, tokens_per_epoch(train_ds), cfg)
    n_epochs = math.ceil(total_steps / steps_per_epoch)
    optimizer = model.configure_optimizer(cfg.lr, cfg.weight_decay, cfg.betas, cfg.optimizer)

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
        f"examples={len(train_ds)} steps/epoch={steps_per_epoch} epochs={n_epochs} total_steps={total_steps} "
        f"amp={amp_dtype and str(amp_dtype).removeprefix('torch.') or 'off'} padding={padding} "
        f"optimizer={cfg.optimizer}"
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

    for epoch in range(n_epochs):
        running.zero_()
        seen = 0
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        for micro, batch in enumerate(train_loader):
            x, y, keep = _to_device(batch, device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                # The sparse head needs the loader's index; without one (packed
                # pretraining, where every position is a target) the dense loss
                # is the same arithmetic minus a sync, a gather and a scatter.
                _, loss, _ = model(x, targets=y, loss_only=keep is not None, keep_index=keep)
            scaler.scale(loss / cfg.grad_accum).backward()
            running += loss.detach().float()
            seen += 1
            if cfg.release_cache and device.type == "mps":
                torch.mps.synchronize()

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
            if cfg.release_cache and device.type == "mps":
                # Dynamic padding gives every batch a new shape, and the MPS allocator
                # keeps cached blocks for each. On SmolLM2-360M that cache held ~1 GB
                # between steps; releasing it cost no measurable time.
                torch.mps.empty_cache()
            step += 1

            if cfg.log_every and step % cfg.log_every == 0:
                elapsed = time.time() - t0
                print(
                    f"epoch {epoch+1}/{n_epochs} step {step}/{total_steps} "
                    f"loss {float(running)/max(1,seen):.4f} lr {lr:.2e} "
                    f"{elapsed:.0f}s eta {_fmt_duration(elapsed / step * (total_steps - step))}",
                    flush=True,
                )
                running.zero_()
                seen = 0

            if cfg.eval_every and step % cfg.eval_every == 0 and val_loader is not None:
                val = evaluate(model, val_loader, device, amp_dtype=amp_dtype, sync=cfg.release_cache)
                history.append({"step": step, "val_loss": val})
                print(f"  val loss {val:.4f} (ppl {math.exp(min(20, val)):.2f})", flush=True)
                if val < best_val:
                    best_val, best_step = val, step
                    if cfg.save == "best":
                        save_checkpoint(out_dir, raw_model, tokenizer,
                                        {**(meta or {}), "val_loss": val, "step": step})

            if step >= total_steps:  # a step or token budget ended training mid-epoch
                stop = True
                break

        if val_loader is not None and not cfg.eval_every:
            val = evaluate(model, val_loader, device, amp_dtype=amp_dtype, sync=cfg.release_cache)
            history.append({"epoch": epoch + 1, "step": step, "val_loss": val})
            print(
                f"epoch {epoch+1}/{n_epochs} done | val loss {val:.4f} "
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


def _fingerprint(path: Path) -> dict:
    """Identity of a source file, cheap enough to check on every run."""
    st = path.stat()
    return {"path": str(path.resolve()), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _tokenizer_digest(tok: BPETokenizer) -> str:
    return hashlib.sha256(json.dumps([tok.specials, tok.merges]).encode()).hexdigest()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def prepare_tokenizer(
    source: Path,
    texts: Callable[[], Iterable[str]],
    vocab_size: int,
    sample_mb: float,
    out_dir: Path,
    reuse: str | None = None,
) -> BPETokenizer:
    """Train (or reuse) the tokenizer for ``source``.

    Training is cached in ``out_dir``: when a previous run already trained a
    tokenizer on the same, unmodified file with the same settings, it is loaded
    instead of retrained. ``texts`` is only called on a cache miss.
    """
    if reuse:
        return build_tokenizer((), vocab_size, out_dir, reuse)
    key = {"source": _fingerprint(source), "vocab_size": vocab_size, "sample_mb": sample_mb}
    tok_path, meta_path = out_dir / "tokenizer.json", out_dir / "tokenizer.cache.json"
    meta = _read_json(meta_path)
    if meta and meta.get("key") == key and tok_path.exists():
        tok = BPETokenizer.load(tok_path)
        if _tokenizer_digest(tok) == meta.get("digest"):
            print(f"reusing tokenizer from {tok_path} ({source} is unchanged)")
            return tok

    tok = build_tokenizer(texts(), vocab_size, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save(tok_path)
    meta_path.write_text(json.dumps({"key": key, "digest": _tokenizer_digest(tok)}), encoding="utf-8")
    return tok


def sample_rows(rows: list, max_bytes: int) -> list:
    """Evenly spaced rows totalling about ``max_bytes`` of text (all if 0)."""
    if max_bytes <= 0:
        return rows
    total = sum(len(u) + len(a) + len(s or "") for u, a, s in rows)
    return rows[:: max(1, math.ceil(total / max_bytes))]


def prepare_tokens(text_path: Path, tok: BPETokenizer, out_dir: Path, workers: int) -> np.memmap:
    """Encode the corpus to ``out_dir/tokens.bin``, or reuse it if still valid.

    The cache is keyed by the corpus file and the exact merges, so editing the
    corpus or changing the tokenizer re-encodes, and nothing else does.
    """
    bin_path, meta_path = out_dir / "tokens.bin", out_dir / "tokens.json"
    key = {"source": _fingerprint(text_path), "tokenizer": _tokenizer_digest(tok)}
    dtype = token_dtype(tok.vocab_size)
    meta = _read_json(meta_path)
    if (
        meta and meta.get("key") == key and bin_path.exists()
        and bin_path.stat().st_size == meta.get("n_tokens", -1) * dtype.itemsize
    ):
        print(f"reusing {meta['n_tokens']} encoded tokens from {bin_path}")
        return np.memmap(bin_path, dtype=dtype, mode="r", shape=(meta["n_tokens"],))

    meta_path.unlink(missing_ok=True)  # never leave a valid-looking key over a partial file
    print(f"encoding corpus with {workers} worker(s) ...", flush=True)
    t0 = time.time()
    ids = encode_corpus_to_file(iter_text_chunks(text_path), tok, bin_path, workers=workers)
    tok._cache.clear()
    print(f"  encoded in {time.time() - t0:.0f}s")
    meta_path.write_text(json.dumps({"key": key, "n_tokens": len(ids)}), encoding="utf-8")
    return ids


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

    sample_bytes = int(args.tokenizer_sample_mb * 1e6)
    tok = prepare_tokenizer(
        Path(args.data), lambda: iter_texts(sample_rows(rows, sample_bytes)),
        preset["vocab_size"], args.tokenizer_sample_mb, out_dir, args.init_from,
    )
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

    epochs = args.epochs
    if epochs is None:
        steps_per_epoch = math.ceil(len(train_ds) / args.batch_size / args.grad_accum)
        epochs = auto_epochs(steps_per_epoch)
        print(f"epochs: {epochs} (auto for {steps_per_epoch} steps/epoch; set --epochs to override)")

    tcfg = TrainConfig(
        epochs=epochs, max_steps=args.max_steps, max_tokens=args.max_tokens,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        val_ratio=args.val_ratio, seed=args.seed, device=args.device,
        num_workers=args.num_workers, early_stop_patience=args.patience,
        log_every=args.log_every, eval_every=args.eval_every, save=args.save,
        amp=args.amp, dynamic_padding=not args.no_dynamic_padding, compile=args.compile,
        grad_checkpoint=args.grad_checkpoint, optimizer=args.optimizer,
    )
    meta = {"task": "sft","dataset": str(args.data), "rows": n_rows, "size": args.size}
    train_model(model, tok, train_ds, val_ds, tcfg, out_dir, meta)


def run_lora(args) -> None:
    """`minigpt lora` - LoRA fine-tuning of a pretrained model on a Q&A CSV."""
    from .hf import default_dtype, load_pretrained
    from .lora import ALL_TARGETS, ATTN_TARGETS, apply_lora

    if args.optimizer != "adamw":
        raise SystemExit("--optimizer muon is for full training; LoRA adapters use AdamW")
    if args.init_from:
        raise SystemExit("--init-from does not apply to lora; choose the starting model with --base")
    out_dir = Path(args.out)
    rows = load_csv(args.data, args.input_col, args.output_col, args.system_col, args.delimiter)
    print(f"loaded {len(rows)} rows from {args.data}")

    device = resolve_device(args.device)
    dtype = default_dtype(device) if args.base_dtype == "auto" else getattr(torch, args.base_dtype)
    model, tok, ref = load_pretrained(args.base, dtype)
    targets = ALL_TARGETS if args.targets == "all" else ATTN_TARGETS
    n_train = apply_lora(model, args.rank, args.alpha, args.lora_dropout, targets)
    print(f"base {ref['name']} ({model.num_parameters()/1e6:.0f}M params, {str(dtype).removeprefix('torch.')}, "
          f"frozen); training {n_train/1e6:.2f}M LoRA params (rank {args.rank}, {args.targets})")

    n_rows = len(rows)
    train_rows, val_rows = train_val_split(rows, args.val_ratio, args.seed)
    del rows
    mask = not args.train_on_prompt
    val_ds = ChatDataset(val_rows, tok, args.block_size, mask_prompt=mask) if val_rows else None
    train_ds = ChatDataset(train_rows, tok, args.block_size, mask_prompt=mask)
    del train_rows, val_rows
    if train_ds.n_truncated:
        print(f"warning: {train_ds.n_truncated} example(s) exceeded --block-size={args.block_size} and were truncated")
    print(f"train {len(train_ds)} / val {len(val_ds) if val_ds else 0} examples, {train_ds.n_tokens} train tokens")

    tcfg = TrainConfig(
        epochs=args.epochs, max_steps=args.max_steps, max_tokens=args.max_tokens,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        val_ratio=args.val_ratio, seed=args.seed, device=args.device,
        num_workers=args.num_workers, log_every=args.log_every, eval_every=args.eval_every,
        save=args.save, amp="off", dynamic_padding=not args.no_dynamic_padding,
        compile=args.compile, grad_checkpoint=args.grad_checkpoint, release_cache=True,
    )
    meta = {"task": "lora", "dataset": str(args.data), "rows": n_rows, "base": ref["name"]}
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

    # The corpus is never held in memory: the tokenizer counts pieces from an
    # evenly spaced sample, and the ids are streamed to a compact on-disk file
    # that is memmapped. Both are cached in out_dir for the next run.
    sample_bytes = int(args.tokenizer_sample_mb * 1e6)
    tok = prepare_tokenizer(
        text_path, lambda: sample_text_chunks(text_path, sample_bytes),
        preset["vocab_size"], args.tokenizer_sample_mb, out_dir, args.init_from,
    )
    preset["vocab_size"] = tok.vocab_size
    cfg = GPTConfig(**preset)

    workers = args.encode_workers if args.encode_workers is not None else default_encode_workers()
    ids = prepare_tokens(text_path, tok, out_dir, workers)
    print(f"corpus = {len(ids)} tokens")
    n_val = int(len(ids) * args.val_ratio)
    train_ids, val_ids = (ids[:-n_val], ids[-n_val:]) if n_val > cfg.block_size else (ids, None)
    train_ds = PackedTextDataset(train_ids, cfg.block_size, args.stride)
    val_ds = PackedTextDataset(val_ids, cfg.block_size) if val_ids is not None else None

    model = GPT(cfg)
    tcfg = TrainConfig(
        epochs=args.epochs, max_steps=args.max_steps, max_tokens=args.max_tokens,
        batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        seed=args.seed, device=args.device, num_workers=args.num_workers,
        log_every=args.log_every, eval_every=args.eval_every, save=args.save,
        amp=args.amp, dynamic_padding=not args.no_dynamic_padding, compile=args.compile,
        grad_checkpoint=args.grad_checkpoint, optimizer=args.optimizer,
    )
    train_model(model, tok, train_ds, val_ds, tcfg, out_dir, {"task": "pretrain","corpus": str(args.text)})
