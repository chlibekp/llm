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

import torch
from torch.utils.data import DataLoader, Dataset

from .checkpoint import resolve_device, save_checkpoint
from .chat import iter_texts
from .config import PRESETS, GPTConfig
from .data import ChatDataset, PackedTextDataset, load_csv, train_val_split
from .model import GPT
from .tokenizer import BPETokenizer


@dataclass
class TrainConfig:
    """Optimisation hyper-parameters."""

    epochs: int = 30
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 3e-4
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


@torch.no_grad()
def evaluate(model: GPT, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss, _ = model(x, targets=y)
        total += float(loss)
        n += 1
    model.train()
    return total / max(1, n)


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

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
        num_workers=cfg.num_workers, pin_memory=False,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    optimizer = model.configure_optimizer(cfg.lr, cfg.weight_decay, cfg.betas)

    if cfg.compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(model)  # torch.compile is not reliable on MPS yet

    print(
        f"device={device.type} params={model.num_parameters()/1e6:.2f}M "
        f"examples={len(train_ds)} steps/epoch={steps_per_epoch} total_steps={total_steps}"
    )

    history: list[dict] = []
    best_val = float("inf")
    best_step = 0
    stale_epochs = 0
    step = 0
    t0 = time.time()
    stop = False

    for epoch in range(cfg.epochs):
        running, seen = 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        for micro, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            _, loss, _ = model(x, targets=y)
            (loss / cfg.grad_accum).backward()
            running += loss.detach().item()
            seen += 1

            is_last = micro == len(train_loader) - 1
            if (micro + 1) % cfg.grad_accum != 0 and not is_last:
                continue

            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if cfg.log_every and step % cfg.log_every == 0:
                print(
                    f"epoch {epoch+1}/{cfg.epochs} step {step}/{total_steps} "
                    f"loss {running/max(1,seen):.4f} lr {lr:.2e} "
                    f"{time.time()-t0:.0f}s",
                    flush=True,
                )
                running, seen = 0.0, 0

            if cfg.eval_every and step % cfg.eval_every == 0 and val_loader is not None:
                val = evaluate(model, val_loader, device)
                history.append({"step": step, "val_loss": val})
                print(f"  val loss {val:.4f} (ppl {math.exp(min(20, val)):.2f})", flush=True)
                if val < best_val:
                    best_val, best_step = val, step
                    if cfg.save == "best":
                        save_checkpoint(out_dir, model, tokenizer,
                                        {**(meta or {}), "val_loss": val, "step": step})

        if val_loader is not None and not cfg.eval_every:
            val = evaluate(model, val_loader, device)
            history.append({"epoch": epoch + 1, "step": step, "val_loss": val})
            print(
                f"epoch {epoch+1}/{cfg.epochs} done | val loss {val:.4f} "
                f"(ppl {math.exp(min(20, val)):.2f})",
                flush=True,
            )
            if val < best_val - 1e-4:
                best_val, best_step, stale_epochs = val, step, 0
                if cfg.save == "best":
                    save_checkpoint(out_dir, model, tokenizer,
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
        save_checkpoint(out_dir, model, tokenizer,
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
    texts: list[str], vocab_size: int, out_dir: Path, reuse: str | None = None
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

    tok = build_tokenizer(list(iter_texts(rows)), preset["vocab_size"], out_dir, args.init_from)
    preset["vocab_size"] = tok.vocab_size
    cfg = GPTConfig(**preset)

    train_rows, val_rows = train_val_split(rows, args.val_ratio, args.seed)
    train_ds = ChatDataset(train_rows, tok, cfg.block_size, mask_prompt=not args.train_on_prompt)
    val_ds = ChatDataset(val_rows, tok, cfg.block_size, mask_prompt=not args.train_on_prompt) if val_rows else None
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
    )
    meta = {"task": "sft", "dataset": str(args.data), "rows": len(rows), "size": args.size}
    train_model(model, tok, train_ds, val_ds, tcfg, out_dir, meta)


def run_pretrain(args) -> None:
    """`minigpt pretrain` - optional next-token pretraining on a raw text file."""
    out_dir = Path(args.out)
    text = Path(args.text).read_text(encoding="utf-8")
    print(f"loaded {len(text)} characters from {args.text}")

    preset = dict(PRESETS[args.size])
    for key in ("n_layer", "n_head", "n_embd", "block_size", "vocab_size", "dropout", "n_kv_head"):
        val = getattr(args, key, None)
        if val is not None:
            preset[key] = val

    tok = build_tokenizer([text], preset["vocab_size"], out_dir, args.init_from)
    preset["vocab_size"] = tok.vocab_size
    cfg = GPTConfig(**preset)

    ids = [tok.bos_id] + tok.encode(text)
    print(f"corpus = {len(ids)} tokens")
    n_val = int(len(ids) * args.val_ratio)
    train_ids, val_ids = (ids[:-n_val], ids[-n_val:]) if n_val > cfg.block_size else (ids, [])
    train_ds = PackedTextDataset(train_ids, cfg.block_size, args.stride)
    val_ds = PackedTextDataset(val_ids, cfg.block_size) if val_ids else None

    model = GPT(cfg)
    tcfg = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
        seed=args.seed, device=args.device, num_workers=args.num_workers,
        log_every=args.log_every, eval_every=args.eval_every, save=args.save,
    )
    train_model(model, tok, train_ds, val_ds, tcfg, out_dir, {"task": "pretrain", "corpus": str(args.text)})
