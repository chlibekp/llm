"""Command line interface.

    minigpt train     --data data/sample_qa.csv --out runs/demo
    minigpt pretrain  --text corpus.txt         --out runs/base
    minigpt chat      --model runs/demo
    minigpt generate  --model runs/demo --prompt "What is minigpt?"
    minigpt serve     --model runs/demo --port 8000
    minigpt info      --model runs/demo
    minigpt tokenize  --model runs/demo --text "hello world"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import PRESETS


# ------------------------------------------------------------------ arguments
def _add_model_shape_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--size", choices=sorted(PRESETS), default="small",
                   help="architecture preset (default: small, ~11M params)")
    p.add_argument("--n-layer", type=int, default=None, help="override: transformer blocks")
    p.add_argument("--n-head", type=int, default=None, help="override: attention heads")
    p.add_argument("--n-kv-head", type=int, default=None, help="override: KV heads (grouped-query attention)")
    p.add_argument("--n-embd", type=int, default=None, help="override: embedding width")
    p.add_argument("--block-size", type=int, default=None, help="override: context length in tokens")
    p.add_argument("--vocab-size", type=int, default=None, help="override: target BPE vocabulary size")
    p.add_argument("--dropout", type=float, default=None, help="override: dropout probability")
    p.add_argument("--rope-contiguous", action="store_true",
                   help="faster contiguous-halves RoPE layout; incompatible with "
                        "checkpoints trained without it")


def _add_optim_args(p: argparse.ArgumentParser, epochs: int) -> None:
    p.add_argument("--epochs", type=int, default=epochs)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimiser step")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--val-ratio", type=float, default=0.1, help="fraction held out for validation")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", default="auto", choices=["auto", "bf16", "fp16", "off"],
                   help="mixed-precision autocast dtype ('auto' picks bf16 on mps/cuda)")
    p.add_argument("--no-dynamic-padding", action="store_true",
                   help="pad every batch to block_size instead of to its longest example")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (CUDA/MPS; fuses the small elementwise kernels)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=0, help="evaluate every N steps (0 = once per epoch)")
    p.add_argument("--save", choices=["last", "best"], default="last",
                   help="'last' keeps the final weights, 'best' keeps the lowest validation loss")
    p.add_argument("--init-from", default=None, help="checkpoint dir to warm-start weights and tokenizer from")


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--seed", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="minigpt",
        description="Train and serve a small from-scratch GPT on your own question/answer CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # train ------------------------------------------------------------------
    t = sub.add_parser("train", help="supervised fine-tuning on a Q&A CSV",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t.add_argument("--data", required=True, help="path to the CSV file")
    t.add_argument("--out", default="runs/model", help="checkpoint directory to write")
    t.add_argument("--input-col", default=None, help="question column (auto-detected by default)")
    t.add_argument("--output-col", default=None, help="answer column (auto-detected by default)")
    t.add_argument("--system-col", default=None, help="optional per-row system prompt column")
    t.add_argument("--delimiter", default=None, help="CSV delimiter (sniffed by default)")
    t.add_argument("--train-on-prompt", action="store_true",
                   help="also compute loss on the question (default: answer only)")
    t.add_argument("--patience", type=int, default=0, help="early-stop after N epochs without val improvement")
    _add_model_shape_args(t)
    _add_optim_args(t, epochs=30)

    # pretrain ---------------------------------------------------------------
    pt = sub.add_parser("pretrain", help="next-token pretraining on a raw .txt corpus",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pt.add_argument("--text", required=True, help="path to a UTF-8 text file")
    pt.add_argument("--out", default="runs/base")
    pt.add_argument("--stride", type=int, default=None, help="window stride (default: block_size, no overlap)")
    _add_model_shape_args(pt)
    _add_optim_args(pt, epochs=5)

    # chat -------------------------------------------------------------------
    c = sub.add_parser("chat", help="interactive REPL", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    c.add_argument("--model", required=True, help="checkpoint directory")
    c.add_argument("--system", default=None, help="system prompt")
    c.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    c.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    c.add_argument("--no-history", action="store_true", help="treat every turn independently")
    _add_sampling_args(c)

    # generate ---------------------------------------------------------------
    g = sub.add_parser("generate", help="answer a single prompt and exit",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", default=None, help="question (reads stdin when omitted)")
    g.add_argument("--system", default=None)
    g.add_argument("--raw", action="store_true", help="skip the chat template, complete the text as-is")
    g.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    g.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    _add_sampling_args(g)

    # serve ------------------------------------------------------------------
    s = sub.add_parser("serve", help="start the OpenAI-compatible HTTP server",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--model-name", default="minigpt", help="id reported by /v1/models")
    s.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    s.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    s.add_argument("--reload", action="store_true", help="uvicorn autoreload (development only)")

    # info / tokenize --------------------------------------------------------
    i = sub.add_parser("info", help="print checkpoint metadata")
    i.add_argument("--model", required=True)

    tk = sub.add_parser("tokenize", help="show how text is tokenised")
    tk.add_argument("--model", required=True)
    tk.add_argument("--text", required=True)
    return p


# ------------------------------------------------------------------- commands
def cmd_chat(args) -> None:
    from .checkpoint import load_checkpoint
    from .chat import render_prompt
    from .generate import SamplingParams, stream_text

    model, tok, meta, device = load_checkpoint(args.model, args.device, args.dtype)
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
    )
    print(f"minigpt chat - {model.num_parameters()/1e6:.1f}M params on {device.type}. "
          f"Ctrl-C or /exit to quit, /reset to clear history.")
    history: list[dict] = []
    while True:
        try:
            user = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in ("/exit", "/quit"):
            return
        if user == "/reset":
            history.clear()
            print("(history cleared)")
            continue

        msgs = ([] if args.no_history else list(history)) + [{"role": "user", "content": user}]
        prompt = render_prompt(msgs, system=args.system)
        # Drop the oldest turns until the prompt fits the context window.
        while len(tok.encode(prompt)) > model.cfg.block_size - 16 and len(msgs) > 1:
            msgs = msgs[2:]
            prompt = render_prompt(msgs, system=args.system)

        print("bot> ", end="", flush=True)
        answer = ""
        for piece in stream_text(model, tok, prompt, params, device):
            answer += piece
            print(piece, end="", flush=True)
        print()
        history = msgs + [{"role": "assistant", "content": answer.strip()}]


def cmd_generate(args) -> None:
    from .checkpoint import load_checkpoint
    from .chat import render_prompt
    from .generate import SamplingParams, stream_text

    prompt_text = args.prompt if args.prompt is not None else sys.stdin.read().strip()
    if not prompt_text:
        raise SystemExit("no prompt given (use --prompt or pipe text on stdin)")

    model, tok, meta, device = load_checkpoint(args.model, args.device, args.dtype)
    params = SamplingParams(
        max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty, seed=args.seed,
    )
    prompt = prompt_text if args.raw else render_prompt(
        [{"role": "user", "content": prompt_text}], system=args.system
    )
    for piece in stream_text(model, tok, prompt, params, device):
        print(piece, end="", flush=True)
    print()


def cmd_serve(args) -> None:
    import uvicorn

    from .server import create_app

    app = create_app(args.model, args.model_name, args.device, args.dtype)
    served = app.state.served
    print(f"minigpt serving {args.model} as '{args.model_name}' "
          f"({served.model.num_parameters()/1e6:.1f}M params, {served.device.type})")
    print(f"  OpenAI base URL: http://{args.host}:{args.port}/v1")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


def cmd_info(args) -> None:
    d = Path(args.model)
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    log_path = d / "train_log.json"
    print(json.dumps(cfg, indent=2))
    if log_path.exists():
        log = json.loads(log_path.read_text(encoding="utf-8"))
        log.pop("history", None)
        print(json.dumps({"training": log}, indent=2))


def cmd_tokenize(args) -> None:
    from .tokenizer import BPETokenizer

    tok = BPETokenizer.load(Path(args.model) / "tokenizer.json")
    ids = tok.encode(args.text)
    print(f"vocab_size = {tok.vocab_size}")
    print(f"{len(ids)} tokens: {ids}")
    print("pieces:", [tok.decode([i], skip_special=False) for i in ids])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        from .train import run_sft
        run_sft(args)
    elif args.command == "pretrain":
        from .train import run_pretrain
        run_pretrain(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "info":
        cmd_info(args)
    elif args.command == "tokenize":
        cmd_tokenize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
