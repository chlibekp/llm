# minigpt

A complete, from-scratch **generative pretrained transformer** in Python. You give it a
CSV of questions and answers, it trains its own tokenizer and its own model, and then you
can talk to it — from a terminal REPL or through an **OpenAI-compatible HTTP API**.

Everything is written to run on a **MacBook Air M2**: no GPU cluster, no downloads, no
pretrained weights. Training the bundled demo takes about **2.5 minutes** on an M2 and
about 420 MB of memory.

```
$ minigpt train --data data/sample_qa.csv --out runs/demo
$ minigpt chat --model runs/demo

you> What is attention?
bot> Attention lets each token look at other tokens and mix in the information
     that is most relevant to it.
```

---

## Table of contents

- [What this actually is](#what-this-actually-is)
- [Install](#install)
- [Quickstart](#quickstart)
- [Your data: the CSV format](#your-data-the-csv-format)
- [How it works](#how-it-works)
  - [1. Tokenizer](#1-tokenizer-byte-level-bpe-trained-on-your-data)
  - [2. Prompt format](#2-prompt-format)
  - [3. Model architecture](#3-model-architecture)
  - [4. Training](#4-training)
  - [5. Generation](#5-generation)
- [CLI reference](#cli-reference)
- [OpenAI-compatible API](#openai-compatible-api)
- [Model presets and sizing](#model-presets-and-sizing)
- [Training tips](#training-tips-getting-good-answers)
- [Performance on a MacBook Air M2](#performance-on-a-macbook-air-m2)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations-read-this)
- [License](#license)

---

## What this actually is

Every piece of the pipeline is implemented in this repo, in readable Python:

| Step | Where | What happens |
|---|---|---|
| Tokenization | `minigpt/tokenizer.py` | A byte-level BPE tokenizer is **trained from scratch** on your CSV |
| Prompt format | `minigpt/chat.py` | Q&A pairs are rendered into a chat template with role control tokens |
| Data pipeline | `minigpt/data.py` | CSV parsing, column auto-detection, label masking, padding, train/val split |
| Model | `minigpt/model.py` | Decoder-only transformer: RoPE, RMSNorm, grouped-query attention, SwiGLU, weight tying |
| Training | `minigpt/train.py` | AdamW, cosine LR schedule with warmup, gradient accumulation and clipping, checkpointing |
| Generation | `minigpt/generate.py` | KV-cached sampling with temperature / top-k / top-p / repetition penalty |
| CLI | `minigpt/cli.py` | `train`, `pretrain`, `chat`, `generate`, `serve`, `info`, `tokenize` |
| API | `minigpt/server.py` | FastAPI server speaking the OpenAI REST dialect, including SSE streaming |

The only heavy dependency is PyTorch (used for tensors, autograd and the fused attention
kernel). There is no HuggingFace, no `tokenizers`, no pretrained anything.

---

## Install

Requires Python 3.10+ and about 2.5 GB of disk for PyTorch.

```bash
git clone <this-repo> && cd llm

python3 -m venv .venv
source .venv/bin/activate

pip install -e .            # installs deps and the `minigpt` command
# or, without installing the package:
pip install -r requirements.txt   # then use `python -m minigpt ...`
```

Verify that Apple Silicon acceleration is picked up:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
```

`minigpt` selects the backend automatically: **MPS** (Apple Silicon) → **CUDA** → **CPU**.
Force one with `--device cpu` if you need to.

---

## Quickstart

A 102-row demo dataset ships in `data/sample_qa.csv`.

```bash
# 1. Train (~2.5 min on an M2 Air; add --block-size 192 to halve that)
minigpt train --data data/sample_qa.csv --out runs/demo --epochs 40

# 2. Ask one question
minigpt generate --model runs/demo --prompt "What is a token?"
# → A token is a small chunk of text, usually part of a word, that the model
#   reads and predicts.

# 3. Interactive chat
minigpt chat --model runs/demo

# 4. Serve an OpenAI-compatible API
minigpt serve --model runs/demo --port 8000
```

Then, from any OpenAI client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
r = client.chat.completions.create(
    model="minigpt",
    messages=[{"role": "user", "content": "What is BPE?"}],
)
print(r.choices[0].message.content)
# → Byte pair encoding is a method that repeatedly merges the most frequent
#   pair of symbols to build a vocabulary.
```

---

## Your data: the CSV format

A UTF-8 CSV with a **header row**, one question/answer pair per row:

```csv
input,output
"What is the capital of France?","The capital of France is Paris."
"Who are you?","I am a small assistant trained on a CSV file."
```

**Column names are auto-detected.** Any of these work as the question column:
`input`, `question`, `prompt`, `instruction`, `user`, `query`, `q`.
And as the answer column: `output`, `answer`, `response`, `completion`, `assistant`,
`target`, `a`. Override explicitly if your headers are unusual:

```bash
minigpt train --data mydata.csv --input-col "Frage" --output-col "Antwort"
```

**Optional per-row system prompt.** Add a `system` column (or `context` /
`system_prompt`) and each row can carry its own instruction:

```csv
input,output,system
"Summarize: ...","...","You are a terse summarizer."
```

Other details:

- The delimiter is sniffed automatically; override with `--delimiter ';'`.
- Rows with an empty question or answer are skipped.
- A UTF-8 BOM is handled.
- Quoted fields may contain newlines and commas (standard CSV rules).
- Pairs longer than the context window are truncated, with a warning telling you how many.

---

## How it works

### 1. Tokenizer: byte-level BPE, trained on your data

`minigpt` does not ship a vocabulary. It learns one from your CSV every time you train.

1. Text is split with a GPT-2 style regex into "pieces" — words with their leading space,
   digit runs, punctuation runs, whitespace. Splitting first stops merges from spanning
   word boundaries.
2. Each piece becomes a sequence of **raw bytes**, so any input is representable and there
   is no `<unk>` token.
3. The most frequent adjacent pair of symbols is merged into a new symbol, repeatedly,
   until the vocabulary reaches `--vocab-size`.

The training loop keeps an incremental `pair → pieces containing it` index, so each merge
only rescans the pieces it actually affects. That is what makes a pure-Python BPE trainer
fast enough to be practical.

Six control tokens occupy ids `0–5` and can never be produced by a merge:
`<|pad|>`, `<|bos|>`, `<|eos|>`, `<|system|>`, `<|user|>`, `<|assistant|>`.

Inspect the result on any checkpoint:

```bash
minigpt tokenize --model runs/demo --text "What is attention?"
# vocab_size = 991
# 4 tokens: [296, 277, 568, 69]
# pieces: ['What', ' is', ' attention', '?']
```

> If your dataset is small, BPE runs out of pairs that occur at least twice and stops
> early — the demo's `--vocab-size 4096` yields an actual vocabulary of 991. That is
> correct behaviour, not a bug.

### 2. Prompt format

A conversation is rendered as one flat token stream:

```
<|bos|><|system|>You are helpful.<|user|>What is 2+2?<|assistant|>4<|eos|>
```

Because the roles are single reserved tokens rather than text like `"User:"`, the model
learns them as unambiguous structural markers, and generation has a reliable stop signal
(`<|eos|>`).

**Loss masking.** During training the loss is computed **only on the answer tokens** — the
prompt positions get the label `-100`, which `cross_entropy` ignores. The model is never
rewarded for memorizing questions, only for producing the right answer. Use
`--train-on-prompt` to also learn the question distribution (occasionally useful on very
small datasets).

### 3. Model architecture

A standard decoder-only transformer with the modern refinements, all in `minigpt/model.py`:

| Component | Choice | Why |
|---|---|---|
| Normalization | **RMSNorm**, pre-norm | Cheaper than LayerNorm and stable in fp16 — which matters on MPS |
| Positions | **RoPE** (rotary) | Relative positions for free; no learned position table to overfit on small data |
| Attention | `F.scaled_dot_product_attention` | Dispatches to the fastest fused kernel for the device |
| KV heads | **Grouped-query attention** (optional, `--n-kv-head`) | Shrinks the KV cache during generation |
| Feed-forward | **SwiGLU** | Consistently beats a GELU MLP at equal parameter count |
| Output head | **Tied to the input embedding** | Saves ~1.5M parameters at `d=384, vocab=4096` |
| Init | N(0, 0.02), residual projections scaled by `1/sqrt(2L)` | Keeps residual-stream variance stable with depth |

RoPE tables are kept in float32 and cached per device, so casting the model to fp16 never
blurs the position angles.

### 4. Training

`minigpt train` runs supervised fine-tuning directly from random initialization — for a
model this small, and a dataset this small, a separate pretraining stage is optional.

- **Optimizer** — AdamW, `betas=(0.9, 0.95)`. Weight decay is applied to matrices only,
  never to norms or biases.
- **Schedule** — linear warmup (5% of steps) then cosine decay to 10% of the peak LR.
- **Gradient accumulation** — `--grad-accum N` gives you the effect of an `N×` larger
  batch inside the same unified memory.
- **Gradient clipping** — global norm 1.0.
- **Validation** — `--val-ratio` (default 0.1) is held out with a fixed seed, so splits
  are reproducible across runs.
- **Checkpointing** — `--save last` (default) writes the final weights; `--save best`
  writes the lowest-validation-loss weights instead.

> **Why `last` is the default.** On a few-hundred-row Q&A set, validation loss bottoms out
> after a handful of epochs while the model has not yet absorbed your answers. What you
> want from a small factual dataset is closer to memorization than to generalization, so
> keeping the final weights gives markedly better answers. Switch to `--save best` once
> your dataset is large enough that the validation curve means something.

A checkpoint is a plain directory:

```
runs/demo/
  config.json      # architecture + metadata (what `minigpt info` prints)
  model.pt         # state_dict
  tokenizer.json   # merges + control tokens
  train_log.json   # loss history
```

**Optional pretraining.** If you have a raw text corpus in the same domain, you can
pretrain on it and then fine-tune:

```bash
minigpt pretrain --text corpus.txt --out runs/base --epochs 5
minigpt train --data qa.csv --out runs/chat --init-from runs/base
```

`--init-from` reuses the base checkpoint's tokenizer and copies every weight whose shape
still matches, so changing depth or width between the two stages is allowed.

### 5. Generation

`generate_stream` decodes one token at a time with a **KV cache**, so step *n* only
attends over cached keys and values rather than recomputing the whole prefix.
(`tests/test_model.py` asserts that cached decoding matches a full forward pass to 1e-4.)

Supported controls: `temperature` (0 = greedy/deterministic), `top_k`, `top_p`,
`repetition_penalty`, OpenAI-style `presence_penalty` / `frequency_penalty`, and `seed`
for reproducible sampling.

Because the vocabulary is byte-level, a single token can end mid-character. The streaming
path buffers incomplete UTF-8 sequences until they are valid, so you never see `�` in the
output.

---

## CLI reference

Run `minigpt <command> --help` for the full list of flags.

### `minigpt train` — supervised fine-tuning on a Q&A CSV

```bash
minigpt train --data data/sample_qa.csv --out runs/demo \
  --size small --epochs 40 --batch-size 16 --lr 6e-4
```

| Flag | Default | Meaning |
|---|---|---|
| `--data` | *required* | Path to the CSV |
| `--out` | `runs/model` | Checkpoint directory to write |
| `--input-col` / `--output-col` / `--system-col` | auto | CSV column names |
| `--delimiter` | sniffed | CSV delimiter |
| `--size` | `small` | Preset: `tiny`, `small`, `medium`, `large` |
| `--n-layer` / `--n-head` / `--n-kv-head` / `--n-embd` / `--block-size` / `--vocab-size` / `--dropout` | preset | Individual overrides |
| `--epochs` | `30` | Passes over the dataset |
| `--batch-size` | `16` | Examples per micro-batch |
| `--grad-accum` | `1` | Micro-batches per optimizer step |
| `--lr` | `3e-4` | Peak learning rate |
| `--weight-decay` / `--warmup-ratio` | `0.1` / `0.05` | Regularization and schedule |
| `--val-ratio` | `0.1` | Held-out fraction (`0` disables validation) |
| `--save` | `last` | `last` or `best` |
| `--patience` | `0` | Early-stop after N epochs without improvement (0 = off) |
| `--train-on-prompt` | off | Also compute loss on the question |
| `--init-from` | — | Warm-start weights and tokenizer from another checkpoint |
| `--device` | `auto` | `auto`, `mps`, `cuda`, `cpu` |
| `--seed` | `1337` | Reproducibility |

### `minigpt pretrain` — next-token pretraining on raw text

```bash
minigpt pretrain --text corpus.txt --out runs/base --size small --epochs 5
```

Same optimization flags, plus `--stride` to control window overlap when packing the corpus
into training blocks.

### `minigpt chat` — interactive REPL

```bash
minigpt chat --model runs/demo --temperature 0.7
```

Multi-turn history is kept and trimmed to fit the context window automatically.
`/reset` clears it, `/exit` quits, `--no-history` makes every turn independent.

### `minigpt generate` — one prompt, then exit

```bash
minigpt generate --model runs/demo --prompt "Who are you?" --temperature 0.2
echo "What is BPE?" | minigpt generate --model runs/demo          # reads stdin
minigpt generate --model runs/demo --raw --prompt "Paris is"      # no chat template
```

Sampling flags on `chat` and `generate`: `--max-new-tokens`, `--temperature`, `--top-k`,
`--top-p`, `--repetition-penalty`, `--seed`, `--device`, `--dtype`.

### `minigpt serve` — OpenAI-compatible API

```bash
minigpt serve --model runs/demo --host 127.0.0.1 --port 8000 --model-name my-bot
```

### `minigpt info` / `minigpt tokenize` — inspection

```bash
minigpt info --model runs/demo                        # config + training summary
minigpt tokenize --model runs/demo --text "hello!"    # token ids and pieces
```

---

## OpenAI-compatible API

`minigpt serve` exposes the subset of the OpenAI REST API that matters for a chat model,
so existing clients work unchanged — just point `base_url` at it. Any API key is accepted
(the server is intended for localhost; it does no authentication).

| Endpoint | Notes |
|---|---|
| `GET /v1/models` | Lists the loaded model, with its context window and parameter count |
| `GET /v1/models/{id}` | 404s on an unknown id |
| `POST /v1/chat/completions` | Non-streaming and SSE streaming |
| `POST /v1/completions` | Legacy raw-text completions |
| `POST /v1/embeddings` | L2-normalized mean-pooled final hidden states |
| `GET /health` | Liveness probe: model name and device |

Supported request fields: `messages`, `prompt`, `temperature`, `top_p`, `top_k`,
`max_tokens` / `max_completion_tokens`, `stream`, `stop` (string or list),
`presence_penalty`, `frequency_penalty`, `repetition_penalty`, `seed`.

Fields this server does not implement — `tools`, `logprobs`, `n > 1`, `response_format` —
are **accepted and ignored** rather than rejected, so SDK calls that set them keep working.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "minigpt",
        "messages": [{"role": "user", "content": "What is a token?"}],
        "temperature": 0.2,
        "max_tokens": 60
      }'
```

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "model": "minigpt",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "A token is a small chunk of text ..."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 8, "completion_tokens": 32, "total_tokens": 40}
}
```

Streaming emits standard `chat.completion.chunk` events terminated by `data: [DONE]`:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"minigpt","messages":[{"role":"user","content":"Hi"}],"stream":true}'
```

Requests are serialized around each decoding step, so one shared model can safely handle
concurrent clients. Interactive API docs are at `http://localhost:8000/docs`.

---

## Model presets and sizing

| Preset | Layers | Heads | Width | Context | Vocab | Params | Notes |
|---|---|---|---|---|---|---|---|
| `tiny` | 4 | 4 | 256 | 256 | 2048 | 3.7 M | Fast iteration, tiny datasets |
| `small` | 6 | 6 | 384 | 256 | 4096 | 12.2 M | **Default.** Good balance on an M2 Air |
| `medium` | 8 | 8 | 512 | 512 | 8192 | 29.5 M | Needs a few thousand rows to be worth it |
| `large` | 12 | 12 | 768 | 512 | 16384 | 97.5 M | Slow on 8 GB; use with a real corpus |

Rule of thumb: pick the **smallest** preset first. A bigger model on 100 rows does not
give better answers, it just overfits faster.

---

## Training tips: getting good answers

**Dataset size dominates everything else.**

| Rows | What to expect |
|---|---|
| < 100 | Memorizes the exact questions; unseen phrasings fail |
| 100–1,000 | Answers trained questions reliably, tolerates small rewordings |
| 1,000–10,000 | Starts generalizing across phrasings; `--size medium` becomes worthwhile |
| 10,000+ | Genuinely useful within its domain |

**Add paraphrases.** Three or four phrasings of the same question, all mapping to the same
answer, buy far more robustness than three more epochs.

**Keep answers consistent.** Contradictory answers to similar questions produce mush.
Pick one style — short and factual works best at this scale — and stick to it.

**Read the loss.** Training loss should fall below ~0.5. If it plateaus high, train
longer or raise `--lr`. If it hits ~0.01 while answers are still bad, your dataset is too
small, not your model.

**If the model repeats itself**, raise `--repetition-penalty` to 1.15–1.3, or lower
`--temperature`. Persistent looping means undertraining.

**If answers are cut off**, raise `--max-new-tokens`, and check the truncation warning at
training time — you may need a larger `--block-size`.

**If it ignores your question and answers a different one**, that question was in the
validation split and the model never trained on it. Expected on a small dataset; fix it
with more data, or `--val-ratio 0` once you have stopped tuning.

---

## Performance on a MacBook Air M2

Measured on an M2 (8 GB), `data/sample_qa.csv` (102 rows), `--size small`, `--block-size 192`:

| Task | Time |
|---|---|
| Tokenizer training | < 1 s |
| 40 epochs of training | ~83 s at `--block-size 192`, ~140 s at the default 256 |
| Generation | ~225 tokens/s (greedy, `small`, fp32) |
| Peak process memory | ~420 MB RSS |
| Checkpoint on disk | 43 MB |

Notes for Apple Silicon specifically:

- The backend defaults to **MPS** and the dtype to **float32** — MPS autocast is only
  reliable in fp16, and at this model size fp32 is fast enough and much more stable.
  Serve in half precision with `--dtype float16` if you want the smaller footprint.
- `torch.compile` is applied only on CUDA; it is not yet dependable on MPS.
- Fused AdamW is CUDA-only and is skipped automatically.
- If you run out of memory, lower `--batch-size` and raise `--grad-accum` to compensate —
  the effective batch stays the same.

---

## Project layout

```
minigpt/
  __init__.py      package exports
  cli.py           argparse entry point for every command
  config.py        GPTConfig dataclass + size presets
  tokenizer.py     byte-level BPE: training, encode, decode, save/load
  chat.py          prompt template and loss-mask construction
  data.py          CSV loading, datasets, train/val split
  model.py         the transformer: RMSNorm, RoPE, attention, SwiGLU, GPT
  train.py         training loops (SFT and pretraining)
  generate.py      KV-cached sampling
  checkpoint.py    save/load a checkpoint directory, device/dtype resolution
  server.py        FastAPI OpenAI-compatible API
data/
  sample_qa.csv    102-row demo dataset
tests/             pytest suite (73 tests)
```

---

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

The suite runs in a few seconds on CPU and covers, among other things:

- BPE round-trips (including emoji, accents and unseen words) and that merges compress.
- That cached incremental decoding is numerically identical to a full forward pass.
- That the model can overfit a single batch — the clearest signal gradients flow.
- That prompt tokens really are masked out of the loss.
- That the LR schedule warms up, decays monotonically, and lands on `min_lr`.
- Full API surface against `fastapi.TestClient`, including that streaming and
  non-streaming produce the same text, and that a client disconnecting mid-stream does not
  wedge the server.

---

## Troubleshooting

**`could not auto-detect the input column`** — your headers are not in the alias list. Pass
`--input-col` and `--output-col` explicitly.

**`every example was longer than block_size`** — raise `--block-size` (it must be a
multiple that fits memory) or shorten your answers.

**`sequence length N exceeds block_size M`** at generation time — the prompt plus history
exceeds the trained context window. `minigpt chat` trims history automatically; if you are
calling the API, send fewer messages.

**MPS out of memory** — lower `--batch-size`, raise `--grad-accum`, or drop to
`--size tiny`.

**Answers are nonsense** — check that training loss actually fell (`minigpt info --model
runs/demo`). If it is still above ~2, train more epochs or raise `--lr`.

**Port already in use** — `minigpt serve --port 8001`.

---

## Limitations (read this)

This is a **small model trained on your data only**. It is a working, honest
implementation of the GPT pipeline, not a competitor to a frontier model.

- It knows nothing that is not in your CSV. It will confidently make things up.
- With a few hundred rows it largely memorizes rather than generalizes.
- The default context window is 256 tokens — a few paragraphs.
- No tool use, no function calling, no retrieval, no safety filtering.
- The API server has no authentication or rate limiting. Bind it to `127.0.0.1`.

What it is genuinely good for: learning exactly how a transformer LLM works end to end,
and building a small, fast, fully local Q&A bot over a dataset you control.

---

## License

MIT.
