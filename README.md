# minigpt

A complete, from-scratch **generative pretrained transformer** in Python. You give it a
CSV of questions and answers, it trains its own tokenizer and its own model, and then you
can talk to it — from a terminal REPL or through an **OpenAI-compatible HTTP API**.

Everything is written to run on a **MacBook Air M2**: no GPU cluster, no downloads, no
pretrained weights. Training the bundled demo takes about **2.5 minutes** on an M2 and
about 420 MB of memory.

When a from-scratch model is too small to make sense of your data, there is a second path.
`minigpt lora` loads a small **pretrained** model (SmolLM2-360M-Instruct by default) into
the same transformer code and fine-tunes it with LoRA adapters. See
[Fine-tuning a pretrained model (LoRA)](#fine-tuning-a-pretrained-model-lora).

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
- [Fine-tuning a pretrained model (LoRA)](#fine-tuning-a-pretrained-model-lora)
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
| Pretrained models | `minigpt/hf.py` | Loads Llama-architecture Hugging Face weights (e.g. SmolLM2) into the same model code |
| LoRA | `minigpt/lora.py` | Low-rank adapters on a frozen base, saved as a small `adapter.pt` |
| CLI | `minigpt/cli.py` | `train`, `pretrain`, `lora`, `chat`, `generate`, `serve`, `info`, `tokenize` |
| API | `minigpt/server.py` | FastAPI server speaking the OpenAI REST dialect, including SSE streaming |

The only heavy dependency is PyTorch (used for tensors, autograd and the fused attention
kernel). The from-scratch path uses no pretrained weights and no Hugging Face code. The
LoRA path adds three small libraries: `huggingface_hub` for the download, `safetensors`
for the weights and `tokenizers` for the pretrained tokenizer. `transformers` is not
used. The model itself still runs on `minigpt/model.py`.

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

## Fine-tuning a pretrained model (LoRA)

A model trained from zero on a laptop has never read enough text to understand language,
so it mostly repeats what it memorized. A pretrained model already writes fluent,
sensible English. LoRA teaches it your data's answers without retraining its weights.

```bash
# see what the base model says before any fine-tuning
minigpt chat --model HuggingFaceTB/SmolLM2-360M-Instruct

# fine-tune it on your CSV (the ~700 MB base is downloaded once and cached)
minigpt lora --data data/qa.csv --out runs/lora

# use the result exactly like any other checkpoint
minigpt chat --model runs/lora --temperature 0.3
minigpt serve --model runs/lora
```

**How it works.**

- `minigpt/hf.py` downloads only `config.json`, the tokenizer files and
  `model.safetensors`, then renames the weights onto `minigpt/model.py`. SmolLM2 is a
  Llama-architecture model: RMSNorm, RoPE, GQA, SwiGLU and tied embeddings, the same
  design minigpt implements. No second model implementation is involved.
  `tests/test_lora.py` checks the logits against an independent Llama reference
  implementation.
- `minigpt/lora.py` freezes the base and wraps every attention and MLP projection:
  `W x + (alpha/r) · B A x`, with `B` starting at zero so step 0 *is* the pretrained model.
  At rank 16 that trains **8.7M parameters while 362M stay frozen**. The frozen weights
  stay in bfloat16 and need no gradients or AdamW state, so optimizer memory drops
  from ~2.9 GB to ~70 MB.
- Prompts use the base model's own ChatML template (`<|im_start|>user … <|im_end|>`),
  including its default system prompt, so the fine-tune starts from the format the model
  was instruction-tuned on. The loss covers only the answer tokens, as with `train`.
- The checkpoint holds only `adapter.pt` (~35 MB), the tokenizer, and a reference to the
  base model pinned to its exact commit. Loading it reloads the cached base, applies the
  adapters and **merges** them into the weights, so generation runs at base-model speed.

**Memory and speed on an 8 GB M2.** Defaults are `--batch-size 4 --grad-accum 8`
(effective batch 32), `--lr 2e-4`, `--rank 16`, with activation checkpointing on.
Measured on an 8 GB M2 with SmolLM2-360M on a 2,000-row sample of `data/qa.csv`
(~100 tokens per example):

| | |
|---|---|
| Peak GPU memory, whole run including validation | **~2.5 GB** |
| Time per optimizer step (32 examples) | ~11 s (~8 s with `--no-grad-checkpoint`, ~2× the activation memory) |
| One epoch of the full 85k-row `data/qa.csv` | **~7–8 hours** |
| Adapter checkpoint | 35 MB |

Two things keep that memory flat. MPS runs asynchronously, so `lora` waits for the device
after every micro-batch; otherwise queued batches pile up (the first measurement without
this peaked at 5.6 GB). It also returns the allocator's cached blocks after every step.
Neither costs measurable time at this model size.

To fit a time budget, cap the run: `--max-tokens 3000000` is about 30k examples, or
~3 h. Or train on fewer rows, or pick a smaller base with
`--base HuggingFaceTB/SmolLM2-135M-Instruct` (~3× less compute per token).
`--eval-every 200` shows validation progress on long runs, and `--save best` (the default
here) keeps the best adapter.

| Flag | Default | Meaning |
|---|---|---|
| `--base` | `HuggingFaceTB/SmolLM2-360M-Instruct` | Hub id or local directory of a Llama-architecture model |
| `--rank` / `--alpha` | `16` / `32` | Adapter rank and scale (update is `alpha/rank · BA`) |
| `--targets` | `all` | `all` projections, or `attn` only (q/k/v/o: fewer params, less memory) |
| `--lora-dropout` | `0.0` | Dropout on the adapter input |
| `--block-size` | `512` | Truncate training examples to this many tokens |
| `--base-dtype` | `auto` | Frozen-weight precision (`auto`: bfloat16 on MPS/CUDA, float32 on CPU) |
| `--epochs` | `1` | Passes over the CSV; `--max-steps` / `--max-tokens` cut it short |
| `--val-ratio` / `--save` | `0.02` / `best` | Held-out rows, and keep the lowest-validation-loss adapter |

All the usual optimization flags from `train` apply too (`--grad-checkpoint`,
`--eval-every`, `--log-every`, …). `--optimizer muon` is rejected, because LoRA adapters use
AdamW. Only Llama-architecture models load (SmolLM2 135M/360M/1.7B, and other
`LlamaForCausalLM` checkpoints without RoPE scaling). Anything else is refused with a
message, not loaded wrong.

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
| QK-norm *(`compact`)* | RMSNorm on each head's queries and keys | Bounds attention logits, so training stays stable at higher LR |
| Logit soft-cap *(`compact`)* | `30 · tanh(logits / 30)` | Stops a small model from pushing a few logits to extremes |
| Zero-init projections *(`compact`)* | Residual output projections start at 0 | Every block starts as the identity, so early training is faster |

The `compact` preset also changes the shape. It is deep and thin (12 layers × 320 wide,
8 query heads sharing 2 KV heads) with an 8192-token vocabulary. At this parameter count,
depth beats width (MobileLLM, 2024). A bigger vocabulary packs more text into each
256-token window. All three switches are fields in `config.json`. Older checkpoints
have none of these fields, load with all three off, and behave exactly as before.

**Muon optimizer** (`--optimizer muon`, `minigpt/optim.py`). The 2-D weight matrices inside
the blocks are updated with Muon: momentum SGD whose update is orthogonalized by five
Newton–Schulz iterations. The embedding (and so the tied head) and the norm gains stay on
fused AdamW. Muon updates are rescaled to match AdamW's RMS, so both share `--lr` and
the schedule. It keeps one state buffer per matrix instead of AdamW's two.

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
- **Dynamic padding** — each batch is padded to its own longest example rather than to
  `--block-size`, and a length-grouped sampler keeps every batch internally near-uniform
  in length. Disable with `--no-dynamic-padding`.
- **Sparse loss head** — during SFT the question and the padding carry the ignore label,
  so only the answer positions are projected through the output head.
- **Precision** — `--amp auto` trains in float32 on MPS and CPU, and in bfloat16 (or
  float16 with loss scaling) on CUDA. On an M2, bf16 autocast was *slower* than float32
  (502 vs 378 ms/step), because the per-op casts cost more than the narrower arithmetic
  saves.
- **No per-step device sync** — the supervised positions are located on the CPU by the
  data loader and passed in. Finding them on the accelerator would mean `nonzero`, whose
  output shape depends on the data and so stalls the pipeline once per step. Packed
  pretraining supervises every position, so it uses the plain dense loss.
- **Epochs** — when `--epochs` is not given, `train` picks enough epochs for ~3000
  optimizer steps, capped at 30: the 100-row demo gets 30, an 80k-row CSV gets 3.
  `--max-steps` and `--max-tokens` stop a run early, even mid-epoch, with the LR schedule
  fitted to the shortened run.

At this model size an M2 is **compute-bound**: tokens per second barely move with the
micro-batch (8.1k tok/s at 16 sequences vs 8.4k at 32), while activation memory grows in
proportion to it (~1.4 GB vs ~3.5 GB of GPU memory). That shapes the defaults:

- `--batch-size 16 --grad-accum 4`: the same effective batch of 64 (and the same
  `--lr 6e-4`) as one 64-sequence step, at a quarter of the activation memory. A single
  64-sequence step does not fit comfortably next to macOS on an 8 GB machine, and once it
  swaps, training slows by orders of magnitude. Want even less memory? `--batch-size 8
  --grad-accum 8` halves it again for ~10% speed.
- `RMSNorm` uses the fused `F.rms_norm`, RoPE tables are cached per `(device, dtype)`
  instead of being re-cast on each of the 12 calls per forward, and grouped-query
  attention uses SDPA's own KV broadcast rather than materializing the expanded tensors.

`--compile` is worth trying on MPS as well as CUDA: fusing the remaining elementwise
chains attacks the dispatch count directly. It compiles with `dynamic=True`, since
dynamic padding means shapes vary between batches.

`--rope-contiguous` pairs channel `i` with `i + head_dim/2` instead of pairing neighbors,
which makes the rotation read contiguous slices rather than strided ones. It is a
different convention, not a drop-in: a model trained one way produces garbage under the
other, so it is off by default and recorded in `config.json`.

On a dataset small enough to memorize, `--dropout 0` is often both better *and* faster —
it removes ~19 kernels and their mask allocations from every forward pass.
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
| `--size` | `small` | Preset: `tiny`, `small`, `medium`, `large`, `compact` |
| `--n-layer` / `--n-head` / `--n-kv-head` / `--n-embd` / `--block-size` / `--vocab-size` / `--dropout` | preset | Individual overrides |
| `--epochs` | auto | Passes over the dataset (auto: ~3000 steps, at most 30 epochs) |
| `--max-steps` / `--max-tokens` | `0` | Stop after this many steps / tokens, even mid-epoch (`0` = no cap) |
| `--batch-size` | `16` | Examples per micro-batch — activation memory scales with this |
| `--grad-accum` | `4` | Micro-batches per optimizer step (effective batch 64) |
| `--tokenizer-sample-mb` | `20` | Train the tokenizer on an evenly spaced sample of this many MB (`0` = all) |
| `--lr` | `6e-4` | Peak learning rate |
| `--optimizer` | `adamw` | `muon`: Muon on the hidden matrices, AdamW on embeddings and norms |
| `--weight-decay` / `--warmup-ratio` | `0.1` / `0.05` | Regularization and schedule |
| `--val-ratio` | `0.1` | Held-out fraction (`0` disables validation) |
| `--amp` | `auto` | Autocast dtype: `auto` (float32 on MPS/CPU, bf16 on CUDA), `bf16`, `fp16`, `off` |
| `--no-dynamic-padding` | off | Pad every batch to `--block-size` instead |
| `--num-workers` | `0` | DataLoader worker processes |
| `--compile` | off | `torch.compile` the model (CUDA/MPS) |
| `--grad-checkpoint` | off | Recompute activations in backward: much less memory, ~30% slower |
| `--rope-contiguous` | off | Faster RoPE layout; **breaks older checkpoints** |
| `--save` | `last` | `last` or `best` |
| `--patience` | `0` | Early-stop after N epochs without improvement (0 = off) |
| `--train-on-prompt` | off | Also compute loss on the question |
| `--init-from` | — | Warm-start weights and tokenizer from another checkpoint |
| `--device` | `auto` | `auto`, `mps`, `cuda`, `cpu` |
| `--seed` | `1337` | Reproducibility |

### `minigpt lora` — LoRA fine-tuning of a pretrained model

```bash
minigpt lora --data data/qa.csv --out runs/lora --base HuggingFaceTB/SmolLM2-360M-Instruct
```

Flags and memory/speed numbers are in
[Fine-tuning a pretrained model (LoRA)](#fine-tuning-a-pretrained-model-lora). `chat`,
`generate`, `serve` and `tokenize` accept the resulting directory, or a Hugging Face model id
directly, e.g. `--model HuggingFaceTB/SmolLM2-360M-Instruct`.

### `minigpt pretrain` — next-token pretraining on raw text

```bash
minigpt pretrain --text corpus.txt --out runs/base --size small
```

Same optimization flags (default `--epochs 1`), plus `--stride` to control window overlap
when packing the corpus into training blocks, and `--encode-workers` for the encoder.

The corpus is never loaded into RAM:

- The tokenizer is trained on an evenly spaced ~20 MB sample (`--tokenizer-sample-mb`).
  A few thousand merges are settled long before hundreds of MB.
- Token ids are encoded by up to 4 worker processes that import only the tokenizer (no
  torch). They are handed a few 1 MB chunks at a time, and the ids are streamed to
  `<out>/tokens.bin` (uint16 when the vocab fits), which is then memory-mapped.
- Both are **cached** in `<out>`. Rerunning on the same, unmodified corpus with the same
  settings skips straight to training. Edit the corpus or change the vocabulary and they
  are rebuilt.

On an M2 (8 GB), with the full 206 MB `data/train.txt`, a first `--size tiny --max-steps
20` run finished in 17 s end to end: tokenizer training, encoding (6 s) and training. With
the cache warm, a rerun reached training in ~3 s. The main process stayed under 700 MB RSS.

**Fastest way to pretrain on a large corpus (Apple Silicon):**

```bash
minigpt pretrain --text data/train.txt --out runs/base \
  --size tiny --dropout 0 --rope-contiguous --log-every 100 --val-ratio 0.01
```

| `--size` | Params | Speed on M2 | One epoch of `data/train.txt` (~60M tokens) |
|---|---|---|---|
| `tiny` | 3.7M | ~0.55 s / step of 64×256 | ~40 min |
| `small` | 12.2M | ~1.4 s / step of 64×256 | ~80 min |

- Every run prints an `eta`. To fit a time budget, pass `--max-tokens` (or `--max-steps`).
  The LR schedule is fitted to the shortened run, so stopping early still anneals properly.
- `tiny` on ~60M tokens is close to the classic ~20-tokens-per-parameter budget. `small`
  would want several times more data than one epoch provides, so use it when you can
  afford the time.
- `--dropout 0`: one epoch over a big corpus will not overfit; dropout only costs time.
- `--rope-contiguous`: faster RoPE kernel. A later `train --init-from runs/base` must pass
  it too, along with the same `--size`.
- Do **not** use `--grad-checkpoint` unless you run out of memory; it trades ~30% speed
  for memory.
- `--val-ratio 0.01` keeps evaluation short while still validating on plenty of text.

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
| `compact` | 12 | 8 (2 KV) | 320 | 256 | 8192 | 15.6 M | Deep and thin, with QK-norm, soft-cap and zero-init; for pretrain → fine-tune |

Rule of thumb: pick the **smallest** preset first. A bigger model on 100 rows does not
give better answers, it just overfits faster.

---

## Training tips: getting good answers

### Why it only answers questions copied from the CSV

A model trained from random initialization knows no English at all. Train it on a few
hundred rows and it can only **memorize** them. A question copied from the CSV gets
the stored answer back. Anything else comes out as noise. The tell is in `minigpt info`:
validation loss bottoms out after a couple of epochs and then *rises*.

The architecture is not the bottleneck. RoPE, RMSNorm, SwiGLU and GQA are already the
modern defaults, and a better design buys maybe 1.3–1.5× more out of the same data. More
tokens buy orders of magnitude. The model has to learn the language first, from raw text,
and only then learn the Q&A format.

### Recipe: sensible answers on an 8 GB M2

**1. Pretrain on raw text** so the model learns English. `data/train.txt` is ~206 MB
(~60M tokens):

```bash
minigpt pretrain --text data/train.txt --out runs/base --size compact --optimizer muon \
  --batch-size 8 --grad-accum 8 --val-ratio 0.01
```

`compact` + Muon is the architecture built for this workflow (see
[Model architecture](#3-model-architecture)). It has not been benchmarked against
`small` on this corpus yet. `--size small --dropout 0` with the default AdamW is the
measured fallback, at about 80 min per epoch. `compact` does roughly 1.3× the compute
per token of `small`.

`--batch-size 8 --grad-accum 8` keeps the effective batch at 64 with about half the
activation memory of the defaults, for ~10% speed. Pretraining 10 steps on
`data/sample_train.txt` does nothing useful. You need the full corpus.

**2. Fine-tune on the full Q&A set**, starting from those weights:

```bash
minigpt train --data data/qa.csv --out runs/chat --init-from runs/base \
  --size compact --optimizer muon --batch-size 8 --grad-accum 8 --save best
```

Use the 85k-row `data/qa.csv`, not the 102-row `data/sample_qa.csv`. At this size
validation loss means something, so `--save best` is the right choice. Pass the same
`--size` to both stages. Add `--rope-contiguous` only if you pretrained a non-`compact`
preset with it; `compact` already uses that layout.

**3. Sample conservatively.** A small model drifts fast at high temperature:

```bash
minigpt chat --model runs/chat --temperature 0.3 --repetition-penalty 1.15
```

**Set expectations.** `data/qa.csv` covers 641 different projects. A ~12M-parameter model
cannot store that many specific facts. The best case after a few hours on an M2 is
fluent, on-topic answers that are often **factually wrong**. For answers that make sense
from the first step, use [the LoRA path](#fine-tuning-a-pretrained-model-lora) instead.

### General tips

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

- The backend defaults to **MPS**, and training and inference both default to
  **float32**. On an M2, bf16 autocast trained ~25% slower than float32. Serve in half
  precision with `--dtype float16` if you want the smaller footprint.
- Fused AdamW is used on CUDA and MPS when the installed torch supports it.
- Fine-tuning on the 85k-row `data/qa.csv` runs ~0.55 s/step at the defaults, and auto
  picks 3 epochs: ~35 min in total.
- `minigpt lora` on SmolLM2-360M peaks at ~2.5 GB of GPU memory and takes ~11 s per
  32-example step. Inference on the merged model runs at ~38 tokens/s in bfloat16.
- If you run out of memory, lower `--batch-size` and raise `--grad-accum` to compensate —
  the effective batch stays the same, and on this hardware so does the speed.

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
  optim.py         Muon (+ AdamW for embeddings and norms)
  hf.py            load pretrained Llama-style Hugging Face models + their tokenizer
  lora.py          LoRA adapters: apply, save, merge
  train.py         training loops (SFT and pretraining)
  generate.py      KV-cached sampling
  checkpoint.py    save/load a checkpoint directory, device/dtype resolution
  server.py        FastAPI OpenAI-compatible API
data/
  sample_qa.csv    102-row demo dataset
  qa.csv           85k-row Q&A set for real fine-tuning
  train.txt        ~206 MB raw-text pretraining corpus (not committed)
tests/             pytest suite (115 tests)
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
- That pretrained Llama weights loaded into minigpt reproduce an independent Llama
  forward pass, and that LoRA starts as the base model, trains only the adapters, and
  merges back exactly. This runs against a tiny model built offline, with no downloads.
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

**Answers are nonsense** — if you need sensible answers quickly, use `minigpt lora` on a
pretrained base. For a from-scratch model, check that training loss actually fell (`minigpt info --model
runs/demo`). If it is still above ~2, train more epochs or raise `--lr`.

**Good answers only when the prompt is copied from the CSV** — the model memorized a
small dataset and never learned the language. Pretrain on raw text first, then fine-tune
on more rows. See [Why it only answers questions copied from the
CSV](#why-it-only-answers-questions-copied-from-the-csv).

**Port already in use** — `minigpt serve --port 8001`.

---

## Limitations (read this)

This is a **small model trained on your data only**. It is a working, honest
implementation of the GPT pipeline, not a competitor to a frontier model.

- It knows nothing that is not in your CSV. It will confidently make things up.
- With a few hundred rows it largely memorizes rather than generalizes.
- Without pretraining on raw text it does not know the language at all, only your rows.
- Even with pretraining, a model this small learns style faster than facts. Expect
  fluent but often wrong answers on a broad dataset.
- The default context window is 256 tokens — a few paragraphs.
- No tool use, no function calling, no retrieval, no safety filtering.
- The API server has no authentication or rate limiting. Bind it to `127.0.0.1`.

What it is genuinely good for: learning exactly how a transformer LLM works end to end,
and building a small, fast, fully local Q&A bot over a dataset you control.

---

## License

MIT.
