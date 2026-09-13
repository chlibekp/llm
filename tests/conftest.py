"""Shared fixtures: a tiny real checkpoint, trained for a handful of steps.

Training for real (rather than using random weights) keeps the end-to-end tests
honest - they exercise the same code path a user does, just at a size that runs
in a couple of seconds on CPU.
"""

import pytest
import torch

from minigpt.checkpoint import save_checkpoint
from minigpt.config import GPTConfig
from minigpt.data import ChatDataset
from minigpt.model import GPT
from minigpt.tokenizer import BPETokenizer

PAIRS = [
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Japan?", "Tokyo."),
    ("Who are you?", "I am minigpt."),
    ("Hello", "Hello! How can I help?"),
]


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory):
    torch.manual_seed(0)
    rows = [(q, a, None) for q, a in PAIRS] * 8
    corpus = [q + " " + a for q, a in PAIRS] * 10
    tok = BPETokenizer.train(corpus, vocab_size=400)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2,
                    n_head=2, n_embd=64, dropout=0.0)
    model = GPT(cfg)
    ds = ChatDataset(rows, tok, cfg.block_size)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(60):
        _, loss, _ = model(x, targets=y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = tmp_path_factory.mktemp("ckpt")
    save_checkpoint(out, model, tok, {"task": "test"})
    return out
