import math

import torch

from minigpt.checkpoint import load_checkpoint, resolve_device, resolve_dtype, save_checkpoint
from minigpt.config import GPTConfig
from minigpt.data import ChatDataset
from minigpt.model import GPT
from minigpt.tokenizer import BPETokenizer
from minigpt.train import TrainConfig, lr_at


def test_lr_warms_up_then_decays():
    cfg = TrainConfig(lr=1e-3, warmup_ratio=0.1, min_lr_ratio=0.1)
    total = 100
    assert lr_at(0, total, cfg) < cfg.lr
    assert math.isclose(lr_at(9, total, cfg), cfg.lr, rel_tol=1e-6)     # end of warmup
    assert lr_at(50, total, cfg) < cfg.lr
    assert math.isclose(lr_at(total - 1, total, cfg), cfg.lr * 0.1, rel_tol=1e-2)


def test_lr_is_monotonic_after_warmup():
    cfg = TrainConfig(lr=1e-3, warmup_ratio=0.1)
    values = [lr_at(s, 100, cfg) for s in range(10, 100)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_optimizer_excludes_norms_from_weight_decay():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32))
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95))
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])


def test_checkpoint_roundtrip(tmp_path):
    tok = BPETokenizer.train(["hello world " * 50], vocab_size=300)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg).eval()
    save_checkpoint(tmp_path, model, tok, {"task": "test"})
    for name in ("config.json", "model.pt", "tokenizer.json"):
        assert (tmp_path / name).exists()

    loaded, loaded_tok, meta, device = load_checkpoint(tmp_path, device="cpu")
    assert meta["task"] == "test"
    assert loaded_tok.vocab_size == tok.vocab_size
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(x)[0], loaded(x)[0], atol=1e-5)


def test_loading_a_non_checkpoint_directory_fails(tmp_path):
    try:
        load_checkpoint(tmp_path, device="cpu")
    except SystemExit as exc:
        assert "minigpt checkpoint" in str(exc)
    else:
        raise AssertionError("expected SystemExit")


def test_resolve_device_and_dtype():
    assert resolve_device("cpu").type == "cpu"
    assert resolve_dtype("float32", torch.device("cpu")) is torch.float32
    assert resolve_dtype("auto", torch.device("cpu")) is torch.float32


def test_end_to_end_training_reduces_loss():
    torch.manual_seed(0)
    tok = BPETokenizer.train(["capital of France is Paris " * 40], vocab_size=350)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=48, n_layer=2, n_head=2, n_embd=64, dropout=0.0)
    model = GPT(cfg)
    ds = ChatDataset([("capital of France", "Paris", None)] * 4, tok, cfg.block_size)
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    y = torch.stack([ds[i][1] for i in range(len(ds))])
    opt = model.configure_optimizer(3e-3, 0.1, (0.9, 0.95))
    losses = []
    for _ in range(50):
        _, loss, _ = model(x, targets=y)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0] * 0.3
