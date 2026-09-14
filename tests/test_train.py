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


def test_train_model_end_to_end_with_dynamic_padding(tmp_path):
    """Drives the real loop: flat-tensor raw view, length sampler, keep index."""
    from minigpt.data import train_val_split
    from minigpt.train import train_model

    torch.manual_seed(0)
    rows = [(f"question number {i}", f"answer {i}", None) for i in range(40)]
    tok = BPETokenizer.train([f"question number {i} answer {i}" for i in range(40)], vocab_size=400)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=48, n_layer=2,
                    n_head=2, n_embd=64, dropout=0.0)
    train_rows, val_rows = train_val_split(rows, 0.1, seed=0)
    train_ds = ChatDataset(train_rows, tok, cfg.block_size)
    val_ds = ChatDataset(val_rows, tok, cfg.block_size)

    tcfg = TrainConfig(epochs=2, batch_size=4, lr=3e-3, device="cpu",
                       log_every=0, seed=0, amp="off")
    summary = train_model(GPT(cfg), tok, train_ds, val_ds, tcfg, tmp_path)

    assert summary["steps"] > 0
    assert len(summary["history"]) == 2                 # one eval per epoch
    assert (tmp_path / "model.pt").exists()
    loaded, _, _, _ = load_checkpoint(tmp_path, device="cpu")
    assert loaded.cfg.n_layer == 2                      # checkpoint is loadable


def test_raw_view_round_trips_through_flat_storage():
    from minigpt.data import dynamic_collate

    tok = BPETokenizer.train(["alpha beta gamma delta " * 40], vocab_size=400)
    rows = [("alpha beta", "gamma", None), ("beta", "delta gamma beta", None)]
    ds = ChatDataset(rows, tok, block_size=64)
    view = ds.raw_view()
    assert len(view) == len(ds)
    for i, (ids, labels) in enumerate(ds.examples):
        v_ids, v_labels = view[i]
        assert v_ids.tolist() == ids
        assert v_labels.tolist() == labels

    # collating the tensor view must equal collating the original lists
    a = dynamic_collate(ds.pad_id, 8)([view[0], view[1]])
    b = dynamic_collate(ds.pad_id, 8)(ds.examples)
    for t1, t2 in zip(a, b):
        assert torch.equal(t1, t2)


def test_step_and_token_budgets_cap_the_schedule():
    from minigpt.train import auto_epochs, plan_steps

    cfg = TrainConfig(epochs=5)
    assert plan_steps(100, 10_000, cfg) == 500
    assert plan_steps(100, 10_000, TrainConfig(epochs=5, max_steps=42)) == 42
    # 100 tokens per step -> 2_050 tokens need 21 steps
    assert plan_steps(100, 10_000, TrainConfig(epochs=5, max_tokens=2_050)) == 21
    assert auto_epochs(2) == 30          # tiny dataset: capped
    assert auto_epochs(1191) == 3        # ~80k rows at an effective batch of 64
    assert auto_epochs(10**6) == 1


def test_max_steps_stops_training_mid_epoch(tmp_path):
    from minigpt.train import train_model

    tok = BPETokenizer.train([f"question {i} answer {i}" for i in range(40)], vocab_size=400)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    ds = ChatDataset([(f"question {i}", f"answer {i}", None) for i in range(40)], tok, cfg.block_size)
    tcfg = TrainConfig(epochs=3, max_steps=3, batch_size=4, grad_accum=2, device="cpu",
                       log_every=0, amp="off")
    summary = train_model(GPT(cfg), tok, ds, None, tcfg, tmp_path)
    assert summary["steps"] == 3


def test_pretrain_reuses_tokenizer_and_tokens_until_the_corpus_changes(tmp_path, capsys):
    import os

    from minigpt.cli import main

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("".join(f"the fox number {i} jumps.\n" for i in range(400)), encoding="utf-8")
    out = tmp_path / "base"
    argv = ["pretrain", "--text", str(corpus), "--out", str(out), "--vocab-size", "300",
            "--n-layer", "1", "--n-head", "2", "--n-embd", "32", "--block-size", "32",
            "--max-steps", "1", "--batch-size", "4", "--grad-accum", "1", "--device", "cpu",
            "--encode-workers", "1", "--log-every", "0"]

    main(argv)
    first = capsys.readouterr().out
    assert "reusing" not in first
    main(argv)
    second = capsys.readouterr().out
    assert "reusing tokenizer" in second and "encoded tokens" in second

    with corpus.open("a", encoding="utf-8") as fh:
        fh.write("a brand new line.\n")
    os.utime(corpus, ns=(0, os.stat(corpus).st_mtime_ns + 10**9))
    main(argv)
    third = capsys.readouterr().out
    assert "reusing" not in third
