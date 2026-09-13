import torch

from minigpt.config import GPTConfig
from minigpt.model import GPT


def make_model(**kw):
    cfg = GPTConfig(vocab_size=128, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0, **kw)
    return GPT(cfg).eval()


def test_forward_shapes_and_loss():
    m = make_model()
    x = torch.randint(0, 128, (2, 16))
    logits, loss, _ = m(x, targets=x)
    assert logits.shape == (2, 16, 128)
    assert loss.item() > 0


def test_inference_returns_only_last_position():
    m = make_model()
    logits, loss, _ = m(torch.randint(0, 128, (1, 16)))
    assert logits.shape == (1, 1, 128)
    assert loss is None


def test_ignore_index_is_respected():
    m = make_model()
    x = torch.randint(0, 128, (1, 8))
    y = torch.full_like(x, -100)
    y[0, -1] = x[0, -1]
    _, loss, _ = m(x, targets=y)
    assert torch.isfinite(loss)


def test_kv_cache_matches_full_forward():
    m = make_model()
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        full, _, _ = m(x, targets=x)
        caches = m.empty_cache(1, torch.device("cpu"), torch.float32)
        outs, pos = [], 0
        for i in range(12):
            logits, _, caches = m(x[:, i : i + 1], kv_caches=caches, pos_offset=pos)
            pos += 1
            outs.append(logits[:, -1])
        incremental = torch.stack(outs, dim=1)
    assert torch.allclose(incremental, full, atol=1e-4)


def test_prefill_then_continue_matches_full_forward():
    m = make_model()
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        full, _, _ = m(x, targets=x)
        caches = m.empty_cache(1, torch.device("cpu"), torch.float32)
        _, _, caches = m(x[:, :8], kv_caches=caches, pos_offset=0)
        tail, _, _ = m(x[:, 8:], kv_caches=caches, pos_offset=8)
    assert torch.allclose(tail[:, -1], full[:, -1], atol=1e-4)


def test_grouped_query_attention_runs():
    m = make_model(n_kv_head=2)
    logits, _, _ = m(torch.randint(0, 128, (1, 8)))
    assert logits.shape == (1, 1, 128)


def test_weight_tying():
    m = make_model()
    assert m.lm_head.weight is m.tok_emb.weight


def test_sequence_longer_than_block_size_raises():
    m = make_model()
    try:
        m(torch.randint(0, 128, (1, 64)))
    except ValueError as exc:
        assert "block_size" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_model_can_overfit_a_single_batch():
    """The clearest end-to-end signal that gradients flow correctly."""
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    m = GPT(cfg)
    x = torch.randint(0, 64, (4, 16))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = None
    for _ in range(120):
        _, loss, _ = m(x, targets=x)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2


def test_loss_only_matches_the_dense_loss():
    import torch

    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (3, 10))
    y = torch.randint(0, cfg.vocab_size, (3, 10))
    y[:, :4] = -100                      # prompt positions
    y[2, :] = -100                       # a fully masked row
    with torch.no_grad():
        _, dense, _ = model(x, targets=y)
        logits, sparse, _ = model(x, targets=y, loss_only=True)
    assert logits is None
    assert torch.allclose(dense, sparse, atol=1e-5)


def test_loss_only_with_no_supervised_positions_still_backprops():
    import torch

    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    y = torch.full((2, 8), -100)
    _, loss, _ = model(x, targets=y, loss_only=True)
    loss.backward()
    assert float(loss.detach()) == 0.0
