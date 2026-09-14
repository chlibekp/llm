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


def test_precomputed_keep_index_matches_the_dense_loss():
    """The collate-supplied index must be exactly what nonzero would have found."""
    import torch

    from minigpt.config import GPTConfig
    from minigpt.data import dynamic_collate
    from minigpt.model import GPT

    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(cfg).eval()
    batch = [([5, 6, 7, 8], [-100, -100, 7, 8]), ([9, 10, 11], [-100, -100, 11])]
    x, y, keep = dynamic_collate(pad_id=0, multiple_of=8)(batch)
    with torch.no_grad():
        _, dense, _ = model(x, targets=y)
        _, sparse, _ = model(x, targets=y, loss_only=True, keep_index=keep)
        _, derived, _ = model(x, targets=y, loss_only=True)   # nonzero fallback
    assert torch.allclose(dense, sparse, atol=1e-5)
    assert torch.allclose(sparse, derived, atol=1e-6)


def test_rope_tables_are_cached_per_device_and_dtype():
    import torch

    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32))
    dev = torch.device("cpu")
    a = model.rope_tables(dev, torch.float32)
    assert model.rope_tables(dev, torch.float32)[0] is a[0]      # same object, no recast
    b = model.rope_tables(dev, torch.bfloat16)
    assert b[0].dtype is torch.bfloat16 and b[0] is not a[0]


def test_contiguous_rope_layout_runs_and_differs_from_interleaved():
    import torch

    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    kw = dict(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0)
    x = torch.randint(0, 64, (1, 8))
    torch.manual_seed(0)
    a = GPT(GPTConfig(**kw)).eval()
    torch.manual_seed(0)
    b = GPT(GPTConfig(**kw, rope_interleaved=False)).eval()
    with torch.no_grad():
        ya, yb = a(x)[0], b(x)[0]
    assert ya.shape == yb.shape
    assert not torch.allclose(ya, yb)   # different convention => not interchangeable


def test_rmsnorm_matches_the_manual_formula():
    import torch

    from minigpt import model as M

    norm = M.RMSNorm(16)
    with torch.no_grad():
        norm.weight.normal_()
    x = torch.randn(2, 4, 16)
    fused = norm(x)
    orig, M._HAS_F_RMS_NORM = M._HAS_F_RMS_NORM, False
    try:
        manual = norm(x)
    finally:
        M._HAS_F_RMS_NORM = orig
    assert torch.allclose(fused, manual, atol=1e-5)


def test_grouped_query_attention_matches_the_materialised_path():
    import torch

    from minigpt import model as M
    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4,
                    n_kv_head=2, n_embd=32, dropout=0.0)
    torch.manual_seed(0)
    model = GPT(cfg).eval()
    x = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        fast = model(x)[0]
        orig, M._HAS_ENABLE_GQA = M._HAS_ENABLE_GQA, False
        try:
            materialised = model(x)[0]
        finally:
            M._HAS_ENABLE_GQA = orig
    assert torch.allclose(fast, materialised, atol=1e-5)


def test_qk_norm_and_softcap_keep_cached_decoding_exact():
    import torch

    m = make_model(qk_norm=True, logit_softcap=5.0, n_kv_head=2)
    x = torch.randint(0, 128, (1, 12))
    with torch.no_grad():
        full, _, _ = m(x, targets=x)
        caches = m.empty_cache(1, torch.device("cpu"), torch.float32)
        outs = []
        for i in range(12):
            logits, _, caches = m(x[:, i : i + 1], kv_caches=caches, pos_offset=i)
            outs.append(logits[:, -1])
    assert torch.allclose(torch.stack(outs, dim=1), full, atol=1e-4)


def test_logit_softcap_bounds_logits_and_the_sparse_loss_agrees():
    import torch

    torch.manual_seed(0)
    m = make_model(logit_softcap=2.0)
    with torch.no_grad():
        for p in m.parameters():
            p.mul_(20)                       # blow the raw logits far past the cap
        x = torch.randint(0, 128, (2, 10))
        y = x.clone()
        y[:, :4] = -100
        logits, dense, _ = m(x, targets=y)
        _, sparse, _ = m(x, targets=y, loss_only=True)
    assert logits.abs().max() <= 2.0
    assert torch.allclose(dense, sparse, atol=1e-5)


def test_zero_init_proj_makes_every_block_start_as_identity():
    import torch

    m = make_model(zero_init_proj=True)
    for block in m.blocks:
        assert torch.count_nonzero(block.attn.o_proj.weight) == 0
        assert torch.count_nonzero(block.mlp.down_proj.weight) == 0


def test_configs_saved_before_the_new_fields_load_with_them_off():
    from minigpt.config import GPTConfig

    old = {"vocab_size": 64, "block_size": 16, "n_layer": 1, "n_head": 2, "n_kv_head": 2,
           "n_embd": 32, "mlp_ratio": 8 / 3, "dropout": 0.0, "bias": False,
           "rope_theta": 10000.0, "rope_interleaved": True, "tie_weights": True}
    cfg = GPTConfig.from_dict(old)
    assert not cfg.qk_norm and cfg.logit_softcap == 0.0 and not cfg.zero_init_proj
    assert not any("q_norm" in k for k in make_model().state_dict())


def test_compact_preset_builds():
    from minigpt.config import PRESETS, GPTConfig
    from minigpt.model import GPT

    m = GPT(GPTConfig(**PRESETS["compact"]))
    assert 14e6 < m.num_parameters() < 17e6
