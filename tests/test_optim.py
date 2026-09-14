import pytest
import torch

from minigpt.config import GPTConfig
from minigpt.model import GPT
from minigpt.optim import MuonAdamW, orthogonalize


def test_orthogonalize_flattens_the_singular_values():
    torch.manual_seed(0)
    for shape in [(8, 32), (32, 8)]:  # rectangular Gaussians are well conditioned
        G = torch.randn(shape) * torch.linspace(0.2, 5.0, shape[1])  # uneven column scales
        X = orthogonalize(G)
        assert X.shape == G.shape
        s = torch.linalg.svdvals(X)
        assert s.min() > 0.5 and s.max() < 1.3


def test_muon_takes_the_block_matrices_and_leaves_the_rest_to_adamw():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, qk_norm=True))
    opt = model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "muon")
    muon = [g for g in opt.param_groups if g["use_muon"]]
    adam = [g for g in opt.param_groups if not g["use_muon"]]
    muon_ids = {id(p) for g in muon for p in g["params"]}
    assert all(p.dim() == 2 for g in muon for p in g["params"])
    assert id(model.tok_emb.weight) not in muon_ids          # embedding / tied head
    assert len(muon_ids) == 2 * 7                             # q k v o gate up down per block
    everything = {id(p) for g in opt.param_groups for p in g["params"]}
    assert everything == {id(p) for p in model.parameters()}
    assert all(g["weight_decay"] == 0.0 for g in adam if g["params"][0].dim() < 2)


def test_muon_rejects_non_matrix_parameters():
    with pytest.raises(ValueError):
        MuonAdamW([{"params": [torch.nn.Parameter(torch.ones(4))], "use_muon": True}], lr=1e-3)


def test_unknown_optimizer_name_fails():
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32))
    with pytest.raises(ValueError):
        model.configure_optimizer(1e-3, 0.1, (0.9, 0.95), "sgd")


def test_muon_follows_the_lr_set_on_its_groups():
    """The training loop writes lr into param_groups; both halves must obey it."""
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32))
    opt = model.configure_optimizer(1e-3, 0.0, (0.9, 0.95), "muon")
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randint(0, 64, (2, 16))
    model(x, targets=x)[1].backward()
    for g in opt.param_groups:
        g["lr"] = 0.0
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_muon_can_overfit_a_single_batch():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
                    qk_norm=True, logit_softcap=30.0, zero_init_proj=True)
    model = GPT(cfg)
    opt = model.configure_optimizer(3e-3, 0.0, (0.9, 0.95), "muon")
    x = torch.randint(0, 64, (4, 16))
    losses = []
    for _ in range(120):
        _, loss, _ = model(x, targets=x)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0] * 0.2


def test_train_model_runs_with_muon(tmp_path):
    from minigpt.checkpoint import load_checkpoint
    from minigpt.data import ChatDataset
    from minigpt.tokenizer import BPETokenizer
    from minigpt.train import TrainConfig, train_model

    tok = BPETokenizer.train([f"question {i} answer {i}" for i in range(40)], vocab_size=400)
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=32, n_layer=1, n_head=2, n_embd=32,
                    dropout=0.0, qk_norm=True, logit_softcap=30.0)
    ds = ChatDataset([(f"question {i}", f"answer {i}", None) for i in range(40)], tok, cfg.block_size)
    tcfg = TrainConfig(epochs=1, max_steps=3, batch_size=4, grad_accum=2, device="cpu",
                       log_every=0, amp="off", optimizer="muon")
    summary = train_model(GPT(cfg), tok, ds, None, tcfg, tmp_path)
    assert summary["steps"] == 3
    loaded, _, _, _ = load_checkpoint(tmp_path, device="cpu")
    assert loaded.cfg.qk_norm and loaded.cfg.logit_softcap == 30.0
