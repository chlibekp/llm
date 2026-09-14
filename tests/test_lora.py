"""The pretrained + LoRA path, against a tiny Llama-style checkpoint built offline.

The fixture writes the same files a Hugging Face model repo ships (config.json,
tokenizer.json, tokenizer_config.json, model.safetensors), so nothing here
touches the network. A plain-torch re-implementation of the Llama forward pass,
written the way Hugging Face writes it, pins down the weight mapping and the
RoPE convention.
"""

import json

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")

from safetensors.torch import save_file  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers  # noqa: E402

from minigpt.chat import render_example, render_prompt  # noqa: E402
from minigpt.checkpoint import load_checkpoint, load_tokenizer, save_checkpoint  # noqa: E402
from minigpt.hf import HFTokenizer, load_pretrained, looks_like_pretrained  # noqa: E402
from minigpt.lora import LoRALinear, apply_lora, lora_state_dict, merge_lora  # noqa: E402

SYSTEM = "You are a tiny test model"
CONFIG = dict(model_type="llama", hidden_size=32, intermediate_size=72, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
              rms_norm_eps=1e-6, rope_theta=10000.0, tie_word_embeddings=True,
              hidden_act="silu", attention_bias=False, mlp_bias=False, rope_scaling=None)


@pytest.fixture(scope="module")
def hf_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-llama")
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    corpus = [f"question {i} is about topic {i % 7}. answer {i} explains it." for i in range(200)]
    tok.train_from_iterator(corpus, trainer)
    tok.save(str(d / "tokenizer.json"))
    template = ("{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
                "{{ '<|im_start|>system\n" + SYSTEM + "<|im_end|>\n' }}{% endif %}"
                "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
                "{% endfor %}")
    (d / "tokenizer_config.json").write_text(json.dumps(
        {"bos_token": "<|im_start|>", "eos_token": "<|im_end|>", "pad_token": "<|im_end|>",
         "chat_template": template}))

    cfg = dict(CONFIG, vocab_size=tok.get_vocab_size())
    (d / "config.json").write_text(json.dumps(cfg))
    g = torch.Generator().manual_seed(0)

    def w(*shape):
        return torch.randn(*shape, generator=g) * 0.2

    C, H, kv, hd, I = 32, 4, 2, 8, 72
    state = {"model.embed_tokens.weight": w(cfg["vocab_size"], C), "model.norm.weight": 1 + w(C)}
    for i in range(2):
        p = f"model.layers.{i}."
        state.update({
            p + "input_layernorm.weight": 1 + w(C), p + "post_attention_layernorm.weight": 1 + w(C),
            p + "self_attn.q_proj.weight": w(H * hd, C), p + "self_attn.k_proj.weight": w(kv * hd, C),
            p + "self_attn.v_proj.weight": w(kv * hd, C), p + "self_attn.o_proj.weight": w(C, H * hd),
            p + "mlp.gate_proj.weight": w(I, C), p + "mlp.up_proj.weight": w(I, C),
            p + "mlp.down_proj.weight": w(C, I),
        })
    save_file(state, str(d / "model.safetensors"))
    return d


def reference_llama_logits(d, ids: torch.Tensor) -> torch.Tensor:
    """Llama as Hugging Face computes it: rotate_half RoPE, repeat_kv GQA."""
    from safetensors.torch import load_file

    s, cfg = load_file(str(d / "model.safetensors")), json.loads((d / "config.json").read_text())
    C, H, kv, eps = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["rms_norm_eps"]
    hd, T = C // H, ids.size(1)

    def rms(x, weight):
        return weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

    def rotate_half(x):
        x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    inv = 1.0 / (cfg["rope_theta"] ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(T).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()

    x = s["model.embed_tokens.weight"][ids]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        h = rms(x, s[p + "input_layernorm.weight"])
        q = (h @ s[p + "self_attn.q_proj.weight"].T).view(1, T, H, hd).transpose(1, 2)
        k = (h @ s[p + "self_attn.k_proj.weight"].T).view(1, T, kv, hd).transpose(1, 2)
        v = (h @ s[p + "self_attn.v_proj.weight"].T).view(1, T, kv, hd).transpose(1, 2)
        q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
        k, v = k.repeat_interleave(H // kv, dim=1), v.repeat_interleave(H // kv, dim=1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + att.transpose(1, 2).reshape(1, T, C) @ s[p + "self_attn.o_proj.weight"].T
        h = rms(x, s[p + "post_attention_layernorm.weight"])
        mlp = F.silu(h @ s[p + "mlp.gate_proj.weight"].T) * (h @ s[p + "mlp.up_proj.weight"].T)
        x = x + mlp @ s[p + "mlp.down_proj.weight"].T
    x = rms(x, s["model.norm.weight"])
    return x @ s["model.embed_tokens.weight"].T


def test_pretrained_weights_reproduce_the_reference_llama(hf_dir):
    model, tok, ref = load_pretrained(str(hf_dir))
    model.eval()
    ids = torch.tensor([tok.encode("question 12 is about topic 5.")])
    with torch.no_grad():
        ours, _, _ = model(ids, targets=ids)
    assert torch.allclose(ours, reference_llama_logits(hf_dir, ids), atol=1e-4)
    assert model.lm_head.weight is model.tok_emb.weight
    assert ref["name"] == str(hf_dir.resolve())


def test_unsupported_architectures_are_refused(tmp_path):
    from minigpt.hf import config_from_hf

    with pytest.raises(SystemExit, match="model_type"):
        config_from_hf(dict(CONFIG, model_type="gpt2", vocab_size=10))
    with pytest.raises(SystemExit, match="rope_scaling"):
        config_from_hf(dict(CONFIG, vocab_size=10, rope_scaling={"type": "linear"}))


def test_hf_tokenizer_reads_specials_and_default_system(hf_dir):
    tok = HFTokenizer.from_pretrained_dir(hf_dir)
    assert tok.default_system == SYSTEM
    assert tok.eos_id == tok.pad_id == tok._tok.token_to_id("<|im_end|>")
    ids = tok.encode("<|im_start|>user\nquestion 3<|im_end|>")
    assert ids[0] == tok.bos_id and ids[-1] == tok.eos_id
    assert tok.decode(ids) == "user\nquestion 3"
    assert tok.decode(ids, skip_special=False) == "<|im_start|>user\nquestion 3<|im_end|>"


def test_chatml_rendering_matches_the_model_template(hf_dir):
    tok = HFTokenizer.from_pretrained_dir(hf_dir)
    prompt = render_prompt([{"role": "user", "content": "hi"}], tokenizer=tok)
    assert prompt == (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
                      "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    own = render_prompt([{"role": "system", "content": "Be terse."},
                         {"role": "user", "content": "hi"}], tokenizer=tok)
    assert SYSTEM not in own and own.startswith("<|im_start|>system\nBe terse.<|im_end|>\n")
    p, c = render_example("q", "a", tokenizer=tok)
    assert p.endswith("<|im_start|>assistant\n") and c == "a<|im_end|>"
    # minigpt's own format is untouched when no pretrained tokenizer is passed
    assert render_prompt([{"role": "user", "content": "hi"}]).startswith("<|bos|>")


def test_lora_starts_as_the_base_model_and_merges_exactly():
    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()
    x = torch.randint(0, 64, (2, 10))
    with torch.no_grad():
        base = model(x, targets=x)[0]
    n = apply_lora(model, rank=4, alpha=8)
    assert n == sum(p.numel() for name, p in model.named_parameters() if ".lora_" in name)
    assert all(not p.requires_grad for name, p in model.named_parameters() if ".lora_" not in name)
    with torch.no_grad():
        assert torch.equal(model(x, targets=x)[0], base)       # B = 0: identical at start
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_()
        adapted = model(x, targets=x)[0]
    assert not torch.allclose(adapted, base)
    assert set(lora_state_dict(model)) and all(".lora_" in k for k in lora_state_dict(model))
    merge_lora(model)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())
    assert not hasattr(model, "lora_config")
    with torch.no_grad():
        assert torch.allclose(model(x, targets=x)[0], adapted, atol=1e-5)


def test_lora_training_only_moves_the_adapters():
    from minigpt.config import GPTConfig
    from minigpt.model import GPT

    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
    apply_lora(model, rank=4, alpha=8)
    frozen = {k: v.clone() for k, v in model.state_dict().items() if ".lora_" not in k}
    opt = model.configure_optimizer(1e-2, 0.0, (0.9, 0.95))
    x = torch.randint(0, 64, (4, 16))
    losses = []
    for _ in range(30):
        _, loss, _ = model(x, targets=x)
        losses.append(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert losses[-1] < losses[0]
    assert all(torch.equal(model.state_dict()[k], v) for k, v in frozen.items())


def test_lora_in_half_precision_keeps_float32_adapters_with_gradients(hf_dir):
    model, tok, _ = load_pretrained(str(hf_dir), torch.bfloat16)
    apply_lora(model, rank=4, alpha=8)
    ids = torch.tensor([tok.encode("answer 7 explains it.")])
    _, loss, _ = model(ids, targets=ids)
    assert loss.dtype == torch.float32
    loss.backward()
    adapters = [p for n, p in model.named_parameters() if ".lora_A" in n]
    assert adapters and all(p.dtype == torch.float32 and p.grad is not None for p in adapters)


def test_lora_cli_end_to_end(hf_dir, tmp_path):
    from minigpt.cli import main
    from minigpt.generate import SamplingParams, generate_text

    csv = tmp_path / "qa.csv"
    csv.write_text("input,output\n" + "".join(f"question {i},answer {i} explains it.\n" for i in range(24)))
    out = tmp_path / "run"
    main(["lora", "--data", str(csv), "--out", str(out), "--base", str(hf_dir), "--rank", "4",
          "--max-steps", "3", "--batch-size", "4", "--grad-accum", "1", "--device", "cpu",
          "--log-every", "0", "--val-ratio", "0.2", "--save", "last"])

    assert (out / "adapter.pt").exists() and not (out / "model.pt").exists()
    payload = json.loads((out / "config.json").read_text())
    assert payload["lora"]["rank"] == 4 and payload["tokenizer"]["type"] == "hf"
    assert payload["base"]["name"] == str(hf_dir.resolve())

    model, tok, meta, device = load_checkpoint(out, device="cpu", dtype="float32")
    assert meta["task"] == "lora" and isinstance(tok, HFTokenizer)
    assert not any(isinstance(m, LoRALinear) for m in model.modules())   # merged
    base, _, _ = load_pretrained(str(hf_dir))
    ids = torch.tensor([tok.encode("question 3")])
    with torch.no_grad():
        assert not torch.allclose(model(ids, targets=ids)[0], base.eval()(ids, targets=ids)[0])
    prompt = render_prompt([{"role": "user", "content": "question 3"}], tokenizer=tok)
    generate_text(model, tok, prompt, SamplingParams(max_new_tokens=5, temperature=0), device)
    assert isinstance(load_tokenizer(out), HFTokenizer)


def test_saved_adapter_reloads_to_the_trained_weights(hf_dir, tmp_path):
    torch.manual_seed(0)
    model, tok, _ = load_pretrained(str(hf_dir))
    apply_lora(model, rank=4, alpha=8)
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, LoRALinear):
                m.lora_B.normal_()
    save_checkpoint(tmp_path, model, tok, {"task": "lora"})
    ids = torch.tensor([tok.encode("question 9 is about topic 2.")])
    with torch.no_grad():
        expected = model.eval()(ids, targets=ids)[0]
        loaded, _, _, _ = load_checkpoint(tmp_path, device="cpu", dtype="float32")
        assert torch.allclose(loaded(ids, targets=ids)[0], expected, atol=1e-5)


def test_pretrained_directory_loads_directly(hf_dir, tmp_path):
    assert looks_like_pretrained(hf_dir)
    assert not looks_like_pretrained(tmp_path)          # existing dir without a HF config
    model, tok, meta, _ = load_checkpoint(hf_dir, device="cpu")
    assert meta["task"] == "pretrained" and isinstance(tok, HFTokenizer)
