import torch

from minigpt.chat import render_prompt
from minigpt.checkpoint import load_checkpoint
from minigpt.generate import SamplingParams, _filter, generate_stream, generate_text, stream_text


def test_top_k_keeps_exactly_k_tokens():
    logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
    out = _filter(logits.clone(), top_k=2, top_p=1.0)
    assert torch.isfinite(out).sum() == 2


def test_top_p_keeps_the_smallest_sufficient_set():
    logits = torch.log(torch.tensor([0.6, 0.3, 0.05, 0.05]))
    out = _filter(logits.clone(), top_k=None, top_p=0.8)
    assert torch.isfinite(out).sum() == 2  # 0.6 then 0.3 crosses 0.8


def test_top_p_always_keeps_the_argmax():
    logits = torch.log(torch.tensor([0.9, 0.05, 0.05]))
    out = _filter(logits.clone(), top_k=None, top_p=0.0)
    assert torch.isfinite(out).sum() == 1
    assert torch.argmax(out).item() == 0


def test_greedy_generation_is_deterministic(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    p = SamplingParams(max_new_tokens=16, temperature=0.0, repetition_penalty=1.0)
    prompt = render_prompt([{"role": "user", "content": "What is the capital of France?"}])
    a = generate_text(model, tok, prompt, p, device)
    b = generate_text(model, tok, prompt, p, device)
    assert a == b


def test_seeded_sampling_is_reproducible(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    prompt = render_prompt([{"role": "user", "content": "Hello"}])
    p = SamplingParams(max_new_tokens=16, temperature=1.0, seed=123)
    assert generate_text(model, tok, prompt, p, device) == generate_text(model, tok, prompt, p, device)


def test_generation_respects_max_new_tokens(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    prompt = render_prompt([{"role": "user", "content": "Hello"}])
    ids = tok.encode(prompt)
    out = list(generate_stream(model, tok, ids, SamplingParams(max_new_tokens=5, temperature=1.0), device))
    assert len(out) <= 5


def test_generation_stops_at_eos_and_never_emits_it(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    prompt = render_prompt([{"role": "user", "content": "Who are you?"}])
    ids = tok.encode(prompt)
    out = list(generate_stream(model, tok, ids, SamplingParams(max_new_tokens=64, temperature=0.0,
                                                               repetition_penalty=1.0), device))
    assert tok.eos_id not in out
    assert len(out) < 64  # stopped early because EOS was produced


def test_stream_text_matches_batch_text(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    prompt = render_prompt([{"role": "user", "content": "What is the capital of Japan?"}])
    p = SamplingParams(max_new_tokens=24, temperature=0.0, repetition_penalty=1.0)
    assert "".join(stream_text(model, tok, prompt, p, device)) == generate_text(model, tok, prompt, p, device)


def test_trained_model_answers_a_memorised_question(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    prompt = render_prompt([{"role": "user", "content": "What is the capital of France?"}])
    p = SamplingParams(max_new_tokens=24, temperature=0.0, repetition_penalty=1.0)
    assert "Paris" in generate_text(model, tok, prompt, p, device)


def test_generation_never_exceeds_the_context_window(checkpoint_dir):
    model, tok, _, device = load_checkpoint(checkpoint_dir, device="cpu")
    prompt = render_prompt([{"role": "user", "content": "word " * 200}])
    ids = tok.encode(prompt)[-model.cfg.block_size:]
    out = list(generate_stream(model, tok, ids, SamplingParams(max_new_tokens=32, temperature=0.0), device))
    assert len(ids) + len(out) <= model.cfg.block_size
