import json

import pytest
from fastapi.testclient import TestClient

from minigpt.server import create_app


@pytest.fixture(scope="module")
def client(checkpoint_dir):
    app = create_app(str(checkpoint_dir), model_name="test-model", device="cpu")
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_list_models(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "test-model"


def test_unknown_model_404s(client):
    assert client.get("/v1/models/nope").status_code == 404


def test_chat_completion_shape(client):
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0, "max_tokens": 24,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["finish_reason"] in ("stop", "length")
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_chat_completion_answers_memorised_question(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
        "temperature": 0, "max_tokens": 24,
    })
    assert "Paris" in r.json()["choices"][0]["message"]["content"]


def test_chat_completion_accepts_unknown_openai_fields(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 8, "temperature": 0,
        "tools": [], "tool_choice": "auto", "user": "someone", "logprobs": False,
    })
    assert r.status_code == 200


def test_chat_completion_rejects_empty_messages(client):
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_chat_streaming_sse_protocol(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Who are you?"}],
        "temperature": 0, "max_tokens": 24, "stream": True,
    })
    assert r.status_code == 200
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] in ("stop", "length")
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)


def test_streaming_and_non_streaming_agree(client):
    payload = {"messages": [{"role": "user", "content": "What is the capital of Japan?"}],
               "temperature": 0, "max_tokens": 24}
    once = client.post("/v1/chat/completions", json=payload).json()["choices"][0]["message"]["content"]
    r = client.post("/v1/chat/completions", json={**payload, "stream": True})
    streamed = "".join(
        json.loads(l[6:])["choices"][0]["delta"].get("content", "")
        for l in r.text.splitlines() if l.startswith("data: ") and l != "data: [DONE]"
    )
    assert streamed.strip() == once


def test_stop_string_truncates_the_answer(client):
    payload = {"messages": [{"role": "user", "content": "What is the capital of France?"}],
               "temperature": 0, "max_tokens": 24}
    full = client.post("/v1/chat/completions", json=payload).json()["choices"][0]["message"]["content"]
    marker = full[:2]
    cut = client.post("/v1/chat/completions", json={**payload, "stop": [marker]})
    assert marker not in cut.json()["choices"][0]["message"]["content"]


def test_legacy_completions(client):
    r = client.post("/v1/completions", json={"prompt": "Paris is", "max_tokens": 8, "temperature": 0})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert isinstance(body["choices"][0]["text"], str)


def test_completions_rejects_empty_prompt(client):
    assert client.post("/v1/completions", json={"prompt": ""}).status_code == 400


def test_embeddings(client):
    r = client.post("/v1/embeddings", json={"input": ["hello", "world"]})
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2
    assert len(data[0]["embedding"]) > 0
    norm = sum(v * v for v in data[0]["embedding"]) ** 0.5
    assert abs(norm - 1.0) < 1e-3


def test_abandoned_stream_does_not_leak_the_lock(checkpoint_dir):
    """A client that disconnects mid-stream must not wedge the server.

    Holding the model lock across a ``yield`` used to deadlock every later
    request, so this consumes one chunk, drops the generator, and checks that
    the next request still completes.
    """
    from minigpt.generate import SamplingParams
    from minigpt.server import ServedModel

    served = ServedModel(str(checkpoint_dir), "t", device="cpu")
    params = SamplingParams(max_new_tokens=16, temperature=0.0, repetition_penalty=1.0)
    gen = served.stream_tokens("<|bos|><|user|>Hello<|assistant|>", params)
    next(gen)
    del gen  # simulate the client going away
    import gc

    gc.collect()
    assert not served.lock.locked()
    assert list(served.stream_tokens("<|bos|><|user|>Hello<|assistant|>", params))
