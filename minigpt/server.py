"""OpenAI-compatible HTTP API.

Implemented endpoints:

============================  ==================================================
``GET  /v1/models``           list the loaded model
``POST /v1/chat/completions`` chat, with optional SSE streaming
``POST /v1/completions``      raw text completion (legacy format)
``POST /v1/embeddings``       mean-pooled final hidden states
``GET  /health``              liveness probe
============================  ==================================================

Anything the OpenAI SDK sends that this server does not implement (``tools``,
``logprobs``, ``n>1`` ...) is accepted and ignored rather than rejected, so
existing client code keeps working. Requests are serialised behind a lock
because a single small model is shared by every caller.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Iterator, Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .chat import render_prompt
from .checkpoint import load_checkpoint
from .generate import SamplingParams, generate_stream
from .model import GPT
from .tokenizer import BPETokenizer


# --------------------------------------------------------------------- schemas
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | None = ""


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float | None = None
    seed: int | None = None
    n: int = 1

    model_config = {"extra": "allow", "protected_namespaces": ()}


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str | list[str] = ""
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    max_tokens: int | None = 128
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float | None = None
    seed: int | None = None

    model_config = {"extra": "allow", "protected_namespaces": ()}


class EmbeddingRequest(BaseModel):
    model: str | None = None
    input: str | list[str]

    model_config = {"extra": "allow", "protected_namespaces": ()}


class ServedModel:
    """Holds the weights and turns API requests into sampling calls."""

    def __init__(self, path: str, name: str, device: str = "auto", dtype: str = "auto"):
        self.model: GPT
        self.tokenizer: BPETokenizer
        self.model, self.tokenizer, self.meta, self.device = load_checkpoint(path, device, dtype)
        self.name = name
        self.lock = threading.Lock()
        self.created = int(time.time())

    def params(self, req: ChatRequest | CompletionRequest, default_max: int) -> SamplingParams:
        requested = getattr(req, "max_completion_tokens", None) or req.max_tokens or default_max
        return SamplingParams(
            max_new_tokens=min(int(requested), self.model.cfg.block_size),
            temperature=float(req.temperature),
            top_k=req.top_k,
            top_p=float(req.top_p),
            repetition_penalty=1.0 if req.repetition_penalty is None else float(req.repetition_penalty),
            presence_penalty=float(req.presence_penalty or 0.0),
            frequency_penalty=float(req.frequency_penalty or 0.0),
            seed=req.seed,
        )

    def stream_tokens(self, prompt: str, params: SamplingParams) -> Iterator[tuple[str, int]]:
        """Yield ``(text_delta, n_tokens_so_far)``, buffering partial UTF-8.

        The lock is taken around each decoding step rather than around the whole
        generation: holding it across a ``yield`` deadlocks the server when a
        client disconnects mid-stream and the abandoned generator is never
        resumed. Per-step locking still serialises the actual model calls.
        """
        ids = self.tokenizer.encode(prompt)
        tokens = generate_stream(self.model, self.tokenizer, ids, params, self.device)
        pending: list[int] = []
        count = 0
        try:
            while True:
                with self.lock:
                    try:
                        tid = next(tokens)
                    except StopIteration:
                        break
                count += 1
                pending.append(tid)
                text = self.tokenizer.decode(pending)
                if "\N{REPLACEMENT CHARACTER}" in text:
                    continue
                pending.clear()
                yield text, count
            if pending:
                yield self.tokenizer.decode(pending), count
        finally:
            tokens.close()

    def n_prompt_tokens(self, prompt: str) -> int:
        return len(self.tokenizer.encode(prompt))

    @torch.inference_mode()
    def embed(self, text: str) -> list[float]:
        ids = self.tokenizer.encode(text)[: self.model.cfg.block_size] or [self.tokenizer.bos_id]
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        with self.lock:
            h = self.model.drop(self.model.tok_emb(x))
            cos, sin = self.model.rope_tables(self.device, self.model.tok_emb.weight.dtype)
            cos, sin = cos[: len(ids)], sin[: len(ids)]
            for block in self.model.blocks:
                h, _ = block(h, cos, sin, None)
            h = self.model.norm(h)
        vec = h[0].float().mean(dim=0)
        vec = vec / vec.norm().clamp_min(1e-8)
        return vec.tolist()


def _stop_strings(stop: str | list[str] | None) -> list[str]:
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _truncate_at_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    cut = len(text)
    hit = False
    for s in stops:
        if s and (i := text.find(s)) != -1:
            cut = min(cut, i)
            hit = True
    return text[:cut], hit


def create_app(model_path: str, model_name: str = "minigpt", device: str = "auto", dtype: str = "auto") -> FastAPI:
    served = ServedModel(model_path, model_name, device, dtype)
    app = FastAPI(title="minigpt", version="0.1.0", description="OpenAI-compatible API for a local minigpt model")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )
    app.state.served = served

    def model_card() -> dict[str, Any]:
        return {
            "id": served.name,
            "object": "model",
            "created": served.created,
            "owned_by": "local",
            "context_window": served.model.cfg.block_size,
            "parameters": served.model.num_parameters(),
        }

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": served.name, "device": served.device.type}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {"object": "list", "data": [model_card()]}

    @app.get("/v1/models/{model_id}")
    def get_model(model_id: str) -> dict[str, Any]:
        if model_id != served.name:
            raise HTTPException(status_code=404, detail="model not found")
        return model_card()

    # ------------------------------------------------------------------- chat
    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatRequest):
        if not req.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty")
        prompt = render_prompt([m.model_dump() for m in req.messages])  # type: ignore[arg-type]
        params = served.params(req, default_max=256)
        stops = _stop_strings(req.stop)
        cid = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        n_prompt = served.n_prompt_tokens(prompt)

        if not req.stream:
            text, n_gen = "", 0
            for delta, n_gen in served.stream_tokens(prompt, params):
                text += delta
                trimmed, hit = _truncate_at_stop(text, stops)
                if hit:
                    text = trimmed
                    break
            finish = "length" if n_gen >= params.max_new_tokens else "stop"
            return {
                "id": cid, "object": "chat.completion", "created": created, "model": served.name,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text.strip()},
                    "logprobs": None,
                    "finish_reason": finish,
                }],
                "usage": {
                    "prompt_tokens": n_prompt,
                    "completion_tokens": n_gen,
                    "total_tokens": n_prompt + n_gen,
                },
            }

        def sse() -> Iterator[str]:
            def chunk(delta: dict, finish: str | None = None) -> str:
                payload = {
                    "id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": served.name,
                    "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}],
                }
                return f"data: {json.dumps(payload)}\n\n"

            yield chunk({"role": "assistant", "content": ""})
            text, n_gen, finish = "", 0, "stop"
            for delta, n_gen in served.stream_tokens(prompt, params):
                text += delta
                trimmed, hit = _truncate_at_stop(text, stops)
                if hit:
                    remainder = trimmed[len(text) - len(delta):]
                    if remainder:
                        yield chunk({"content": remainder})
                    break
                yield chunk({"content": delta})
            else:
                if n_gen >= params.max_new_tokens:
                    finish = "length"
            yield chunk({}, finish)
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------- legacy completions
    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        prompt = req.prompt if isinstance(req.prompt, str) else "".join(req.prompt)
        if not prompt:
            raise HTTPException(status_code=400, detail="prompt must not be empty")
        params = served.params(req, default_max=128)
        stops = _stop_strings(req.stop)
        cid = f"cmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        n_prompt = served.n_prompt_tokens(prompt)

        if not req.stream:
            text, n_gen = "", 0
            for delta, n_gen in served.stream_tokens(prompt, params):
                text += delta
                trimmed, hit = _truncate_at_stop(text, stops)
                if hit:
                    text = trimmed
                    break
            return {
                "id": cid, "object": "text_completion", "created": created, "model": served.name,
                "choices": [{"index": 0, "text": text, "logprobs": None,
                             "finish_reason": "length" if n_gen >= params.max_new_tokens else "stop"}],
                "usage": {"prompt_tokens": n_prompt, "completion_tokens": n_gen,
                          "total_tokens": n_prompt + n_gen},
            }

        def sse() -> Iterator[str]:
            text = ""
            for delta, _ in served.stream_tokens(prompt, params):
                text += delta
                trimmed, hit = _truncate_at_stop(text, stops)
                if hit:
                    break
                payload = {"id": cid, "object": "text_completion", "created": created,
                           "model": served.name,
                           "choices": [{"index": 0, "text": delta, "finish_reason": None}]}
                yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ------------------------------------------------------------- embeddings
    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingRequest):
        texts = [req.input] if isinstance(req.input, str) else list(req.input)
        data = [
            {"object": "embedding", "index": i, "embedding": served.embed(t)}
            for i, t in enumerate(texts)
        ]
        n = sum(served.n_prompt_tokens(t) for t in texts)
        return {"object": "list", "data": data, "model": served.name,
                "usage": {"prompt_tokens": n, "total_tokens": n}}

    return app
