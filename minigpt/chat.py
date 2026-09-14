"""The prompt format (a.k.a. chat template) the model is trained and served with.

A conversation is rendered as a flat token stream:

    <|bos|><|system|>You are helpful.<|user|>What is 2+2?<|assistant|>4<|eos|>

The control tokens are single tokens from the tokenizer's reserved range, so the
model learns them as unambiguous role markers rather than as ordinary text. At
inference time we render everything up to and including the final
``<|assistant|>`` and let the model continue until it emits ``<|eos|>``.

A pretrained base brings its own format instead. Its tokenizer sets
``chat_format = "chatml"``, and the same functions render the turns that model
was instruction-tuned on::

    <|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n4<|im_end|>
"""

from __future__ import annotations

from typing import Iterable, Sequence, TypedDict

from .tokenizer import ASSISTANT, BOS, EOS, SYSTEM, USER, BPETokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


class Message(TypedDict):
    role: str
    content: str


IM_START, IM_END = "<|im_start|>", "<|im_end|>"


def _is_chatml(tokenizer) -> bool:
    return getattr(tokenizer, "chat_format", None) == "chatml"


def _render_chatml(messages: list[Message], system: str | None, default_system: str | None) -> str:
    # Mirrors the SmolLM2 template: the model's default system prompt is used
    # only when the conversation does not bring one of its own.
    if messages and messages[0]["role"] == "system":
        system, messages = messages[0]["content"], messages[1:]
    system = system or default_system
    parts = [f"{IM_START}system\n{system}{IM_END}\n"] if system else []
    for m in messages:
        if m["role"] not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported role: {m['role']!r}")
        parts.append(f"{IM_START}{m['role']}\n{m['content'] or ''}{IM_END}\n")
    parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def render_prompt(
    messages: Sequence[Message], system: str | None = None, tokenizer=None
) -> str:
    """Render a conversation, ending with an open assistant turn.

    Pass the ``tokenizer`` so a pretrained model gets its own chat format.
    """
    if _is_chatml(tokenizer):
        return _render_chatml(list(messages), system, tokenizer.default_system)
    parts = [BOS]
    msgs = list(messages)
    sys_text = system
    if msgs and msgs[0]["role"] == "system":
        sys_text = msgs[0]["content"]
        msgs = msgs[1:]
    if sys_text:
        parts.append(SYSTEM + sys_text)
    for m in msgs:
        role = m["role"]
        content = m["content"] or ""
        if role == "user":
            parts.append(USER + content)
        elif role == "assistant":
            parts.append(ASSISTANT + content + EOS)
        elif role == "system":
            parts.append(SYSTEM + content)
        else:
            raise ValueError(f"unsupported role: {role!r}")
    parts.append(ASSISTANT)
    return "".join(parts)


def render_example(
    user: str, assistant: str, system: str | None = None, tokenizer=None
) -> tuple[str, str]:
    """Render one training pair as ``(prompt, completion)``.

    The split matters: during supervised fine-tuning we only compute the loss on
    the completion, so the model is never rewarded for memorising the questions.
    """
    prompt = render_prompt([{"role": "user", "content": user}], system=system, tokenizer=tokenizer)
    return prompt, assistant + (IM_END if _is_chatml(tokenizer) else EOS)


def encode_example(
    tok: BPETokenizer, user: str, assistant: str, system: str | None = None
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` where prompt labels are ``-100`` (ignored)."""
    prompt, completion = render_example(user, assistant, system, tok)
    p_ids = tok.encode(prompt)
    c_ids = tok.encode(completion)
    ids = p_ids + c_ids
    labels = [-100] * len(p_ids) + c_ids
    return ids, labels


def iter_texts(rows: Iterable[tuple[str, str, str | None]]) -> Iterable[str]:
    """Full rendered text for every row - used to train the tokenizer."""
    for user, assistant, system in rows:
        prompt, completion = render_example(user, assistant, system)
        yield prompt + completion
