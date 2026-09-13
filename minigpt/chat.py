"""The prompt format (a.k.a. chat template) the model is trained and served with.

A conversation is rendered as a flat token stream:

    <|bos|><|system|>You are helpful.<|user|>What is 2+2?<|assistant|>4<|eos|>

The control tokens are single tokens from the tokenizer's reserved range, so the
model learns them as unambiguous role markers rather than as ordinary text. At
inference time we render everything up to and including the final
``<|assistant|>`` and let the model continue until it emits ``<|eos|>``.
"""

from __future__ import annotations

from typing import Iterable, Sequence, TypedDict

from .tokenizer import ASSISTANT, BOS, EOS, SYSTEM, USER, BPETokenizer

DEFAULT_SYSTEM = "You are a helpful assistant."


class Message(TypedDict):
    role: str
    content: str


def render_prompt(messages: Sequence[Message], system: str | None = None) -> str:
    """Render a conversation, ending with an open ``<|assistant|>`` turn."""
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


def render_example(user: str, assistant: str, system: str | None = None) -> tuple[str, str]:
    """Render one training pair as ``(prompt, completion)``.

    The split matters: during supervised fine-tuning we only compute the loss on
    the completion, so the model is never rewarded for memorising the questions.
    """
    prompt = render_prompt([{"role": "user", "content": user}], system=system)
    return prompt, assistant + EOS


def encode_example(
    tok: BPETokenizer, user: str, assistant: str, system: str | None = None
) -> tuple[list[int], list[int]]:
    """Return ``(input_ids, labels)`` where prompt labels are ``-100`` (ignored)."""
    prompt, completion = render_example(user, assistant, system)
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
