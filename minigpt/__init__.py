"""minigpt - a small, from-scratch GPT trained on your own question/answer CSV."""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["GPT", "GPTConfig", "PRESETS", "BPETokenizer", "__version__"]

# Exports are resolved lazily. Tokenizer worker processes import
# ``minigpt.tokenizer``, which runs this file first; importing the model here
# would load torch into every worker for nothing.
_EXPORTS = {
    "GPT": "model",
    "GPTConfig": "config",
    "PRESETS": "config",
    "BPETokenizer": "tokenizer",
}


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'minigpt' has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(f".{module}", __name__), name)
