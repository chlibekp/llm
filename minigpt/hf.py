"""Load a pretrained Hugging Face Llama-style model into minigpt's own ``GPT``.

SmolLM2 (and other ``LlamaForCausalLM`` checkpoints) use exactly the design
``minigpt/model.py`` already implements: pre-norm RMSNorm blocks, RoPE with the
contiguous-halves layout, grouped-query attention, a SwiGLU MLP and optionally
tied embeddings. So no second model implementation is needed. The config is
translated, the tensors are renamed, and generation, chat and the OpenAI server
keep running on the same code.

Only three small libraries are involved, and none of them is ``transformers``:
``huggingface_hub`` downloads the files, ``safetensors`` reads the weights and
``tokenizers`` runs the pretrained tokenizer.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from .config import GPTConfig
from .model import GPT

DEFAULT_BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json", "*.safetensors"]

# minigpt name <- Hugging Face Llama name (per layer ``{i}``).
_LAYER_NAMES = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn.q_proj.weight": "self_attn.q_proj.weight",
    "attn.k_proj.weight": "self_attn.k_proj.weight",
    "attn.v_proj.weight": "self_attn.v_proj.weight",
    "attn.o_proj.weight": "self_attn.o_proj.weight",
    "mlp_norm.weight": "post_attention_layernorm.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}


def _import(module: str):
    try:
        return __import__(module)
    except ImportError as exc:
        raise SystemExit(
            f"pretrained models need the '{module}' package: pip install -e .  "
            "(or: pip install safetensors tokenizers huggingface_hub)"
        ) from exc


def looks_like_pretrained(name: str | Path) -> bool:
    """A local Hugging Face model directory, or a ``org/name`` hub id."""
    p = Path(name)
    if p.is_dir():
        cfg = p / "config.json"
        return cfg.exists() and "model_type" in json.loads(cfg.read_text(encoding="utf-8"))
    return not p.exists() and re.fullmatch(r"[\w.-]+/[\w.-]+", str(name)) is not None


def resolve_files(name: str, revision: str | None = None) -> tuple[Path, dict]:
    """Local directory with the model files, plus a reference that pins them.

    Hub downloads are cached by ``huggingface_hub``; the cache is tried first so
    a model that is already present never touches the network.
    """
    if Path(name).is_dir():
        path = Path(name).resolve()
        return path, {"name": str(path)}
    _import("huggingface_hub")
    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(name, revision=revision, allow_patterns=_FILES, local_files_only=True)
    except Exception:  # not cached yet (or only partially): fetch it
        print(f"downloading {name} from the Hugging Face hub ...", flush=True)
        try:
            path = snapshot_download(name, revision=revision, allow_patterns=_FILES)
        except Exception as exc:
            raise SystemExit(
                f"{name!r} is neither a local checkpoint directory nor a model on the "
                f"Hugging Face hub ({type(exc).__name__}: {exc})"
            ) from exc
    path = Path(path)
    return path, {"name": name, "revision": path.name}  # snapshot dirs are named by commit


def config_from_hf(hf: dict) -> GPTConfig:
    """Translate a Llama ``config.json``, refusing anything ``GPT`` cannot express."""
    if hf.get("model_type") != "llama":
        raise SystemExit(
            f"unsupported model_type {hf.get('model_type')!r}: only Llama-architecture "
            "models (e.g. SmolLM2) load into minigpt"
        )
    n_embd, n_head = hf["hidden_size"], hf["num_attention_heads"]
    problems = []
    if hf.get("rope_scaling"):
        problems.append("rope_scaling")
    if hf.get("attention_bias") or hf.get("mlp_bias"):
        problems.append("biases")
    if hf.get("hidden_act", "silu") != "silu":
        problems.append(f"hidden_act={hf['hidden_act']}")
    if hf.get("head_dim") not in (None, n_embd // n_head):
        problems.append("a head_dim other than hidden_size / num_attention_heads")
    if problems:
        raise SystemExit(f"this checkpoint uses {', '.join(problems)}, which minigpt does not implement")
    return GPTConfig(
        vocab_size=hf["vocab_size"],
        block_size=min(hf.get("max_position_embeddings", 2048), 8192),
        n_layer=hf["num_hidden_layers"],
        n_head=n_head,
        n_kv_head=hf.get("num_key_value_heads") or n_head,
        n_embd=n_embd,
        intermediate_size=hf["intermediate_size"],
        dropout=0.0,
        bias=False,
        rope_theta=float(hf.get("rope_theta", 10000.0)),
        rope_interleaved=False,  # Hugging Face's rotate_half pairs i with i + d/2
        tie_weights=bool(hf.get("tie_word_embeddings", False)),
        norm_eps=float(hf.get("rms_norm_eps", 1e-5)),
    )


def load_llama_state(path: Path, cfg: GPTConfig, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """Read every ``*.safetensors`` shard and rename it to minigpt's keys."""
    _import("safetensors")
    from safetensors.torch import load_file

    raw: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        raw.update(load_file(shard))
    if not raw:
        raise SystemExit(f"no .safetensors weights in {path}")

    state = {"tok_emb.weight": raw.pop("model.embed_tokens.weight"),
             "norm.weight": raw.pop("model.norm.weight")}
    for i in range(cfg.n_layer):
        for ours, theirs in _LAYER_NAMES.items():
            state[f"blocks.{i}.{ours}"] = raw.pop(f"model.layers.{i}.{theirs}")
    head = raw.pop("lm_head.weight", None)
    state["lm_head.weight"] = state["tok_emb.weight"] if cfg.tie_weights or head is None else head
    raw = {k: v for k, v in raw.items() if not k.endswith("rotary_emb.inv_freq")}
    if raw:
        raise SystemExit(f"unexpected tensors in {path}: {sorted(raw)[:5]}")
    return {k: v.to(dtype) for k, v in state.items()}


class HFTokenizer:
    """A pretrained ``tokenizer.json`` behind the interface minigpt expects.

    It also carries the chat format: ``chat_format = "chatml"`` makes
    :mod:`minigpt.chat` render ``<|im_start|>role ... <|im_end|>`` turns, with
    the model's own default system prompt.
    """

    chat_format = "chatml"

    def __init__(self, tokenizer_json: str, meta: dict):
        _import("tokenizers")
        from tokenizers import Tokenizer

        self._json = tokenizer_json
        self._tok = Tokenizer.from_str(tokenizer_json)
        self.meta = dict(meta, type="hf")
        self.default_system: str | None = meta.get("default_system")
        self.vocab_size = self._tok.get_vocab_size()
        ids = {}
        for role in ("bos", "eos", "pad"):
            token = meta.get(role)
            ids[role] = self._tok.token_to_id(token) if token else None
        if ids["eos"] is None:
            raise SystemExit("the pretrained tokenizer defines no end-of-turn token")
        self.eos_id = ids["eos"]
        self.pad_id = ids["pad"] if ids["pad"] is not None else self.eos_id
        self.bos_id = ids["bos"] if ids["bos"] is not None else self.eos_id

    @classmethod
    def from_pretrained_dir(cls, path: Path) -> "HFTokenizer":
        tc_path = path / "tokenizer_config.json"
        tc = json.loads(tc_path.read_text(encoding="utf-8")) if tc_path.exists() else {}
        template = tc.get("chat_template") or ""
        if template and "<|im_start|>" not in template:
            raise SystemExit("only ChatML chat templates (<|im_start|> ... <|im_end|>) are supported")
        default = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", template, re.S)

        def token(v):  # tokenizer_config stores tokens as strings or AddedToken dicts
            return v.get("content") if isinstance(v, dict) else v

        meta = {"bos": token(tc.get("bos_token")), "eos": token(tc.get("eos_token")) or "<|im_end|>",
                "pad": token(tc.get("pad_token")), "default_system": default.group(1) if default else None}
        return cls((path / "tokenizer.json").read_text(encoding="utf-8"), meta)

    @classmethod
    def load(cls, path: str | Path, meta: dict) -> "HFTokenizer":
        return cls(Path(path).read_text(encoding="utf-8"), meta)

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        # Control tokens written in the text (<|im_start|> ...) are always matched.
        return self._tok.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special: bool = True) -> str:
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=skip_special)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self._json, encoding="utf-8")


def default_dtype(device: torch.device) -> torch.dtype:
    """Half precision wherever it runs on the accelerator, since the weights ship in bf16."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def load_pretrained(
    name: str, dtype: torch.dtype = torch.float32, revision: str | None = None
) -> tuple[GPT, HFTokenizer, dict]:
    """Build a ``GPT`` holding pretrained weights (on the CPU, in ``dtype``).

    Returns ``(model, tokenizer, base_ref)``; ``base_ref`` pins the exact files,
    so a LoRA checkpoint can later reload the base it was trained on.
    """
    path, ref = resolve_files(name, revision)
    cfg = config_from_hf(json.loads((path / "config.json").read_text(encoding="utf-8")))
    state = load_llama_state(path, cfg, dtype)
    with torch.device("meta"):  # no 1.4 GB of random init that is overwritten at once
        model = GPT(cfg)
    model.load_state_dict(state, assign=True)
    if cfg.tie_weights:
        model.lm_head.weight = model.tok_emb.weight  # assign=True unties them
    model.base_ref = ref
    tok = HFTokenizer.from_pretrained_dir(path)
    if tok.vocab_size > cfg.vocab_size:
        raise SystemExit(f"tokenizer has {tok.vocab_size} ids but the model only {cfg.vocab_size}")
    return model, tok, ref
