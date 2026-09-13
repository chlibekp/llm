"""CSV loading and dataset construction.

Input format
------------
A UTF-8 CSV with a header row. Two columns are required - the question and the
answer - and an optional ``system`` column overrides the system prompt per row::

    input,output
    "What is the capital of France?","Paris."

Column names are auto-detected from a list of common aliases, or given
explicitly with ``--input-col`` / ``--output-col``.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .chat import encode_example
from .tokenizer import BPETokenizer

INPUT_ALIASES = ["input", "question", "prompt", "instruction", "user", "query", "q"]
OUTPUT_ALIASES = ["output", "answer", "response", "completion", "assistant", "target", "a"]
SYSTEM_ALIASES = ["system", "context", "system_prompt"]

Row = tuple[str, str, str | None]


def _pick(header: list[str], aliases: list[str], explicit: str | None, what: str) -> str:
    lower = {h.strip().lower(): h for h in header}
    if explicit:
        if explicit in header:
            return explicit
        if explicit.lower() in lower:
            return lower[explicit.lower()]
        raise SystemExit(f"column {explicit!r} not found in CSV header {header}")
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    raise SystemExit(
        f"could not auto-detect the {what} column in {header}. "
        f"Pass --{what}-col explicitly (tried: {', '.join(aliases)})"
    )


def load_csv(
    path: str | Path,
    input_col: str | None = None,
    output_col: str | None = None,
    system_col: str | None = None,
    delimiter: str | None = None,
) -> list[Row]:
    """Read ``path`` into ``(user, assistant, system)`` triples, skipping empty rows."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"dataset not found: {path}")
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(8192)
        fh.seek(0)
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        reader = csv.DictReader(fh, delimiter=delimiter)
        if not reader.fieldnames:
            raise SystemExit(f"{path} has no header row")
        header = list(reader.fieldnames)
        in_key = _pick(header, INPUT_ALIASES, input_col, "input")
        out_key = _pick(header, OUTPUT_ALIASES, output_col, "output")
        sys_key: str | None = None
        if system_col:
            sys_key = _pick(header, SYSTEM_ALIASES, system_col, "system")
        else:
            for alias in SYSTEM_ALIASES:
                if alias in {h.strip().lower() for h in header}:
                    sys_key = {h.strip().lower(): h for h in header}[alias]
                    break

        rows: list[Row] = []
        for rec in reader:
            user = (rec.get(in_key) or "").strip()
            assistant = (rec.get(out_key) or "").strip()
            if not user or not assistant:
                continue
            system = (rec.get(sys_key) or "").strip() if sys_key else ""
            rows.append((user, assistant, system or None))
    if not rows:
        raise SystemExit(f"no usable rows in {path} (need non-empty {in_key!r} and {out_key!r})")
    return rows


class ChatDataset(Dataset):
    """Tokenised Q&A pairs, right-padded to a fixed length.

    Each item is ``(input_ids[:-1], labels[1:])`` - i.e. the standard
    shifted-by-one next-token objective. Padding and prompt tokens carry the
    label ``-100`` so ``cross_entropy`` ignores them.
    """

    def __init__(
        self,
        rows: list[Row],
        tokenizer: BPETokenizer,
        block_size: int,
        mask_prompt: bool = True,
        drop_truncated: bool = False,
    ):
        self.block_size = block_size
        self.pad_id = tokenizer.pad_id
        self.examples: list[tuple[list[int], list[int]]] = []
        self.n_truncated = 0
        for user, assistant, system in rows:
            ids, labels = encode_example(tokenizer, user, assistant, system)
            if not mask_prompt:
                labels = list(ids)
            if len(ids) > block_size:
                self.n_truncated += 1
                if drop_truncated:
                    continue
                ids, labels = ids[:block_size], labels[:block_size]
            self.examples.append((ids, labels))
        if not self.examples:
            raise SystemExit("every example was longer than block_size; increase --block-size")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ids, labels = self.examples[i]
        pad = self.block_size - len(ids)
        x = torch.tensor(ids + [self.pad_id] * pad, dtype=torch.long)
        y = torch.tensor(labels + [-100] * pad, dtype=torch.long)
        return x[:-1], y[1:]

    @property
    def n_tokens(self) -> int:
        return sum(len(ids) for ids, _ in self.examples)


class PackedTextDataset(Dataset):
    """Contiguous blocks of a raw token stream - used for optional pretraining."""

    def __init__(self, ids: list[int], block_size: int, stride: int | None = None):
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size
        self.stride = stride or block_size
        n = len(ids) - 1
        if n < block_size:
            raise SystemExit(f"corpus too small: {len(ids)} tokens < block_size {block_size}")
        self.starts = list(range(0, n - block_size + 1, self.stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        chunk = self.ids[s : s + self.block_size + 1]
        return chunk[:-1], chunk[1:].clone()

    @property
    def n_tokens(self) -> int:
        return len(self.ids)


def train_val_split(rows: list, val_ratio: float, seed: int = 1337) -> tuple[list, list]:
    """Shuffle deterministically and hold out ``val_ratio`` of the rows."""
    if val_ratio <= 0 or len(rows) < 10:
        return rows, []
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]
