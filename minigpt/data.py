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
import math
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

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
        self._lengths: list[int] | None = None
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

    @property
    def lengths(self) -> list[int]:
        """Real (unpadded) token count of every example."""
        if self._lengths is None:
            self._lengths = [len(ids) for ids, _ in self.examples]
        return self._lengths

    def raw_view(self) -> "RawChatView":
        """A view whose items are the unpadded id/label sequences.

        Used together with :func:`dynamic_collate`, which pads each batch to its
        own longest member instead of to ``block_size``. On a typical Q&A set the
        mean example is a fraction of ``block_size``, so this removes most of the
        padding - and most of the compute that was spent on it.

        The view holds flat tensors rather than the Python lists, for two reasons:
        slicing them is a C-level copy instead of a per-element conversion, and
        DataLoader workers share tensor storage instead of pickling 2N lists.
        """
        flat_ids = torch.empty(sum(self.lengths), dtype=torch.long)
        flat_labels = torch.empty(sum(self.lengths), dtype=torch.long)
        offsets = torch.empty(len(self.examples) + 1, dtype=torch.long)
        at = 0
        for i, (ids, labels) in enumerate(self.examples):
            offsets[i] = at
            n = len(ids)
            flat_ids[at : at + n] = torch.tensor(ids, dtype=torch.long)
            flat_labels[at : at + n] = torch.tensor(labels, dtype=torch.long)
            at += n
        offsets[-1] = at
        return RawChatView(flat_ids, flat_labels, offsets, self.pad_id)


class RawChatView(Dataset):
    """Unpadded ``(ids, labels)`` pairs backing :meth:`ChatDataset.raw_view`."""

    def __init__(
        self,
        flat_ids: torch.Tensor,
        flat_labels: torch.Tensor,
        offsets: torch.Tensor,
        pad_id: int,
    ):
        self.flat_ids = flat_ids
        self.flat_labels = flat_labels
        self.offsets = offsets
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def __getitem__(self, i: int):
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        return self.flat_ids[a:b], self.flat_labels[a:b]


def dynamic_collate(pad_id: int, multiple_of: int = 8):
    """Build a ``collate_fn`` that pads a batch to its own longest example.

    The length is rounded up to ``multiple_of`` so the matmul shapes stay in a
    small, cache-friendly set rather than changing on every batch.
    """

    def collate(batch):
        n = max(len(ids) for ids, _ in batch)
        n = multiple_of * math.ceil(n / multiple_of)
        n = max(n, 2)  # need at least one input and one target after the shift
        x = torch.full((len(batch), n), pad_id, dtype=torch.long)
        y = torch.full((len(batch), n), -100, dtype=torch.long)
        for r, (ids, labels) in enumerate(batch):
            k = len(ids)
            x[r, :k] = ids if torch.is_tensor(ids) else torch.tensor(ids, dtype=torch.long)
            y[r, :k] = labels if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
        x, y = x[:, :-1], y[:, 1:]
        # Locate the supervised positions here, on the CPU, where nonzero is just
        # a scan. Doing it inside the model would mean a data-dependent output
        # shape on the accelerator, which forces a full device sync every step.
        keep = (y.reshape(-1) != -100).nonzero(as_tuple=True)[0]
        return x, y, keep

    return collate


class LengthGroupedSampler(Sampler[list[int]]):
    """Batch sampler that puts examples of similar length in the same batch.

    Without it, dynamic padding is only as good as the longest example in each
    random batch. The classic fix: shuffle, cut the stream into "megabatches" of
    ``batch_size * pool`` examples, sort each megabatch by length, slice it into
    batches, then shuffle the batch order. Every epoch still sees a different
    partition, so this costs no randomness that matters, but each batch is
    internally near-uniform in length.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 1337,
        pool: int = 64,
        drop_last: bool = False,
    ):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.pool = max(1, pool)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        n = len(self.lengths)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)

        mega = self.batch_size * self.pool
        batches: list[list[int]] = []
        for start in range(0, len(idx), mega):
            # list.__getitem__ sorts at C level; a lambda would re-enter Python
            # once per element, and this runs over the whole dataset every epoch.
            chunk = sorted(idx[start : start + mega], key=self.lengths.__getitem__)
            for b in range(0, len(chunk), self.batch_size):
                batches.append(chunk[b : b + self.batch_size])
        if self.drop_last and batches and len(batches[-1]) < self.batch_size:
            batches.pop()
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)


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
