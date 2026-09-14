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
import multiprocessing as mp
import os
import random
import sys
from array import array
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .chat import render_example
from .tokenizer import BPETokenizer, encode_chunk_to_bytes, id_typecode, init_encode_worker

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

    Storage is compact: every token lives in one flat ``int32`` tensor, and the
    labels are not stored at all - they are the ids with the first
    ``prompt_len`` positions masked. Python lists of ints cost ~70 bytes per
    token (two list slots plus int objects); this costs 4.
    """

    def __init__(
        self,
        rows: Iterable[Row],
        tokenizer: BPETokenizer,
        block_size: int,
        mask_prompt: bool = True,
        drop_truncated: bool = False,
    ):
        self.block_size = block_size
        self.pad_id = tokenizer.pad_id
        self.n_truncated = 0
        ids_buf = array("i")
        offsets = array("q", [0])
        prompt_lens = array("i")
        for user, assistant, system in rows:
            prompt, completion = render_example(user, assistant, system)
            p_ids = tokenizer.encode(prompt)
            c_ids = tokenizer.encode(completion)
            n = len(p_ids) + len(c_ids)
            if n > block_size:
                self.n_truncated += 1
                if drop_truncated:
                    continue
            ids_buf.extend(p_ids[:block_size])
            if len(p_ids) < block_size:
                ids_buf.extend(c_ids[: block_size - len(p_ids)])
            offsets.append(len(ids_buf))
            prompt_lens.append(min(len(p_ids), block_size) if mask_prompt else 0)
        if len(offsets) == 1:
            raise SystemExit("every example was longer than block_size; increase --block-size")
        self.flat_ids = torch.frombuffer(ids_buf, dtype=torch.int32).clone()
        self.offsets = torch.frombuffer(offsets, dtype=torch.int64).clone()
        self.prompt_lens = torch.frombuffer(prompt_lens, dtype=torch.int32).clone()
        self._lengths: list[int] | None = None

    def __len__(self) -> int:
        return len(self.offsets) - 1

    def example(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Unpadded ``(ids, labels)`` of example ``i`` as int64 tensors."""
        a, b = int(self.offsets[i]), int(self.offsets[i + 1])
        ids = self.flat_ids[a:b].long()
        labels = ids.clone()
        labels[: int(self.prompt_lens[i])] = -100
        return ids, labels

    @property
    def examples(self) -> list[tuple[list[int], list[int]]]:
        """Every example as Python lists. Materialises everything - debug/tests only."""
        return [tuple(t.tolist() for t in self.example(i)) for i in range(len(self))]

    def __getitem__(self, i: int):
        ids, labels = self.example(i)
        x = torch.full((self.block_size,), self.pad_id, dtype=torch.long)
        y = torch.full((self.block_size,), -100, dtype=torch.long)
        x[: len(ids)] = ids
        y[: len(labels)] = labels
        return x[:-1], y[1:]

    @property
    def n_tokens(self) -> int:
        return len(self.flat_ids)

    @property
    def lengths(self) -> list[int]:
        """Real (unpadded) token count of every example."""
        if self._lengths is None:
            self._lengths = (self.offsets[1:] - self.offsets[:-1]).tolist()
        return self._lengths

    def raw_view(self) -> "RawChatView":
        """A view whose items are the unpadded id/label sequences.

        Used together with :func:`dynamic_collate`, which pads each batch to its
        own longest member instead of to ``block_size``. On a typical Q&A set the
        mean example is a fraction of ``block_size``, so this removes most of the
        padding - and most of the compute that was spent on it.

        The view shares this dataset's flat tensors (no copy), and DataLoader
        workers share tensor storage instead of pickling per-example lists.
        """
        return RawChatView(self)


class RawChatView(Dataset):
    """Unpadded ``(ids, labels)`` pairs backing :meth:`ChatDataset.raw_view`."""

    def __init__(self, ds: ChatDataset):
        self.ds = ds
        self.pad_id = ds.pad_id

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        return self.ds.example(i)


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


def token_dtype(vocab_size: int) -> np.dtype:
    """Smallest unsigned dtype that holds every token id."""
    return np.dtype(np.uint16) if vocab_size <= 2**16 else np.dtype(np.uint32)


def iter_text_chunks(path: str | Path, chunk_chars: int = 1 << 20) -> Iterator[str]:
    """Yield ``path`` in ~``chunk_chars`` pieces, each ending on a line break.

    Cutting on newlines keeps words intact, so neither tokenizer training nor
    encoding ever needs the whole corpus in memory at once.
    """
    with Path(path).open("r", encoding="utf-8") as fh:
        carry = ""
        while True:
            block = fh.read(chunk_chars)
            if not block:
                break
            block = carry + block
            cut = block.rfind("\n") + 1
            if cut == 0:
                carry = block
                continue
            carry = block[cut:]
            yield block[:cut]
        if carry:
            yield carry


def sample_text_chunks(
    path: str | Path, max_bytes: int, chunk_bytes: int = 1 << 20
) -> Iterator[str]:
    """Yield about ``max_bytes`` of ``path``, spread evenly across the file.

    Used to train the tokenizer: BPE merges for a few thousand tokens are settled
    long before hundreds of MB, and reading evenly spaced windows (rather than
    the head of the file) keeps the sample representative. Every window is cut
    on line breaks, so no word and no UTF-8 sequence is split.
    """
    path = Path(path)
    size = path.stat().st_size
    if max_bytes <= 0 or size <= max_bytes:
        yield from iter_text_chunks(path, chunk_bytes)
        return
    n_windows = max(1, max_bytes // chunk_bytes)
    with path.open("rb") as fh:
        for k in range(n_windows):
            start = k * size // n_windows
            fh.seek(start)
            block = fh.read(chunk_bytes)
            if start > 0:  # drop the partial line we landed in
                block = block[block.find(b"\n") + 1 :]
            end = block.rfind(b"\n") + 1
            if end > 0:
                yield block[:end].decode("utf-8", errors="replace")


def default_encode_workers() -> int:
    """Encoding processes to use: leave a core for the OS, cap memory use."""
    return max(1, min(4, (os.cpu_count() or 1) - 1))


def _batched(items: Iterable, n: int) -> Iterator[list]:
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_corpus_to_file(
    chunks: Iterable[str],
    tokenizer: BPETokenizer,
    out_path: str | Path,
    workers: int = 1,
) -> np.memmap:
    """Stream-encode ``chunks`` into a flat binary token file and memory-map it.

    The file starts with ``<|bos|>``. With ``workers > 1`` chunks are encoded in
    parallel by spawned processes that import only the tokenizer (no torch).
    Chunks are handed out a small window at a time - ``Pool.imap`` would read
    the entire corpus into its task queue up front - so at most ``2 * workers``
    chunks are in RAM. The returned memmap is paged in by the OS on demand.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dtype = token_dtype(tokenizer.vocab_size)
    typecode = id_typecode(tokenizer.vocab_size)
    assert array(typecode).itemsize == dtype.itemsize
    n = 0
    with out_path.open("wb") as fh:
        fh.write(np.array([tokenizer.bos_id], dtype=dtype).tobytes())
        n += 1
        if workers <= 1:
            for chunk in chunks:
                buf = array(typecode, tokenizer.encode(chunk)).tobytes()
                fh.write(buf)
                n += len(buf) // dtype.itemsize
        else:
            ctx = mp.get_context("spawn")  # fork is unsafe once torch has loaded
            with ctx.Pool(
                workers,
                initializer=init_encode_worker,
                initargs=(tokenizer.merges, tokenizer.specials),
            ) as pool:
                for window in _batched(chunks, 2 * workers):
                    jobs = [(chunk, typecode) for chunk in window]
                    for buf in pool.map(encode_chunk_to_bytes, jobs, chunksize=1):
                        fh.write(buf)
                        n += len(buf) // dtype.itemsize
    return np.memmap(out_path, dtype=dtype, mode="r", shape=(n,))


class PackedTextDataset(Dataset):
    """Contiguous blocks of a raw token stream - used for optional pretraining.

    ``ids`` may be a list, a numpy array, or a ``np.memmap`` of a token file. It
    is kept in its compact dtype and only each sampled window is widened to
    int64, so a memmapped corpus costs almost no resident memory.
    """

    def __init__(self, ids, block_size: int, stride: int | None = None):
        self.ids = ids if isinstance(ids, np.ndarray) else np.asarray(ids, dtype=np.int64)
        self.block_size = block_size
        self.stride = stride or block_size
        n = len(self.ids) - 1
        if n < block_size:
            raise SystemExit(f"corpus too small: {len(self.ids)} tokens < block_size {block_size}")
        self._len = (n - block_size) // self.stride + 1

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, i: int):
        if i < 0:
            i += self._len
        if not 0 <= i < self._len:
            raise IndexError(i)
        s = i * self.stride
        chunk = torch.from_numpy(self.ids[s : s + self.block_size + 1].astype(np.int64))
        return chunk[:-1], chunk[1:]

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
