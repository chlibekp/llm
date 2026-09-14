"""Byte-level BPE tokenizer, trained from scratch on the user's own data.

Why byte-level: every possible input is representable, so the tokenizer can never
emit an "unknown token" and we do not need a separate normalisation step.

The training algorithm is the classic BPE loop (Sennrich et al., 2016):

1. Split the corpus into "pieces" with a GPT-2 style regex (keeps whitespace
   attached to the following word, splits punctuation, digits, etc.).
2. Represent every piece as a sequence of raw bytes (ids 0..255).
3. Repeatedly find the most frequent adjacent pair of ids and merge it into a
   new id, until the vocabulary reaches ``vocab_size``.

The naive loop recounts every pair after every merge, which is far too slow in
pure Python. We keep an incremental index (``pair -> word indices``) so each
merge only touches the pieces that actually contain the merged pair, and a heap
of pair counts so finding the next merge does not scan every pair.

This module deliberately imports nothing heavy (no torch, no numpy): encoding
worker processes import only this file, which keeps their memory small.
"""

from __future__ import annotations

import heapq
import json
import re
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

# GPT-2's pre-tokenisation pattern. Splitting before BPE stops merges from
# spanning word boundaries, which keeps the learned vocabulary sane.
SPLIT_PATTERN = re.compile(
    "|".join(
        [
            r"'(?:[sdmt]|ll|ve|re)",       # English contractions
            r" ?[^\W\d_]+",                # (space +) letters
            r" ?\d+",                       # (space +) digits
            r" ?(?:(?![^\W\d_])[^\s\d])+",  # (space +) punctuation / symbols
            r"\s+(?!\S)",                  # trailing whitespace runs
            r"\s+",                         # any other whitespace
        ]
    ),
    re.UNICODE,
)

# Control tokens. They are never produced by BPE (they are matched before the
# text is split), so their ids are reserved at the very start of the vocabulary.
PAD = "<|pad|>"
BOS = "<|bos|>"
EOS = "<|eos|>"
SYSTEM = "<|system|>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
DEFAULT_SPECIALS = [PAD, BOS, EOS, SYSTEM, USER, ASSISTANT]


def _pieces(text: str) -> list[str]:
    return SPLIT_PATTERN.findall(text)


class BPETokenizer:
    """A trainable byte-level BPE tokenizer.

    Attributes:
        merges: ordered list of merged pairs; index == merge rank.
        specials: control tokens, occupying ids ``0..len(specials)-1``.
    """

    def __init__(self, merges: Sequence[tuple[int, int]], specials: Sequence[str] | None = None):
        self.specials: list[str] = list(specials if specials is not None else DEFAULT_SPECIALS)
        self.merges: list[tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        n_special = len(self.specials)
        self.special_to_id = {tok: i for i, tok in enumerate(self.specials)}
        self.id_to_special = {i: tok for tok, i in self.special_to_id.items()}
        # Base vocabulary: one id per byte value, offset past the specials.
        self._byte_offset = n_special
        # id -> bytes, used for decoding.
        self.vocab: dict[int, bytes] = {i: bytes([b]) for b, i in enumerate(range(n_special, n_special + 256))}
        self.ranks: dict[tuple[int, int], int] = {}
        self.merge_to_id: dict[tuple[int, int], int] = {}
        next_id = n_special + 256
        for rank, (a, b) in enumerate(self.merges):
            self.ranks[(a, b)] = rank
            self.merge_to_id[(a, b)] = next_id
            self.vocab[next_id] = self.vocab[a] + self.vocab[b]
            next_id += 1
        self.vocab_size = next_id
        self._cache: dict[str, list[int]] = {}
        # Regex that matches any control token verbatim.
        self._special_re = (
            re.compile("(" + "|".join(re.escape(s) for s in self.specials) + ")") if self.specials else None
        )

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        specials: Sequence[str] | None = None,
        min_frequency: int = 2,
        verbose: bool = False,
    ) -> "BPETokenizer":
        """Learn merges from ``texts`` until the vocabulary reaches ``vocab_size``."""
        specials = list(specials if specials is not None else DEFAULT_SPECIALS)
        offset = len(specials)
        n_merges = vocab_size - offset - 256
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {offset + 256} (specials + 256 byte values)")

        # Count identical pieces once instead of storing the whole corpus.
        # finditer, not findall: a big text would otherwise become one list with
        # a string object per piece before a single one is counted.
        piece_freq: dict[str, int] = defaultdict(int)
        for text in texts:
            for m in SPLIT_PATTERN.finditer(text):
                piece_freq[m.group()] += 1

        words: list[list[int]] = []
        freqs: list[int] = []
        for piece, f in piece_freq.items():
            words.append([b + offset for b in piece.encode("utf-8")])
            freqs.append(f)
        del piece_freq

        pair_counts: dict[tuple[int, int], int] = defaultdict(int)
        pair_to_words: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i, word in enumerate(words):
            f = freqs[i]
            for pair in zip(word, word[1:]):
                pair_counts[pair] += f
                pair_to_words[pair].add(i)
        pair_counts = dict(pair_counts)

        # Max-heap of counts. Ties go to the larger pair, hence the negated ids.
        # A heap cannot update entries in place, so entries go stale when a count
        # changes: a count that grows pushes a fresh entry, and a stale entry is
        # re-queued at its current count when it reaches the top.
        heap = [(-c, (-a, -b)) for (a, b), c in pair_counts.items()]
        heapq.heapify(heap)

        merges: list[tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            neg_count, (na, nb) = heapq.heappop(heap)
            best = (-na, -nb)
            count = pair_counts.get(best, 0)
            if count != -neg_count:
                if count > 0:
                    heapq.heappush(heap, (-count, (na, nb)))
                continue
            if count < min_frequency:
                break
            new_id = offset + 256 + len(merges)
            merges.append(best)
            a, b = best

            delta: dict[tuple[int, int], int] = defaultdict(int)
            for i in pair_to_words.pop(best, ()):
                word = words[i]
                if not _contains(word, a, b):
                    continue
                f = freqs[i]
                for pair in zip(word, word[1:]):
                    delta[pair] -= f
                word = _merge_word(word, a, b, new_id)
                words[i] = word
                for pair in zip(word, word[1:]):
                    delta[pair] += f
                    # Only pairs touching the new id are new to this word.
                    if pair[0] == new_id or pair[1] == new_id:
                        pair_to_words[pair].add(i)
            pair_counts.pop(best, None)

            for pair, d in delta.items():
                if d == 0 or pair == best:
                    continue
                c = pair_counts.get(pair, 0) + d
                if c > 0:
                    pair_counts[pair] = c
                    if d > 0:
                        heapq.heappush(heap, (-c, (-pair[0], -pair[1])))
                else:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)
            if verbose and len(merges) % 500 == 0:
                print(f"  bpe merge {len(merges)}/{n_merges}", flush=True)
        return cls(merges, specials)

    # ----------------------------------------------------------------- encode
    def _bpe(self, piece: str) -> list[int]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = [b + self._byte_offset for b in piece.encode("utf-8")]
        while len(ids) >= 2:
            best_rank: int | None = None
            best_pair: tuple[int, int] | None = None
            for pair in zip(ids, ids[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            ids = _merge_word(ids, best_pair[0], best_pair[1], self.merge_to_id[best_pair])
        if len(self._cache) < 200_000:
            self._cache[piece] = ids
        return ids

    def encode(self, text: str, allowed_special: bool = True) -> list[int]:
        """Encode ``text``. Control tokens in the string are honoured verbatim
        when ``allowed_special`` is true, otherwise they are encoded as plain text."""
        if not allowed_special or self._special_re is None:
            chunks = [text]
        else:
            chunks = self._special_re.split(text)
        out: list[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if allowed_special and chunk in self.special_to_id:
                out.append(self.special_to_id[chunk])
                continue
            for piece in _pieces(chunk):
                out.extend(self._bpe(piece))
        return out

    # ----------------------------------------------------------------- decode
    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        buf = bytearray()
        text: list[str] = []
        for i in ids:
            i = int(i)
            if i in self.id_to_special:
                if buf:
                    text.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                if not skip_special:
                    text.append(self.id_to_special[i])
                continue
            piece = self.vocab.get(i)
            if piece is not None:
                buf.extend(piece)
        if buf:
            text.append(buf.decode("utf-8", errors="replace"))
        return "".join(text)

    # ------------------------------------------------------------- properties
    @property
    def pad_id(self) -> int:
        return self.special_to_id[PAD]

    @property
    def bos_id(self) -> int:
        return self.special_to_id[BOS]

    @property
    def eos_id(self) -> int:
        return self.special_to_id[EOS]

    # ------------------------------------------------------------------- i/o
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"version": 1, "specials": self.specials, "merges": [list(m) for m in self.merges]},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([tuple(m) for m in data["merges"]], data["specials"])


def id_typecode(vocab_size: int) -> str:
    """``array`` typecode of the smallest unsigned int that holds every id."""
    if vocab_size <= 2**16:
        return "H"
    return "I" if array("I").itemsize == 4 else "L"


# One tokenizer per encoding worker process, built once by the initializer.
_worker_tokenizer: BPETokenizer | None = None


def init_encode_worker(merges: Sequence[tuple[int, int]], specials: Sequence[str]) -> None:
    global _worker_tokenizer
    _worker_tokenizer = BPETokenizer(merges, specials)


def encode_chunk_to_bytes(args: tuple[str, str]) -> bytes:
    """Worker entry point: encode one text chunk to packed ids.

    Returning packed bytes instead of a list of ints makes the result ~20x
    cheaper to pickle back to the parent.
    """
    text, typecode = args
    assert _worker_tokenizer is not None, "init_encode_worker was not called"
    return array(typecode, _worker_tokenizer.encode(text)).tobytes()


def _contains(word: list[int], a: int, b: int) -> bool:
    return any(word[i] == a and word[i + 1] == b for i in range(len(word) - 1))


def _merge_word(word: list[int], a: int, b: int, new_id: int) -> list[int]:
    out: list[int] = []
    i, n = 0, len(word)
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out
