import pytest

from minigpt.tokenizer import DEFAULT_SPECIALS, BPETokenizer, _pieces

CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the quick brown cat sleeps under the lazy sun",
    "hello world, hello again! 123 456.",
] * 20


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=512)


def test_pretokenizer_is_lossless():
    for text in ["Hello, world! 42 things.", "snake_case CAFÉ\n\nnew", "don't  stop", "  leading"]:
        assert "".join(_pieces(text)) == text


@pytest.mark.parametrize(
    "text",
    ["the quick brown fox", "hello world", "unseen words appear here", "émoji ☕ ok", "", "12345"],
)
def test_encode_decode_roundtrip(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_merges_actually_compress(tok):
    text = "the quick brown fox jumps over the lazy dog"
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_special_tokens_are_single_ids(tok):
    for s in DEFAULT_SPECIALS:
        assert tok.encode(s) == [tok.special_to_id[s]]


def test_specials_can_be_disabled(tok):
    assert len(tok.encode("<|eos|>", allowed_special=False)) > 1


def test_decode_skips_specials_by_default(tok):
    ids = tok.encode("<|user|>hi<|eos|>")
    assert tok.decode(ids) == "hi"
    assert tok.decode(ids, skip_special=False) == "<|user|>hi<|eos|>"


def test_save_load_roundtrip(tok, tmp_path):
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = BPETokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("the quick brown fox") == tok.encode("the quick brown fox")


def test_vocab_size_floor():
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, vocab_size=10)


def test_merges_follow_count_then_largest_pair():
    """The heap must pick exactly what a full scan with max((count, pair)) picks."""
    from collections import Counter

    corpus = ["abab xyxy"] * 5  # (a,b) and (x,y) tie on count; (x,y) is the larger pair
    tok = BPETokenizer.train(corpus, vocab_size=len(DEFAULT_SPECIALS) + 256 + 1)
    counts = Counter()
    for piece in _pieces(corpus[0]):
        ids = [b + len(DEFAULT_SPECIALS) for b in piece.encode()]
        counts.update(zip(ids, ids[1:]))
    best = max(counts, key=lambda p: (counts[p], p))
    assert tok.merges == [best]
