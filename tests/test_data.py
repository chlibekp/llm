import csv

import pytest

from minigpt.chat import ASSISTANT, USER, encode_example, render_prompt
from minigpt.data import ChatDataset, load_csv, train_val_split
from minigpt.tokenizer import BPETokenizer


def write_csv(path, header, rows, delimiter=","):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter=delimiter)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_load_csv_default_columns(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["input", "output"], [["q1", "a1"], ["q2", "a2"]])
    assert load_csv(p) == [("q1", "a1", None), ("q2", "a2", None)]


def test_load_csv_alias_columns(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["question", "answer"], [["q", "a"]])
    assert load_csv(p) == [("q", "a", None)]


def test_load_csv_explicit_columns(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["col_a", "col_b"], [["q", "a"]])
    assert load_csv(p, input_col="col_a", output_col="col_b") == [("q", "a", None)]


def test_load_csv_system_column(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["input", "output", "system"], [["q", "a", "be terse"]])
    assert load_csv(p) == [("q", "a", "be terse")]


def test_load_csv_semicolon_delimiter(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["input", "output"], [["q", "a"], ["q2", "a2"]], delimiter=";")
    assert load_csv(p, delimiter=";") == [("q", "a", None), ("q2", "a2", None)]


def test_load_csv_skips_blank_rows(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["input", "output"], [["q", "a"], ["", "a2"], ["q3", ""]])
    assert load_csv(p) == [("q", "a", None)]


def test_load_csv_unknown_columns_errors(tmp_path):
    p = write_csv(tmp_path / "d.csv", ["foo", "bar"], [["q", "a"]])
    with pytest.raises(SystemExit):
        load_csv(p)


def test_load_csv_missing_file():
    with pytest.raises(SystemExit):
        load_csv("does-not-exist.csv")


def test_render_prompt_ends_with_open_assistant_turn():
    prompt = render_prompt([{"role": "user", "content": "hi"}])
    assert prompt.endswith(ASSISTANT)
    assert USER + "hi" in prompt


def test_prompt_tokens_are_masked_out():
    tok = BPETokenizer.train(["question answer " * 50], vocab_size=400)
    ids, labels = encode_example(tok, "question", "answer")
    assert len(ids) == len(labels)
    assert labels[0] == -100
    supervised = [l for l in labels if l != -100]
    assert supervised == ids[len(labels) - len(supervised):]
    assert supervised[-1] == tok.eos_id


def test_chat_dataset_shapes_and_padding():
    tok = BPETokenizer.train(["question answer " * 50], vocab_size=400)
    ds = ChatDataset([("question", "answer", None)], tok, block_size=64)
    x, y = ds[0]
    assert x.shape == y.shape == (63,)
    assert (y == -100).sum() > 0            # prompt + padding are ignored
    assert x[-1].item() == tok.pad_id


def test_chat_dataset_truncates_long_examples():
    tok = BPETokenizer.train(["word " * 200], vocab_size=400)
    ds = ChatDataset([("word " * 100, "word " * 100, None)], tok, block_size=32)
    assert ds.n_truncated == 1
    assert ds[0][0].shape == (31,)


def test_train_val_split_is_deterministic_and_disjoint():
    rows = [(f"q{i}", f"a{i}", None) for i in range(100)]
    tr1, va1 = train_val_split(rows, 0.1, seed=7)
    tr2, va2 = train_val_split(rows, 0.1, seed=7)
    assert (tr1, va1) == (tr2, va2)
    assert len(va1) == 10 and len(tr1) == 90
    assert not set(tr1) & set(va1)


def test_dynamic_collate_pads_to_batch_max_not_block_size():
    from minigpt.data import dynamic_collate

    collate = dynamic_collate(pad_id=0, multiple_of=8)
    x, y, keep = collate([([1, 2, 3], [-100, 2, 3]), ([4, 5], [-100, 5])])
    assert x.shape == y.shape == (2, 7)     # max len 3 -> rounded to 8, minus the shift
    assert x[0].tolist() == [1, 2, 3, 0, 0, 0, 0]
    assert y[0].tolist() == [2, 3, -100, -100, -100, -100, -100]
    # keep indexes the supervised positions of the flattened (2, 7) label grid
    assert keep.tolist() == [0, 1, 7]
    assert y.reshape(-1)[keep].tolist() == [2, 3, 5]


def test_dynamic_collate_matches_fixed_padding_on_the_real_tokens():
    from minigpt.data import dynamic_collate

    tok = BPETokenizer.train(["question answer " * 50], vocab_size=400)
    ds = ChatDataset([("question", "answer", None)], tok, block_size=64)
    fixed_x, fixed_y = ds[0]
    dyn_x, dyn_y, _ = dynamic_collate(tok.pad_id, multiple_of=8)([ds.examples[0]])
    n = dyn_x.shape[1]
    assert dyn_x[0].tolist() == fixed_x[:n].tolist()
    assert dyn_y[0].tolist() == fixed_y[:n].tolist()


def test_length_grouped_sampler_covers_every_index_once():
    from minigpt.data import LengthGroupedSampler

    lengths = [(i * 7) % 50 + 1 for i in range(103)]
    s = LengthGroupedSampler(lengths, batch_size=8, shuffle=True, seed=3)
    batches = list(s)
    assert len(batches) == len(s)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(103))


def test_length_grouped_sampler_reshuffles_per_epoch():
    from minigpt.data import LengthGroupedSampler

    lengths = [(i * 7) % 50 + 1 for i in range(103)]
    s = LengthGroupedSampler(lengths, batch_size=8, shuffle=True, seed=3)
    s.set_epoch(0)
    first = list(s)
    s.set_epoch(1)
    assert list(s) != first


def test_length_grouped_sampler_batches_are_length_homogeneous():
    from minigpt.data import LengthGroupedSampler

    lengths = [(i * 13) % 200 + 1 for i in range(512)]
    grouped = LengthGroupedSampler(lengths, batch_size=16, shuffle=True, seed=1)
    spread = [max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in grouped]
    # Random batches of 16 drawn from 1..200 would span most of that range.
    assert sum(spread) / len(spread) < 40


def test_sample_text_chunks_is_line_aligned_and_bounded(tmp_path):
    from minigpt.data import sample_text_chunks

    lines = [f"line {i} café\n" for i in range(20000)]
    path = tmp_path / "corpus.txt"
    path.write_text("".join(lines), encoding="utf-8")
    chunks = list(sample_text_chunks(path, max_bytes=40_000, chunk_bytes=10_000))
    assert len(chunks) == 4
    assert sum(len(c.encode()) for c in chunks) <= 40_000
    known = set(lines)
    for chunk in chunks:
        assert chunk.endswith("\n")
        assert all(line + "\n" in known for line in chunk.splitlines())
    # A small file is returned whole.
    assert "".join(sample_text_chunks(path, max_bytes=10**9)) == "".join(lines)


def test_parallel_encoding_matches_sequential(tmp_path):
    from minigpt.data import encode_corpus_to_file, iter_text_chunks

    text = "".join(f"sentence number {i} about the quick brown fox.\n" for i in range(3000))
    path = tmp_path / "corpus.txt"
    path.write_text(text, encoding="utf-8")
    tok = BPETokenizer.train([text], vocab_size=400)
    seq = encode_corpus_to_file(iter_text_chunks(path, 4096), tok, tmp_path / "a.bin", workers=1)
    par = encode_corpus_to_file(iter_text_chunks(path, 4096), tok, tmp_path / "b.bin", workers=2)
    assert seq.tolist() == par.tolist()
    assert tok.decode(seq.tolist()) == text
