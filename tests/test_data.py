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
