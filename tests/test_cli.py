import pytest

from minigpt.cli import build_parser, main


def test_train_parser_defaults():
    args = build_parser().parse_args(["train", "--data", "d.csv"])
    assert args.command == "train"
    assert args.size == "small"
    assert args.save == "last"
    assert args.val_ratio == 0.1


def test_overrides_are_parsed():
    args = build_parser().parse_args(
        ["train", "--data", "d.csv", "--size", "tiny", "--n-layer", "3", "--block-size", "128"]
    )
    assert (args.size, args.n_layer, args.block_size) == ("tiny", 3, 128)


def test_train_requires_data():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train"])


def test_serve_defaults():
    args = build_parser().parse_args(["serve", "--model", "runs/demo"])
    assert (args.host, args.port, args.model_name) == ("127.0.0.1", 8000, "minigpt")


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["nope"])


def test_info_and_tokenize_commands(checkpoint_dir, capsys):
    assert main(["info", "--model", str(checkpoint_dir)]) == 0
    assert "n_layer" in capsys.readouterr().out
    assert main(["tokenize", "--model", str(checkpoint_dir), "--text", "hello"]) == 0
    assert "tokens" in capsys.readouterr().out


def test_generate_command(checkpoint_dir, capsys):
    code = main([
        "generate", "--model", str(checkpoint_dir), "--prompt", "What is the capital of France?",
        "--device", "cpu", "--temperature", "0", "--max-new-tokens", "24", "--repetition-penalty", "1.0",
    ])
    assert code == 0
    assert "Paris" in capsys.readouterr().out
