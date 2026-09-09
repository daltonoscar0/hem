"""Tests for the command line.

The model is never loaded: every case here either uses ``--rules``, which needs
no checkpoint, or stops before a model would be reached.
"""

import pytest

from hem import levels as L
from hem.__main__ import build_parser, main


def run(capsys, argv):
    code = main(argv)
    return code, capsys.readouterr().out


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


def test_the_default_level_is_natural():
    assert build_parser().parse_args(["hello"]).level == 2


def test_only_the_four_levels_are_accepted():
    for k in L.LEVELS:
        assert build_parser().parse_args(["--level", str(k), "x"]).level == k
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--level", "4", "x"])


def test_the_text_is_joined_from_the_positional_arguments():
    args = build_parser().parse_args(["Send", "the", "report."])
    assert args.text == ["Send", "the", "report."]


def test_the_help_lists_the_dial():
    text = build_parser().format_help()
    for token in L.LEVEL_TOKENS:
        assert token in text


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


SENTENCE = "Send the quarterly report to Sarah before Friday and copy Michael."


def test_it_prints_one_line_per_input(capsys):
    code, out = run(capsys, ["--rules", "--level", "2", SENTENCE])
    assert code == 0
    assert len(out.strip().splitlines()) == 1


def test_level_zero_returns_the_script_in_spoken_register(capsys):
    _, out = run(capsys, ["--rules", "--level", "0", SENTENCE])
    said = out.strip()
    assert said == said.lower()
    assert "," not in said and "." not in said
    assert said.split() == [w.strip(".,").lower() for w in SENTENCE.split()]


def test_a_higher_level_says_more(capsys):
    _, low = run(capsys, ["--rules", "--level", "1", "--seed", "5", SENTENCE])
    _, high = run(capsys, ["--rules", "--level", "3", "--seed", "5", SENTENCE])
    assert len(high.split()) > len(low.split())


def test_all_levels_prints_four_labelled_lines(capsys):
    _, out = run(capsys, ["--rules", "--all-levels", SENTENCE])
    lines = out.strip().splitlines()
    assert len(lines) == 4
    for k in L.LEVELS:
        assert lines[k].startswith(f"d{k}: ")


def test_levels_prints_the_dial_and_stops(capsys):
    code, out = run(capsys, ["--levels"])
    assert code == 0
    assert "| Level |" in out
    for token in L.LEVEL_TOKENS:
        assert token in out


def test_no_input_is_an_error(capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert main(["--rules"]) == 1


def test_stdin_is_read_when_no_text_is_given(capsys, monkeypatch):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(SENTENCE + "\n"))
    code, out = run(capsys, ["--rules", "--level", "1"])
    assert code == 0
    assert out.strip()


# --------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------


def test_trace_lists_the_insertions(capsys):
    _, out = run(capsys, ["--rules", "--level", "3", "--seed", "3", "--trace", SENTENCE])
    lines = out.strip().splitlines()
    assert len(lines) > 1
    for line in lines[1:]:
        assert any(t in line for t in ("filler", "repetition", "repair", "false_start"))
        assert "before clean word" in line


def test_trace_says_so_when_nothing_was_inserted(capsys):
    _, out = run(capsys, ["--rules", "--level", "0", "--trace", SENTENCE])
    assert "(nothing inserted)" in out


def test_the_traced_count_matches_the_words_that_appeared(capsys):
    _, out = run(capsys, ["--rules", "--level", "3", "--seed", "11", "--trace", SENTENCE])
    lines = out.strip().splitlines()
    said = lines[0]
    extra = len(said.split()) - len([w for w in SENTENCE.split()])
    assert extra > 0
    assert len(lines) - 1 > 0
