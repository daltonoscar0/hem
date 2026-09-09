"""Tests for the generator interface.

The model is never loaded. What is covered is the rule-based generator, which
implements the same interface, and the path resolution that decides whether a
``--model`` argument is a missing local checkpoint or a Hub identifier.
"""

import pytest

from hem import levels as L
from hem.generate import HUB_MODEL, Rules, _looks_local, load
from hem.injector import spoken_text

SENTENCES = [
    "Send the quarterly report to Sarah before Friday and copy Michael.",
    "Move the blue folder from the basement to the upstairs office today.",
]


# --------------------------------------------------------------------------
# the rule generator
# --------------------------------------------------------------------------


def test_it_returns_one_line_per_input():
    assert len(Rules().speak(SENTENCES, 2)) == len(SENTENCES)


def test_level_zero_is_the_script_in_spoken_register():
    assert Rules().speak(SENTENCES, 0) == [spoken_text(t) for t in SENTENCES]


def test_output_is_lowercase_and_unpunctuated():
    for said in Rules().speak(SENTENCES, 3):
        assert said == said.lower()
        assert not any(c in said for c in ".,;:!?")


def test_higher_levels_add_more_words():
    lengths = [
        sum(len(s.split()) for s in Rules(seed=4).speak(SENTENCES * 20, k))
        for k in L.LEVELS
    ]
    assert lengths == sorted(lengths)


def test_the_seed_makes_it_reproducible():
    assert Rules(seed=9).speak(SENTENCES, 3) == Rules(seed=9).speak(SENTENCES, 3)


def test_different_seeds_give_different_speech():
    a = Rules(seed=1).speak(SENTENCES * 5, 3)
    b = Rules(seed=2).speak(SENTENCES * 5, 3)
    assert a != b


def test_the_offset_shifts_the_draw():
    assert Rules(seed=1).speak(SENTENCES, 3) != Rules(seed=1).speak(SENTENCES, 3, offset=7)


def test_an_empty_input_gives_an_empty_output():
    assert Rules().speak([], 2) == []


# --------------------------------------------------------------------------
# resolving --model
# --------------------------------------------------------------------------


def test_a_checkpoint_directory_looks_local():
    assert _looks_local("outputs/model")
    assert _looks_local("./somewhere")
    assert _looks_local("/tmp/model")


def test_a_hub_identifier_does_not_look_local():
    assert not _looks_local(HUB_MODEL)
    assert not _looks_local("google/flan-t5-small")


def test_a_missing_local_checkpoint_says_how_to_get_one():
    with pytest.raises(SystemExit) as excinfo:
        load("model", "outputs/definitely-not-here")
    message = str(excinfo.value)
    assert "hem.train" in message
    assert HUB_MODEL in message


def test_rules_are_loadable_by_name():
    assert isinstance(load("rules", seed=3), Rules)
