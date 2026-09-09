"""Tests for the surprisal analysis.

The language model is never loaded here. ``collect`` takes its scorer as an
argument, so a stub that returns one known value per word covers the part that
is easy to get wrong: which word a subword belongs to, which position the
control window is read at, and whether the two stay paired instance by instance.
"""

import numpy as np
import pytest

from hem.surprisal import (
    WINDOW,
    _fold_to_words,
    _word_starts,
    bootstrap_ci,
    collect,
    matched_clean_position,
    summarise,
)


class StubScorer:
    """Returns a caller-supplied surprisal per word, keyed by sentence."""

    def __init__(self, table):
        self.table = table
        self.seen = []

    def word_surprisals(self, sentences, batch=16):
        self.seen.extend(sentences)
        return [np.asarray(self.table[s], dtype=float) for s in sentences]


# --------------------------------------------------------------------------
# folding subwords into words
# --------------------------------------------------------------------------


def test_word_starts_gives_character_spans():
    assert _word_starts("send it now") == [(0, 4), (5, 7), (8, 11)]


def test_subwords_sum_into_their_word():
    text = "ab cd"
    offsets = [(0, 1), (1, 2), (2, 5)]  # "ab" arrives as two subwords
    assert _fold_to_words(text, offsets, np.array([1.0, 2.0, 4.0])).tolist() == [3.0, 4.0]


def test_a_leading_space_does_not_push_a_subword_into_the_previous_word():
    """The tokeniser folds the preceding space into the token, so a raw offset
    points at the end of the previous word."""
    text = "ab cd"
    offsets = [(0, 2), (2, 5)]
    assert _fold_to_words(text, offsets, np.array([1.0, 5.0])).tolist() == [1.0, 5.0]


def test_folding_returns_one_value_per_word():
    text = "one two three four"
    offsets = [(0, 3), (3, 7), (7, 13), (13, 18)]
    assert _fold_to_words(text, offsets, np.ones(4)).shape == (4,)


# --------------------------------------------------------------------------
# locating the control window
# --------------------------------------------------------------------------


def test_the_control_position_counts_the_surviving_words():
    alignment = [0, 1, None, None, 2, 3]
    assert matched_clean_position(alignment, 2) == 2
    assert matched_clean_position(alignment, 4) == 2


def test_an_out_of_order_alignment_has_no_matching_position():
    """A delayed repair holds a word back and emits it later, so the words
    before the boundary are no longer a prefix of the clean sentence and there
    is nothing to read the control at."""
    assert matched_clean_position([0, 2, 3, None], 3) is None


def test_an_in_order_alignment_always_matches():
    assert matched_clean_position([0, 1, 2, None], 3) == 3


def test_a_boundary_at_the_very_start_maps_to_the_start():
    assert matched_clean_position([None, 0, 1], 0) == 0


# --------------------------------------------------------------------------
# collecting windows
# --------------------------------------------------------------------------


def make_row(kind="repair"):
    """A row with a one-token insertion at index 6, with room either side."""
    clean = [f"c{i}" for i in range(14)]
    disfluent = clean[:6] + ["wrong"] + clean[6:]
    alignment = list(range(6)) + [None] + list(range(6, 14))
    return {
        "disfluent": " ".join(disfluent),
        "clean": " ".join(clean),
        "disfluent_tokens": disfluent,
        "clean_tokens": clean,
        "alignment": alignment,
        "annotations": [
            {
                "type": kind,
                "reparandum": {"start": 6, "end": 7},
                "interregnum": None,
                "repair": {"start": 7, "end": 8},
            }
        ],
    }


def scorer_for(row, disfluent_values, clean_values):
    return StubScorer(
        {row["disfluent"]: disfluent_values, row["clean"]: clean_values}
    )


def test_a_window_is_centred_on_the_boundary():
    """With no editing phrase, the boundary is the first token of the *repair*,
    at index 7, not the abandoned word at index 6. That is the position a
    listener learns something went wrong, and getting it wrong would shift every
    curve in the figure by one."""
    row = make_row()
    dis = [0.0] * len(row["disfluent_tokens"])
    dis[7] = 9.0
    scorer = scorer_for(row, dis, [0.0] * len(row["clean_tokens"]))
    data, control = collect([row], scorer, batch=8, verbose=False)
    assert data.shape == (1, 2 * WINDOW + 1)
    assert data[0][WINDOW] == 9.0
    assert data[0][WINDOW - 1] == 0.0  # the reparandum sits one position earlier
    assert control.shape == data.shape


def test_an_editing_phrase_moves_the_boundary_to_its_first_token():
    row = make_row()
    row["annotations"][0]["interregnum"] = {"start": 6, "end": 7}
    dis = [0.0] * len(row["disfluent_tokens"])
    dis[6] = 9.0
    scorer = scorer_for(row, dis, [0.0] * len(row["clean_tokens"]))
    data, _ = collect([row], scorer, batch=8, verbose=False)
    assert data[0][WINDOW] == 9.0


def test_the_control_is_read_at_the_matching_clean_position():
    row = make_row()
    clean = [0.0] * len(row["clean_tokens"])
    clean[6] = 4.0
    scorer = scorer_for(row, [0.0] * len(row["disfluent_tokens"]), clean)
    _, control = collect([row], scorer, batch=8, verbose=False)
    assert control[0][WINDOW] == 4.0


def test_only_the_requested_type_is_collected():
    row = make_row(kind="filler")
    scorer = scorer_for(row, [0.0] * len(row["disfluent_tokens"]),
                        [0.0] * len(row["clean_tokens"]))
    data, _ = collect([row], scorer, batch=8, kind="repair", verbose=False)
    assert data.shape[0] == 0
    data, _ = collect([row], scorer, batch=8, kind="filler", verbose=False)
    assert data.shape[0] == 1


def test_a_disfluency_too_close_to_the_edge_is_skipped():
    clean = [f"c{i}" for i in range(6)]
    disfluent = ["wrong"] + clean
    row = {
        "disfluent": " ".join(disfluent),
        "clean": " ".join(clean),
        "disfluent_tokens": disfluent,
        "clean_tokens": clean,
        "alignment": [None] + list(range(6)),
        "annotations": [
            {"type": "repair", "reparandum": {"start": 0, "end": 1},
             "interregnum": None, "repair": {"start": 1, "end": 2}}
        ],
    }
    scorer = StubScorer({row["disfluent"]: [0.0] * 7, row["clean"]: [0.0] * 6})
    data, _ = collect([row], scorer, batch=8, verbose=False)
    assert data.shape[0] == 0


def test_a_row_with_no_annotations_contributes_nothing():
    row = make_row()
    row["annotations"] = []
    scorer = StubScorer({})
    data, _ = collect([row], scorer, batch=8, verbose=False)
    assert data.shape[0] == 0


def test_each_sentence_is_scored_once_however_many_disfluencies_it_has():
    row = make_row()
    row["annotations"] = row["annotations"] * 2
    scorer = scorer_for(row, [0.0] * len(row["disfluent_tokens"]),
                        [0.0] * len(row["clean_tokens"]))
    collect([row], scorer, batch=8, verbose=False)
    assert len(scorer.seen) == 2  # the disfluent sentence and its control


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def test_the_bootstrap_mean_is_the_sample_mean():
    data = np.array([[1.0, 2.0], [3.0, 4.0]])
    mean, lo, hi = bootstrap_ci(data, n_draws=200)
    assert mean.tolist() == [2.0, 3.0]
    assert (lo <= mean).all() and (mean <= hi).all()


def test_the_bootstrap_is_seeded():
    data = np.random.default_rng(0).normal(size=(50, 3))
    a = bootstrap_ci(data, n_draws=100, seed=7)
    b = bootstrap_ci(data, n_draws=100, seed=7)
    assert np.allclose(a[1], b[1]) and np.allclose(a[2], b[2])


def test_the_peak_is_found_where_the_difference_is_largest():
    n = 2 * WINDOW + 1
    data = np.zeros((30, n))
    data[:, WINDOW + 1] = 5.0  # one position after the boundary
    control = np.zeros((30, n))
    out = summarise(data, control, n_draws=100, seed=1)
    assert out["peak_position"] == 1
    assert out["peak_effect_bits"] == pytest.approx(5.0)
    assert out["n"] == 30


def test_the_effect_at_the_boundary_is_reported_separately_from_the_peak():
    n = 2 * WINDOW + 1
    data = np.zeros((10, n))
    data[:, WINDOW] = 2.0
    data[:, WINDOW - 1] = 6.0
    out = summarise(data, np.zeros((10, n)), n_draws=100, seed=1)
    assert out["peak_position"] == -1
    assert out["effect_at_boundary_bits"] == pytest.approx(2.0)


def test_an_empty_set_summarises_to_nothing():
    assert summarise(np.zeros((0, 11)), np.zeros((0, 11)), 100, 1) == {"n": 0}


def test_the_control_is_subtracted_instance_by_instance():
    n = 2 * WINDOW + 1
    data = np.zeros((4, n))
    control = np.zeros((4, n))
    data[:, WINDOW] = [10.0, 20.0, 30.0, 40.0]
    control[:, WINDOW] = [9.0, 19.0, 29.0, 39.0]
    out = summarise(data, control, n_draws=100, seed=1)
    assert out["effect_at_boundary_bits"] == pytest.approx(1.0)
