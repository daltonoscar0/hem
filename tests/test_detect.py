"""Tests for the insertion detector.

Every number in the evaluation is a count the detector produced, so its rules
are worth pinning down one at a time: which runs are peeled off as filled
pauses, when a run counts as a repetition, and what happens to the leftovers.

The end-to-end property is that the detector run over the injector's output
should recover the injector's own annotations, because there the right answer is
known exactly.
"""

import pytest

from hem import levels as L
from hem.detect import classify_run, detect, trace
from hem.injector import InjectionConfig, inject

CLEAN = "Send the quarterly report to Sarah before Friday."


def types_of(clean, spoken):
    result, _ = detect(clean, spoken)
    return [a.type for a in result.annotations]


# --------------------------------------------------------------------------
# the four types
# --------------------------------------------------------------------------


def test_a_filled_pause_is_found():
    assert types_of(CLEAN, "send the uh quarterly report to sarah before friday") == [
        "filler"
    ]


def test_a_multi_word_filled_pause_is_one_event_not_two():
    assert types_of(
        CLEAN, "send the you know quarterly report to sarah before friday"
    ) == ["filler"]


def test_a_repetition_is_found():
    assert types_of(
        CLEAN, "send the the quarterly report to sarah before friday"
    ) == ["repetition"]


def test_a_two_word_repetition_is_found():
    assert types_of(
        CLEAN, "send the quarterly send the quarterly report to sarah before friday"
    ) == ["repetition"]


def test_a_substitution_with_an_editing_phrase_is_found():
    assert types_of(
        CLEAN, "send the quarterly report to john no wait sarah before friday"
    ) == ["repair"]


def test_a_substitution_without_an_editing_phrase_is_found():
    assert types_of(
        CLEAN, "send the quarterly report to john sarah before friday"
    ) == ["repair"]


def test_a_restart_at_the_front_is_found():
    assert types_of(
        CLEAN, "can you send the quarterly report to sarah before friday"
    ) == ["false_start"]


def test_nothing_inserted_gives_nothing():
    assert types_of(CLEAN, "send the quarterly report to sarah before friday") == []


# --------------------------------------------------------------------------
# splitting a mixed run
# --------------------------------------------------------------------------


def test_a_filler_in_front_of_a_substitution_is_two_events():
    assert types_of(
        CLEAN, "send the quarterly report to uh john no wait sarah before friday"
    ) == ["filler", "repair"]


def test_a_filler_after_a_substitution_is_peeled_off_too():
    out = classify_run(["john", "no", "wait", "uh"], 0, 4, 0)
    assert [t for t, _, _ in out] == ["repair", "filler"]


def test_fillers_on_both_ends_are_both_peeled():
    out = classify_run(["uh", "john", "you", "know"], 0, 4, 0)
    assert [t for t, _, _ in out] == ["filler", "repair", "filler"]


def test_a_run_that_is_entirely_fillers_yields_only_fillers():
    out = classify_run(["uh", "you", "know"], 0, 3, 0)
    assert [t for t, _, _ in out] == ["filler", "filler"]


def test_you_know_is_matched_before_you():
    """Longest phrase first, or "you know" becomes a filler plus a leftover."""
    out = classify_run(["you", "know"], 0, 2, 0)
    assert len(out) == 1 and out[0][0] == "filler"


# --------------------------------------------------------------------------
# fidelity
# --------------------------------------------------------------------------


def test_dropped_words_are_reported():
    _, report = detect(CLEAN, "send the report to sarah before friday")
    assert report["dropped"] == ["quarterly"]
    assert report["n_dropped"] == 1


def test_nothing_dropped_when_the_script_survives():
    _, report = detect(CLEAN, "send the uh quarterly report to sarah before friday")
    assert report["n_dropped"] == 0
    assert report["kept"] == report["clean_words"]


def test_a_rewritten_word_counts_as_dropped_and_inserted():
    result, report = detect(CLEAN, "send the annual report to sarah before friday")
    assert report["dropped"] == ["quarterly"]
    assert [a.type for a in result.annotations] == ["repair"]


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def test_alignment_marks_inserted_tokens_as_none():
    result, _ = detect(CLEAN, "send the uh quarterly report to sarah before friday")
    assert result.alignment[2] is None
    assert result.alignment[:2] == [0, 1]


def test_alignment_maps_survivors_back_to_the_clean_index():
    result, _ = detect(CLEAN, "send the uh quarterly report to sarah before friday")
    for i, src in enumerate(result.alignment):
        if src is not None:
            assert result.disfluent_tokens[i] == result.clean_tokens[src].strip(",.").lower()


# --------------------------------------------------------------------------
# trace
# --------------------------------------------------------------------------


def test_trace_reports_type_position_and_words():
    rows = trace(CLEAN, "send the uh quarterly report to sarah before friday")
    assert rows == [
        {"type": "filler", "position": 2, "before_clean_word": 2, "words": "uh"}
    ]


def test_trace_is_empty_when_nothing_was_inserted():
    assert trace(CLEAN, "send the quarterly report to sarah before friday") == []


# --------------------------------------------------------------------------
# against the injector, where the answer is known
# --------------------------------------------------------------------------


LONG = (
    "Send the quarterly report to Sarah before Friday, and copy Michael on the "
    "final version of the proposal."
)


def test_it_finds_about_as_many_events_as_the_injector_inserted():
    injected = detected = 0
    cfg = InjectionConfig(filler_per_100w=10.0, repetition_per_100w=5.0,
                          repair_per_100w=5.0, false_start_per_100w=1.0)
    for seed in range(120):
        result = inject(LONG, cfg, rng_seed=seed)
        injected += len(result.annotations)
        found, _ = detect(result.clean_text, result.disfluent_text)
        detected += len(found.annotations)
    assert 0.85 < detected / injected < 1.15


def test_fillers_are_recovered_almost_exactly():
    cfg = InjectionConfig(filler_per_100w=15.0)
    put = got = 0
    for seed in range(150):
        result = inject(LONG, cfg, rng_seed=seed)
        put += sum(1 for a in result.annotations if a.type == "filler")
        found, _ = detect(result.clean_text, result.disfluent_text)
        got += sum(1 for a in found.annotations if a.type == "filler")
    assert put > 0
    assert 0.95 < got / put < 1.05


def test_a_delayed_repair_is_the_only_thing_that_looks_like_a_dropped_word():
    """The injector never deletes a script word, so every dropped word the
    detector reports on its output is a false positive. They all come from one
    place: a delayed repair emits the corrected word several positions late, the
    aligner sees it leave its clean position and arrive somewhere else, and
    calls that a deletion plus an insertion. It stays under a few per cent even
    at the heaviest setting, which is the bound worth holding."""
    cfg = L.config(3)
    rows = 400
    bad = 0
    for seed in range(rows):
        result = inject(LONG, cfg, rng_seed=seed)
        _, report = detect(result.clean_text, result.disfluent_text)
        if report["n_dropped"]:
            bad += 1
            # the word reported as dropped is always one the injector kept
            assert all(w in result.clean_text.lower() for w in report["dropped"])
    assert bad / rows < 0.05


def test_no_dropped_words_when_no_repair_can_be_delayed():
    cfg = InjectionConfig(filler_per_100w=15.0, repetition_per_100w=8.0)
    for seed in range(120):
        result = inject(LONG, cfg, rng_seed=seed)
        _, report = detect(result.clean_text, result.disfluent_text)
        assert report["n_dropped"] == 0
