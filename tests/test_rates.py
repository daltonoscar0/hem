"""Tests for the Switchboard measurement that the dial is built from.

The dial is only as trustworthy as this arithmetic, and the arithmetic is easy
to get subtly wrong: rates against the wrong denominator, percentiles that do
not interpolate, buckets that overlap.
"""

import math

import pytest

from hem.injector import TYPES
from hem.rates import (
    HEAVY_PERCENTILE,
    LIGHT_PERCENTILE,
    MIN_CLEAN_WORDS,
    build_levels,
    bucket_of,
    eligible,
    measure,
    percentile,
    rate_table,
    shape_parameters,
    unigram_log_frequencies,
)


def row(clean_words, annotations, spoken=None):
    clean = [f"w{i}" for i in range(clean_words)]
    return {
        "clean": " ".join(clean),
        "clean_tokens": clean,
        "disfluent": " ".join(spoken or clean),
        "disfluent_tokens": list(spoken or clean),
        "alignment": list(range(clean_words)),
        "annotations": annotations,
    }


def ann(kind, start, end, interregnum=None):
    out = {"type": kind, "reparandum": None, "interregnum": None, "repair": None}
    if kind == "filler":
        out["interregnum"] = {"start": start, "end": end}
    else:
        out["reparandum"] = {"start": start, "end": end}
        out["repair"] = {"start": end, "end": end + (end - start)}
        if interregnum:
            out["interregnum"] = {"start": interregnum[0], "end": interregnum[1]}
    return out


# --------------------------------------------------------------------------
# per-utterance measurement
# --------------------------------------------------------------------------


def test_rate_is_events_per_hundred_clean_words():
    m = measure(row(20, [ann("filler", 0, 1), ann("repetition", 3, 4)]))
    assert m["total"] == 2
    assert m["rate"] == pytest.approx(10.0)


def test_a_clean_utterance_measures_zero():
    m = measure(row(20, []))
    assert m["total"] == 0 and m["rate"] == 0.0


def test_types_are_counted_separately():
    m = measure(row(10, [ann("filler", 0, 1), ann("filler", 2, 3), ann("repair", 5, 6)]))
    assert m["counts"]["filler"] == 2
    assert m["counts"]["repair"] == 1


def test_structural_excludes_filled_pauses():
    m = measure(row(10, [ann("filler", 0, 1), ann("repetition", 2, 3)]))
    assert m["structural"] == 1


def test_inserted_length_counts_reparandum_and_interregnum_only():
    m = measure(row(10, [ann("repair", 0, 2, interregnum=(2, 4))]))
    assert m["inserted"] == 4  # two abandoned words plus a two-word editing phrase


def test_an_unknown_annotation_type_is_not_counted():
    m = measure(row(10, [{"type": "aside", "reparandum": {"start": 0, "end": 1},
                          "interregnum": None, "repair": None}]))
    assert m["total"] == 0


# --------------------------------------------------------------------------
# percentiles
# --------------------------------------------------------------------------


def test_percentile_interpolates():
    assert percentile([0, 10], 50) == pytest.approx(5.0)
    assert percentile([0, 1, 2, 3, 4], 50) == pytest.approx(2.0)


def test_percentile_hits_the_ends():
    values = [3, 1, 2]
    assert percentile(values, 0) == 1
    assert percentile(values, 100) == 3


def test_percentile_of_nothing_is_zero():
    assert percentile([], 90) == 0.0


def test_percentile_of_one_value_is_that_value():
    assert percentile([7.5], 90) == 7.5


# --------------------------------------------------------------------------
# bucketing
# --------------------------------------------------------------------------


def test_buckets_are_cut_at_the_two_percentiles():
    assert bucket_of({"total": 0, "rate": 0.0}, 10.0, 20.0) == 0
    assert bucket_of({"total": 1, "rate": 5.0}, 10.0, 20.0) == 1
    assert bucket_of({"total": 2, "rate": 10.0}, 10.0, 20.0) == 2
    assert bucket_of({"total": 3, "rate": 20.0}, 10.0, 20.0) == 3


def test_the_cuts_are_inclusive_at_the_bottom():
    """A rate exactly on a cut belongs to the higher bucket, so the four are a
    partition rather than three plus a gap."""
    assert bucket_of({"total": 1, "rate": 10.0}, 10.0, 20.0) == 2
    assert bucket_of({"total": 1, "rate": 9.999}, 10.0, 20.0) == 1


def test_anything_disfluent_is_above_bucket_zero():
    assert bucket_of({"total": 1, "rate": 0.001}, 10.0, 20.0) == 1


def test_short_utterances_are_not_eligible():
    assert not eligible(row(MIN_CLEAN_WORDS - 1, []))
    assert eligible(row(MIN_CLEAN_WORDS, []))


# --------------------------------------------------------------------------
# the level table
# --------------------------------------------------------------------------


def corpus():
    """A small corpus with a deliberate spread of rates."""
    rows = []
    for _ in range(20):
        rows.append(row(20, []))                                   # rate 0
    for _ in range(20):
        rows.append(row(20, [ann("filler", 0, 1)]))                # rate 5
    for _ in range(20):
        rows.append(row(20, [ann("filler", 0, 1), ann("repetition", 2, 3),
                             ann("repair", 4, 5)]))                # rate 15
    for _ in range(20):
        rows.append(row(20, [ann("filler", i, i + 1) for i in range(8)]))  # rate 40
    rows.append(row(MIN_CLEAN_WORDS - 1, [ann("filler", 0, 1)]))   # too short
    return rows


def test_short_utterances_are_left_out_of_the_pool():
    spec = build_levels(corpus())
    assert spec["utterances"] == 80
    assert spec["utterances_before_length_filter"] == 81


def test_every_utterance_lands_in_exactly_one_bucket():
    spec = build_levels(corpus())
    assert sum(spec["levels"][str(k)]["utterances"] for k in range(4)) == 80


def test_the_clean_bucket_has_no_events():
    spec = build_levels(corpus())
    assert spec["levels"]["0"]["events"] == 0
    assert spec["levels"]["0"]["total_rate"] == 0.0


def test_rates_rise_with_the_level():
    spec = build_levels(corpus())
    totals = [spec["levels"][str(k)]["total_rate"] for k in range(4)]
    assert totals == sorted(totals)
    assert totals[0] == 0.0


def test_the_mix_sums_to_one_wherever_there_are_events():
    spec = build_levels(corpus())
    for k in range(1, 4):
        assert sum(spec["levels"][str(k)]["mix"].values()) == pytest.approx(1.0)


def test_the_cuts_are_the_named_percentiles_of_the_disfluent_utterances():
    spec = build_levels(corpus())
    assert spec["light_percentile"] == LIGHT_PERCENTILE
    assert spec["heavy_percentile"] == HEAVY_PERCENTILE
    assert spec["light_cut_per_100w"] <= spec["heavy_cut_per_100w"]


def test_the_table_has_a_row_per_level_plus_a_total():
    spec = build_levels(corpus())
    lines = rate_table(spec).splitlines()
    assert len(lines) == 2 + 4 + 1


# --------------------------------------------------------------------------
# shape parameters
# --------------------------------------------------------------------------


def test_interregnum_rate_counts_repairs_with_an_editing_phrase():
    rows = [
        row(10, [ann("repair", 0, 1, interregnum=(1, 3))]),
        row(10, [ann("repair", 0, 1)]),
    ]
    assert shape_parameters(rows)["interregnum_rate"] == pytest.approx(0.5)


def test_span_repair_rate_counts_multi_token_reparandums():
    rows = [row(10, [ann("repair", 0, 2)]), row(10, [ann("repair", 0, 1)])]
    assert shape_parameters(rows)["span_repair_rate"] == pytest.approx(0.5)


def test_filler_phrases_are_counted_from_the_spoken_tokens():
    r = row(10, [ann("filler", 0, 2)], spoken=["you", "know"] + [f"w{i}" for i in range(10)])
    phrases = shape_parameters([r])["filler_phrases"]
    assert phrases[0]["phrase"] == "you know"
    assert phrases[0]["share"] == pytest.approx(1.0)


def test_shape_of_an_empty_corpus_does_not_divide_by_zero():
    out = shape_parameters([])
    assert out["interregnum_rate"] == 0.0
    assert out["filler_phrases"] == []


# --------------------------------------------------------------------------
# frequencies
# --------------------------------------------------------------------------


def test_log_frequencies_are_negative_and_ordered():
    rows = [
        {"clean": "the the the cat", "clean_tokens": ["the", "the", "the", "cat"]},
    ]
    freqs = unigram_log_frequencies(rows)
    assert freqs["the"] > freqs["cat"]
    assert freqs["the"] == pytest.approx(math.log10(3 / 4))


def test_frequencies_are_case_and_punctuation_insensitive():
    rows = [{"clean": "The cat, the CAT.", "clean_tokens": ["The", "cat,", "the", "CAT."]}]
    freqs = unigram_log_frequencies(rows)
    assert set(freqs) == {"the", "cat"}
