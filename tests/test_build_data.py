"""Tests for the training-pair builder.

The one property worth protecting here is that the control token means the same
thing on both sides of the mix. Synthetic pairs are generated at a setting and
then relabelled by what they measured; real pairs are labelled by what they
measured. If those two ever come apart, the model is being taught that ``<d3>``
sometimes means light, and no amount of training fixes that.
"""

import json

import pytest

from hem import levels as L
from hem.build_data import (
    PREFIX,
    encode_source,
    level_of_row,
    make_synthetic,
    mix_real,
    real_pairs,
    summarise,
    write_jsonl,
)
from hem.corpus import SourceSentence
from hem.injector import inject
from hem.rates import MIN_CLEAN_WORDS, measure

SENTENCES = [
    SourceSentence(
        "Send the quarterly report to Sarah before Friday and copy Michael.",
        "template", 0,
    ),
    SourceSentence(
        "Move the blue folder from the basement to the upstairs office today.",
        "template", 1,
    ),
    SourceSentence(
        "Remind me to confirm the numbers with the supplier on Tuesday morning.",
        "template", 2,
    ),
]


# --------------------------------------------------------------------------
# the input encoding
# --------------------------------------------------------------------------


def test_the_source_starts_with_the_control_token():
    text = "Send the report."
    assert encode_source(2, text) == f"<d2> {PREFIX}{text}"


def test_every_level_gets_a_distinct_prefix():
    prefixes = {encode_source(k, "x")[:4] for k in L.LEVELS}
    assert len(prefixes) == 4


def test_the_level_can_be_read_back_off_the_source():
    for k in L.LEVELS:
        assert L.level_of_token(encode_source(k, "Send the report.")) == k


# --------------------------------------------------------------------------
# labelling
# --------------------------------------------------------------------------


def test_a_pair_is_labelled_by_what_it_measured():
    row = inject(SENTENCES[0].text, L.config(3), rng_seed=4).to_dict()
    spec = L.spec()
    rate = measure(row)["rate"]
    label = level_of_row(row)
    if rate >= spec["heavy_cut_per_100w"]:
        assert label == 3
    elif rate >= spec["light_cut_per_100w"]:
        assert label == 2
    elif rate > 0:
        assert label == 1
    else:
        assert label == 0


def test_a_pair_with_nothing_injected_is_labelled_zero():
    row = inject(SENTENCES[0].text, L.config(0), rng_seed=1).to_dict()
    assert level_of_row(row) == 0


def test_the_label_can_differ_from_the_setting_that_was_asked_for():
    """This is the whole point of relabelling: the injector is stochastic, so a
    heavy setting sometimes produces a light sentence."""
    asked = [
        (r["asked_level"], r["level"])
        for r in make_synthetic(SENTENCES, 200, 7, balance=False)
    ]
    assert any(a != g for a, g in asked)


def test_generated_pairs_all_carry_a_valid_label():
    rows = make_synthetic(SENTENCES, 60, 3)
    assert rows
    for row in rows:
        assert row["level"] in L.LEVELS
        assert row["level"] == level_of_row(row)


def test_balancing_spreads_the_labels_across_the_four_levels():
    rows = make_synthetic(SENTENCES, 400, 11, balance=True)
    counts = {k: sum(1 for r in rows if r["level"] == k) for k in L.LEVELS}
    assert all(counts[k] > 0 for k in L.LEVELS)
    assert min(counts.values()) > 0.5 * max(counts.values())


def test_short_sentences_are_not_turned_into_pairs():
    short = [SourceSentence("Send it now.", "template", 0)]
    assert make_synthetic(short, 20, 1) == []


def test_an_empty_pool_produces_nothing():
    assert make_synthetic([], 10, 1) == []


# --------------------------------------------------------------------------
# real pairs
# --------------------------------------------------------------------------


def swda_row(clean_words, disfluent_words, annotations):
    clean = [f"w{i}" for i in range(clean_words)]
    return {
        "clean": " ".join(clean),
        "clean_tokens": clean,
        "disfluent": " ".join(disfluent_words),
        "disfluent_tokens": disfluent_words,
        "alignment": [None] * (len(disfluent_words) - clean_words)
        + list(range(clean_words)),
        "annotations": annotations,
    }


def test_real_pairs_drop_the_short_utterances(tmp_path):
    path = tmp_path / "real.jsonl"
    write_jsonl(
        path,
        [
            swda_row(MIN_CLEAN_WORDS, [f"w{i}" for i in range(MIN_CLEAN_WORDS)], []),
            swda_row(MIN_CLEAN_WORDS - 1,
                     [f"w{i}" for i in range(MIN_CLEAN_WORDS - 1)], []),
        ],
    )
    rows = real_pairs(path)
    assert len(rows) == 1
    assert len(rows[0]["clean_tokens"]) == MIN_CLEAN_WORDS


def test_real_pairs_are_labelled_the_same_way_synthetic_ones_are(tmp_path):
    path = tmp_path / "real.jsonl"
    tokens = ["uh"] + [f"w{i}" for i in range(12)]
    row = swda_row(12, tokens, [{"type": "filler", "reparandum": None,
                                 "interregnum": {"start": 0, "end": 1},
                                 "repair": None}])
    write_jsonl(path, [row])
    got = real_pairs(path)[0]
    assert got["level"] == level_of_row(row)
    assert got["level"] > 0


def test_a_missing_real_file_is_not_an_error(tmp_path):
    assert real_pairs(tmp_path / "nope.jsonl") == []


def test_the_cap_is_applied(tmp_path):
    path = tmp_path / "real.jsonl"
    write_jsonl(path, [swda_row(12, [f"w{i}" for i in range(12)], []) for _ in range(10)])
    assert len(real_pairs(path, limit=3)) == 3


# --------------------------------------------------------------------------
# mixing
# --------------------------------------------------------------------------


def test_mixing_keeps_every_row_from_both_sides():
    synthetic = make_synthetic(SENTENCES, 20, 5)
    real = [{"clean": "a", "disfluent": "a", "level": 0} for _ in range(10)]
    mixed = mix_real(synthetic, real, limit=10, seed=1)
    assert len(mixed) == len(synthetic) + 10


def test_a_zero_limit_mixes_nothing_in():
    synthetic = make_synthetic(SENTENCES, 20, 5)
    assert mix_real(synthetic, [{"x": 1}], limit=0, seed=1) == synthetic


def test_mixing_is_seeded():
    synthetic = make_synthetic(SENTENCES, 20, 5)
    real = [{"clean": str(i), "disfluent": str(i), "level": 0} for i in range(10)]
    a = mix_real(list(synthetic), list(real), 5, seed=3)
    b = mix_real(list(synthetic), list(real), 5, seed=3)
    assert [r.get("clean") for r in a] == [r.get("clean") for r in b]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_the_summary_names_every_level_and_type():
    text = summarise(make_synthetic(SENTENCES, 40, 2))
    for k in L.LEVELS:
        assert f"d{k}=" in text
    assert "events per 100 words" in text
