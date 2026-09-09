"""Tests for the rule-based injector.

The injector is the baseline the model is measured against, and it is also what
generates most of the training data, so a bug here is a bug in both halves of
the repo. The properties that matter are that it is invertible, that its spans
are exact, and that its output rate tracks the rate it was asked for.
"""

import random
from collections import Counter

import pytest

from hem.injector import (
    FILLERS,
    FILLER_VOCAB,
    TYPES,
    InjectionConfig,
    canonical_clean_tokens,
    draw_count,
    inject,
    remove_disfluencies,
    spoken_form,
    spoken_text,
)

SENTENCE = (
    "Send the quarterly report to Sarah before Friday, and copy Michael on the "
    "final version of the proposal."
)


def heavy(**kw):
    base = dict(
        filler_per_100w=12.0,
        repetition_per_100w=7.0,
        repair_per_100w=7.0,
        false_start_per_100w=1.5,
    )
    base.update(kw)
    return InjectionConfig(**base)


# --------------------------------------------------------------------------
# invertibility
# --------------------------------------------------------------------------


def test_removing_the_annotated_spans_reproduces_the_clean_text():
    for seed in range(60):
        result = inject(SENTENCE, heavy(), rng_seed=seed)
        assert remove_disfluencies(result) == result.clean_text


def test_clean_text_is_the_input_minus_nothing():
    result = inject(SENTENCE, heavy(), rng_seed=3)
    assert result.clean_tokens == canonical_clean_tokens(SENTENCE)


def test_alignment_is_one_entry_per_disfluent_token():
    result = inject(SENTENCE, heavy(), rng_seed=5)
    assert len(result.alignment) == len(result.disfluent_tokens)
    assert result.disfluent_text.split() == result.disfluent_tokens


def test_every_kept_token_maps_to_the_clean_token_it_came_from():
    result = inject(SENTENCE, heavy(), rng_seed=11)
    for i, source in enumerate(result.alignment):
        if source is None:
            continue
        assert result.disfluent_tokens[i] == spoken_form(result.clean_tokens[source])


def test_empty_input_gives_an_empty_result():
    result = inject("", heavy(), rng_seed=1)
    assert result.disfluent_text == ""
    assert result.clean_text == ""


# --------------------------------------------------------------------------
# spans
# --------------------------------------------------------------------------


def test_spans_stay_inside_the_disfluent_sequence():
    for seed in range(40):
        result = inject(SENTENCE, heavy(), rng_seed=seed)
        n = len(result.disfluent_tokens)
        for ann in result.annotations:
            for span in (ann.reparandum, ann.interregnum, ann.repair):
                if span is None:
                    continue
                assert 0 <= span.start <= span.end <= n


def test_deleted_positions_never_overlap_between_annotations():
    for seed in range(40):
        result = inject(SENTENCE, heavy(), rng_seed=seed)
        seen = set()
        for ann in result.annotations:
            here = set(ann.deleted_indices())
            assert not (here & seen)
            seen |= here


def test_deleted_positions_are_exactly_the_injected_ones():
    for seed in range(40):
        result = inject(SENTENCE, heavy(), rng_seed=seed)
        deleted = set()
        for ann in result.annotations:
            deleted.update(ann.deleted_indices())
        injected = {i for i, src in enumerate(result.alignment) if src is None}
        assert deleted == injected


def test_annotations_come_back_in_position_order():
    result = inject(SENTENCE, heavy(), rng_seed=17)
    boundaries = [a.boundary_index() for a in result.annotations]
    assert boundaries == sorted(boundaries)


def test_every_annotation_has_a_known_type():
    for seed in range(30):
        result = inject(SENTENCE, heavy(), rng_seed=seed)
        for ann in result.annotations:
            assert ann.type in TYPES


# --------------------------------------------------------------------------
# rate control
# --------------------------------------------------------------------------


def test_draw_count_has_the_right_expectation():
    rng = random.Random(0)
    draws = [draw_count(10.0, 15, rng) for _ in range(20_000)]
    assert 1.4 < sum(draws) / len(draws) < 1.6  # 10 per 100 words over 15 words


def test_draw_count_does_not_round_short_sentences_to_zero():
    """At three events per 100 words a twelve-word sentence expects 0.36. Round
    that and the dial is dead below twenty words."""
    rng = random.Random(1)
    draws = [draw_count(3.0, 12, rng) for _ in range(5_000)]
    assert any(draws)
    assert 0.3 < sum(draws) / len(draws) < 0.42


def test_draw_count_is_zero_at_a_zero_rate():
    rng = random.Random(2)
    assert all(draw_count(0.0, n, rng) == 0 for n in range(1, 40))


def test_measured_rate_tracks_the_requested_rate():
    cfg = heavy(filler_per_100w=8.0, repetition_per_100w=0.0,
                repair_per_100w=0.0, false_start_per_100w=0.0)
    words = fillers = 0
    for seed in range(400):
        result = inject(SENTENCE, cfg, rng_seed=seed)
        words += len(result.clean_tokens)
        fillers += sum(1 for a in result.annotations if a.type == "filler")
    assert 6.5 < fillers * 100 / words < 9.5


def test_a_longer_sentence_gets_proportionally_more():
    cfg = heavy(filler_per_100w=20.0, repetition_per_100w=0.0,
                repair_per_100w=0.0, false_start_per_100w=0.0)
    short = "Send the report to Sarah before Friday today."
    long = " ".join([short.rstrip(".")] * 4) + "."
    def count(text):
        return sum(
            len([a for a in inject(text, cfg, rng_seed=s).annotations
                 if a.type == "filler"])
            for s in range(200)
        )
    assert count(long) > 2.5 * count(short)


def test_nothing_is_injected_at_a_zero_configuration():
    result = inject(SENTENCE, InjectionConfig(), rng_seed=9)
    assert result.annotations == []
    assert result.disfluent_text == spoken_text(SENTENCE)


# --------------------------------------------------------------------------
# spoken register
# --------------------------------------------------------------------------


def test_output_is_lowercase_and_unpunctuated():
    for seed in range(20):
        text = inject(SENTENCE, heavy(), rng_seed=seed).disfluent_text
        assert text == text.lower()
        assert not any(c in text for c in ".,;:!?")


def test_spoken_text_keeps_every_word_and_their_order():
    assert spoken_text(SENTENCE).split() == [
        spoken_form(t) for t in canonical_clean_tokens(SENTENCE)
    ]


def test_internal_apostrophes_survive():
    assert spoken_form("don't") == "don't"
    assert spoken_form("Sarah's,") == "sarah's"


def test_a_token_with_no_letters_is_dropped_from_the_clean_side():
    assert canonical_clean_tokens("send it -- now") == ["send", "it", "now"]


# --------------------------------------------------------------------------
# lexicon
# --------------------------------------------------------------------------


def test_filler_weights_are_a_distribution():
    total = sum(w for _, w in FILLERS)
    assert abs(total - 1.0) < 1e-9


def test_filler_vocab_covers_every_filler_phrase():
    for phrase, _ in FILLERS:
        assert all(w in FILLER_VOCAB for w in phrase)


def test_uh_is_the_commonest_filler_in_the_output():
    cfg = heavy(filler_per_100w=40.0, repetition_per_100w=0.0,
                repair_per_100w=0.0, false_start_per_100w=0.0)
    counts = Counter()
    for seed in range(300):
        result = inject(SENTENCE, cfg, rng_seed=seed)
        for ann in result.annotations:
            if ann.type == "filler" and ann.interregnum is not None:
                span = ann.interregnum
                counts[" ".join(result.disfluent_tokens[span.start : span.end])] += 1
    assert counts.most_common(1)[0][0] == "uh"


def test_a_filler_is_never_placed_beside_a_literal_use_of_itself():
    cfg = heavy(filler_per_100w=60.0, repetition_per_100w=0.0,
                repair_per_100w=0.0, false_start_per_100w=0.0)
    text = "Something like the blue folder would work well for the launch."
    for seed in range(80):
        result = inject(text, cfg, rng_seed=seed)
        toks = result.disfluent_tokens
        for ann in result.annotations:
            if ann.type != "filler" or ann.interregnum is None:
                continue
            span = ann.interregnum
            neighbours = toks[max(0, span.start - 1) : span.start] + toks[span.end : span.end + 1]
            assert not (set(neighbours) & set(toks[span.start : span.end]))


# --------------------------------------------------------------------------
# repairs on arbitrary text
# --------------------------------------------------------------------------


def test_a_repair_can_be_placed_in_text_naming_no_days_or_colours():
    """Mend's injector only put a repair where a closed word category appeared.
    Hem has to work on whatever script it is handed."""
    text = (
        "The committee published its findings after reviewing every submission "
        "received during the consultation period."
    )
    cfg = heavy(filler_per_100w=0.0, repetition_per_100w=0.0,
                repair_per_100w=20.0, false_start_per_100w=0.0)
    found = sum(
        1
        for seed in range(60)
        for a in inject(text, cfg, rng_seed=seed).annotations
        if a.type == "repair"
    )
    assert found > 20


def test_a_repair_always_has_a_reparandum_and_a_repair():
    for seed in range(60):
        for ann in inject(SENTENCE, heavy(), rng_seed=seed).annotations:
            if ann.type == "repair":
                assert ann.reparandum is not None
                assert ann.repair is not None


def test_the_same_seed_gives_the_same_output():
    a = inject(SENTENCE, heavy(), rng_seed=42).disfluent_text
    b = inject(SENTENCE, heavy(), rng_seed=42).disfluent_text
    assert a == b


def test_different_seeds_give_different_output():
    outputs = {inject(SENTENCE, heavy(), rng_seed=s).disfluent_text for s in range(20)}
    assert len(outputs) > 10
