"""Tests for the dial itself.

``hem/levels.json`` is checked in, so these run against the real measured
numbers rather than a fixture. The point is to catch a regenerated spec that
would quietly break the dial: a level whose rate went down, a mix that stopped
summing to one, a config that no longer wires the measured rates through to the
injector.
"""

import json

import pytest

from hem import levels as L
from hem.injector import TYPES, InjectionConfig, inject


# --------------------------------------------------------------------------
# the spec file
# --------------------------------------------------------------------------


def test_the_spec_loads_and_names_its_source():
    spec = L.spec()
    assert "Switchboard" in spec["source"]
    assert spec["utterances"] > 10_000


def test_there_is_an_entry_for_every_level():
    spec = L.spec()
    assert set(spec["levels"]) == {"0", "1", "2", "3"}


def test_the_buckets_partition_the_pool():
    spec = L.spec()
    counted = sum(spec["levels"][str(k)]["utterances"] for k in L.LEVELS)
    assert counted == spec["utterances"]


def test_shares_of_the_corpus_sum_to_one():
    spec = L.spec()
    total = sum(spec["levels"][str(k)]["share_of_corpus"] for k in L.LEVELS)
    assert total == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# the dial is a dial
# --------------------------------------------------------------------------


def test_the_total_rate_rises_at_every_step():
    totals = [L.targets(k)["total"] for k in L.LEVELS]
    assert totals == sorted(totals)
    assert len(set(totals)) == 4


def test_every_type_rate_rises_at_every_step():
    """A dial that adds substitutions while removing hesitations is not a dial.
    An earlier bucketing did exactly that and this is the test that would have
    caught it."""
    for kind in TYPES:
        rates = [L.targets(k)[kind] for k in L.LEVELS]
        assert rates == sorted(rates), kind


def test_level_zero_is_silent():
    t = L.targets(0)
    assert t["total"] == 0.0
    assert all(t[kind] == 0.0 for kind in TYPES)


def test_the_mix_sums_to_one_above_level_zero():
    for k in (1, 2, 3):
        assert sum(L.mix(k).values()) == pytest.approx(1.0)


def test_substitutions_take_a_larger_share_as_the_dial_goes_up():
    shares = [L.mix(k)["repair"] for k in (1, 2, 3)]
    assert shares == sorted(shares)


def test_filled_pauses_are_the_largest_share_at_every_level():
    for k in (1, 2, 3):
        mix = L.mix(k)
        assert mix["filler"] == max(mix.values())


# --------------------------------------------------------------------------
# tokens
# --------------------------------------------------------------------------


def test_there_is_one_token_per_level():
    assert L.LEVEL_TOKENS == ("<d0>", "<d1>", "<d2>", "<d3>")
    assert [L.token(k) for k in L.LEVELS] == list(L.LEVEL_TOKENS)


def test_an_unknown_level_is_rejected():
    with pytest.raises(ValueError):
        L.token(4)
    with pytest.raises(ValueError):
        L.token(-1)


def test_a_token_is_recognised_at_the_front_of_a_string():
    assert L.level_of_token("<d2> say aloud: hello") == 2
    assert L.level_of_token("say aloud: hello") is None


# --------------------------------------------------------------------------
# the injector configuration
# --------------------------------------------------------------------------


def test_config_carries_the_measured_rates_through():
    for k in L.LEVELS:
        cfg = L.config(k)
        t = L.targets(k)
        assert cfg.filler_per_100w == pytest.approx(t["filler"])
        assert cfg.repetition_per_100w == pytest.approx(t["repetition"])
        assert cfg.repair_per_100w == pytest.approx(t["repair"])
        assert cfg.false_start_per_100w == pytest.approx(t["false_start"])


def test_config_carries_the_measured_shape_through():
    cfg = L.config(2)
    shape = L.spec()["shape"]
    assert cfg.interregnum_rate == pytest.approx(shape["interregnum_rate"])
    assert cfg.span_repair_rate == pytest.approx(shape["span_repair_rate"])
    assert cfg.repetition_bigram_rate == pytest.approx(shape["repetition_bigram_rate"])


def test_overrides_win():
    assert L.config(2, filler_per_100w=99.0).filler_per_100w == 99.0


def test_level_zero_injects_nothing():
    text = "Send the quarterly report to Sarah before Friday and copy Michael."
    for seed in range(20):
        assert inject(text, L.config(0), rng_seed=seed).annotations == []


def test_the_injector_at_each_level_produces_more_than_the_one_below():
    text = (
        "Send the quarterly report to Sarah before Friday, and copy Michael on "
        "the final version of the proposal before the meeting."
    )
    counts = []
    for k in L.LEVELS:
        counts.append(
            sum(len(inject(text, L.config(k), rng_seed=s).annotations)
                for s in range(150))
        )
    assert counts == sorted(counts)
    assert counts[0] == 0


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def test_describe_has_a_line_per_level():
    lines = L.describe()
    assert len(lines) == 4
    assert all(L.LEVEL_TOKENS[k] in lines[k] for k in L.LEVELS)


def test_the_rate_table_is_markdown_with_every_level_in_it():
    table = L.rate_table()
    assert table.startswith("| Level |")
    for token in L.LEVEL_TOKENS:
        assert f"`{token}`" in table
