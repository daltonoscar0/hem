"""Tests for the scoring functions.

No model is loaded. Every function here takes text in and gives numbers out, and
these are the numbers the README quotes, so the arithmetic is worth pinning down
separately from the systems that feed it.
"""

import math

import pytest

from hem import levels as L
from hem.evaluate import (
    annotated_rows,
    baseline_placement,
    eval_sentences,
    gold_records,
    insertion_records,
    is_content,
    kl_divergence,
    placement_stats,
    rate_error,
    recovery,
    summarise_run,
    words_only,
)
from hem.injector import TYPES

CLEAN = "Send the quarterly report to Sarah before Friday."


# --------------------------------------------------------------------------
# insertion records
# --------------------------------------------------------------------------


def test_an_insertion_records_the_clean_word_it_sits_before():
    rows, _ = insertion_records(CLEAN, "send the uh quarterly report to sarah before friday")
    assert len(rows) == 1
    assert rows[0]["type"] == "filler"
    assert rows[0]["clean_index"] == 2
    assert rows[0]["next_word"] == "quarterly"


def test_an_insertion_at_the_front_is_at_a_clause_onset():
    rows, _ = insertion_records(CLEAN, "uh send the quarterly report to sarah before friday")
    assert rows[0]["at_onset"] is True
    assert rows[0]["clean_index"] == 0


def test_an_insertion_after_a_comma_is_at_a_clause_onset():
    clean = "Send the report, and copy Michael on it."
    rows, _ = insertion_records(clean, "send the report uh and copy michael on it")
    assert rows[0]["at_onset"] is True


def test_an_insertion_mid_phrase_is_not_at_a_clause_onset():
    rows, _ = insertion_records(CLEAN, "send the uh quarterly report to sarah before friday")
    assert rows[0]["at_onset"] is False


def test_relative_position_is_a_fraction_of_the_sentence():
    rows, _ = insertion_records(CLEAN, "send the quarterly report uh to sarah before friday")
    assert 0.0 < rows[0]["relative"] < 1.0
    assert rows[0]["relative"] == pytest.approx(4 / 8)


def test_content_word_detection():
    assert is_content("quarterly")
    assert not is_content("the")
    assert not is_content("to")


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def test_rates_are_per_hundred_clean_words():
    texts = [" ".join(["word"] * 50)] * 2  # 100 clean words in total
    records = [[{"type": "filler"}] * 3, [{"type": "repair"}] * 2]
    run = summarise_run(texts, records)
    assert run["clean_words"] == 100
    assert run["total_rate"] == pytest.approx(5.0)
    assert run["rates"]["filler"] == pytest.approx(3.0)
    assert run["rates"]["repair"] == pytest.approx(2.0)


def test_the_mix_sums_to_one():
    run = summarise_run([" ".join(["w"] * 20)],
                        [[{"type": "filler"}, {"type": "repair"}]])
    assert sum(run["mix"].values()) == pytest.approx(1.0)


def test_an_empty_run_does_not_divide_by_zero():
    run = summarise_run([], [])
    assert run["total_rate"] == 0.0
    assert all(v == 0.0 for v in run["mix"].values())


def test_rate_error_is_measured_against_the_targets():
    runs = {
        ("X", k): {"total_rate": L.targets(k)["total"], "rates": L.targets(k)}
        for k in L.LEVELS
    }
    err = rate_error(runs, "X")
    assert err["mean_abs_error"] == pytest.approx(0.0)
    assert err["mean_rel_error"] == pytest.approx(0.0)


def test_rate_error_grows_when_the_dial_misses():
    runs = {
        ("X", k): {"total_rate": L.targets(k)["total"] * 2, "rates": L.targets(k)}
        for k in L.LEVELS
    }
    assert rate_error(runs, "X")["mean_rel_error"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# KL
# --------------------------------------------------------------------------


def test_kl_of_a_distribution_from_itself_is_about_zero():
    p = {"filler": 0.5, "repetition": 0.25, "repair": 0.2, "false_start": 0.05}
    assert kl_divergence(p, p) < 0.02


def test_kl_grows_as_the_distributions_separate():
    p = {"filler": 1.0, "repetition": 0.0, "repair": 0.0, "false_start": 0.0}
    near = {"filler": 0.8, "repetition": 0.2, "repair": 0.0, "false_start": 0.0}
    far = {"filler": 0.1, "repetition": 0.9, "repair": 0.0, "false_start": 0.0}
    assert kl_divergence(p, near) < kl_divergence(p, far)


def test_kl_is_finite_when_the_reference_has_an_empty_cell():
    p = {"filler": 0.5, "repetition": 0.5, "repair": 0.0, "false_start": 0.0}
    q = {"filler": 1.0, "repetition": 0.0, "repair": 0.0, "false_start": 0.0}
    assert math.isfinite(kl_divergence(p, q))


def test_kl_is_zero_for_an_empty_system():
    assert kl_divergence({t: 0.0 for t in TYPES}, {t: 0.25 for t in TYPES}) == 0.0


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


FREQS = {"the": -1.0, "report": -3.0, "quarterly": -5.0, "sarah": -4.0}


def test_placement_splits_onset_from_mid_clause():
    records = [
        {"type": "filler", "at_onset": True, "next_is_content": False,
         "next_word": "the", "relative": 0.0},
        {"type": "filler", "at_onset": False, "next_is_content": True,
         "next_word": "quarterly", "relative": 0.3},
    ]
    stats = placement_stats(records, FREQS)
    assert stats["n"] == 2
    assert stats["at_clause_onset"] == pytest.approx(0.5)
    assert stats["mean_next_log_freq"] == pytest.approx(-3.0)
    assert stats["n_mid_clause"] == 1
    assert stats["mid_clause_next_log_freq"] == pytest.approx(-5.0)


def test_placement_only_counts_the_type_it_was_asked_for():
    records = [
        {"type": "filler", "at_onset": True, "next_is_content": False,
         "next_word": "the", "relative": 0.0},
        {"type": "repair", "at_onset": True, "next_is_content": False,
         "next_word": "the", "relative": 0.0},
    ]
    assert placement_stats(records, FREQS, only="filler")["n"] == 1
    assert placement_stats(records, FREQS, only=None)["n"] == 2


def test_placement_of_nothing_reports_zero():
    assert placement_stats([], FREQS)["n"] == 0


def test_the_baseline_counts_every_word_in_the_text():
    base = baseline_placement(["Send the report, and copy it."], FREQS)
    assert base["n"] == 6


def test_the_baseline_finds_the_clause_onsets():
    base = baseline_placement(["Send the report, and copy it."], FREQS)
    # the first word, and the word after the comma
    assert base["at_clause_onset"] == pytest.approx(2 / 6)


def test_the_baseline_excludes_onsets_from_the_mid_clause_figure():
    base = baseline_placement(["Send the report, and copy it."], FREQS)
    assert base["n_mid_clause"] == 4


# --------------------------------------------------------------------------
# round trip
# --------------------------------------------------------------------------


def test_words_only_strips_case_and_punctuation():
    assert words_only("Send the Report, please!") == "send the report please"


def test_words_only_keeps_apostrophes():
    assert words_only("Don't send it.") == "don't send it"


def test_a_perfect_recovery_scores_one():
    out = recovery(["Send the report."], ["Send the report."])
    assert out["exact_match"] == 1.0
    assert out["words_only_match"] == 1.0
    assert out["wer_words_only"] == 0.0


def test_casing_alone_costs_exact_match_but_not_the_words_only_score():
    out = recovery(["Send the report."], ["send the report"])
    assert out["exact_match"] == 0.0
    assert out["words_only_match"] == 1.0


def test_a_missing_word_costs_both():
    out = recovery(["Send the quarterly report."], ["Send the report."])
    assert out["words_only_match"] == 0.0
    assert out["wer_words_only"] > 0.0


def test_the_final_stop_is_optional_for_exact_match():
    assert recovery(["Send the report."], ["Send the report"])["exact_match"] == 1.0


# --------------------------------------------------------------------------
# dumping for the surprisal analysis
# --------------------------------------------------------------------------


def test_annotated_rows_carry_the_spans_the_surprisal_pass_needs():
    rows = annotated_rows(
        [CLEAN], ["send the quarterly report to john no wait sarah before friday"]
    )
    assert len(rows) == 1
    assert rows[0]["annotations"]
    assert rows[0]["annotations"][0]["type"] in TYPES
    assert len(rows[0]["alignment"]) == len(rows[0]["disfluent_tokens"])


def test_annotated_rows_keep_only_known_types():
    rows = annotated_rows([CLEAN], ["send the uh quarterly report to sarah before friday"])
    assert {a["type"] for a in rows[0]["annotations"]} <= set(TYPES)


# --------------------------------------------------------------------------
# gold records
# --------------------------------------------------------------------------


def test_gold_records_read_the_same_fields_the_detector_produces():
    row = annotated_rows(
        [CLEAN], ["send the uh quarterly report to sarah before friday"]
    )[0]
    gold = gold_records(row)
    detected, _ = insertion_records(row["clean"], row["disfluent"])
    assert [g["type"] for g in gold] == [d["type"] for d in detected]
    assert [g["clean_index"] for g in gold] == [d["clean_index"] for d in detected]


# --------------------------------------------------------------------------
# the evaluation set
# --------------------------------------------------------------------------


def test_eval_sentences_deduplicate_and_are_capped(tmp_path):
    import json

    path = tmp_path / "test.jsonl"
    with path.open("w") as fh:
        for text in ["a b c", "a b c", "d e f", "g h i"]:
            fh.write(json.dumps({"clean": text}) + "\n")
    assert len(eval_sentences(path, 0, 13)) == 3
    assert len(eval_sentences(path, 2, 13)) == 2


def test_eval_sentences_are_seeded(tmp_path):
    import json

    path = tmp_path / "test.jsonl"
    with path.open("w") as fh:
        for i in range(20):
            fh.write(json.dumps({"clean": f"sentence number {i}"}) + "\n")
    assert eval_sentences(path, 5, 13) == eval_sentences(path, 5, 13)
