"""Where the dial's four settings come from.

Nothing in this file is chosen. Every rate, every type proportion and every
shape parameter is counted off the Switchboard parse and written to
``hem/levels.json``, which is what :mod:`hem.levels` loads and what the README
table reports. Rerunning

    python -m hem.rates --data data/swda_all.jsonl

regenerates the file from the corpus.

Rates are **events per 100 clean words**, where a clean word is a word that
survives into the written form. Counting against the clean side rather than the
spoken side is what makes the number a property of the text being spoken rather
than of the disfluency itself: "the the report" has one repetition per three
clean words, not per four spoken ones, and a heavier dial should not shrink its
own denominator.

The four buckets are quantiles of that rate:

* ``d0``  no disfluency at all. 38% of the pool.
* ``d1``  below the median of the utterances that carry anything.
* ``d2``  from the median to the 90th percentile.
* ``d3``  at or above the 90th percentile.

Both cuts are percentiles of the corpus rather than round numbers, so "natural"
and "heavy" mean what Switchboard's middle and heaviest tenth actually sound
like. Bucketing on the total rate, rather than on which types are present, is
what makes the dial monotone: cutting ``d1`` as "filled pauses only" and ``d2``
as "has a repetition or a substitution" was tried first and gave a ``d1`` with a
*higher* filler rate than ``d2``, because the first definition guarantees a
filler and the second does not. Turning that dial up would have removed
hesitations, which is not a dial.

Only utterances of at least ``MIN_CLEAN_WORDS`` clean words are used, for the
measurement and for training. Switchboard's short slash units are mostly
backchannels ("uh-huh", "yeah that's right"), which is not the register Hem is
aimed at, and a per-word rate computed over five words is quantised so coarsely
that one filled pause already reads as 20 events per 100 words. That cut halves
the corpus and leaves 35,208 utterances.

What the corpus disagrees with: ``d1`` was specified as filled pauses only, and
the lightest half of disfluent Switchboard is 58% filled pauses, not 100%. It
still carries about one repetition per 70 words. The measured mix is what goes
in the table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from hem.injector import TYPES

LEVELS_PATH = Path(__file__).with_name("levels.json")

#: Structural disfluencies: the ones that involve going back over something
#: already said, as opposed to just hesitating.
STRUCTURAL = ("repetition", "repair", "false_start")

#: Percentiles of the disfluent utterances that d1/d2 and d2/d3 are cut at.
LIGHT_PERCENTILE = 50.0
HEAVY_PERCENTILE = 90.0

#: Shortest utterance a per-word rate is computed over. See the module
#: docstring: below this, the rate is quantised too coarsely to mean anything
#: and the register is backchannels rather than sentences.
MIN_CLEAN_WORDS = 10


# --------------------------------------------------------------------------
# per-utterance measurement
# --------------------------------------------------------------------------


def measure(row: dict) -> dict:
    """Counts and rates for one annotated utterance."""
    n_clean = len(row.get("clean_tokens") or row["clean"].split())
    counts = Counter()
    inserted = 0
    for ann in row.get("annotations", []):
        if ann["type"] in TYPES:
            counts[ann["type"]] += 1
        for key in ("reparandum", "interregnum"):
            span = ann.get(key)
            if span is not None:
                inserted += span["end"] - span["start"]
    total = sum(counts.values())
    scale = 100.0 / n_clean if n_clean else 0.0
    return {
        "n_clean": n_clean,
        "counts": counts,
        "total": total,
        "inserted": inserted,
        "rate": total * scale,
        "structural": sum(counts[t] for t in STRUCTURAL),
    }


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, so the cut does not depend on numpy."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[int(pos)])
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


def bucket_of(m: dict, light_cut: float, heavy_cut: float) -> int:
    """Which dial level this utterance is an example of."""
    if m["total"] == 0:
        return 0
    if m["rate"] >= heavy_cut:
        return 3
    return 2 if m["rate"] >= light_cut else 1


def eligible(row: dict) -> bool:
    """Whether this utterance is long enough to carry a meaningful rate."""
    n = len(row.get("clean_tokens") or row["clean"].split())
    return n >= MIN_CLEAN_WORDS


def level_of(row: dict, spec: dict) -> int:
    """Label one annotated utterance with its dial level."""
    return bucket_of(measure(row), spec["light_cut_per_100w"], spec["heavy_cut_per_100w"])


# --------------------------------------------------------------------------
# corpus-level shape parameters
# --------------------------------------------------------------------------


def shape_parameters(rows: Sequence[dict]) -> dict:
    """What a disfluency of each type looks like once its position is fixed.

    These are the parameters the injector needs beyond the four rates: how
    often a repair announces itself with an editing phrase, how long a
    reparandum runs, and which words get used as filled pauses.
    """
    repairs = 0
    repairs_with_interregnum = 0
    repair_lengths: Counter = Counter()
    repetition_lengths: Counter = Counter()
    filler_words: Counter = Counter()

    for row in rows:
        tokens = row.get("disfluent_tokens") or []
        for ann in row.get("annotations", []):
            rep = ann.get("reparandum")
            if ann["type"] == "repair":
                repairs += 1
                if ann.get("interregnum") is not None:
                    repairs_with_interregnum += 1
                if rep is not None:
                    repair_lengths[rep["end"] - rep["start"]] += 1
            elif ann["type"] == "repetition" and rep is not None:
                repetition_lengths[rep["end"] - rep["start"]] += 1
            elif ann["type"] == "filler":
                span = ann.get("interregnum")
                if span is None or not tokens:
                    continue
                phrase = " ".join(tokens[span["start"] : span["end"]])
                if phrase:
                    filler_words[phrase] += 1

    def share(counter: Counter, predicate) -> float:
        total = sum(counter.values())
        if not total:
            return 0.0
        return sum(v for k, v in counter.items() if predicate(k)) / total

    top = filler_words.most_common(25)
    filler_total = sum(filler_words.values())
    return {
        "repairs": repairs,
        "interregnum_rate": repairs_with_interregnum / repairs if repairs else 0.0,
        "span_repair_rate": share(repair_lengths, lambda k: k >= 2),
        "mean_repair_span": (
            sum(k * v for k, v in repair_lengths.items()) / sum(repair_lengths.values())
            if repair_lengths
            else 0.0
        ),
        "repetition_bigram_rate": share(repetition_lengths, lambda k: k >= 2),
        "filler_phrases": [
            {"phrase": p, "count": c, "share": c / filler_total} for p, c in top
        ],
    }


# --------------------------------------------------------------------------
# building the level table
# --------------------------------------------------------------------------


def build_levels(rows: Sequence[dict]) -> dict:
    """Measure Switchboard and turn it into the four dial settings."""
    pool = [r for r in rows if eligible(r)]
    measured = [measure(r) for r in pool]
    disfluent = [m["rate"] for m in measured if m["total"] > 0]
    light_cut = percentile(disfluent, LIGHT_PERCENTILE)
    heavy_cut = percentile(disfluent, HEAVY_PERCENTILE)

    buckets: Dict[int, List[dict]] = {0: [], 1: [], 2: [], 3: []}
    for m in measured:
        buckets[bucket_of(m, light_cut, heavy_cut)].append(m)

    levels = {}
    for k, members in buckets.items():
        words = sum(m["n_clean"] for m in members)
        counts = Counter()
        for m in members:
            counts.update(m["counts"])
        events = sum(counts.values())
        scale = 100.0 / words if words else 0.0
        levels[str(k)] = {
            "utterances": len(members),
            "share_of_corpus": len(members) / len(measured) if measured else 0.0,
            "clean_words": words,
            "events": events,
            # events per 100 clean words, per type and in total
            "rates": {t: counts[t] * scale for t in TYPES},
            "total_rate": events * scale,
            # what fraction of this bucket's events are of each type
            "mix": {t: (counts[t] / events if events else 0.0) for t in TYPES},
            "inserted_per_100w": sum(m["inserted"] for m in members) * scale,
            "mean_words_per_event": words / events if events else float("inf"),
        }

    corpus_words = sum(m["n_clean"] for m in measured)
    corpus_counts = Counter()
    for m in measured:
        corpus_counts.update(m["counts"])
    corpus_events = sum(corpus_counts.values())
    corpus_scale = 100.0 / corpus_words if corpus_words else 0.0

    return {
        "source": "Switchboard Dialog Act corpus, parsed by hem/swda.py",
        "utterances": len(measured),
        "utterances_before_length_filter": len(rows),
        "min_clean_words": MIN_CLEAN_WORDS,
        "clean_words": corpus_words,
        "light_percentile": LIGHT_PERCENTILE,
        "heavy_percentile": HEAVY_PERCENTILE,
        "light_cut_per_100w": light_cut,
        "heavy_cut_per_100w": heavy_cut,
        "corpus": {
            "rates": {t: corpus_counts[t] * corpus_scale for t in TYPES},
            "total_rate": corpus_events * corpus_scale,
            "mix": {
                t: (corpus_counts[t] / corpus_events if corpus_events else 0.0)
                for t in TYPES
            },
        },
        "levels": levels,
        "shape": shape_parameters(pool),
    }


# --------------------------------------------------------------------------
# unigram frequencies, for the placement analysis
# --------------------------------------------------------------------------


def unigram_log_frequencies(rows: Sequence[dict], floor: int = 1) -> Dict[str, float]:
    """Log10 relative frequency of every word in the corpus's clean side.

    The placement analysis asks whether fillers land in front of *low-frequency*
    content words, which needs a frequency estimate. Taking it from the same
    corpus keeps the repo self-contained and keeps the estimate in the right
    register: conversational word frequencies are not written-English ones.
    """
    counts: Counter = Counter()
    for row in rows:
        for tok in row.get("clean_tokens") or row["clean"].split():
            word = tok.strip(" \t\n.,;:!?\"'()[]{}").lower()
            if word:
                counts[word] += 1
    total = sum(counts.values()) or 1
    return {w: math.log10(max(c, floor) / total) for w, c in counts.items()}


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def rate_table(spec: dict) -> str:
    """The markdown table the README carries."""
    names = {0: "`<d0>` clean", 1: "`<d1>` light", 2: "`<d2>` natural", 3: "`<d3>` heavy"}
    head = (
        "| Level | Switchboard utterances | Filled pause | Repetition | "
        "Substitution | Restart | All types | One event per |"
    )
    lines = [head, "|" + "|".join("---" for _ in range(8)) + "|"]
    for k in (0, 1, 2, 3):
        entry = spec["levels"][str(k)]
        r = entry["rates"]
        per = entry["mean_words_per_event"]
        lines.append(
            f"| {names[k]} | {entry['utterances']:,} "
            f"({entry['share_of_corpus']:.0%}) | "
            f"{r['filler']:.2f} | {r['repetition']:.2f} | {r['repair']:.2f} | "
            f"{r['false_start']:.2f} | {entry['total_rate']:.2f} | "
            + ("n/a" if math.isinf(per) else f"{per:.0f} words")
            + " |"
        )
    corpus = spec["corpus"]
    lines.append(
        f"| all four | {spec['utterances']:,} (100%) | "
        f"{corpus['rates']['filler']:.2f} | {corpus['rates']['repetition']:.2f} | "
        f"{corpus['rates']['repair']:.2f} | {corpus['rates']['false_start']:.2f} | "
        f"{corpus['total_rate']:.2f} | "
        f"{100 / corpus['total_rate']:.0f} words |"
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Measure Switchboard, write the dial")
    ap.add_argument("--data", type=Path, default=Path("data/swda_all.jsonl"))
    ap.add_argument("--out", type=Path, default=LEVELS_PATH)
    ap.add_argument("--frequencies", type=Path, default=Path("data/unigrams.json"))
    args = ap.parse_args(argv)

    with args.data.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    print(f"read {len(rows)} utterances from {args.data}", file=sys.stderr)

    spec = build_levels(rows)
    args.out.write_text(json.dumps(spec, indent=2) + "\n")
    print(f"wrote {args.out}")

    # Frequencies come from the whole parse, not the length-filtered pool: the
    # more text the estimate rests on the better, and nothing about a word's
    # frequency depends on how long the utterances it appeared in were.
    args.frequencies.parent.mkdir(parents=True, exist_ok=True)
    freqs = unigram_log_frequencies(rows)
    args.frequencies.write_text(json.dumps(freqs))
    print(f"wrote {args.frequencies} ({len(freqs)} word types)")

    print()
    print(rate_table(spec))
    print()
    print(
        f"pool: {spec['utterances']:,} of {spec['utterances_before_length_filter']:,} "
        f"utterances with at least {MIN_CLEAN_WORDS} clean words"
    )
    print(
        f"cuts, in events per 100 clean words over the disfluent utterances: "
        f"p{LIGHT_PERCENTILE:g}={spec['light_cut_per_100w']:.2f}  "
        f"p{HEAVY_PERCENTILE:g}={spec['heavy_cut_per_100w']:.2f}"
    )
    print("\ntype mix per level (share of that level's events)")
    for k in ("0", "1", "2", "3"):
        mix = spec["levels"][k]["mix"]
        print(f"  d{k}: " + "  ".join(f"{t}={mix[t]:.3f}" for t in TYPES))
    print("  corpus: " + "  ".join(f"{t}={spec['corpus']['mix'][t]:.3f}" for t in TYPES))

    shape = spec["shape"]
    print("\nshape parameters")
    print(f"  repairs with an editing phrase: {shape['interregnum_rate']:.3f}")
    print(f"  repairs with a multi-token reparandum: {shape['span_repair_rate']:.3f}")
    print(f"  mean reparandum length: {shape['mean_repair_span']:.2f} tokens")
    print(f"  repetitions of two or more words: {shape['repetition_bigram_rate']:.3f}")
    print("\n  most common filled pauses")
    for entry in shape["filler_phrases"][:12]:
        print(f"    {entry['phrase']:<12s} {entry['count']:>6d}  {entry['share']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
