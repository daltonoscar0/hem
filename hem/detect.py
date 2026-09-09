"""Recover what was inserted, from the clean text and the disfluent text alone.

The injector knows where it put things; the model does not tell you. Every
number in the evaluation is a count of insertions by type and position, so
without a detector there is nothing to count and no way to compare a model's
output against the injector's or against Switchboard's.

The detector aligns the spoken form of the clean text against the disfluent
text, reads the insertions off the alignment, and classifies each one as a
filled pause, a repetition, a substitution or a restart. It emits the same
``InjectionResult`` the injector does, so the evaluation, the trace output and
the surprisal analysis all read model output and rule output through one path.

Because everything rests on it, the detector is checked against Switchboard's
own annotation rather than trusted: ``python -m hem.detect --check`` runs it
over the parsed corpus, where the gold spans are known, and reports how often
it agrees. That number belongs in the README next to anything the detector
measured.

It cannot be exact, and the ways it is wrong are worth stating up front:

* A repetition is only visible as one when the repeated words are identical.
  "the report, the report" is caught; "the report, that report" is read as a
  substitution, which is arguably right anyway.
* A restart is a run of function words at a clause onset. A speaker who
  abandons a content phrase and starts again is read as a substitution.
* Anything left over after the other three rules is called a substitution,
  because a run of inserted material that is not a hesitation and not a repeat
  is the speaker having said something else. That makes substitution the
  bucket that absorbs the detector's uncertainty, so it is the count to trust
  least.
* A *delayed* repair, where the corrected word arrives several positions after
  the material it belongs in front of, breaks the alignment's assumption that
  the script is visited in order. The aligner sees the word leave its place and
  turn up elsewhere and reports a deletion. On the injector's own heaviest
  output, where no word is ever really deleted, that misfires on under 5% of
  sentences.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hem.injector import (
    FALSE_START_FRAGMENTS,
    FILLER_PHRASES,
    FILLER_VOCAB,
    INTERREGNA,
    STOPWORDS,
    TYPES,
    Annotation,
    InjectionResult,
    Span,
    canonical_clean_tokens,
    spoken_form,
)

#: Filler phrases longest first, so "you know" is matched before "you".
_FILLER_BY_LENGTH: Tuple[Tuple[str, ...], ...] = tuple(
    sorted(FILLER_PHRASES, key=len, reverse=True)
)

#: Editing phrases that announce a repair. A run containing one of these is a
#: substitution whatever else it looks like.
_EDITING: Tuple[Tuple[str, ...], ...] = tuple(
    sorted(INTERREGNA, key=len, reverse=True)
)

#: Words a restart can be built from. A restart is an abandoned clause opener,
#: so it is function words and auxiliaries, never a content phrase.
_OPENER_WORDS = frozenset(
    STOPWORDS
    | {w for frag in FALSE_START_FRAGMENTS for w in frag}
    | set("what when where why how which who am are is was were let want need going".split())
)

#: Longest run of inserted function words still read as a restart rather than a
#: substitution. Switchboard's restarts run 1 to 4 words; beyond that the
#: speaker abandoned something with content in it.
MAX_RESTART = 4


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def _opcodes(clean: Sequence[str], out: Sequence[str]):
    """Alignment opcodes with ``replace`` broken into a delete and an insert.

    A replace is the model having put different words where the clean text had
    some. Keeping it whole would hide the inserted side from the classifier, so
    it is split, and the deleted side is reported separately as lost material.
    """
    matcher = SequenceMatcher(a=list(clean), b=list(out), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            yield ("delete", i1, i2, j1, j1)
            yield ("insert", i2, i2, j1, j2)
        else:
            yield (tag, i1, i2, j1, j2)


# --------------------------------------------------------------------------
# classifying one inserted run
# --------------------------------------------------------------------------


def _strip_fillers(run: Sequence[str]) -> Tuple[List[Tuple[int, int]], int, int]:
    """Peel filler phrases off both ends of an inserted run.

    Returns the (start, end) offsets of the phrases found, and the offsets of
    whatever is left in the middle. A run like "uh john no wait" is a filled
    pause followed by a substitution, and scoring it as one thing would put the
    filler in the wrong bucket and lose it from the rate.
    """
    lo, hi = 0, len(run)
    found: List[Tuple[int, int]] = []
    changed = True
    while changed and lo < hi:
        changed = False
        for phrase in _FILLER_BY_LENGTH:
            n = len(phrase)
            if hi - lo >= n and tuple(run[lo : lo + n]) == phrase:
                found.append((lo, lo + n))
                lo += n
                changed = True
                break
        if changed:
            continue
        for phrase in _FILLER_BY_LENGTH:
            n = len(phrase)
            if hi - lo >= n and tuple(run[hi - n : hi]) == phrase:
                found.append((hi - n, hi))
                hi -= n
                changed = True
                break
    return sorted(found), lo, hi


def _is_repetition(
    out: Sequence[str], start: int, end: int
) -> bool:
    """Whether the inserted run repeats the words either side of it."""
    n = end - start
    if n == 0:
        return False
    ahead = out[end : end + n]
    if len(ahead) == n and list(ahead) == list(out[start:end]):
        return True
    behind = out[start - n : start]
    return start - n >= 0 and len(behind) == n and list(behind) == list(out[start:end])


def _contains_editing(run: Sequence[str]) -> Optional[Tuple[int, int]]:
    for phrase in _EDITING:
        n = len(phrase)
        for i in range(len(run) - n + 1):
            if tuple(run[i : i + n]) == phrase:
                return (i, i + n)
    return None


def _is_restart(
    out: Sequence[str], start: int, end: int, clean_index: int
) -> bool:
    """A run of function words abandoned at a clause onset."""
    n = end - start
    if not 1 <= n <= MAX_RESTART:
        return False
    if not all(w in _OPENER_WORDS for w in out[start:end]):
        return False
    # at the very start of the utterance, or resuming the clean text from its
    # beginning after something was already said
    return start == 0 or clean_index == 0


def classify_run(
    out: Sequence[str], start: int, end: int, clean_index: int
) -> List[Tuple[str, int, int]]:
    """Split one inserted run into ``(type, start, end)`` pieces."""
    run = list(out[start:end])
    if not run:
        return []

    pieces: List[Tuple[str, int, int]] = []
    filler_spans, lo, hi = _strip_fillers(run)
    for a, b in filler_spans:
        pieces.append(("filler", start + a, start + b))

    if lo >= hi:
        return sorted(pieces, key=lambda p: p[1])

    core_start, core_end = start + lo, start + hi

    editing = _contains_editing(run[lo:hi])
    if editing is not None:
        # the editing phrase is the interregnum of a substitution; the whole
        # core is one repair, reparandum and editing term together
        pieces.append(("repair", core_start, core_end))
    elif _is_repetition(out, core_start, core_end):
        pieces.append(("repetition", core_start, core_end))
    elif _is_restart(out, core_start, core_end, clean_index):
        pieces.append(("false_start", core_start, core_end))
    else:
        pieces.append(("repair", core_start, core_end))
    return sorted(pieces, key=lambda p: p[1])


# --------------------------------------------------------------------------
# the detector
# --------------------------------------------------------------------------


def detect(clean_text: str, disfluent_text: str) -> Tuple[InjectionResult, dict]:
    """Recover the insertions that turn ``clean_text`` into ``disfluent_text``.

    Returns the annotated result plus a small report of what did not line up:
    clean words the disfluent side dropped, which is the fidelity failure mode
    that matters most, since a generator that quietly deletes half the script
    is worse than one that inserts nothing.
    """
    clean_tokens = canonical_clean_tokens(clean_text)
    clean_spoken = [spoken_form(t) for t in clean_tokens]
    out = [w for w in disfluent_text.split() if w]

    alignment: List[Optional[int]] = [None] * len(out)
    inserted: List[Tuple[int, int, int]] = []  # (start, end, clean index here)
    dropped: List[str] = []

    for tag, i1, i2, j1, j2 in _opcodes(clean_spoken, out):
        if tag == "equal":
            for k in range(j2 - j1):
                alignment[j1 + k] = i1 + k
        elif tag == "insert":
            inserted.append((j1, j2, i1))
        elif tag == "delete":
            dropped.extend(clean_spoken[i1:i2])

    annotations: List[Annotation] = []
    for start, end, clean_index in inserted:
        for kind, a, b in classify_run(out, start, end, clean_index):
            span = Span(a, b)
            if kind == "filler":
                annotations.append(Annotation(type="filler", interregnum=span))
            elif kind == "repetition":
                annotations.append(
                    Annotation(
                        type="repetition",
                        reparandum=span,
                        repair=Span(b, b + (b - a)),
                    )
                )
            elif kind == "false_start":
                annotations.append(
                    Annotation(
                        type="false_start",
                        reparandum=span,
                        repair=Span(b, min(b + 3, len(out))),
                    )
                )
            else:
                editing = _contains_editing(out[a:b])
                if editing is None:
                    annotations.append(
                        Annotation(
                            type="repair",
                            reparandum=span,
                            repair=Span(b, min(b + (b - a), len(out))),
                        )
                    )
                else:
                    e0, e1 = a + editing[0], a + editing[1]
                    annotations.append(
                        Annotation(
                            type="repair",
                            reparandum=Span(a, e0) if e0 > a else None,
                            interregnum=Span(e0, e1),
                            repair=Span(b, min(b + max(1, e0 - a), len(out))),
                        )
                    )

    annotations.sort(key=lambda x: x.boundary_index() or 0)
    result = InjectionResult(
        disfluent_text=" ".join(out),
        clean_text=" ".join(clean_tokens),
        disfluent_tokens=out,
        clean_tokens=clean_tokens,
        alignment=alignment,
        annotations=annotations,
    )
    report = {
        "clean_words": len(clean_tokens),
        "spoken_words": len(out),
        "dropped": dropped,
        "n_dropped": len(dropped),
        "kept": len(clean_tokens) - len(dropped),
    }
    return result, report


def trace(clean_text: str, disfluent_text: str) -> List[dict]:
    """One row per detected insertion, for ``hem --trace``."""
    result, _ = detect(clean_text, disfluent_text)
    rows = []
    for ann in result.annotations:
        spans = [s for s in (ann.reparandum, ann.interregnum) if s is not None]
        if not spans:
            continue
        start = min(s.start for s in spans)
        end = max(s.end for s in spans)
        before = next(
            (result.alignment[i] for i in range(end, len(result.alignment))
             if result.alignment[i] is not None),
            len(result.clean_tokens),
        )
        rows.append(
            {
                "type": ann.type,
                "position": start,
                "before_clean_word": before,
                "words": " ".join(result.disfluent_tokens[start:end]),
            }
        )
    return rows


# --------------------------------------------------------------------------
# checking the detector against gold annotation
# --------------------------------------------------------------------------


def gold_types(row: dict) -> Counter:
    counts: Counter = Counter()
    for ann in row.get("annotations", []):
        if ann["type"] in TYPES:
            counts[ann["type"]] += 1
    return counts


def check(rows: Sequence[dict]) -> dict:
    """Run the detector over annotated utterances and score it against them.

    Two questions. Does it find the right *number* of disfluencies, which is
    what the rate metric depends on, and does it give them the right *type*,
    which is what the type-mix metric depends on. Both are scored by matching
    detected boundary positions to gold ones, so a detection only counts if it
    landed in the right place.
    """
    matched = 0
    detected_total = 0
    gold_total = 0
    confusion: Counter = Counter()
    per_type_gold: Counter = Counter()
    per_type_detected: Counter = Counter()
    dropped_rows = 0

    for row in rows:
        gold = [a for a in row.get("annotations", []) if a["type"] in TYPES]
        result, report = detect(row["clean"], row["disfluent"])
        if report["n_dropped"]:
            dropped_rows += 1
        gold_total += len(gold)
        detected_total += len(result.annotations)
        for a in gold:
            per_type_gold[a["type"]] += 1
        for a in result.annotations:
            per_type_detected[a.type] += 1

        # match on the first token of the inserted material, which is the one
        # place both sides agree on regardless of how the spans are drawn
        def gold_start(a: dict) -> Optional[int]:
            spans = [a.get("reparandum"), a.get("interregnum")]
            starts = [s["start"] for s in spans if s is not None]
            return min(starts) if starts else None

        pool = []
        for a in gold:
            s = gold_start(a)
            if s is not None:
                pool.append((s, a["type"]))
        used = set()
        for ann in result.annotations:
            spans = [ann.reparandum, ann.interregnum]
            starts = [s.start for s in spans if s is not None]
            if not starts:
                continue
            here = min(starts)
            best = None
            for idx, (s, t) in enumerate(pool):
                if idx in used or abs(s - here) > 1:
                    continue
                if best is None or abs(s - here) < abs(pool[best][0] - here):
                    best = idx
            if best is None:
                confusion[("none", ann.type)] += 1
                continue
            used.add(best)
            matched += 1
            confusion[(pool[best][1], ann.type)] += 1
        for idx, (_, t) in enumerate(pool):
            if idx not in used:
                confusion[(t, "none")] += 1

    correct_type = sum(v for (g, d), v in confusion.items() if g == d and g != "none")
    return {
        "utterances": len(rows),
        "gold_events": gold_total,
        "detected_events": detected_total,
        "matched": matched,
        "detection_recall": matched / gold_total if gold_total else 0.0,
        "detection_precision": matched / detected_total if detected_total else 0.0,
        "type_accuracy_given_match": correct_type / matched if matched else 0.0,
        "type_accuracy_overall": correct_type / gold_total if gold_total else 0.0,
        "rows_with_dropped_words": dropped_rows,
        "gold_by_type": dict(per_type_gold),
        "detected_by_type": dict(per_type_detected),
        "confusion": {f"{g}->{d}": v for (g, d), v in sorted(confusion.items())},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="The insertion detector, and its check")
    ap.add_argument("--check", action="store_true", help="score against gold spans")
    ap.add_argument("--data", type=Path, default=Path("data/swda_eval.jsonl"))
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("outputs/detector_check.json"))
    args = ap.parse_args(argv)

    if not args.check:
        ap.error("nothing to do; pass --check")

    with args.data.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"checking on {len(rows)} annotated utterances from {args.data}",
          file=sys.stderr)

    report = check(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"gold events        {report['gold_events']}")
    print(f"detected events    {report['detected_events']}")
    print(f"detection recall   {report['detection_recall']:.3f}")
    print(f"detection precision{report['detection_precision']:.3f}")
    print(f"type accuracy      {report['type_accuracy_given_match']:.3f} of matched, "
          f"{report['type_accuracy_overall']:.3f} of gold")
    print("\nby type: gold vs detected")
    for t in TYPES:
        print(f"  {t:<12s} {report['gold_by_type'].get(t, 0):>6d} "
              f"{report['detected_by_type'].get(t, 0):>6d}")
    print("\nconfusion (gold -> detected)")
    for key, value in sorted(report["confusion"].items(), key=lambda kv: -kv[1]):
        print(f"  {key:<28s} {value}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
