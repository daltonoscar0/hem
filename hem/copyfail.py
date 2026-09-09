"""How much of Hem's "substitution" rate is really a failure to copy.

The qualitative table makes the problem obvious. Given

    The match consisted of former professional players, as well as current
    professionals such as Leroy Lita, Nicky Shorey, Aaron McLean, ...

Hem returns *leroy lisa*, *scott ferdinand*, *paul meerson*. Those are not
self-repairs. They are a 77M-parameter model failing to copy a name it has
never seen, and the detector cannot tell the difference: an inserted run that is
not a filled pause and not a repeat gets called a substitution, which is exactly
what a mangled name looks like.

That matters for two numbers the evaluation reports. Hem's substitution rate is
roughly double its target, and its type mix leans much further towards
substitution than Switchboard does. If the excess is copy failure rather than
self-repair, both readings are wrong.

The test splits the evaluation sentences on whether they contain a word
Switchboard never says. Switchboard is ten million words of conversation, so a
word absent from it is a rare proper noun or a technical term, which is exactly
the kind of word a small model cannot copy. If the substitution excess is real
self-repair it should not care about that split; if it is copy failure it should
almost all be on one side.

Usage::

    python -m hem.evaluate        # writes outputs/generated/
    python -m hem.copyfail        # reads it, no model needed
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hem.build_data import read_jsonl
from hem.detect import detect
from hem.evaluate import ORDER, normalise
from hem.injector import TYPES


def load_vocabulary(path: Path) -> frozenset:
    """Every word Switchboard says, from the unigram table."""
    if not path.exists():
        raise SystemExit(
            f"no {path}; run `python -m hem.rates` to build the unigram table"
        )
    return frozenset(json.loads(path.read_text()))


def has_unseen_word(clean_tokens: Sequence[str], vocabulary: frozenset) -> bool:
    """Whether the script names something Switchboard never mentions."""
    return any(
        normalise(t) and normalise(t) not in vocabulary for t in clean_tokens
    )


def split_scores(rows: Sequence[dict], vocabulary: frozenset) -> Dict[str, dict]:
    """Rates and fidelity for the two halves of the split."""
    out: Dict[str, dict] = {}
    for name, predicate in (
        ("script has a word Switchboard never says", True),
        ("every word is one Switchboard says", False),
    ):
        here = [
            r for r in rows
            if has_unseen_word(r["clean_tokens"], vocabulary) is predicate
        ]
        words = sum(len(r["clean_tokens"]) for r in here)
        counts: Counter = Counter()
        dropped = 0
        for row in here:
            for ann in row["annotations"]:
                if ann["type"] in TYPES:
                    counts[ann["type"]] += 1
            _, report = detect(row["clean"], row["disfluent"])
            dropped += report["n_dropped"]
        scale = 100.0 / words if words else 0.0
        out[name] = {
            "sentences": len(here),
            "clean_words": words,
            "rates": {t: counts[t] * scale for t in TYPES},
            "total_rate": sum(counts.values()) * scale,
            "kept": 1 - dropped / words if words else 0.0,
        }
    return out


def label_of(path: Path) -> Tuple[str, int]:
    stem = path.stem
    name, _, level = stem.rpartition("_d")
    return name.replace("_", " ").capitalize(), int(level)


def table(scores: Dict[Tuple[str, int], Dict[str, dict]], level: int) -> str:
    header = (
        "| System | Script | Sentences | Substitution rate | All types | "
        "Script words kept |"
    )
    lines = [header, "|" + "|".join("---" for _ in range(6)) + "|"]
    keys = sorted(
        (k for k in scores if k[1] == level),
        key=lambda k: ORDER.index(k[0]) if k[0] in ORDER else len(ORDER),
    )
    for key in keys:
        for which, entry in scores[key].items():
            short = "has an unseen word" if which.startswith("script has") else "all seen"
            lines.append(
                f"| {key[0]} | {short} | {entry['sentences']:,} | "
                f"{entry['rates']['repair']:.2f} | {entry['total_rate']:.2f} | "
                f"{entry['kept']:.3f} |"
            )
    lines.append("")
    lines.append(
        "Split on whether the script contains a word absent from ten million "
        "words of Switchboard, which in practice means a rare proper noun. "
        "Rates are events per 100 clean words, at the natural setting."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Copy failure against self-repair")
    ap.add_argument("--generated", type=Path, default=Path("outputs/generated"))
    ap.add_argument("--vocabulary", type=Path, default=Path("data/unigrams.json"))
    ap.add_argument("--out", type=Path, default=Path("outputs/copyfail.md"))
    ap.add_argument("--level", type=int, default=2)
    args = ap.parse_args(argv)

    vocabulary = load_vocabulary(args.vocabulary)
    files = sorted(args.generated.glob(f"*_d{args.level}.jsonl"))
    if not files:
        raise SystemExit(
            f"no d{args.level} generations in {args.generated}; "
            "run `python -m hem.evaluate` first"
        )

    scores: Dict[Tuple[str, int], Dict[str, dict]] = {}
    for path in files:
        name, level = label_of(path)
        rows = read_jsonl(path)
        print(f"  {name} d{level} ({len(rows)})", file=sys.stderr)
        scores[(name, level)] = split_scores(rows, vocabulary)

    text = table(scores, args.level)
    print("\n## Copy failure against self-repair\n")
    print(text)
    args.out.write_text("## Copy failure against self-repair\n\n" + text + "\n")
    print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
