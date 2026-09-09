"""Build ``<dk> clean -> disfluent`` training pairs from both sources.

Two sources, the same two Mend trained on, run backwards:

* **synthetic** pairs, made by injecting the rule-based generator into the clean
  sentence pool at a level drawn at random. Unlimited, and the level is under
  the generator's control.
* **real** pairs, from the Switchboard parse. SwDA gives (disfluent, clean);
  Hem wants (clean, disfluent), so the pair is simply read the other way round.
  These are the only examples in the mix where a human chose where to hesitate.

**Every pair is labelled by its own measured rate, not by what was asked for.**
The injector is stochastic: a sentence generated at the heavy setting may come
out carrying two events rather than five, and if that pair kept its ``<d3>``
label the model would be taught that ``<d3>`` sometimes means "light". Passing
both sources through the same measurement, ``hem.rates.bucket_of``, makes the
control token mean exactly one thing on both sides of the mix: an utterance
whose disfluency rate falls in this bucket.

The cost of that is a skew. Injecting uniformly across the four settings and
relabelling gives fewer ``<d3>`` pairs than were asked for, because a short
sentence at the heavy setting often lands in ``<d2>``, and never the reverse.
``--balance`` corrects it by generating extra pairs at the settings whose
buckets came up short, until each bucket has the share it was meant to.

Usage::

    python -m hem.build_data --smoke
    python -m hem.build_data --train 50000 --val 2000 --test 10000 --seed 13
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from hem import levels as L
from hem.corpus import Pools, SourceSentence, build_pools
from hem.injector import TYPES, InjectionResult, canonical_clean_tokens, inject, spoken_text
from hem.rates import MIN_CLEAN_WORDS, bucket_of, measure

#: The task prefix, after the control token. flan-t5 was instruction tuned, so
#: the input reads as an instruction rather than as a bare sentence.
PREFIX = "say aloud: "


def encode_source(level: int, clean_text: str) -> str:
    """The model's input side: control token, task prefix, clean written text."""
    return f"{L.token(level)} {PREFIX}{clean_text}"


# --------------------------------------------------------------------------
# synthetic pairs
# --------------------------------------------------------------------------


def synthetic_pair(
    sentence: SourceSentence, level: int, seed: int
) -> Optional[dict]:
    """Inject one clean sentence at one dial setting and label what came out."""
    if level == 0:
        clean_tokens = canonical_clean_tokens(sentence.text)
        if len(clean_tokens) < MIN_CLEAN_WORDS:
            return None
        result = InjectionResult(
            disfluent_text=spoken_text(sentence.text),
            clean_text=" ".join(clean_tokens),
            disfluent_tokens=spoken_text(sentence.text).split(),
            clean_tokens=clean_tokens,
            alignment=list(range(len(clean_tokens))),
            annotations=[],
        )
    else:
        result = inject(sentence.text, L.config(level), rng_seed=seed)
        if len(result.clean_tokens) < MIN_CLEAN_WORDS:
            return None
    if not result.clean_text or not result.disfluent_text:
        return None
    row = result.to_dict()
    row["source"] = sentence.source
    row["template_id"] = sentence.template_id
    row["asked_level"] = level
    row["level"] = level_of_row(row)
    return row


def level_of_row(row: dict) -> int:
    spec = L.spec()
    return bucket_of(measure(row), spec["light_cut_per_100w"], spec["heavy_cut_per_100w"])


def make_synthetic(
    pool: Sequence[SourceSentence],
    n_pairs: int,
    base_seed: int,
    balance: bool = True,
) -> List[dict]:
    """Generate ``n_pairs`` synthetic pairs, roughly even across the four levels.

    The level is drawn, the pair is made, and then the pair is relabelled by
    what it measured. With ``balance`` on, the setting for the next draw is the
    one whose *measured* bucket is furthest behind, which pulls the label
    distribution back towards even without ever overriding a measurement.
    """
    records: List[dict] = []
    if not pool:
        return records
    rng = random.Random(base_seed)
    want = n_pairs / len(L.LEVELS)
    got: Counter = Counter()
    i = 0
    attempts = 0
    while len(records) < n_pairs and attempts < n_pairs * 20:
        attempts += 1
        src = pool[i % len(pool)]
        i += 1
        if balance and records:
            # aim at the emptiest bucket; setting k is the only setting that
            # reliably produces bucket k, so ask for it and take what comes
            level = min(L.LEVELS, key=lambda k: got[k])
        else:
            level = rng.choice(list(L.LEVELS))
        row = synthetic_pair(src, level, base_seed + i)
        if row is None:
            continue
        got[row["level"]] += 1
        records.append(row)
        if balance and got[row["level"]] > want * 1.6 and row["level"] == level:
            # this setting is overshooting its own bucket; stop asking for it
            continue
    return records


# --------------------------------------------------------------------------
# real pairs
# --------------------------------------------------------------------------


def real_pairs(path: Path, limit: int = 0) -> List[dict]:
    """Switchboard utterances, read as (clean, disfluent) and labelled.

    Only utterances of at least ``MIN_CLEAN_WORDS`` clean words are kept, the
    same pool the dial was measured on. Below that a per-word rate is quantised
    too coarsely to sit in the right bucket, and the register is backchannels
    rather than sentences.
    """
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row.get("clean_tokens") or row["clean"].split()) < MIN_CLEAN_WORDS:
                continue
            row["level"] = level_of_row(row)
            row["asked_level"] = row["level"]
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def mix_real(
    synthetic: List[dict], real: List[dict], limit: int, seed: int
) -> List[dict]:
    """Fold real speech into the synthetic pairs.

    The real rows are shuffled before the cap is applied, so the sample keeps
    the corpus's own composition across the four levels rather than an
    oversampled one. A model taught that every scripted sentence needs heavy
    disfluency would be a worse failure than one that hesitates too rarely.
    """
    if limit == 0 or not real:
        return synthetic
    pool = list(real)
    random.Random(seed).shuffle(pool)
    if limit > 0:
        pool = pool[:limit]
    mixed = synthetic + pool
    random.Random(seed).shuffle(mixed)
    return mixed


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(rows: Sequence[dict]) -> str:
    by_level = Counter(r["level"] for r in rows)
    types: Counter = Counter()
    words = 0
    for row in rows:
        words += len(row["clean_tokens"])
        for ann in row["annotations"]:
            types[ann["type"]] += 1
    parts = [f"d{k}={by_level[k]}" for k in L.LEVELS]
    rate = sum(types.values()) * 100.0 / words if words else 0.0
    return (
        f"{len(rows)} pairs  " + " ".join(parts)
        + f"  |  {rate:.2f} events per 100 words  "
        + " ".join(f"{t}={types[t]}" for t in TYPES)
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate (clean, disfluent) pairs")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--train", type=int, default=50_000)
    ap.add_argument("--val", type=int, default=2_000)
    ap.add_argument("--test", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="generate 120 pairs from templates only and print a sample",
    )
    args = ap.parse_args(argv)

    if args.smoke:
        pools = build_pools(100, 10, 10, args.seed, template_share=1.0, verbose=True)
        rows = make_synthetic(pools.train, 100, args.seed)
        rows += make_synthetic(pools.val, 10, args.seed + 10_000)
        rows += make_synthetic(pools.test, 10, args.seed + 20_000)
        path = args.out / "smoke.jsonl"
        write_jsonl(path, rows)
        print(f"wrote {len(rows)} pairs to {path}")
        print(summarise(rows))
        print()
        for row in rows[:10]:
            print(f" IN  <d{row['level']}>: {row['clean']}")
            print(f" OUT         : {row['disfluent']}")
            print("             ", [a["type"] for a in row["annotations"]])
            print()
        return 0

    pools = build_pools(args.train, args.val, args.test, args.seed)
    print(
        f"source sentences: train={len(pools.train)} val={len(pools.val)} "
        f"test={len(pools.test)}",
        file=sys.stderr,
    )
    for name, pool, n, offset in (
        ("train", pools.train, args.train, 0),
        ("val", pools.val, args.val, 1_000_000),
        ("test", pools.test, args.test, 2_000_000),
    ):
        rows = make_synthetic(pool, n, args.seed + offset)
        written = write_jsonl(args.out / f"{name}.jsonl", rows)
        print(f"{name}: {summarise(rows)} -> {args.out / (name + '.jsonl')} ({written})")

    # The real side needs no generation, only labelling, so it is written once
    # here rather than recomputed by every stage that reads it.
    for name in ("train", "val", "eval"):
        src = args.out / f"swda_{name}.jsonl"
        if not src.exists():
            print(f"no {src}; skipping the real pairs", file=sys.stderr)
            continue
        rows = real_pairs(src)
        written = write_jsonl(args.out / f"real_{name}.jsonl", rows)
        print(f"real_{name}: {summarise(rows)} -> ({written})")
    return 0


if __name__ == "__main__":
    code = main()
    # The streaming WikiText reader leaves a live worker thread behind, which
    # keeps the interpreter alive long after the files are written.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
