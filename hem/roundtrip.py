"""Run Mend over Hem's output and see how much of the script comes back.

Hem and Mend are meant to be inverses. Hem writes a script out as speech, Mend
reads speech back as writing, and if the two are really inverses the text that
comes out of the pair is the text that went in. This is the only measurement in
the repo that tests the claim end to end, and it is the one the README leads
with.

It runs as its own command rather than inside ``hem.evaluate`` for a boring
reason: on a 16 GB machine, holding two Hem checkpoints and their MPS decode
buffers while loading a third model gets the process killed. Splitting it means
Mend starts with the machine to itself and reads Hem's output off disk, which
also makes the round trip re-runnable without regenerating anything.

Usage::

    python -m hem.evaluate          # writes outputs/generated/
    python -m hem.roundtrip         # reads it, runs Mend, writes the table
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from hem import levels as L
from hem.build_data import read_jsonl
from hem.evaluate import ORDER, Mend, recovery, round_trip_table


def label_of(path: Path) -> tuple:
    """``hem_greedy_d2.jsonl`` -> ``("Hem greedy", 2)``."""
    stem = path.stem
    name, _, level = stem.rpartition("_d")
    pretty = name.replace("_", " ").capitalize().replace("Hem greedy", "Hem greedy")
    return pretty, int(level)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Round-trip Hem's output through Mend")
    ap.add_argument("--generated", type=Path, default=Path("outputs/generated"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--mend", default="daltonoscar0/mend-flan-t5-small")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int, default=1200,
                    help="cap the Switchboard ceiling row")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    files = sorted(args.generated.glob("*_d*.jsonl"))
    if not files:
        raise SystemExit(
            f"no generations in {args.generated}; run `python -m hem.evaluate` first"
        )

    mend = Mend(args.mend, args.device, args.batch)
    print(f"mend: {args.mend} on {mend.device}", file=sys.stderr)

    results: List[dict] = []
    for path in files:
        name, level = label_of(path)
        rows = read_jsonl(path)
        print(f"  {name} d{level} ({len(rows)})", file=sys.stderr)
        back = mend.clean([r["disfluent"] for r in rows], progress=True)
        results.append(
            {"system": f"Mend after {name}", "level": level,
             **recovery([r["clean"] for r in rows], back)}
        )

    # The ceiling. Mend on real Switchboard is the job Mend was trained for, so
    # no round trip through a generator should be expected to beat it, and a
    # round trip that comes close is a round trip that is working.
    real_path = args.data / "real_eval.jsonl"
    if real_path.exists():
        real = read_jsonl(real_path)
        random.Random(args.seed).shuffle(real)
        real = real[: args.limit]
        print(f"  real Switchboard ({len(real)})", file=sys.stderr)
        back = mend.clean([r["disfluent"] for r in real], progress=True)
        results.append(
            {"system": "Mend on real Switchboard", "level": None,
             **recovery([r["clean"] for r in real], back)}
        )

    def order(entry: dict) -> tuple:
        name = entry["system"].replace("Mend after ", "")
        rank = ORDER.index(name) if name in ORDER else len(ORDER)
        return (rank, entry["level"] if entry["level"] is not None else 99)

    results.sort(key=order)
    table = round_trip_table(results)
    print("\n## Round trip\n")
    print(table)

    (args.out / "round_trip.md").write_text("## Round trip\n\n" + table + "\n")
    path = args.out / "eval.json"
    if path.exists():
        report = json.loads(path.read_text())
        report["round_trip"] = results
        path.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\nwrote {args.out / 'round_trip.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
