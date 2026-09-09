"""The Hem command.

    $ python -m hem --level 2 "Send the quarterly report to Sarah before Friday."
    send the uh quarterly report to sarah no wait to sarah before friday

    $ python -m hem --level 2 --trace "Send the report to Sarah before Friday."
    send the uh report to sarah before before friday
      filler       at 2   before clean word 2   "uh"
      repetition   at 7   before clean word 6   "before"

    $ python -m hem --eval

``--trace`` runs the detector over the generated text rather than asking the
generator what it did, so it says the same thing about model output and rule
output, and so the numbers it prints are the ones the evaluation counts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from hem import levels as L
from hem.generate import DEFAULT_MODEL


def read_lines(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [" ".join(args.text)]
    return [line.strip() for line in sys.stdin if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="hem",
        description="Insert realistic disfluency into clean text, on a dial",
        epilog="levels:\n  " + "\n  ".join(L.describe()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("text", nargs="*", help="text to speak; omit to read stdin")
    ap.add_argument(
        "--level", type=int, default=2, choices=list(L.LEVELS),
        help="how disfluent, 0 clean to 3 heavy (default 2)",
    )
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--rules", action="store_true",
        help="use the rule-based injector instead of the model",
    )
    ap.add_argument(
        "--trace", action="store_true",
        help="also list the detected insertions with their type and position",
    )
    ap.add_argument("--all-levels", dest="all_levels", action="store_true",
                    help="print every level, so the dial can be seen moving")
    ap.add_argument("--levels", action="store_true", help="print the dial and exit")
    ap.add_argument("--eval", action="store_true", help="run the evaluation")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--eval" in argv:
        from hem.evaluate import main as evaluate_main

        argv.remove("--eval")
        return evaluate_main(argv)

    args = build_parser().parse_args(argv)

    if args.levels:
        print("\n".join(L.describe()))
        print()
        print(L.rate_table())
        return 0

    lines = read_lines(args)
    if not lines:
        print("nothing to say; pass text as arguments or on stdin", file=sys.stderr)
        return 1

    from hem.detect import trace as trace_insertions
    from hem.generate import load

    speaker = load(
        "rules" if args.rules else "model",
        path=args.model,
        device=args.device,
        batch=args.batch,
        seed=args.seed,
    )

    wanted = list(L.LEVELS) if args.all_levels else [args.level]
    for level in wanted:
        spoken = speaker.speak(lines, level)
        for source, said in zip(lines, spoken):
            prefix = f"d{level}: " if args.all_levels else ""
            print(prefix + said)
            if args.trace:
                rows = trace_insertions(source, said)
                if not rows:
                    print("  (nothing inserted)")
                for row in rows:
                    print(
                        f"  {row['type']:<12s} at {row['position']:<4d} "
                        f"before clean word {row['before_clean_word']:<4d} "
                        f"{row['words']!r}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
