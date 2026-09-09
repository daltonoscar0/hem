"""The dial: four levels, and what each one is a target for.

``hem/levels.json`` is measured by :mod:`hem.rates` and checked in, so importing
this module needs no corpus. Everything downstream reads the dial from here: the
control token the model is conditioned on, the injector configuration the
baseline runs at, and the target rates the evaluation scores both against.

    <d0>  clean     nothing inserted
    <d1>  light     below the median disfluent Switchboard utterance
    <d2>  natural   median to 90th percentile
    <d3>  heavy     90th percentile and above

The token goes on the front of the model's input, so one checkpoint serves all
four settings and the dial is a decode-time argument rather than four models.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from hem.injector import TYPES, InjectionConfig

SPEC_PATH = Path(__file__).with_name("levels.json")

#: The dial's settings, low to high.
LEVELS: Sequence[int] = (0, 1, 2, 3)

#: One control token per level, added to the tokenizer as a special token so it
#: survives as a single piece rather than being split into "<", "d", "0", ">".
LEVEL_TOKENS: Sequence[str] = tuple(f"<d{k}>" for k in LEVELS)

LEVEL_NAMES: Dict[int, str] = {0: "clean", 1: "light", 2: "natural", 3: "heavy"}


@lru_cache(maxsize=1)
def spec() -> dict:
    """The measured Switchboard specification the dial is built from."""
    return json.loads(SPEC_PATH.read_text())


def token(level: int) -> str:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {list(LEVELS)}, got {level!r}")
    return LEVEL_TOKENS[level]


def level_of_token(text: str) -> Optional[int]:
    for k, tok in enumerate(LEVEL_TOKENS):
        if text.startswith(tok):
            return k
    return None


def targets(level: int) -> Dict[str, float]:
    """Target rate per type, in events per 100 clean words."""
    entry = spec()["levels"][str(level)]
    out = dict(entry["rates"])
    out["total"] = entry["total_rate"]
    return out


def mix(level: int) -> Dict[str, float]:
    """Target share of each type among that level's events."""
    return dict(spec()["levels"][str(level)]["mix"])


def config(level: int, **overrides) -> InjectionConfig:
    """The rule-based injector, set to this level's measured rates.

    This is the baseline the model has to beat. It gets the same rate targets
    the model is trained towards and the same measured shape parameters, so the
    only thing left for the two to differ on is *where* the disfluencies go.
    """
    rates = targets(level)
    shape = spec()["shape"]
    kwargs = dict(
        filler_per_100w=rates["filler"],
        repetition_per_100w=rates["repetition"],
        repair_per_100w=rates["repair"],
        false_start_per_100w=rates["false_start"],
        interregnum_rate=shape["interregnum_rate"],
        repetition_bigram_rate=shape["repetition_bigram_rate"],
        span_repair_rate=shape["span_repair_rate"],
    )
    kwargs.update(overrides)
    return InjectionConfig(**kwargs)


def describe() -> List[str]:
    """One line per level, for ``hem --levels`` and the CLI's help."""
    out = []
    for k in LEVELS:
        r = targets(k)
        per = 100.0 / r["total"] if r["total"] else 0.0
        tail = "nothing inserted" if not r["total"] else (
            f"{r['total']:.1f} events per 100 words, one per {per:.0f}"
        )
        out.append(f"{token(k)}  {LEVEL_NAMES[k]:<8s} {tail}")
    return out


def rate_table() -> str:
    """The markdown rate table, rebuilt from the checked-in spec."""
    from hem.rates import rate_table as _table

    return _table(spec())


__all__ = [
    "LEVELS",
    "LEVEL_TOKENS",
    "LEVEL_NAMES",
    "TYPES",
    "spec",
    "token",
    "level_of_token",
    "targets",
    "mix",
    "config",
    "describe",
    "rate_table",
]
