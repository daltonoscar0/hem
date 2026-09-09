"""Run the dial over text, with either generator behind it.

Two things implement the same interface, which is what lets the evaluation put
them on one table and the CLI switch between them with a flag:

* :class:`Model` is the fine-tuned flan-t5-small, conditioned on the control
  token.
* :class:`Rules` is the rule-based injector at that level's measured rates,
  which is the baseline.

Both take clean written text and a level and return spoken-register text:
lowercased, no punctuation, disfluencies inserted. That is the register Mend
reads, so either one's output can be fed straight back through Mend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

from hem import levels as L
from hem.build_data import encode_source
from hem.injector import inject, spoken_text

#: Where the trained weights live if nothing else is said.
DEFAULT_MODEL = "outputs/model"
HUB_MODEL = "daltonoscar0/hem-flan-t5-small"


class Rules:
    """The rule-based injector, set to each level's measured rates."""

    name = "Injector"

    def __init__(self, seed: int = 13) -> None:
        self.seed = seed

    def speak(self, texts: Sequence[str], level: int, offset: int = 0) -> List[str]:
        out = []
        for i, text in enumerate(texts):
            if level == 0:
                out.append(spoken_text(text))
            else:
                result = inject(text, L.config(level), rng_seed=self.seed + offset + i)
                out.append(result.disfluent_text)
        return out


class Model:
    """The fine-tuned model, conditioned on the control token.

    Decoding is greedy, which is deliberate. Sampling would let the dial's rate
    be tuned after the fact by turning the temperature up, and the whole point
    of the control token is that the rate is a property the model learned rather
    than one imposed on it at decode time. Greedy also makes every number in the
    evaluation reproducible.
    """

    name = "Hem"

    def __init__(
        self,
        path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from hem.train import pick_device

        self.device = device or pick_device()
        self.batch = batch
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)
        self.torch = torch
        missing = [t for t in L.LEVEL_TOKENS if t not in self.tokenizer.get_vocab()]
        if missing:
            raise SystemExit(
                f"{path} has no control tokens in its vocabulary ({missing}); "
                "this is not a Hem checkpoint"
            )

    def speak(
        self,
        texts: Sequence[str],
        level: int,
        offset: int = 0,
        progress: bool = False,
    ) -> List[str]:
        import sys

        out: List[str] = []
        sources = [encode_source(level, t) for t in texts]
        for i in range(0, len(sources), self.batch):
            enc = self.tokenizer(
                sources[i : i + self.batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=192,
                pad_to_multiple_of=8,
            ).to(self.device)
            gen = self.model.generate(
                **enc, max_new_tokens=192, num_beams=1, do_sample=False
            )
            out.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            if progress:
                print(f"  d{level}: {min(i + self.batch, len(sources))}/{len(sources)}",
                      end="\r", file=sys.stderr)
            # See the note in train.py: the MPS allocator pools per shape and
            # holds on to everything unless it is told otherwise.
            if self.device == "mps" and (i // self.batch) % 20 == 19:
                self.torch.mps.empty_cache()
        if progress:
            print(file=sys.stderr)
        return [" ".join(t.split()) for t in out]


def load(which: str = "model", path: str = DEFAULT_MODEL, **kwargs):
    """``"model"`` or ``"rules"``, resolved to something with ``.speak``."""
    if which == "rules":
        return Rules(**{k: v for k, v in kwargs.items() if k == "seed"})
    target = Path(path)
    if not target.exists() and "/" not in str(path):
        raise SystemExit(
            f"no model at {path}. Either train one:\n"
            "  python -m hem.build_data --train 50000 --val 2000 --test 10000\n"
            "  python -m hem.train --batch 32 --epochs 2 --real 25000\n"
            f"or point at the Hub copy with --model {HUB_MODEL}"
        )
    return Model(str(path), **{k: v for k, v in kwargs.items()
                               if k in ("device", "batch")})
