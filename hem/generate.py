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

    Decoding is **ancestral sampling at temperature 1**, which is the model's
    own distribution with nothing tuned. Greedy decoding was the first choice,
    on the reasoning that sampling would let the rate be adjusted after the fact
    by turning the temperature up, and the whole point of the control token is
    that the rate is something the model learned. Measured, greedy destroys the
    dial:

        decode          d1          d2           d3        script words lost
        greedy          1.8 / 6.7   2.3 / 14.2    6.2 / 28.1   1.2% to 2.2%
        sample t=1.0    4.9 / 6.7  11.1 / 14.2   22.4 / 28.1   3.2% to 6.1%
        sample t=1.3   13.6 / 6.7  24.4 / 14.2   35.5 / 28.1   9.9% to 22.7%

    Measured over the same 1,200 held-out scripts, rate against target, in
    events per 100 clean words.

    The reason is not a bug. The model is a distribution over ways of saying a
    sentence, and the single most likely way to say any sentence is fluently: a
    filled pause has to go *somewhere*, and no one position carries as much
    probability as inserting nothing. Greedy takes the mode and the mode is
    nearly fluent. Sampling asks the model the question it was trained to
    answer.

    Temperature 1 is not a tuned value, it is the absence of one, and the third
    row is why it stays there. Pushing the temperature up does close the
    remaining 20% of the rate gap, and costs four times as much of the script to
    do it.

    Sampling is seeded, so every number in the evaluation still reproduces.
    """

    name = "Hem"

    def __init__(
        self,
        path: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        batch: int = 32,
        seed: int = 13,
        greedy: bool = False,
        temperature: float = 1.0,
    ) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from hem.train import pick_device

        self.device = device or pick_device()
        self.batch = batch
        self.seed = seed
        self.greedy = greedy
        self.temperature = temperature
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
        # Seeded per call, so a level's output does not depend on which other
        # levels were generated first.
        self.torch.manual_seed(self.seed + offset + 1000 * level)
        if self.greedy:
            settings = dict(num_beams=1, do_sample=False)
        else:
            # top_k and top_p are switched off on purpose: any truncation of the
            # tail is a tuned knob, and the point is to sample the distribution
            # the model actually learned.
            settings = dict(
                do_sample=True, temperature=self.temperature, top_k=0, top_p=1.0
            )
        for i in range(0, len(sources), self.batch):
            enc = self.tokenizer(
                sources[i : i + self.batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=192,
                pad_to_multiple_of=8,
            ).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=192, **settings)
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


#: Directories a local checkpoint would sit under. A path starting with one of
#: these and not existing is a missing checkpoint, not a Hub identifier, and
#: saying so beats letting `transformers` report a 404 on "outputs/model".
_LOCAL_ROOTS = ("outputs", "data", "checkpoints")


def _looks_local(path: str) -> bool:
    if path.startswith((".", "/", "~")):
        return True
    head = Path(path).parts[0] if Path(path).parts else ""
    return head in _LOCAL_ROOTS


def load(which: str = "model", path: str = DEFAULT_MODEL, **kwargs):
    """``"model"`` or ``"rules"``, resolved to something with ``.speak``."""
    if which == "rules":
        return Rules(**{k: v for k, v in kwargs.items() if k == "seed"})
    if not Path(path).exists() and _looks_local(str(path)):
        raise SystemExit(
            f"no model at {path}. Either train one:\n"
            "  python -m hem.swda --root swda --out data/swda_all.jsonl --split\n"
            "  python -m hem.build_data --train 50000 --val 2000 --test 10000\n"
            "  python -m hem.train --batch 32 --epochs 2 --real 25000\n"
            f"or point at the Hub copy with --model {HUB_MODEL}"
        )
    return Model(str(path), **{k: v for k, v in kwargs.items()
                               if k in ("device", "batch", "seed", "greedy",
                                        "temperature")})
