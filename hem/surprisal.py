"""Surprisal at inserted repair points: Hem, the injector, and real speech.

The probing toolkit is Mend's, reused unchanged in its mechanics. What it is
pointed at is different. Mend asked whether a language model is surprised at the
interruption point of a self-repair, and found that it is, by about five bits,
on 548 real ones. Hem asks a sharper question: **do the repairs Hem inserts sit
where real ones sit**, or do they carry the signature of a generator.

That signature is visible in Mend's own figures. On real speech the surprisal
peak lands exactly on the interruption; on the injector's output it lands one
position *early*, on the reparandum, because the injector's abandoned word is
drawn at random from a category and a random substitution is surprising in
itself. A real speaker's abandoned word is something they were plausibly about
to say. So the position of the peak is a test of whether a generated repair is
the kind of mistake a person makes.

The figure plots the **paired difference** from each instance's own fluent
control, not raw surprisal. Raw surprisal would put the three sources on
incomparable scales: telephone conversation and WikiText prose differ by bits
before any disfluency is inserted, and the figure would show that difference
rather than the one being asked about. Differencing each disfluent window
against the same words in the same sentence with the disfluency removed cancels
it.

Usage::

    python -m hem.surprisal --draws 4000
    python -m hem.surprisal --limit 400 --draws 200    # quick pass
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hem.build_data import read_jsonl
from hem.injector import InjectionResult, spoken_form

DEFAULT_LM = "EleutherAI/pythia-160m"
WINDOW = 5

#: Okabe-Ito, checked for deuteranopia and protanopia separation.
COLOURS = {
    "Hem": "#0072B2",
    "Injector": "#D55E00",
    "Switchboard": "#009E73",
}


# --------------------------------------------------------------------------
# surprisal, from mend/surprisal.py
# --------------------------------------------------------------------------


class Surprisal:
    """Word-level surprisal in bits from a causal LM."""

    def __init__(self, model_name: str = DEFAULT_LM, device: Optional[str] = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from hem.train import pick_device

        self.torch = torch
        self.device = device or pick_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)
        self.bos = self.tokenizer.eos_token_id or 0

    def word_surprisals(self, sentences: Sequence[str], batch: int = 16) -> List[np.ndarray]:
        """One array of per-word surprisals (bits) per sentence."""
        torch = self.torch
        out: List[np.ndarray] = []
        for start in range(0, len(sentences), batch):
            chunk = list(sentences[start : start + batch])
            encoded = [
                self.tokenizer(text, return_offsets_mapping=True, truncation=True,
                               max_length=256)
                for text in chunk
            ]
            lengths = [len(e["input_ids"]) for e in encoded]
            width = max(lengths) + 1  # room for the primed first position
            ids = torch.full((len(chunk), width), self.bos, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for i, enc in enumerate(encoded):
                n = len(enc["input_ids"])
                ids[i, 1 : n + 1] = torch.tensor(enc["input_ids"], dtype=torch.long)
                mask[i, : n + 1] = 1
            ids = ids.to(self.device)
            mask = mask.to(self.device)

            logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = ids[:, 1:]
            token_bits = -(
                logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1) / math.log(2)
            )
            token_bits = token_bits.cpu().numpy()

            for i, (text, enc) in enumerate(zip(chunk, encoded)):
                bits = token_bits[i, : lengths[i]]
                out.append(_fold_to_words(text, enc["offset_mapping"], bits))
        return out


def _word_starts(text: str) -> List[Tuple[int, int]]:
    """Character spans of the whitespace-separated words."""
    spans = []
    pos = 0
    for word in text.split(" "):
        if word:
            spans.append((pos, pos + len(word)))
        pos += len(word) + 1
    return spans


def _fold_to_words(
    text: str, offsets: Sequence[Tuple[int, int]], bits: np.ndarray
) -> np.ndarray:
    """Sum subword surprisals into the word whose characters they cover."""
    spans = _word_starts(text)
    totals = np.zeros(len(spans), dtype=np.float64)
    edges = [s for s, _ in spans]
    for (start, end), value in zip(offsets, bits):
        # This tokenizer folds the preceding space into the token, so the raw
        # offset points at the end of the previous word. Skip past the space
        # before deciding which word the subword belongs to.
        while start < end and text[start].isspace():
            start += 1
        if start >= end:
            continue
        idx = int(np.searchsorted(edges, start, side="right")) - 1
        if 0 <= idx < len(totals):
            totals[idx] += value
    return totals


def matched_clean_position(
    alignment: Sequence[Optional[int]], boundary: int
) -> Optional[int]:
    """Where the boundary falls in the fluent control, or None if nowhere.

    Counting survivors only locates the boundary if the disfluent stream visits
    the clean tokens in clean order. A delayed repair breaks that on purpose: it
    holds the corrected token back and emits it a few words later, so the words
    before the boundary are no longer a prefix of the clean sentence and the
    count lands on an unrelated word. Requiring the prefix property and giving
    up when it fails is what stops the control window being read at whatever
    happened to sit at that index.
    """
    seen = {src for src in alignment[:boundary] if src is not None}
    return len(seen) if seen == set(range(len(seen))) else None


# --------------------------------------------------------------------------
# collecting windows at repair points
# --------------------------------------------------------------------------


def collect(
    rows: Sequence[dict], scorer: Surprisal, batch: int, kind: str = "repair",
    verbose: bool = True, label: str = "",
) -> Tuple[np.ndarray, np.ndarray]:
    """Paired ±WINDOW profiles for every repair with a full window on both sides."""
    wanted: List[dict] = []
    skipped = 0
    for row in rows:
        result = InjectionResult.from_dict(row)
        if not result.annotations:
            continue
        clean_spoken = [spoken_form(t) for t in result.clean_tokens]
        for ann in result.annotations:
            if ann.type != kind:
                continue
            boundary = ann.boundary_index()
            if boundary is None:
                continue
            if boundary - WINDOW < 0 or boundary + WINDOW >= len(result.disfluent_tokens):
                continue
            centre = matched_clean_position(result.alignment, boundary)
            if centre is None:
                skipped += 1
                continue
            if centre - WINDOW < 0 or centre + WINDOW >= len(clean_spoken):
                continue
            wanted.append(
                {
                    "disfluent": " ".join(result.disfluent_tokens),
                    "clean": " ".join(clean_spoken),
                    "boundary": boundary,
                    "centre": centre,
                }
            )

    if verbose:
        print(f"{label}: {len(wanted)} {kind}s with a full window"
              + (f", {skipped} with no matching fluent position" if skipped else ""),
              file=sys.stderr)
    if not wanted:
        return np.zeros((0, 2 * WINDOW + 1)), np.zeros((0, 2 * WINDOW + 1))

    texts: List[str] = []
    index: Dict[str, int] = {}
    for item in wanted:
        for key in ("disfluent", "clean"):
            if item[key] not in index:
                index[item[key]] = len(texts)
                texts.append(item[key])
    if verbose:
        print(f"  scoring {len(texts)} sentences", file=sys.stderr)

    scores: List[np.ndarray] = []
    for start in range(0, len(texts), batch):
        scores.extend(scorer.word_surprisals(texts[start : start + batch], batch))
        if verbose:
            print(f"  {min(start + batch, len(texts))}/{len(texts)}", end="\r",
                  file=sys.stderr)
    if verbose:
        print(file=sys.stderr)

    data, control = [], []
    for item in wanted:
        dis = scores[index[item["disfluent"]]]
        cln = scores[index[item["clean"]]]
        b, c = item["boundary"], item["centre"]
        if b + WINDOW >= len(dis) or c + WINDOW >= len(cln):
            continue  # truncated by the LM's context window
        data.append(dis[b - WINDOW : b + WINDOW + 1])
        control.append(cln[c - WINDOW : c + WINDOW + 1])
    if not data:
        return np.zeros((0, 2 * WINDOW + 1)), np.zeros((0, 2 * WINDOW + 1))
    return np.stack(data), np.stack(control)


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap_ci(
    data: np.ndarray, n_draws: int = 2000, seed: int = 13
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and percentile 95% CI over instances, per relative position."""
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    means = data.mean(axis=0)
    draws = np.empty((n_draws, data.shape[1]))
    for i in range(n_draws):
        draws[i] = data[rng.integers(0, n, n)].mean(axis=0)
    return means, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)


def summarise(data: np.ndarray, control: np.ndarray, n_draws: int, seed: int) -> dict:
    """Effect size and where along the window it peaks.

    The peak is reported wherever it falls. Its landing somewhere other than 0
    is the finding: a generator whose peak sits one position early is announcing
    that its abandoned word was drawn at random rather than chosen.
    """
    if data.shape[0] == 0:
        return {"n": 0}
    mean, lo, hi = bootstrap_ci(data - control, n_draws, seed)
    peak = int(np.argmax(np.abs(mean)))
    return {
        "n": int(data.shape[0]),
        "peak_position": peak - WINDOW,
        "peak_effect_bits": float(mean[peak]),
        "peak_effect_ci95": [float(lo[peak]), float(hi[peak])],
        "effect_at_boundary_bits": float(mean[WINDOW]),
        "effect_at_boundary_ci95": [float(lo[WINDOW]), float(hi[WINDOW])],
    }


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------


def make_figure(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
    n_draws: int = 2000,
    tag: str = "",
    title: str = "",
) -> Tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.6,
            "figure.dpi": 300,
        }
    )

    positions = np.arange(-WINDOW, WINDOW + 1)
    fig, ax = plt.subplots(figsize=(3.6, 2.7))

    for name in ("Switchboard", "Hem", "Injector"):
        if name not in series:
            continue
        data, control = series[name]
        if data.shape[0] == 0:
            continue
        mean, lo, hi = bootstrap_ci(data - control, n_draws)
        ax.fill_between(positions, lo, hi, color=COLOURS[name], alpha=0.18, linewidth=0)
        ax.plot(positions, mean, color=COLOURS[name], linewidth=1.4, marker="o",
                markersize=2.8, label=f"{name} (n={data.shape[0]})")

    ax.axhline(0, color="#999999", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#999999", linewidth=0.6, zorder=0)
    ax.set_xlabel("Position relative to the repair boundary")
    ax.set_ylabel("Surprisal above the fluent control (bits)")
    ax.set_xticks(positions[::2])
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.margins(y=0.16)
    ax.legend(frameon=False, loc="best", handlelength=1.6)
    if title:
        ax.set_title(title, pad=4)
    fig.tight_layout(pad=0.4)

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"surprisal_at_repair{tag}.pdf"
    png = out_dir / f"surprisal_at_repair{tag}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


SOURCES = (
    ("Hem", "outputs/generated/hem_d2.jsonl"),
    ("Injector", "outputs/generated/injector_d2.jsonl"),
    ("Switchboard", "data/real_eval.jsonl"),
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Surprisal at inserted repair points")
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--lm", default=DEFAULT_LM)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=20000,
                    help="cap the utterances read per source")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tag", default="")
    ap.add_argument("--title", default="Repairs, Hem against the injector and real speech")
    ap.add_argument("--seed", type=int, default=13)
    for name, default in SOURCES:
        ap.add_argument(f"--{name.lower()}", type=Path, default=Path(default))
    args = ap.parse_args(argv)

    scorer = Surprisal(args.lm, args.device)
    print(f"surprisal model: {args.lm} on {scorer.device}", file=sys.stderr)

    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    report: Dict[str, dict] = {}
    for name, _ in SOURCES:
        path: Path = getattr(args, name.lower())
        if not path.exists():
            print(f"no {path}; skipping {name}", file=sys.stderr)
            continue
        rows = read_jsonl(path)
        # The real-speech file is ordered by conversation, so sample across the
        # whole of it rather than off the front.
        random.Random(args.seed).shuffle(rows)
        if args.limit:
            rows = rows[: args.limit]
        data, control = collect(rows, scorer, args.batch, label=name)
        if data.shape[0] == 0:
            print(f"{name}: no repairs with a full window", file=sys.stderr)
            continue
        series[name] = (data, control)
        report[name] = summarise(data, control, args.draws, args.seed)

    if not series:
        print("nothing to plot", file=sys.stderr)
        return 1

    pdf, png = make_figure(series, args.out, args.draws, args.tag, args.title)

    print("\nSurprisal at the repair boundary, against each instance's own control:")
    for name, entry in report.items():
        lo, hi = entry["peak_effect_ci95"]
        blo, bhi = entry["effect_at_boundary_ci95"]
        print(
            f"  {name:<13s} n={entry['n']:<5d} peak {entry['peak_effect_bits']:+.2f} "
            f"bits at position {entry['peak_position']:+d} "
            f"[95% CI {lo:+.2f}, {hi:+.2f}]; at the boundary itself "
            f"{entry['effect_at_boundary_bits']:+.2f} [{blo:+.2f}, {bhi:+.2f}]"
        )
    print(f"\nwrote {pdf} and {png}")
    (args.out / f"surprisal_effects{args.tag}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
