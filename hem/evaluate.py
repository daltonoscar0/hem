"""The evaluation: does the dial do what it says, and does it sound real.

Five questions, in the order they matter.

1. **Rate control.** Set the dial to k and measure what comes out, per type,
   against the target measured off Switchboard. A dial that does not move is
   not a dial.
2. **Type mix.** Of the disfluencies that were inserted, what proportion are
   filled pauses, repetitions, substitutions and restarts, against
   Switchboard's proportions at the same level.
3. **Placement.** Shriberg's result is that filled pauses cluster at clause
   onsets and in front of hard words. The rule-based injector encodes the first
   half of that explicitly, so it should not lose there; it knows nothing about
   the second half, so that is where a model that has read real speech can win.
4. **Round trip.** Run Mend over Hem's output and see how much of the original
   script comes back. Hem and Mend are meant to be inverses, and this is the
   only measurement that tests it end to end.
5. **A qualitative table**, thirty sentences at all four settings, because a
   reader should be able to see the dial rather than take the table's word.

Everything counted here goes through :mod:`hem.detect`, including the
Switchboard reference, so no system is scored by a rule another system was not
scored by. The Switchboard row is also reported from its gold annotation, and
the gap between the two is the detector's own error.

Usage::

    python -m hem --eval
    python -m hem --eval --limit 500 --skip-roundtrip
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from hem import levels as L
from hem.build_data import read_jsonl
from hem.detect import detect
from hem.injector import STOPWORDS, TYPES, TYPE_LABELS, _STRIP_CHARS, spoken_text

MEND_HUB = "daltonoscar0/mend-flan-t5-small"
MEND_PREFIX = "clean up: "

#: Okabe-Ito, checked for deuteranopia and protanopia separation.
COLOURS = {
    "Hem": "#0072B2",
    "Injector": "#D55E00",
    "Switchboard": "#009E73",
    "target": "#4D4D4D",
}


def normalise(word: str) -> str:
    return word.strip(_STRIP_CHARS).lower()


def is_content(word: str) -> bool:
    w = normalise(word)
    return len(w) > 3 and w not in STOPWORDS


# --------------------------------------------------------------------------
# counting what a system produced
# --------------------------------------------------------------------------


def annotated_rows(clean_texts: Sequence[str], spoken: Sequence[str]) -> List[dict]:
    """Detected output in the corpus schema, for the surprisal analysis.

    The surprisal figure compares Hem's repair points against the injector's and
    against real ones, and it needs spans to centre its windows on. Writing the
    detector's output in the same shape the corpus parse uses means the analysis
    reads all three through one code path.
    """
    out = []
    for clean, said in zip(clean_texts, spoken):
        result, _ = detect(clean, said)
        row = result.to_dict()
        row["annotations"] = [a for a in row["annotations"] if a["type"] in TYPES]
        out.append(row)
    return out


def insertion_records(clean_text: str, spoken: str) -> Tuple[List[dict], dict]:
    """Every detected insertion in one output, with what sits after it.

    ``before_clean_word`` is the index of the first clean word that survives
    after the insertion, which is where the disfluency was placed relative to
    the script. That index is what the placement analysis is about.
    """
    result, report = detect(clean_text, spoken)
    clean = result.clean_tokens
    rows: List[dict] = []
    for ann in result.annotations:
        spans = [s for s in (ann.reparandum, ann.interregnum) if s is not None]
        if not spans:
            continue
        start = min(s.start for s in spans)
        end = max(s.end for s in spans)
        after = next(
            (result.alignment[i] for i in range(end, len(result.alignment))
             if result.alignment[i] is not None),
            None,
        )
        index = len(clean) if after is None else after
        rows.append(
            {
                "type": ann.type,
                "length": end - start,
                "clean_index": index,
                "relative": index / len(clean) if clean else 0.0,
                "next_word": normalise(clean[index]) if index < len(clean) else "",
                "next_is_content": index < len(clean) and is_content(clean[index]),
                "at_onset": index == 0
                or (
                    0 < index <= len(clean)
                    and clean[index - 1].rstrip("\"”'’)").endswith((",", ";", ":"))
                ),
            }
        )
    return rows, report


def gold_records(row: dict) -> List[dict]:
    """The same rows, read off an annotated corpus instead of detected."""
    clean = row["clean_tokens"]
    alignment = row["alignment"]
    out: List[dict] = []
    for ann in row["annotations"]:
        if ann["type"] not in TYPES:
            continue
        spans = [ann.get("reparandum"), ann.get("interregnum")]
        present = [s for s in spans if s is not None]
        if not present:
            continue
        start = min(s["start"] for s in present)
        end = max(s["end"] for s in present)
        after = next(
            (alignment[i] for i in range(end, len(alignment)) if alignment[i] is not None),
            None,
        )
        index = len(clean) if after is None else after
        out.append(
            {
                "type": ann["type"],
                "length": end - start,
                "clean_index": index,
                "relative": index / len(clean) if clean else 0.0,
                "next_word": normalise(clean[index]) if index < len(clean) else "",
                "next_is_content": index < len(clean) and is_content(clean[index]),
                "at_onset": index == 0
                or (
                    0 < index <= len(clean)
                    and clean[index - 1].rstrip("\"”'’)").endswith((",", ";", ":"))
                ),
            }
        )
    return out


def summarise_run(
    clean_texts: Sequence[str], records: Sequence[Sequence[dict]]
) -> dict:
    """Rates, mix and placement for one system at one level."""
    words = sum(len(t.split()) for t in clean_texts)
    flat = [r for rows in records for r in rows]
    counts = Counter(r["type"] for r in flat)
    scale = 100.0 / words if words else 0.0
    total = len(flat)
    return {
        "sentences": len(clean_texts),
        "clean_words": words,
        "events": total,
        "rates": {t: counts[t] * scale for t in TYPES},
        "total_rate": total * scale,
        "mix": {t: (counts[t] / total if total else 0.0) for t in TYPES},
        "records": flat,
    }


# --------------------------------------------------------------------------
# 1. rate control
# --------------------------------------------------------------------------


def rate_table(runs: Dict[Tuple[str, int], dict]) -> str:
    header = (
        "| System | Level | Filled pause | Repetition | Substitution | Restart | "
        "All types | Target | Ratio |"
    )
    lines = [header, "|" + "|".join("---" for _ in range(9)) + "|"]
    for (system, level), run in sorted(runs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        target = L.targets(level)
        r = run["rates"]
        ratio = (
            f"{run['total_rate'] / target['total']:.2f}" if target["total"] else "n/a"
        )
        lines.append(
            f"| {system} | `<d{level}>` | {r['filler']:.2f} | {r['repetition']:.2f} | "
            f"{r['repair']:.2f} | {r['false_start']:.2f} | "
            f"**{run['total_rate']:.2f}** | {target['total']:.2f} | {ratio} |"
        )
    lines.append("")
    lines.append(
        "Events per 100 clean words, counted by the detector. Target is the rate "
        "measured off Switchboard for that bucket. Ratio is measured over target, "
        "so 1.00 is the dial landing where it was aimed."
    )
    return "\n".join(lines)


def rate_error(runs: Dict[Tuple[str, int], dict], system: str) -> dict:
    """Mean absolute and relative error against the targets, over levels 1-3."""
    abs_err, rel_err = [], []
    for level in (1, 2, 3):
        target = L.targets(level)["total"]
        got = runs[(system, level)]["total_rate"]
        abs_err.append(abs(got - target))
        rel_err.append(abs(got - target) / target)
    return {
        "mean_abs_error": sum(abs_err) / len(abs_err),
        "mean_rel_error": sum(rel_err) / len(rel_err),
    }


def rate_figure(runs: Dict[Tuple[str, int], dict], out_dir: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharex=True)
    xs = list(L.LEVELS)

    ax = axes[0]
    ax.plot(xs, [L.targets(k)["total"] for k in xs], color=COLOURS["target"],
            linestyle="--", dashes=(4, 2), linewidth=1.2, marker="s", markersize=3,
            label="Switchboard target")
    for system in ("Hem", "Injector"):
        if (system, 0) not in runs:
            continue
        ax.plot(xs, [runs[(system, k)]["total_rate"] for k in xs],
                color=COLOURS[system], linewidth=1.4, marker="o", markersize=3.2,
                label=system)
    ax.set_ylabel("Events per 100 words")
    ax.set_title("All disfluency types", pad=4)
    ax.legend(frameon=False, loc="upper left", handlelength=1.8)

    ax = axes[1]
    for name, key in (("Filled pause", "filler"), ("Substitution", "repair")):
        style = "-" if key == "filler" else ":"
        ax.plot(xs, [L.targets(k)[key] for k in xs], color=COLOURS["target"],
                linestyle=style, linewidth=1.0, alpha=0.7)
        for system in ("Hem", "Injector"):
            if (system, 0) not in runs:
                continue
            ax.plot(xs, [runs[(system, k)]["rates"][key] for k in xs],
                    color=COLOURS[system], linewidth=1.3, linestyle=style,
                    marker="o" if key == "filler" else "^", markersize=3.2,
                    label=f"{system}, {name.lower()}")
    ax.set_title("By type, grey is the target", pad=4)
    ax.legend(frameon=False, loc="upper left", handlelength=2.2, fontsize=6)

    for ax in axes:
        ax.set_xlabel("Dial setting")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"d{k}" for k in xs])
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.margins(y=0.16)
    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "rate_control.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / "rate_control.pdf")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------
# 2. type mix
# --------------------------------------------------------------------------


def kl_divergence(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-3) -> float:
    """KL(p || q) in bits over the four types, with the reference smoothed.

    A system that never produces restarts would otherwise score an infinite
    divergence off one empty cell, which says less than the finite number does.
    """
    total_p = sum(p.get(t, 0.0) for t in TYPES) or 1.0
    total_q = sum(q.get(t, 0.0) + eps for t in TYPES)
    out = 0.0
    for t in TYPES:
        pi = p.get(t, 0.0) / total_p
        qi = (q.get(t, 0.0) + eps) / total_q
        if pi > 0:
            out += pi * math.log2(pi / qi)
    return out


def mix_table(
    runs: Dict[Tuple[str, int], dict], reference: Dict[int, dict]
) -> str:
    header = (
        "| System | Level | Filled pause | Repetition | Substitution | Restart | "
        "KL from Switchboard |"
    )
    lines = [header, "|" + "|".join("---" for _ in range(7)) + "|"]
    for level in (1, 2, 3):
        ref = reference[level]["mix"]
        for system in ("Hem", "Injector"):
            if (system, level) not in runs:
                continue
            mix = runs[(system, level)]["mix"]
            lines.append(
                f"| {system} | `<d{level}>` | "
                + " | ".join(f"{mix[t]:.3f}" for t in TYPES)
                + f" | {kl_divergence(mix, ref):.3f} |"
            )
        lines.append(
            f"| Switchboard, detected | `<d{level}>` | "
            + " | ".join(f"{ref[t]:.3f}" for t in TYPES)
            + " | 0 |"
        )
        gold = reference[level]["gold_mix"]
        lines.append(
            f"| Switchboard, gold spans | `<d{level}>` | "
            + " | ".join(f"{gold[t]:.3f}" for t in TYPES)
            + f" | {kl_divergence(gold, ref):.3f} |"
        )
    lines.append("")
    lines.append(
        "Share of that level's events falling in each type, in bits of KL from "
        "the Switchboard row directly above. The gold-span row is the same "
        "utterances read from the corpus annotation instead of through the "
        "detector, so its KL is the detector's own error and is the floor any "
        "system is really being measured against."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 3. placement
# --------------------------------------------------------------------------


def load_frequencies(path: Path) -> Dict[str, float]:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def placement_stats(
    records: Sequence[dict], freqs: Dict[str, float], only: Optional[str] = "filler"
) -> dict:
    """Where a system's insertions land, on the two axes Shriberg names.

    The two axes interact, which is why the mid-clause figure is kept apart.
    Filled pauses cluster at clause onsets, and the word after a clause onset is
    almost always a very common function word, so a system that gets the onset
    behaviour right will look like it prefers *easy* words. The question of
    whether a hesitation precedes a hard word can only be asked of the
    hesitations that are not at an onset.
    """
    rows = [r for r in records if only is None or r["type"] == only]
    if not rows:
        return {"n": 0, "relative_positions": []}
    floor = min(freqs.values()) if freqs else -6.0

    def mean_freq(items):
        vals = [freqs.get(r["next_word"], floor) for r in items if r["next_word"]]
        return sum(vals) / len(vals) if vals else 0.0

    mid = [r for r in rows if not r["at_onset"]]
    return {
        "n": len(rows),
        "at_clause_onset": sum(r["at_onset"] for r in rows) / len(rows),
        "before_content_word": sum(r["next_is_content"] for r in rows) / len(rows),
        "mean_next_log_freq": mean_freq(rows),
        "n_mid_clause": len(mid),
        "mid_clause_next_log_freq": mean_freq(mid),
        "relative_positions": [r["relative"] for r in rows],
    }


def baseline_placement(clean_texts: Sequence[str], freqs: Dict[str, float]) -> dict:
    """The same numbers for a filler dropped at a uniformly random slot.

    Every system needs its own, computed over the text *it* was given. The
    scripted sentences and the Switchboard clean sides do not comma at the same
    rate, and a raw clause-onset share compared across the two would be
    measuring the transcriber's punctuation habits rather than the speaker's
    hesitation habits.
    """
    floor = min(freqs.values()) if freqs else -6.0
    onset = content = total = 0
    logf: List[float] = []
    mid_logf: List[float] = []
    for text in clean_texts:
        toks = text.split()
        for i, tok in enumerate(toks):
            total += 1
            here = i == 0 or toks[i - 1].rstrip("\"”'’)").endswith((",", ";", ":"))
            if here:
                onset += 1
            if is_content(tok):
                content += 1
            value = freqs.get(normalise(tok), floor)
            logf.append(value)
            if not here:
                mid_logf.append(value)
    return {
        "n": total,
        "at_clause_onset": onset / total if total else 0.0,
        "before_content_word": content / total if total else 0.0,
        "mean_next_log_freq": sum(logf) / len(logf) if logf else 0.0,
        "n_mid_clause": len(mid_logf),
        "mid_clause_next_log_freq": sum(mid_logf) / len(mid_logf) if mid_logf else 0.0,
    }


def placement_table(rows: Dict[str, dict], baselines: Dict[str, dict]) -> str:
    header = (
        "| Where filled pauses land | N | At a clause onset | Before a content word | "
        "Log freq of the next word | Same, mid-clause only |"
    )
    lines = [header, "|" + "|".join("---" for _ in range(6)) + "|"]
    for name in ("Hem", "Injector", "Switchboard"):
        s = rows.get(name)
        if not s or not s.get("n"):
            continue
        b = baselines[name]
        lines.append(
            f"| {name} | {s['n']:,} | "
            f"**{s['at_clause_onset']:.3f}** ({b['at_clause_onset']:.3f}) | "
            f"{s['before_content_word']:.3f} ({b['before_content_word']:.3f}) | "
            f"{s['mean_next_log_freq']:.2f} ({b['mean_next_log_freq']:.2f}) | "
            f"**{s['mid_clause_next_log_freq']:.2f}** "
            f"({b['mid_clause_next_log_freq']:.2f}) |"
        )
    lines.append("")
    lines.append(
        "Each cell is the measured figure with, in brackets, what it would be if "
        "the filled pause were dropped in front of a word drawn uniformly from "
        "that system's own text. The bracketed figure differs by row because the "
        "scripted sentences and the Switchboard transcripts do not carry commas "
        "at the same rate, and comparing raw onset shares across them would be "
        "measuring a transcriber's punctuation rather than a speaker's "
        "hesitations."
    )
    lines.append("")
    lines.append(
        "Log frequency is base 10 over the Switchboard clean sides, so a lower "
        "number is a rarer word. The last column is the Shriberg question asked "
        "properly: of the hesitations that are *not* at a clause onset, does the "
        "word after them tend to be a hard one. Asked of all hesitations the "
        "answer is confounded, because the word after a clause onset is nearly "
        "always a very common one."
    )
    return "\n".join(lines)


def placement_figure(
    rows: Dict[str, dict], baseline: dict, out_dir: Path, bins: int = 10
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _style(plt)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))

    ax = axes[0]
    edges = np.linspace(0, 1, bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2
    for name in ("Switchboard", "Hem", "Injector"):
        s = rows.get(name)
        if not s or not s.get("n"):
            continue
        counts, _ = np.histogram(s["relative_positions"], bins=edges)
        share = counts / counts.sum() if counts.sum() else counts
        ax.plot(centres, share, color=COLOURS[name], linewidth=1.4,
                marker="o", markersize=3.0, label=f"{name} (n={s['n']:,})")
    ax.axhline(1 / bins, color="#999999", linewidth=0.7, linestyle="--", dashes=(4, 2))
    ax.set_xlabel("Position in the sentence")
    ax.set_ylabel("Share of filled pauses")
    ax.set_title("Where in the sentence", pad=4)
    ax.legend(frameon=False, loc="best", handlelength=1.6, fontsize=6)

    ax = axes[1]
    names = [n for n in ("Switchboard", "Hem", "Injector") if rows.get(n, {}).get("n")]
    values = [rows[n]["mean_next_log_freq"] for n in names]
    ax.bar(range(len(names)), values, color=[COLOURS[n] for n in names], width=0.55)
    ax.axhline(baseline["mean_next_log_freq"], color="#999999", linewidth=0.9,
               linestyle="--", dashes=(4, 2))
    ax.text(len(names) - 0.5, baseline["mean_next_log_freq"], " a random word",
            va="bottom", ha="right", fontsize=6, color="#666666")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Mean log frequency of the next word")
    ax.set_title("What comes after, lower is harder", pad=4)
    ax.invert_yaxis()

    for ax in axes:
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "placement.png"
    fig.savefig(png, dpi=300)
    fig.savefig(out_dir / "placement.pdf")
    plt.close(fig)
    return png


# --------------------------------------------------------------------------
# 4. round trip through Mend
# --------------------------------------------------------------------------


class Mend:
    """Mend, loaded from the Hub, for the round trip."""

    def __init__(self, path: str = MEND_HUB, device: Optional[str] = None,
                 batch: int = 32) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        from hem.train import pick_device

        self.device = device or pick_device()
        self.batch = batch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)
        self.model.to(self.device).eval()
        torch.set_grad_enabled(False)
        self.torch = torch

    def clean(self, texts: Sequence[str], progress: bool = False) -> List[str]:
        out: List[str] = []
        for i in range(0, len(texts), self.batch):
            chunk = [MEND_PREFIX + t for t in texts[i : i + self.batch]]
            enc = self.tokenizer(chunk, return_tensors="pt", padding=True,
                                 truncation=True, max_length=192,
                                 pad_to_multiple_of=8).to(self.device)
            gen = self.model.generate(**enc, max_new_tokens=192, num_beams=1,
                                      do_sample=False)
            out.extend(self.tokenizer.batch_decode(gen, skip_special_tokens=True))
            if progress:
                print(f"  mend {min(i + self.batch, len(texts))}/{len(texts)}",
                      end="\r", file=sys.stderr)
            if self.device == "mps" and (i // self.batch) % 20 == 19:
                self.torch.mps.empty_cache()
        if progress:
            print(file=sys.stderr)
        return out


def words_only(text: str) -> str:
    import re

    return " ".join(
        w for w in re.sub(r"[^\w\s']|_", "", " ".join(text.split())).lower().split()
    )


def recovery(originals: Sequence[str], recovered: Sequence[str]) -> dict:
    """How much of the script Mend got back out of Hem's speech."""
    import jiwer

    exact = sum(
        1 for a, b in zip(originals, recovered)
        if " ".join(a.split()).rstrip(".!?") == " ".join(b.split()).rstrip(".!?")
    )
    refs = [words_only(a) or "@" for a in originals]
    outs = [words_only(b) or "@" for b in recovered]
    words_exact = sum(1 for a, b in zip(refs, outs) if a == b)
    return {
        "n": len(originals),
        "exact_match": exact / len(originals) if originals else 0.0,
        "words_only_match": words_exact / len(originals) if originals else 0.0,
        "wer_words_only": float(jiwer.wer(refs, outs)),
    }


def round_trip_table(results: List[dict]) -> str:
    header = (
        "| Through | Level | N | Script recovered exactly | "
        "Recovered up to casing and punctuation | Word error rate |"
    )
    lines = [header, "|" + "|".join("---" for _ in range(6)) + "|"]
    for r in results:
        level = "n/a" if r["level"] is None else f"`<d{r['level']}>`"
        lines.append(
            f"| {r['system']} | {level} | {r['n']:,} | {r['exact_match']:.3f} | "
            f"**{r['words_only_match']:.3f}** | {r['wer_words_only']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Hem writes the script out as speech at a given level, Mend reads that "
        "speech back as writing, and the recovered text is compared with the "
        "script Hem started from. The middle column ignores casing and "
        "punctuation, which is the fair comparison: Mend restores those from its "
        "own training distribution and getting a comma back in the wrong place "
        "is not a failure of the round trip."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 6. the qualitative table
# --------------------------------------------------------------------------


def qualitative_table(sentences: Sequence[str], by_level: Dict[int, List[str]]) -> str:
    lines = []
    for i, sentence in enumerate(sentences):
        lines.append(f"**{i + 1}.** `{sentence}`")
        lines.append("")
        for k in L.LEVELS:
            lines.append(f"    d{k}  {by_level[k][i]}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# plumbing
# --------------------------------------------------------------------------


def _style(plt) -> None:
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


def eval_sentences(path: Path, n: int, seed: int) -> List[str]:
    """Clean written sentences from the held-out synthetic split.

    The split is by *source sentence*, so nothing here was seen in training in
    any form, at any level.
    """
    rows = read_jsonl(path)
    seen = set()
    texts = []
    for row in rows:
        text = row["clean"]
        if text in seen:
            continue
        seen.add(text)
        texts.append(text)
    random.Random(seed).shuffle(texts)
    return texts[:n] if n else texts


def switchboard_reference(path: Path, limit: int, seed: int) -> Dict[int, dict]:
    """Per level: the corpus's own mix, detected and from its gold spans."""
    rows = read_jsonl(path)
    random.Random(seed).shuffle(rows)
    rows = rows[:limit] if limit else rows
    out: Dict[int, dict] = {}
    for level in L.LEVELS:
        here = [r for r in rows if r["level"] == level]
        detected = [insertion_records(r["clean"], r["disfluent"])[0] for r in here]
        gold = [gold_records(r) for r in here]
        flat_d = [x for rows_ in detected for x in rows_]
        flat_g = [x for rows_ in gold for x in rows_]
        cd, cg = Counter(x["type"] for x in flat_d), Counter(x["type"] for x in flat_g)
        words = sum(len(r["clean_tokens"]) for r in here)
        out[level] = {
            "utterances": len(here),
            "clean_words": words,
            "events": len(flat_d),
            "mix": {t: (cd[t] / len(flat_d) if flat_d else 0.0) for t in TYPES},
            "gold_mix": {t: (cg[t] / len(flat_g) if flat_g else 0.0) for t in TYPES},
            "rates": {t: cd[t] * 100.0 / words if words else 0.0 for t in TYPES},
            "records": flat_d,
            "gold_records": flat_g,
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate the dial")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--model", default="outputs/model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=1500,
                    help="clean sentences to run the dial over, per level")
    ap.add_argument("--swda-limit", dest="swda_limit", type=int, default=6000)
    ap.add_argument("--qualitative", type=int, default=30)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--skip-roundtrip", dest="skip_roundtrip", action="store_true")
    ap.add_argument("--skip-model", dest="skip_model", action="store_true",
                    help="score the rule-based injector alone")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    from hem.generate import Rules, load

    texts = eval_sentences(args.data / "test.jsonl", args.limit, args.seed)
    print(f"{len(texts)} held-out clean sentences", file=sys.stderr)

    systems: List[Tuple[str, object]] = [("Injector", Rules(seed=args.seed))]
    if not args.skip_model:
        systems.insert(0, ("Hem", load("model", args.model, device=args.device,
                                       batch=args.batch)))

    runs: Dict[Tuple[str, int], dict] = {}
    spoken: Dict[Tuple[str, int], List[str]] = {}
    fidelity: Dict[Tuple[str, int], dict] = {}
    for name, system in systems:
        for level in L.LEVELS:
            print(f"generating {name} at d{level}", file=sys.stderr)
            said = system.speak(texts, level)
            spoken[(name, level)] = said
            pairs = [insertion_records(c, s) for c, s in zip(texts, said)]
            runs[(name, level)] = summarise_run(texts, [p[0] for p in pairs])
            dropped = sum(p[1]["n_dropped"] for p in pairs)
            words = sum(p[1]["clean_words"] for p in pairs)
            fidelity[(name, level)] = {
                "dropped_words": dropped,
                "clean_words": words,
                "kept": 1 - dropped / words if words else 0.0,
                "sentences_with_a_dropped_word": sum(
                    1 for p in pairs if p[1]["n_dropped"]
                ),
            }

    reference = switchboard_reference(
        args.data / "real_eval.jsonl", args.swda_limit, args.seed
    )
    freqs = load_frequencies(args.data / "unigrams.json")

    report: Dict[str, object] = {
        "sentences": len(texts),
        "rate_control": {
            f"{s}/d{k}": {
                "rates": v["rates"], "total_rate": v["total_rate"],
                "target": L.targets(k), "mix": v["mix"], "events": v["events"],
            }
            for (s, k), v in runs.items()
        },
        "rate_error": {
            s: rate_error(runs, s) for s in {k[0] for k in runs}
        },
        "fidelity": {f"{s}/d{k}": v for (s, k), v in fidelity.items()},
        "switchboard": {
            f"d{k}": {x: v[x] for x in ("utterances", "events", "mix", "gold_mix", "rates")}
            for k, v in reference.items()
        },
    }

    # ---- 1 and 2
    print("\n## Rate control\n")
    print(rate_table(runs))
    rate_png = rate_figure(runs, args.out)
    print(f"\nwrote {rate_png}", file=sys.stderr)

    print("\n## Type mix\n")
    print(mix_table(runs, reference))
    report["type_mix_kl"] = {
        f"{s}/d{k}": kl_divergence(v["mix"], reference[k]["mix"])
        for (s, k), v in runs.items() if k > 0
    }
    report["type_mix_kl"]["Switchboard gold/d2"] = kl_divergence(
        reference[2]["gold_mix"], reference[2]["mix"]
    )

    # ---- 3
    natural = 2
    placement = {
        "Hem": placement_stats(runs[("Hem", natural)]["records"], freqs)
        if ("Hem", natural) in runs else {"n": 0},
        "Injector": placement_stats(runs[("Injector", natural)]["records"], freqs),
        "Switchboard": placement_stats(reference[natural]["records"], freqs),
    }
    base = baseline_placement(texts, freqs)
    print("\n## Placement\n")
    print(placement_table(placement, base))
    place_png = placement_figure(placement, base, args.out)
    print(f"\nwrote {place_png}", file=sys.stderr)
    report["placement"] = {
        name: {k: v for k, v in s.items() if k != "relative_positions"}
        for name, s in placement.items()
    }
    report["placement"]["random word"] = base

    # ---- 4
    if not args.skip_roundtrip:
        print("\n## Round trip\n", file=sys.stderr)
        mend = Mend(device=args.device, batch=args.batch)
        results = []
        for name, _ in systems:
            for level in L.LEVELS:
                back = mend.clean(spoken[(name, level)], progress=True)
                entry = {"system": f"Mend after {name}", "level": level,
                         **recovery(texts, back)}
                results.append(entry)
        # the ceiling: Mend on real speech, which is what it was trained for
        real = read_jsonl(args.data / "real_eval.jsonl")
        random.Random(args.seed).shuffle(real)
        real = real[: args.limit]
        back = mend.clean([r["disfluent"] for r in real], progress=True)
        results.append({"system": "Mend on real Switchboard", "level": None,
                        **recovery([r["clean"] for r in real], back)})
        print("\n## Round trip\n")
        print(round_trip_table(results))
        report["round_trip"] = results
        del mend

    # ---- 6
    sample = texts[: args.qualitative]
    by_level = {k: spoken[(systems[0][0], k)][: args.qualitative] for k in L.LEVELS}
    (args.out / "qualitative.md").write_text(
        f"# The dial on {len(sample)} sentences\n\n"
        f"Generated by {systems[0][0]}, held-out sentences, greedy decoding.\n\n"
        + qualitative_table(sample, by_level) + "\n"
    )
    print(f"\nwrote {args.out / 'qualitative.md'}", file=sys.stderr)

    # The annotated generations, for hem.surprisal to read.
    dump = args.out / "generated"
    dump.mkdir(parents=True, exist_ok=True)
    for (name, level), said in spoken.items():
        path = dump / f"{name.lower()}_d{level}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in annotated_rows(texts, said):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(spoken)} generation dumps to {dump}", file=sys.stderr)

    (args.out / "eval.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    tables = args.out / "eval_tables.md"
    parts = [
        "# Evaluation\n",
        "## Rate control\n", rate_table(runs), "",
        "## Type mix\n", mix_table(runs, reference), "",
        "## Placement\n", placement_table(placement, base), "",
    ]
    if not args.skip_roundtrip:
        parts += ["## Round trip\n", round_trip_table(report["round_trip"]), ""]
    tables.write_text("\n".join(parts) + "\n")
    print(f"wrote {tables} and {args.out / 'eval.json'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
