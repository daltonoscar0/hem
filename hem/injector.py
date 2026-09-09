"""Rule-based disfluency injector: the trivial baseline Hem has to beat.

The structure is Shriberg's, the same one Mend's injector uses, because Hem is
Mend run backwards and the two have to agree on what a self-repair is:

    send it to  john   no wait     sarah
                ^^^^   ^^^^^^^     ^^^^^
                reparandum         repair
                       interregnum

The difference from Mend's version is how much gets injected. Mend wanted one
fixed distribution of disfluent training data, so its rates are per-sentence
probabilities. Hem has a dial, and the dial's settings are rates measured off
Switchboard in events per 100 words, so the rates here are per-100-word and the
per-sentence counts are drawn from them. That makes the baseline hit its rate
targets on long and short sentences alike, which it has to, because rate control
is one of the things the model is being scored against it on.

Every injected disfluency is recorded with its span offsets on the disfluent
side, so reparandum / interregnum / repair boundaries stay recoverable. The
disfluent side is lowercased and stripped of punctuation, which is the register
Mend reads, so Hem's output can be fed straight back into Mend.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# lexicons
# --------------------------------------------------------------------------

#: Filled pauses and discourse markers, with the weight each carries in real
#: speech. The weights are Switchboard's, counted by ``python -m hem.rates`` off
#: the same parse the dial comes from and renormalised over the nine phrases
#: kept here. Two thirds of the corpus's filled pauses are "uh" or "you know",
#: so a uniform draw over this list would produce far too many "like"s, and
#: "like" is the one a reader notices.
FILLERS: Tuple[Tuple[Tuple[str, ...], float], ...] = (
    (("uh",), 0.445),
    (("you", "know"), 0.256),
    (("well",), 0.097),
    (("um",), 0.090),
    (("i", "mean"), 0.042),
    (("like",), 0.039),
    (("oh",), 0.019),
    (("now",), 0.007),
    (("actually",), 0.005),
)

FILLER_PHRASES: Tuple[Tuple[str, ...], ...] = tuple(p for p, _ in FILLERS)
FILLER_WEIGHTS: Tuple[float, ...] = tuple(w for _, w in FILLERS)
FILLER_VOCAB = frozenset(w for phrase in FILLER_PHRASES for w in phrase)

INTERREGNA: Tuple[Tuple[str, ...], ...] = (
    ("i", "mean"),
    ("no", "wait"),
    ("sorry",),
    ("or", "rather"),
    ("no",),
    ("uh",),
)

FALSE_START_FRAGMENTS: Tuple[Tuple[str, ...], ...] = (
    ("can", "you"),
    ("could", "you"),
    ("i", "want", "to"),
    ("i", "need", "to"),
    ("we", "should"),
    ("do", "you"),
    ("let", "me"),
    ("i", "was", "going", "to"),
    ("it", "is"),
    ("there", "are"),
    ("what", "i"),
)

# Swap groups for repairs. A reparandum is drawn from the same group as the
# word it stands in for, which is what makes the wrong version plausible.
SWAP_GROUPS: Tuple[Tuple[str, ...], ...] = (
    # people
    (
        "john", "sarah", "michael", "emily", "david", "rachel", "daniel",
        "laura", "kevin", "anna", "peter", "maria", "tom", "julia", "chris",
        "nina", "omar", "priya", "hannah", "marcus",
    ),
    # days
    ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"),
    # months
    (
        "january", "february", "march", "april", "june", "july",
        "august", "september", "october", "november", "december",
    ),
    # number words
    (
        "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "twenty", "thirty", "fifty",
    ),
    # ordinals
    ("first", "second", "third", "fourth", "fifth"),
    # colours
    ("red", "blue", "green", "black", "white", "yellow", "purple", "orange", "grey"),
    # places around a building
    ("office", "kitchen", "garage", "basement", "hallway", "lobby", "warehouse", "studio"),
    # documents
    ("report", "invoice", "contract", "summary", "proposal", "agenda", "memo", "draft"),
    # time of day
    ("morning", "afternoon", "evening", "tonight", "tomorrow"),
    # directions
    ("north", "south", "east", "west"),
    # channels
    ("email", "message", "call", "text", "letter"),
)

_SWAP_INDEX: Dict[str, int] = {}
for _gi, _group in enumerate(SWAP_GROUPS):
    for _w in _group:
        _SWAP_INDEX[_w] = _gi

STOPWORDS = frozenset(
    """a an the and or but if of to in on at by for with from as is are was were
    be been being do does did have has had i you he she it we they me him her us
    them my your his its our their this that these those not no so then than
    there here about into over under up down out off just very can could would
    should will shall may might must""".split()
)

_STRIP_CHARS = " \t\n.,;:!?\"'()[]{}<>—–-…‘’“”"
_QUOTE_TABLE = str.maketrans("", "", "\"“”«»")
_HAS_ALNUM = re.compile(r"[0-9a-zA-Z]")

#: The four disfluency types, in the order the type-mix table reports them.
TYPES: Tuple[str, ...] = ("filler", "repetition", "repair", "false_start")

#: What each type is called in prose, for figures and tables.
TYPE_LABELS: Dict[str, str] = {
    "filler": "Filled pause",
    "repetition": "Repetition",
    "repair": "Substitution",
    "false_start": "Restart",
}


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """Half-open token span ``[start, end)`` on the disfluent side."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start

    def indices(self) -> range:
        return range(self.start, self.end)


@dataclass
class Annotation:
    """One injected disfluency.

    ``reparandum`` is the material the speaker abandons, ``interregnum`` the
    editing phrase between the two, ``repair`` the material that replaces the
    reparandum. Types that lack a component leave it as ``None``: a filled
    pause is interregnum only, a repetition has no interregnum.
    """

    type: str
    reparandum: Optional[Span] = None
    interregnum: Optional[Span] = None
    repair: Optional[Span] = None

    def deleted_indices(self) -> List[int]:
        """Disfluent token positions that should not survive cleanup."""
        out: List[int] = []
        for span in (self.reparandum, self.interregnum):
            if span is not None:
                out.extend(span.indices())
        return out

    def boundary_index(self) -> Optional[int]:
        """The token at which a listener learns something went wrong.

        That is the start of the interregnum when there is one, otherwise the
        start of the repair, otherwise the start of the reparandum.
        """
        for span in (self.interregnum, self.repair, self.reparandum):
            if span is not None:
                return span.start
        return None

    def inserted_length(self) -> int:
        """How many spoken tokens this disfluency added."""
        return len(self.deleted_indices())


@dataclass
class InjectionResult:
    disfluent_text: str
    clean_text: str
    disfluent_tokens: List[str] = field(default_factory=list)
    clean_tokens: List[str] = field(default_factory=list)
    # alignment[i] is the clean-token index that disfluent token i came from,
    # or None when the token was injected.
    alignment: List[Optional[int]] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "disfluent": self.disfluent_text,
            "clean": self.clean_text,
            "disfluent_tokens": list(self.disfluent_tokens),
            "clean_tokens": list(self.clean_tokens),
            "alignment": list(self.alignment),
            "annotations": [asdict(a) for a in self.annotations],
        }

    @staticmethod
    def from_dict(d: dict) -> "InjectionResult":
        def _span(x) -> Optional[Span]:
            return None if x is None else Span(x["start"], x["end"])

        anns = [
            Annotation(
                type=a["type"],
                reparandum=_span(a.get("reparandum")),
                interregnum=_span(a.get("interregnum")),
                repair=_span(a.get("repair")),
            )
            for a in d.get("annotations", [])
        ]
        return InjectionResult(
            disfluent_text=d["disfluent"],
            clean_text=d["clean"],
            disfluent_tokens=list(d.get("disfluent_tokens", [])),
            clean_tokens=list(d.get("clean_tokens", [])),
            alignment=list(d.get("alignment", [])),
            annotations=anns,
        )


@dataclass
class InjectionConfig:
    """How much to inject, and in what shape.

    The four rates are **events per 100 clean words**, which is the unit
    ``hem.rates`` measures Switchboard in, so a level's settings can be read
    straight off the corpus. Per-sentence counts are drawn from them, so a
    forty-word sentence gets about twice what a twenty-word one does.

    The shape parameters below the rates say what a disfluency of each type
    looks like once its position is chosen. Those are also measured, except
    where noted.
    """

    filler_per_100w: float = 0.0
    repetition_per_100w: float = 0.0
    repair_per_100w: float = 0.0
    false_start_per_100w: float = 0.0
    # Measured, by `python -m hem.rates`: 20.4% of Switchboard repairs carry an
    # editing phrase, 56.0% have a reparandum of two tokens or more, and 16.1%
    # of repetitions cover two words rather than one.
    interregnum_rate: float = 0.204
    repetition_bigram_rate: float = 0.161
    span_repair_rate: float = 0.560
    max_repair_span: int = 3
    # Not measured. The SwDA parser closes a reparandum over min..max of the
    # words it covers, so a gap between the two halves of a repair is not
    # recoverable from the parse and neither of these two can be counted off it.
    # They are Mend's injector's values, kept so the two generators agree.
    span_drop_rate: float = 0.50
    delayed_repair_rate: float = 0.15
    max_delay: int = 6
    # given a restart, how often it happens at a later clause boundary rather
    # than at the start of the sentence
    false_start_midsentence_rate: float = 0.40
    lowercase: bool = True
    strip_punctuation: bool = True

    @staticmethod
    def from_dict(d: Optional[dict]) -> "InjectionConfig":
        if not d:
            return InjectionConfig()
        known = set(InjectionConfig.__dataclass_fields__)
        return InjectionConfig(**{k: v for k, v in d.items() if k in known})


# --------------------------------------------------------------------------
# tokenisation helpers
# --------------------------------------------------------------------------


def tokenize(text: str) -> List[str]:
    """Whitespace tokenisation, keeping punctuation attached to its word."""
    return text.split()


def spoken_form(token: str, cfg: Optional[InjectionConfig] = None) -> str:
    """Render one written token the way an ASR transcript would show it."""
    cfg = cfg or InjectionConfig()
    out = token.translate(_QUOTE_TABLE)
    if cfg.strip_punctuation:
        out = out.strip(_STRIP_CHARS)
    if cfg.lowercase:
        out = out.lower()
    return out


def canonical_clean_tokens(text: str, cfg: Optional[InjectionConfig] = None) -> List[str]:
    """Clean tokens that survive to the spoken side.

    Tokens with no alphanumeric content (a stray em dash, say) have no spoken
    form, so they are dropped and the clean text is rebuilt without them.
    """
    return [t for t in tokenize(text) if _HAS_ALNUM.search(t)]


def spoken_text(text: str, cfg: Optional[InjectionConfig] = None) -> str:
    """The whole of ``text`` in spoken register, with nothing injected.

    This is what level 0 produces, and what the round-trip evaluation compares
    against: Hem's job at every level is to add disfluency to *this*, not to
    change the words or the word order.
    """
    words = [spoken_form(t, cfg) for t in canonical_clean_tokens(text, cfg)]
    return " ".join(w for w in words if w)


def _is_content_word(token: str) -> bool:
    stripped = token.strip(_STRIP_CHARS).lower()
    return len(stripped) > 3 and stripped not in STOPWORDS


def _ends_clause(token: str) -> bool:
    return token.rstrip("\"”'’)").endswith((",", ";", ":"))


# --------------------------------------------------------------------------
# drawing counts from a rate
# --------------------------------------------------------------------------


def draw_count(rate_per_100w: float, n_words: int, rng: random.Random) -> int:
    """How many events of one type this sentence should get.

    The expected count is ``rate * n / 100``. Rounding it would quantise the
    dial badly: at 3 events per 100 words a twelve-word sentence expects 0.36
    and would always round to zero, so short sentences would come back clean
    however high the dial went. Taking the integer part and adding one more
    with probability equal to the remainder gives the right expectation at
    every sentence length, which is what makes the measured rate track the
    target.
    """
    if rate_per_100w <= 0 or n_words <= 0:
        return 0
    expected = rate_per_100w * n_words / 100.0
    whole = int(expected)
    return whole + (1 if rng.random() < expected - whole else 0)


def _weighted_filler(rng: random.Random) -> Tuple[str, ...]:
    return rng.choices(FILLER_PHRASES, weights=FILLER_WEIGHTS, k=1)[0]


# --------------------------------------------------------------------------
# repair material
# --------------------------------------------------------------------------


def _alternative_for(word: str, rng: random.Random) -> Optional[str]:
    """A wrong-but-plausible stand-in drawn from the same category."""
    key = word.strip(_STRIP_CHARS).lower()
    if not key:
        return None
    gi = _SWAP_INDEX.get(key)
    if gi is not None:
        options = [w for w in SWAP_GROUPS[gi] if w != key]
        return rng.choice(options) if options else None
    if key.isdigit():
        n = int(key)
        deltas = [d for d in (-3, -2, -1, 1, 2, 3, 10, -10) if n + d > 0]
        if not deltas:
            return None
        return str(n + rng.choice(deltas))
    return None


def _swappable_positions(clean_tokens: Sequence[str]) -> List[int]:
    out = []
    for i, tok in enumerate(clean_tokens):
        key = tok.strip(_STRIP_CHARS).lower()
        if key in _SWAP_INDEX or key.isdigit():
            out.append(i)
    return out


def _span_reparandum(
    spoken: Sequence[str],
    clean_tokens: Sequence[str],
    pos: int,
    length: int,
    rng: random.Random,
    allow_drop: bool,
) -> Optional[List[str]]:
    """Build the abandoned version of a multi-token span.

    Two ways a speaker gets a phrase wrong and says it again. Either a word
    inside it was wrong ("the annual report ... the quarterly report"), or the
    first attempt was less specific and a modifier gets added on the second
    pass ("the report ... the quarterly report"). Both leave the reparandum and
    the repair sharing most of their words, which is what real span repairs
    mostly look like.

    The drop path needs no closed word category, so it is the one that lets an
    arbitrary scripted sentence carry a repair at all. Without it the injector
    could only put a repair where the text happened to name a day or a colour,
    and most scripted text does not.
    """
    span = list(spoken[pos : pos + length])
    if length < 2:
        return None

    if allow_drop:
        inner = list(range(1, length))
        if inner:
            k = rng.choice(inner)
            shorter = span[:k] + span[k + 1 :]
            if shorter and shorter != span:
                return shorter

    swappable = [
        k for k in range(length) if _alternative_for(clean_tokens[pos + k], rng) is not None
    ]
    if not swappable:
        return None
    k = rng.choice(swappable)
    alt = _alternative_for(clean_tokens[pos + k], rng)
    if alt is None:
        return None
    span[k] = alt
    return span


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def _repair_positions(clean_tokens: Sequence[str]) -> List[int]:
    """Where a repair can start.

    A swap-group word can carry a single-token substitution. Anywhere with two
    tokens of room and a content word in them can carry a span repair, which is
    what keeps the injector usable on text that names no days or colours.
    """
    swappable = set(_swappable_positions(clean_tokens))
    out = list(swappable)
    n = len(clean_tokens)
    for p in range(n - 1):
        if p in swappable:
            continue
        if any(_is_content_word(clean_tokens[q]) for q in range(p, min(p + 3, n))):
            out.append(p)
    return sorted(out)


def _plan_edits(
    clean_tokens: Sequence[str], cfg: InjectionConfig, rng: random.Random
) -> Tuple[List[dict], List[Tuple[int, Tuple[str, ...]]]]:
    """Choose non-overlapping edits over clean-token positions."""
    n = len(clean_tokens)
    edits: List[dict] = []
    reserved: set = set()
    spoken = [spoken_form(t, cfg) for t in clean_tokens]

    # ---- restarts: an abandoned clause opener, then the speaker starts again
    fragments: List[Tuple[int, Tuple[str, ...]]] = []
    if n >= 6:
        for _ in range(draw_count(cfg.false_start_per_100w, n, rng)):
            boundaries = [0] + [
                p for p in range(1, n - 3) if _ends_clause(clean_tokens[p - 1])
            ]
            boundaries = [b for b in boundaries if not any(
                q in reserved for q in range(b, min(b + 3, n))
            )]
            if not boundaries:
                break
            if len(boundaries) > 1 and rng.random() < cfg.false_start_midsentence_rate:
                at = rng.choice(boundaries[1:])
            else:
                at = boundaries[0]
            here = " ".join(
                t.strip(_STRIP_CHARS).lower() for t in clean_tokens[at : at + 2]
            )
            options = [f for f in FALSE_START_FRAGMENTS if " ".join(f) != here]
            fragments.append((at, rng.choice(options)))
            # keep the restart's first few positions clear so it resumes cleanly
            reserved.update(range(at, min(at + 3, n)))

    # ---- repairs
    for _ in range(draw_count(cfg.repair_per_100w, n, rng)):
        candidates = [p for p in _repair_positions(clean_tokens) if p not in reserved]
        if not candidates:
            break
        pos = rng.choice(candidates)
        placed = False

        if rng.random() < cfg.span_repair_rate:
            max_len = min(cfg.max_repair_span, n - pos)
            for length in sorted(range(2, max_len + 1), key=lambda _: rng.random()):
                if any(q in reserved for q in range(pos, pos + length)):
                    continue
                wrong_span = _span_reparandum(
                    spoken,
                    clean_tokens,
                    pos,
                    length,
                    rng,
                    allow_drop=rng.random() < cfg.span_drop_rate,
                )
                if wrong_span is None:
                    continue
                edits.append(
                    {
                        "kind": "span_repair",
                        "pos": pos,
                        "length": length,
                        "wrong": wrong_span,
                        "interregnum": (
                            rng.choice(INTERREGNA)
                            if rng.random() < cfg.interregnum_rate
                            else None
                        ),
                    }
                )
                reserved.update(range(pos, pos + length))
                placed = True
                break

        if placed:
            continue

        wrong = _alternative_for(clean_tokens[pos], rng)
        if wrong is None:
            # no single-token alternative here; try the span path once more
            wrong_span = (
                _span_reparandum(spoken, clean_tokens, pos, 2, rng, allow_drop=True)
                if pos + 2 <= n and not any(q in reserved for q in (pos, pos + 1))
                else None
            )
            if wrong_span is None:
                continue
            edits.append(
                {
                    "kind": "span_repair",
                    "pos": pos,
                    "length": 2,
                    "wrong": wrong_span,
                    "interregnum": (
                        rng.choice(INTERREGNA)
                        if rng.random() < cfg.interregnum_rate
                        else None
                    ),
                }
            )
            reserved.update((pos, pos + 1))
            continue

        interregnum = (
            rng.choice(INTERREGNA) if rng.random() < cfg.interregnum_rate else None
        )
        # A delayed repair: the speaker carries on for a few words before going
        # back and correcting, so the correction lands after the material it
        # belongs in front of.
        delay = 0
        if rng.random() < cfg.delayed_repair_rate:
            room = min(cfg.max_delay, n - pos - 1)
            options = [
                d
                for d in range(2, room + 1)
                if not any(q in reserved for q in range(pos, pos + d + 1))
            ]
            if options:
                delay = rng.choice(options)
        if delay:
            # a delayed correction needs an editing phrase, or a listener has
            # no way to tell a repair from a list
            edits.append(
                {
                    "kind": "delayed_repair",
                    "pos": pos,
                    "delay": delay,
                    "wrong": wrong,
                    "interregnum": interregnum or rng.choice(INTERREGNA),
                }
            )
            reserved.update(range(pos, pos + delay + 1))
        else:
            edits.append(
                {"kind": "repair", "pos": pos, "wrong": wrong, "interregnum": interregnum}
            )
            reserved.add(pos)

    # ---- repetitions
    if n >= 3:
        for _ in range(draw_count(cfg.repetition_per_100w, n, rng)):
            length = 2 if (n >= 5 and rng.random() < cfg.repetition_bigram_rate) else 1
            candidates = [
                p
                for p in range(n - length + 1)
                if not any(q in reserved for q in range(p, p + length))
            ]
            if not candidates:
                break
            pos = rng.choice(candidates)
            edits.append({"kind": "repetition", "pos": pos, "length": length})
            reserved.update(range(pos, pos + length))

    # ---- filled pauses, inserted *before* a clean token.
    # Placement follows Shriberg: a filler goes at a clause onset or in front of
    # a content word, never in the middle of a settled phrase. A filler is also
    # never placed next to a literal use of the same word, since "like
    # like-minded people" is noise rather than a disfluency.
    def _clashes(p: int) -> bool:
        neighbours = spoken[max(0, p - 1) : p + 2]
        return any(w in FILLER_VOCAB for w in neighbours)

    slots = [
        p
        for p in range(n)
        if p not in reserved
        and not _clashes(p)
        and (p == 0 or _ends_clause(clean_tokens[p - 1]) or _is_content_word(clean_tokens[p]))
    ]
    rng.shuffle(slots)
    for i in range(min(draw_count(cfg.filler_per_100w, n, rng), len(slots))):
        edits.append({"kind": "filler", "pos": slots[i], "words": _weighted_filler(rng)})

    order = {"filler": 0, "repair": 1, "span_repair": 1, "delayed_repair": 1, "repetition": 1}
    edits.sort(key=lambda e: (e["pos"], order[e["kind"]]))
    return edits, fragments


# --------------------------------------------------------------------------
# the injector
# --------------------------------------------------------------------------


def inject(
    clean_text: str,
    config: Optional[InjectionConfig] = None,
    rng_seed: int = 0,
) -> InjectionResult:
    """Render ``clean_text`` as disfluent speech with annotated repair spans."""
    cfg = config or InjectionConfig()
    rng = random.Random(rng_seed)

    clean_tokens = canonical_clean_tokens(clean_text, cfg)
    if not clean_tokens:
        return InjectionResult(disfluent_text="", clean_text="")

    spoken = [spoken_form(t, cfg) for t in clean_tokens]
    edits, fragments = _plan_edits(clean_tokens, cfg, rng)
    by_pos: Dict[int, List[dict]] = {}
    for e in edits:
        by_pos.setdefault(e["pos"], []).append(e)
    fragment_at: Dict[int, Tuple[str, ...]] = dict(fragments)

    out: List[str] = []
    align: List[Optional[int]] = []
    anns: List[Annotation] = []

    def emit(token: str, source: Optional[int]) -> None:
        out.append(token)
        align.append(source)

    def emit_interregnum(words: Optional[Sequence[str]]) -> Optional[Span]:
        if not words:
            return None
        start = len(out)
        for w in words:
            emit(w, None)
        return Span(start, len(out))

    pending: Optional[dict] = None

    for i, tok in enumerate(spoken):
        if i in fragment_at:
            start = len(out)
            for w in fragment_at[i]:
                emit(w, None)
            anns.append(
                Annotation(
                    type="false_start",
                    reparandum=Span(start, len(out)),
                    # the restart resumes on this clean token, emitted next
                    repair=Span(len(out), len(out) + min(3, len(clean_tokens) - i)),
                )
            )

        for e in by_pos.get(i, []):
            if e["kind"] == "filler":
                span = emit_interregnum(e["words"])
                anns.append(Annotation(type="filler", interregnum=span))
            elif e["kind"] == "repair":
                rep_start = len(out)
                emit(e["wrong"], None)
                rep_end = len(out)
                inter_span = emit_interregnum(e["interregnum"])
                anns.append(
                    Annotation(
                        type="repair",
                        reparandum=Span(rep_start, rep_end),
                        interregnum=inter_span,
                        repair=Span(len(out), len(out) + 1),
                    )
                )
            elif e["kind"] == "span_repair":
                rep_start = len(out)
                for w in e["wrong"]:
                    emit(w, None)
                rep_end = len(out)
                inter_span = emit_interregnum(e["interregnum"])
                anns.append(
                    Annotation(
                        type="repair",
                        reparandum=Span(rep_start, rep_end),
                        interregnum=inter_span,
                        repair=Span(len(out), len(out) + e["length"]),
                    )
                )
            elif e["kind"] == "delayed_repair":
                rep_start = len(out)
                emit(e["wrong"], None)
                pending = {
                    "pos": i,
                    "until": i + e["delay"],
                    "interregnum": e["interregnum"],
                    "reparandum": Span(rep_start, len(out)),
                }
            elif e["kind"] == "repetition":
                rep_start = len(out)
                for k in range(e["length"]):
                    emit(spoken[i + k], None)
                rep_end = len(out)
                anns.append(
                    Annotation(
                        type="repetition",
                        reparandum=Span(rep_start, rep_end),
                        repair=Span(rep_end, rep_end + e["length"]),
                    )
                )

        # A delayed repair holds its corrected token back until the speaker
        # circles round to it; everything else is emitted in place.
        if not (pending is not None and pending["pos"] == i):
            emit(tok, i)

        if pending is not None and i == pending["until"]:
            inter_span = emit_interregnum(pending["interregnum"])
            fix_start = len(out)
            emit(spoken[pending["pos"]], pending["pos"])
            anns.append(
                Annotation(
                    type="repair",
                    reparandum=pending["reparandum"],
                    interregnum=inter_span,
                    repair=Span(fix_start, len(out)),
                )
            )
            pending = None

    anns.sort(key=lambda a: a.boundary_index() or 0)
    return InjectionResult(
        disfluent_text=" ".join(out),
        clean_text=" ".join(clean_tokens),
        disfluent_tokens=out,
        clean_tokens=clean_tokens,
        alignment=align,
        annotations=anns,
    )


# --------------------------------------------------------------------------
# inverse, used by the tests and by the evaluation code
# --------------------------------------------------------------------------


def remove_disfluencies(result: InjectionResult) -> str:
    """Delete annotated reparandum/interregnum spans and restore the writing.

    This is the inverse of :func:`inject`: it reproduces ``result.clean_text``
    exactly, casing and punctuation included.

    The surviving tokens are put back in clean-token order rather than the
    order they were spoken. That matters for a delayed repair, where the
    correction arrives after the words it belongs in front of: deleting the
    reparandum alone would leave the corrected word stranded at the end.
    """
    drop = set()
    for ann in result.annotations:
        drop.update(ann.deleted_indices())
    kept = sorted(
        result.alignment[i]
        for i in range(len(result.disfluent_tokens))
        if i not in drop and result.alignment[i] is not None
    )
    return " ".join(result.clean_tokens[j] for j in kept)
