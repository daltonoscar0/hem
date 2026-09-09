"""Fine-tune the disfluency model.

A seq2seq fine-tune of flan-t5-small, the same base model and the same size as
Mend, so the two are comparable:

    input   <d2> say aloud: Send the report to Sarah by Friday.
    target  send the uh report to sarah no wait to sarah by friday

The four control tokens are added to the tokenizer as special tokens and the
embedding matrix is resized, so ``<d2>`` arrives as one piece. Left as ordinary
text it tokenises to ``<``, ``d``, ``2``, ``>``, which shares the ``d`` and the
``>`` between all four settings and makes the dial harder to learn than it needs
to be.

Usage::

    python -m hem.train --smoke      # 200 steps, checks the loss is moving
    python -m hem.train --real 25000 # the real run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)

from hem import levels as L
from hem.build_data import encode_source

DEFAULT_MODEL = "google/flan-t5-small"


# --------------------------------------------------------------------------
# hardware
# --------------------------------------------------------------------------


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_pairs(path: Path, limit: int = 0) -> List[dict]:
    """Read only the three fields training needs.

    The JSONL also carries tokens, alignments and span annotations for the
    evaluation and surprisal stages. Keeping all of that for 50k rows costs
    hundreds of megabytes for no benefit here.
    """
    rows: List[dict] = []
    if not Path(path).exists():
        return rows
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            rows.append(
                {
                    "source": encode_source(record["level"], record["clean"]),
                    "target": record["disfluent"],
                    "level": record["level"],
                }
            )
            if limit and len(rows) >= limit:
                break
    return rows


def mix_real_speech(
    synthetic: List[dict], path: Path, limit: int, seed: int
) -> Tuple[List[dict], int]:
    """Fold real transcribed speech into a synthetic training set.

    Everything the injector produces is, by construction, a sample from the
    injector's own placement rule: a filled pause goes at a clause onset or in
    front of a content word, chosen uniformly among the slots that qualify. Real
    speakers do not choose uniformly, and placement is the axis Hem is meant to
    win on, so the real rows are the only part of the mix that can teach it.
    """
    if limit == 0 or not path.exists():
        if limit and not path.exists():
            print(f"no {path}; training on synthetic pairs only", file=sys.stderr)
        return synthetic, 0
    real = load_pairs(path)
    random.Random(seed).shuffle(real)
    if limit > 0:
        real = real[:limit]
    mixed = synthetic + real
    random.Random(seed).shuffle(mixed)
    return mixed, len(real)


def to_dataset(rows: Sequence[dict]) -> Dataset:
    return Dataset.from_dict(
        {
            "source": [r["source"] for r in rows],
            "target": [r["target"] for r in rows],
        }
    )


def build_tokenize_fn(tokenizer, max_source: int, max_target: int):
    def fn(batch: Dict[str, List[str]]) -> Dict[str, List]:
        model_inputs = tokenizer(
            batch["source"], max_length=max_source, truncation=True
        )
        labels = tokenizer(
            text_target=batch["target"], max_length=max_target, truncation=True
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return fn


def prepare_tokenizer(name_or_path: str):
    """Load the tokenizer with the four control tokens registered."""
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    missing = [t for t in L.LEVEL_TOKENS if t not in tokenizer.get_vocab()]
    if missing:
        tokenizer.add_special_tokens({"additional_special_tokens": list(L.LEVEL_TOKENS)})
    return tokenizer


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------


class LossTrace(TrainerCallback):
    """Keeps the training losses so the smoke run can check they fall."""

    def __init__(self) -> None:
        self.losses: List[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))


class TrimMPSCache(TrainerCallback):
    """Drop the MPS allocator's cached blocks every so often.

    The allocator keeps a separate pool per tensor shape and does not hand
    memory back. With dynamically padded batches the number of distinct shapes
    keeps growing, so the pool grows all run and eventually pushes the machine
    into swap, which shows up as step times that get steadily worse.

    Mend trimmed every 100 steps and that held there. It did not hold here: a
    first run of this script degraded from 1.5 s/step to 25 s/step once
    something else on the machine wanted memory at the same time, with the
    training process paged almost entirely out. MPS allocations do not show up
    in RSS, so the pool is invisible to every tool that would have shown it.
    Trimming four times as often costs a fraction of a second per hundred steps
    and removes the failure.
    """

    def __init__(self, every: int = 25) -> None:
        self.every = every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every == 0:
            torch.mps.empty_cache()


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


@torch.no_grad()
def greedy_decode(
    model, tokenizer, sources: Sequence[str], device: str, max_new_tokens: int = 160
) -> List[str]:
    model.eval()
    out: List[str] = []
    for i in range(0, len(sources), 16):
        enc = tokenizer(
            list(sources[i : i + 16]), return_tensors="pt", padding=True,
            truncation=True, max_length=192,
        ).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, num_beams=1,
                             do_sample=False)
        out.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return out


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------


def train(args: argparse.Namespace) -> int:
    device = args.device or pick_device()
    print(f"device: {device}", file=sys.stderr)

    train_rows = load_pairs(args.data / "train.jsonl", args.max_train)
    val_rows = load_pairs(args.data / "val.jsonl")

    n_synthetic = len(train_rows)
    train_rows, n_real = mix_real_speech(
        train_rows, args.data / "real_train.jsonl", args.real, args.seed
    )
    # Validate on the same mixture the model is trained on. Selecting the best
    # checkpoint by synthetic loss alone would pick the one that best imitates
    # the injector, which is the thing we are trying to stop measuring.
    real_val = round(len(val_rows) * n_real / max(1, n_synthetic)) if n_real else 0
    val_rows, n_real_val = mix_real_speech(
        val_rows, args.data / "real_val.jsonl", real_val, args.seed + 1
    )
    if n_real:
        share = n_real / len(train_rows)
        print(
            f"training mix: {n_synthetic} synthetic + {n_real} real "
            f"({share:.0%} real); validation +{n_real_val} real",
            file=sys.stderr,
        )

    epochs = args.epochs
    if device == "cpu" and not args.smoke:
        # A full run on CPU is not worth the wall clock; shrink it and say so.
        train_rows = train_rows[:10_000]
        epochs = 1
        print("no GPU: cutting the training set to 10000 pairs and 1 epoch",
              file=sys.stderr)
    if args.smoke:
        train_rows = train_rows[: args.batch * args.smoke_steps]
        val_rows = val_rows[:200]
        epochs = 1

    from collections import Counter

    spread = Counter(r["level"] for r in train_rows)
    print(
        f"train pairs: {len(train_rows)}  val pairs: {len(val_rows)}  "
        + "  ".join(f"d{k}={spread[k]}" for k in L.LEVELS),
        file=sys.stderr,
    )

    tokenizer = prepare_tokenizer(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if len(tokenizer) != model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))
        print(f"resized embeddings to {len(tokenizer)} for the control tokens",
              file=sys.stderr)

    lr = args.lr if args.lr is not None else 5e-5

    tokenize = build_tokenize_fn(tokenizer, args.max_source, args.max_target)
    ds_train = to_dataset(train_rows).map(
        tokenize, batched=True, remove_columns=["source", "target"]
    )
    ds_val = to_dataset(val_rows).map(
        tokenize, batched=True, remove_columns=["source", "target"]
    )

    out_dir = args.out / ("smoke" if args.smoke else "checkpoints")
    steps_per_epoch = max(1, math.ceil(len(ds_train) / args.batch))
    eval_every = args.eval_steps or max(50, min(500, steps_per_epoch // 2))

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        max_steps=args.smoke_steps if args.smoke else -1,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        learning_rate=lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=args.log_steps,
        eval_strategy="steps",
        eval_steps=eval_every,
        save_strategy="no" if args.smoke else "steps",
        save_steps=eval_every,
        save_total_limit=2,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=(device == "cuda"),
        # Step time here is dominated by sequence length rather than batch
        # size, so batching similar lengths together is a large win.
        group_by_length=True,
        # MPS and CPU both want plain fp32 here; half precision on MPS is not
        # reliable for T5.
        fp16=(device == "cuda"),
        disable_tqdm=False,
    )

    trace = LossTrace()
    callbacks: List[TrainerCallback] = [trace]
    if device == "mps":
        callbacks.append(TrimMPSCache())
    if not args.smoke:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        # Quantising the padded width keeps the number of distinct tensor
        # shapes small, which the MPS allocator is much happier with.
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    resume = args.resume or None
    if resume == "auto":
        existing = sorted(
            out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])
        )
        resume = str(existing[-1]) if existing else None
        if resume:
            print(f"resuming from {resume}", file=sys.stderr)
    result = trainer.train(resume_from_checkpoint=resume)
    metrics = trainer.evaluate()
    final_eval_loss = metrics["eval_loss"]

    if args.smoke:
        losses = trace.losses
        if len(losses) < 2:
            print("smoke run produced too few log points to judge", file=sys.stderr)
            return 1
        head = sum(losses[:2]) / 2
        tail = sum(losses[-2:]) / 2
        print(f"\nsmoke: first logged loss {head:.4f} -> last {tail:.4f}")
        print(f"smoke: eval loss {final_eval_loss:.4f}")
        if tail >= head:
            print("smoke: loss did not fall, aborting", file=sys.stderr)
            return 1
        print("smoke: loss is falling")
        sample = "Send the quarterly report to Sarah before Friday."
        sources = [encode_source(k, sample) for k in L.LEVELS]
        for src, hyp in zip(sources, greedy_decode(model, tokenizer, sources, device)):
            print(f"  {src}\n  -> {hyp}")
        return 0

    # Point outputs/model at the best checkpoint.
    best = trainer.state.best_model_checkpoint
    link = args.out / "model"
    if best:
        target = Path(best).resolve()
        if link.is_symlink() or link.exists():
            if link.is_symlink():
                link.unlink()
            else:
                import shutil

                shutil.rmtree(link)
        link.symlink_to(os.path.relpath(target, args.out))
        print(f"best checkpoint: {target}", file=sys.stderr)
    else:
        trainer.save_model(str(link))

    history = [h for h in trainer.state.log_history if "eval_loss" in h]
    first_eval = history[0]["eval_loss"] if history else float("nan")
    best_eval = min(h["eval_loss"] for h in history) if history else float("nan")
    summary = {
        "device": device,
        "model": args.model,
        "learning_rate": lr,
        "batch_size": args.batch,
        "epochs": epochs,
        "train_pairs": len(train_rows),
        "synthetic_pairs": n_synthetic,
        "real_pairs": n_real,
        "real_share": round(n_real / max(1, len(train_rows)), 4),
        "level_spread": {f"d{k}": spread[k] for k in L.LEVELS},
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "first_eval_loss": first_eval,
        "best_eval_loss": best_eval,
        "final_train_loss": result.metrics.get("train_loss"),
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not math.isnan(first_eval) and best_eval >= first_eval:
        print("validation loss did not improve", file=sys.stderr)
        return 1

    print("\nthe dial on one sentence:\n")
    sample = "Send the quarterly report to Sarah before Friday and copy Michael."
    sources = [encode_source(k, sample) for k in L.LEVELS]
    for k, hyp in zip(L.LEVELS, greedy_decode(model, tokenizer, sources, device)):
        print(f"  d{k}: {hyp}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the Hem disfluency model")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-source", dest="max_source", type=int, default=128)
    ap.add_argument("--max-target", dest="max_target", type=int, default=160)
    ap.add_argument("--max-train", dest="max_train", type=int, default=0)
    ap.add_argument(
        "--real",
        type=int,
        default=0,
        help="mix in up to N real Switchboard pairs (-1 for all, 0 for none)",
    )
    ap.add_argument("--log-steps", dest="log_steps", type=int, default=20)
    ap.add_argument("--eval-steps", dest="eval_steps", type=int, default=0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument(
        "--resume",
        default=None,
        help='checkpoint to carry on from, or "auto" for the latest one',
    )
    ap.add_argument("--smoke", action="store_true", help="short run to check the loss falls")
    ap.add_argument("--smoke-steps", dest="smoke_steps", type=int, default=200)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
