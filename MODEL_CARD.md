---
license: mit
language:
  - en
base_model: google/flan-t5-small
library_name: transformers
tags:
  - disfluency
  - speech
  - text-to-speech
  - controllable-generation
datasets:
  - swda
---

# hem-flan-t5-small

Turns clean scripted text into realistically disfluent speech, on a four
position dial. It is the inverse of
[`daltonoscar0/mend-flan-t5-small`](https://huggingface.co/daltonoscar0/mend-flan-t5-small),
which removes fillers and self-repairs from dictation.

Code, evaluation and the measurements behind the dial:
[github.com/daltonoscar0/hem](https://github.com/daltonoscar0/hem)

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("daltonoscar0/hem-flan-t5-small")
tok = AutoTokenizer.from_pretrained("daltonoscar0/hem-flan-t5-small")

text = "Send the quarterly report to Sarah before Friday."
ids = tok("<d2> say aloud: " + text, return_tensors="pt")
print(tok.decode(model.generate(**ids, max_new_tokens=192)[0], skip_special_tokens=True))
```

The input format is not optional: a control token, then `say aloud: `, then the
clean written sentence. The four control tokens were added to the tokenizer as
special tokens during fine-tuning, so a tokenizer loaded from anywhere else will
split them into pieces and the dial will not work.

## The dial

| Token | Setting | Target rate |
|---|---|---|
| `<d0>` | clean | nothing inserted |
| `<d1>` | light | 6.7 events per 100 words, one per 15 |
| `<d2>` | natural | 14.2 events per 100 words, one per 7 |
| `<d3>` | heavy | 28.1 events per 100 words, one per 4 |

The rates are not chosen. They are measured off the Switchboard Dialog Act
corpus: `<d1>` is the median disfluent utterance and below, `<d2>` runs from the
median to the 90th percentile, `<d3>` is the 90th percentile and above. The
per-type breakdown and how it was counted are in the repository README.

## Output register

Output is lowercased with punctuation stripped, which is the register Mend
reads. That is deliberate: the two models are inverses and their output and
input formats have to match.

## Training

flan-t5-small fine-tuned on 50,000 synthetic pairs from a rule-based
Shriberg-structured injector plus 24,619 real Switchboard utterances read
backwards (33% real), 2 epochs, batch 32, learning rate 5e-5, greedy decoding.
Every pair is labelled by its own measured disfluency rate rather than by the
setting it was generated at.

The Switchboard split is by conversation, so the conversations the model was
evaluated on were never trained on.

## Limitations

Read the repository README's "Where it fails" section before using this. In
short: the register is American telephone conversation, so the filled pauses it
inserts can be wrong for a formal script; it does not fix agreement or
determiners across a repair it has just created; restarts are the type it
handles worst; and it will occasionally drop or alter a word of the script
rather than only adding to it, which matters if the script is something someone
has to say verbatim.

## Licence

MIT.
