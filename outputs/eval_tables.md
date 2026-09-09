# Evaluation

## Rate control

| System | Level | Filled pause | Repetition | Substitution | Restart | All types | Target | Ratio |
|---|---|---|---|---|---|---|---|---|
| Injector | `<d0>` | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 | n/a |
| Injector | `<d1>` | 3.98 | 2.24 | 0.50 | 0.13 | **6.85** | 6.67 | 1.03 |
| Injector | `<d2>` | 7.68 | 4.78 | 1.23 | 0.59 | **14.27** | 14.20 | 1.00 |
| Injector | `<d3>` | 12.76 | 10.63 | 3.31 | 1.38 | **28.08** | 28.06 | 1.00 |

Events per 100 clean words, counted by the detector. Target is the rate measured off Switchboard for that bucket. Ratio is measured over target, so 1.00 is the dial landing where it was aimed.

## Type mix

| System | Level | Filled pause | Repetition | Substitution | Restart | KL from Switchboard |
|---|---|---|---|---|---|---|
| Injector | `<d1>` | 0.581 | 0.327 | 0.073 | 0.020 | 0.142 |
| Switchboard, detected | `<d1>` | 0.614 | 0.177 | 0.190 | 0.019 | 0 |
| Switchboard, gold spans | `<d1>` | 0.583 | 0.222 | 0.172 | 0.022 | 0.010 |
| Injector | `<d2>` | 0.538 | 0.335 | 0.086 | 0.042 | 0.143 |
| Switchboard, detected | `<d2>` | 0.613 | 0.171 | 0.184 | 0.032 | 0 |
| Switchboard, gold spans | `<d2>` | 0.543 | 0.243 | 0.180 | 0.034 | 0.025 |
| Injector | `<d3>` | 0.455 | 0.379 | 0.118 | 0.049 | 0.220 |
| Switchboard, detected | `<d3>` | 0.567 | 0.167 | 0.238 | 0.028 | 0 |
| Switchboard, gold spans | `<d3>` | 0.429 | 0.243 | 0.291 | 0.037 | 0.057 |

Share of that level's events falling in each type, in bits of KL from the Switchboard row directly above. The gold-span row is the same utterances read from the corpus annotation instead of through the detector, so its KL is the detector's own error and is the floor any system is really being measured against.

## Placement

| Where filled pauses land | N | At a clause onset | Before a content word | Mean log freq of the next word |
|---|---|---|---|---|
| Injector | 569 | 0.169 | 0.909 | -4.494 |
| Switchboard | 664 | 0.762 | 0.399 | -2.613 |
| a word picked at random | 7,412 | 0.111 | 0.542 | -3.587 |

Log frequency is base 10, over the Switchboard clean sides. Lower means a harder word, so a filled pause preceding harder-than-average words is a figure below the last row. The last row is the scale: what these three columns come to if a filler is dropped in front of a word chosen uniformly from the script.

