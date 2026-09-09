# Evaluation

## Rate control

| System | Level | Filled pause | Repetition | Substitution | Restart | All types | Target | Ratio |
|---|---|---|---|---|---|---|---|---|
| Hem | `<d0>` | 0.05 | 0.00 | 1.59 | 0.00 | **1.65** | 0.00 | n/a |
| Hem | `<d1>` | 1.29 | 0.60 | 2.99 | 0.05 | **4.93** | 6.67 | 0.74 |
| Hem | `<d2>` | 3.54 | 1.91 | 5.51 | 0.12 | **11.09** | 14.20 | 0.78 |
| Hem | `<d3>` | 7.74 | 4.48 | 9.84 | 0.39 | **22.44** | 28.06 | 0.80 |
| Hem greedy | `<d0>` | 0.33 | 0.00 | 1.09 | 0.00 | **1.42** | 0.00 | n/a |
| Hem greedy | `<d1>` | 0.77 | 0.02 | 0.98 | 0.00 | **1.77** | 6.67 | 0.26 |
| Hem greedy | `<d2>` | 1.20 | 0.13 | 1.02 | 0.00 | **2.34** | 14.20 | 0.16 |
| Hem greedy | `<d3>` | 3.36 | 1.63 | 1.16 | 0.00 | **6.17** | 28.06 | 0.22 |
| Injector | `<d0>` | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** | 0.00 | n/a |
| Injector | `<d1>` | 3.95 | 2.09 | 0.54 | 0.15 | **6.73** | 6.67 | 1.01 |
| Injector | `<d2>` | 7.65 | 4.79 | 1.25 | 0.51 | **14.21** | 14.20 | 1.00 |
| Injector | `<d3>` | 12.65 | 10.55 | 3.41 | 1.15 | **27.77** | 28.06 | 0.99 |

Events per 100 clean words, counted by the detector. Target is the rate measured off Switchboard for that bucket. Ratio is measured over target, so 1.00 is the dial landing where it was aimed.

## Script fidelity

| System | Level | Script words kept | Sentences losing a word |
|---|---|---|---|
| Hem | `<d0>` | 0.981 | 0.225 |
| Hem | `<d1>` | 0.968 | 0.357 |
| Hem | `<d2>` | 0.956 | 0.470 |
| Hem | `<d3>` | 0.939 | 0.541 |
| Hem greedy | `<d0>` | 0.987 | 0.163 |
| Hem greedy | `<d1>` | 0.988 | 0.147 |
| Hem greedy | `<d2>` | 0.986 | 0.152 |
| Hem greedy | `<d3>` | 0.978 | 0.171 |
| Injector | `<d0>` | 1.000 | 0.000 |
| Injector | `<d1>` | 1.000 | 0.000 |
| Injector | `<d2>` | 1.000 | 0.001 |
| Injector | `<d3>` | 1.000 | 0.003 |

Hem is a generator, not an editor, so nothing stops it rewriting a word of the script instead of only adding to it. The injector cannot: it copies the clean tokens through and inserts around them, so it keeps 1.000 at every setting by construction. This is the column to read before using Hem on a script somebody has to say verbatim.

## Type mix

| System | Level | Filled pause | Repetition | Substitution | Restart | KL from Switchboard |
|---|---|---|---|---|---|---|
| Hem | `<d1>` | 0.261 | 0.122 | 0.606 | 0.010 | 0.584 |
| Hem greedy | `<d1>` | 0.436 | 0.010 | 0.554 | 0.000 | 0.562 |
| Injector | `<d1>` | 0.586 | 0.311 | 0.081 | 0.022 | 0.139 |
| Switchboard, detected | `<d1>` | 0.611 | 0.166 | 0.200 | 0.024 | 0 |
| Switchboard, gold spans | `<d1>` | 0.576 | 0.215 | 0.179 | 0.029 | 0.013 |
| Hem | `<d2>` | 0.320 | 0.172 | 0.497 | 0.011 | 0.335 |
| Hem greedy | `<d2>` | 0.512 | 0.054 | 0.435 | 0.000 | 0.269 |
| Injector | `<d2>` | 0.539 | 0.337 | 0.088 | 0.036 | 0.158 |
| Switchboard, detected | `<d2>` | 0.597 | 0.170 | 0.205 | 0.028 | 0 |
| Switchboard, gold spans | `<d2>` | 0.527 | 0.240 | 0.195 | 0.039 | 0.027 |
| Hem | `<d3>` | 0.345 | 0.199 | 0.438 | 0.017 | 0.165 |
| Hem greedy | `<d3>` | 0.546 | 0.265 | 0.188 | 0.001 | 0.082 |
| Injector | `<d3>` | 0.456 | 0.380 | 0.123 | 0.041 | 0.207 |
| Switchboard, detected | `<d3>` | 0.556 | 0.170 | 0.244 | 0.030 | 0 |
| Switchboard, gold spans | `<d3>` | 0.433 | 0.268 | 0.257 | 0.042 | 0.059 |

Share of that level's events falling in each type, in bits of KL from the Switchboard row directly above. The gold-span row is the same utterances read from the corpus annotation instead of through the detector, so its KL is the detector's own error and is the floor any system is really being measured against.

## Placement

| Where filled pauses land | N | At a clause onset | Before a content word | Log freq of the next word | Same, mid-clause only |
|---|---|---|---|---|---|
| Hem | 787 | **0.215** (0.109) | 0.689 (0.545) | -4.08 (-3.60) | **-4.34** (-3.62) |
| Hem greedy | 266 | **0.120** (0.109) | 0.395 (0.545) | -4.77 (-3.60) | **-4.94** (-3.62) |
| Injector | 1,699 | **0.169** (0.109) | 0.899 (0.545) | -4.50 (-3.60) | **-4.74** (-3.62) |
| Switchboard | 1,994 | **0.786** (0.144) | 0.397 (0.440) | -2.67 (-2.71) | **-2.77** (-2.77) |

Each cell is the measured figure with, in brackets, what it would be if the filled pause were dropped in front of a word drawn uniformly from that system's own text. The bracketed figure differs by row because the scripted sentences and the Switchboard transcripts do not carry commas at the same rate, and comparing raw onset shares across them would be measuring a transcriber's punctuation rather than a speaker's hesitations.

Log frequency is base 10 over the Switchboard clean sides, so a lower number is a rarer word. The last column is the Shriberg question asked properly: of the hesitations that are *not* at a clause onset, does the word after them tend to be a hard one. Asked of all hesitations the answer is confounded, because the word after a clause onset is nearly always a very common one.

