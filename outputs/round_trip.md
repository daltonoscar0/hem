## Round trip

| Through | Level | N | Script recovered exactly | Recovered up to casing and punctuation | Word error rate |
|---|---|---|---|---|---|
| Mend after Hem | `<d0>` | 1,200 | 0.471 | **0.747** | 0.021 |
| Mend after Hem | `<d1>` | 1,200 | 0.387 | **0.583** | 0.044 |
| Mend after Hem | `<d2>` | 1,200 | 0.299 | **0.403** | 0.077 |
| Mend after Hem | `<d3>` | 1,200 | 0.203 | **0.251** | 0.144 |
| Mend after Hem greedy | `<d0>` | 1,200 | 0.480 | **0.792** | 0.019 |
| Mend after Hem greedy | `<d1>` | 1,200 | 0.482 | **0.799** | 0.027 |
| Mend after Hem greedy | `<d2>` | 1,200 | 0.481 | **0.794** | 0.036 |
| Mend after Hem greedy | `<d3>` | 1,200 | 0.471 | **0.780** | 0.108 |
| Mend after Injector | `<d0>` | 1,200 | 0.500 | **0.868** | 0.009 |
| Mend after Injector | `<d1>` | 1,200 | 0.474 | **0.796** | 0.015 |
| Mend after Injector | `<d2>` | 1,200 | 0.438 | **0.709** | 0.026 |
| Mend after Injector | `<d3>` | 1,200 | 0.378 | **0.603** | 0.041 |
| Mend on real Switchboard | n/a | 1,200 | 0.463 | **0.778** | 0.027 |

Hem writes the script out as speech at a given level, Mend reads that speech back as writing, and the recovered text is compared with the script Hem started from. The middle column ignores casing and punctuation, which is the fair comparison: Mend restores those from its own training distribution and getting a comma back in the wrong place is not a failure of the round trip.
