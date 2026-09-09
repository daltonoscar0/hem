## Copy failure against self-repair

| System | Script | Sentences | Substitution rate | All types | Script words kept |
|---|---|---|---|---|---|
| Hem | has an unseen word | 966 | 5.77 | 11.19 | 0.953 |
| Hem | all seen | 234 | 3.91 | 10.47 | 0.979 |
| Hem greedy | has an unseen word | 966 | 1.17 | 2.67 | 0.984 |
| Hem greedy | all seen | 234 | 0.07 | 0.30 | 0.999 |
| Injector | has an unseen word | 966 | 1.23 | 14.15 | 1.000 |
| Injector | all seen | 234 | 1.38 | 14.57 | 1.000 |

Split on whether the script contains a word absent from ten million words of Switchboard, which in practice means a rare proper noun. Rates are events per 100 clean words, at the natural setting.
