# Grouped 5-fold split of data_v2

2141 scenes in 772 groups (largest [13, 13, 12, 12, 11], median 2); built by tools/build/split.py.

Test year 2026: 486 scenes. Left out: 597 scenes (max gap 5.0 min, buffer 14 days); see excluded.json.

| fold | scenes | water share | cake share | small share | medium share | big share | vast share | ice Mpx |
|---|---|---|---|---|---|---|---|---|
| 0 | 427 | 0.20 | 0.19 | 0.21 | 0.20 | 0.20 | 0.20 | 2243 |
| 1 | 393 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 2226 |
| 2 | 424 | 0.20 | 0.20 | 0.20 | 0.20 | 0.19 | 0.20 | 2220 |
| 3 | 433 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 0.20 | 2226 |
| 4 | 464 | 0.20 | 0.20 | 0.19 | 0.20 | 0.20 | 0.20 | 2197 |

Archive class totals (Mpx): water 10186.3, cake 132.8, small 651.7, medium 944.7, big 1325.0, vast 8056.9
