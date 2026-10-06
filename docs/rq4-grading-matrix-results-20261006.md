# Grading matrix results (Qwen3.5-27B, 1 step, 2 groups x 8)
| shape | problem | job | status | graded local/farm | correctness | best | sampling s | train s |
|---|---|---|---|---|---|---|---|---|
| remote-only v6e-8 asia | qubit | 1967 | PASS | 8 / 6 (14 of 16; 2 format) | 0.50 | 0.5194 | 580 | 170 |
| remote-only v6e-8 asia | erdos | 1965 | PASS | 16 / 0 | 0.25 | C5 0.3824 | 1233 | 46 |
| remote-only v6e-8 asia | ac1 | 1966 | PASS | 16 / 0 | 0.3125 | ac1 2.0 | 1768 | 61 |
| remote-only v6e-8 asia | circuit | 1968 | PASS | 101 / 1 cases (6 programs graded; 6 format fails) | 0.25 | 0.5051 | 1194 | 128 |
| remote-only v5p-32 | erdos | 1969 | PASS | 16 / 0 | 0.375 | C5 0.3877 | 588 | 85 |
| remote-only v5p-32 | ac1 | 1970 | PASS | 16 / 0 | 0.3125 | ac1 1.9501 (improved on seed 2.0) | 1948 | 95 |
| remote-only v5p-32 | qubit | 1971 | PASS | 16 / 0 | 0.625 | 0.5194 | 693 | 164 |
| remote-only v5p-32 | circuit | 1972 | PASS | 136 cases / 0 (8 programs) | 0.625 | 0.5086 | 1059 | 140 |
| train+infer v5p-32 | qubit | 1975 | PASS | Ray local, 4 hosts; pool cores 24-33/76-85 (one NUMA node each), not the legacy 106-207 block | 0.25 | 0.5203 | 777 | 143 |
| train+infer v5p-32 | circuit | 1976 | PASS | Ray local, 4 hosts; every case on 4 pool cores | 0.25 | 0.5054 | 1100 | 149 |
| train+infer v5p-32 | erdos | 1973 | PASS | 14 / 0 via pooled sandbox on the slice (2 format) | 0.3125 | C5 0.3833 | 2012 | 46 |
| train+infer v5p-32 | ac1 | 1974 | PASS | 16 / 0 via pooled sandbox on the slice | 0.50 | ac1 1.8517 | 1862 | 131 |

## Farm probes (all four problems submitted at once under one lease)
| farm | erdos | ac1 | qubit | circuit | generation under load | isolation |
|---|---|---|---|---|---|---|
| v6e-8 asia (1956) | 24/24 verified | 24/24 verified | 2/2 (72 cases) | 2x17 cases, fast proxy | 200 OK | peak 54 units = 48 ac2x2c + 4 placementx4c + 2 routingx10c = 132 = whole pool; 0 overlaps, 0 outside pool |
| v5p-32 east5a (1958) | 48/48 verified | 48/48 verified | 4/4 (72 cases) | 4x17 cases, fast proxy | 200 OK | all 4 hosts ran ac2+placement+routing at once (peak 27 units, 66 cores of 172); 0 overlaps, 0 outside pool |

Cross-hardware: the same qubit seed scored 0.5193640461472073 and the circuit seed 0.42141243779481874 on both the v6e-8 and the v5p-32 farm.
