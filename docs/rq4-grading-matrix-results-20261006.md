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

## Second round (after the review fixes: math family, 4/8 GiB)
| check | result |
|---|---|
| Qwen3.8 v5p-32 farm probe (1979) | PASS: erdos 48/48, ac1 48/48, qubit 4/4 (72 cases), circuit 4x17; generation 200; all admission core-pool-v1; same seed scores as Qwen3.5 farms |
| Muse v5p-32 farm, TP4 x1/host, 16 seqs (2004) | PASS: erdos 48/48, ac1 48/48, qubit 4/4, circuit 4x17; generation 200; core-pool-v1; same seed scores |
| Muse v5p-32 farm, TP2 x2/host, 64 seqs (2005) | PASS: erdos 48/48, ac1 48/48, qubit 4/4, circuit 4x17; generation 200 (identical greedy text to TP4); core-pool-v1 |
| Remote-only v6e-8 bootstrap smoke (1977 farm, 1978 trainer) | PASS: 0 local engines; 64 drafts generated on the leased farm, 20 valid, 8 retained (fixed budget), pool sha recorded; then 1 training step from that tree; 88 grades all via the pooled math grader; best C5 0.3812 |
| Gemma v5p-32 farm, TP4 x1/host (2003) | PASS: erdos 48/48, ac1 48/48, qubit 4/4, circuit 4x17; core-pool-v1. Generation: the raw probe prompt (no <bos>) degenerates; with the Gemma chat template + <bos> (what trainers send) it answers correctly |
| Qwen3.8 remote-only v5p-32 circuit (1986) | PASS: 16 rollouts on leased farms, 136 cases graded on the trainer pool, 4 valid, best 0.4612, training step 202 s |

## Legacy vs pooled grading parity (TPU host, real systemd limits)
92 candidates (78 real Qwen3.5 Erdős/AC1 programs from the smokes + 14 edge cases), graded on a v6e-8 host by the legacy
child (`run_with_timeout`, 2 pinned cores, no memory limit) and by the pooled math grader (core pool, systemd MemoryMax 8 GiB):
- 89/92 identical validity, score and reward.
- The 3 that differ are nondeterministic programs. Re-graded 3x under each grader:
  - Erdős #44 flips valid/invalid inside the same grader (legacy: invalid/0.418781/invalid; pooled: 0.418781/invalid/0.418781).
  - Erdős #47 C5: legacy 0.4023/0.4016/0.4011, pooled 0.4002/0.3994/0.4027 (overlapping).
  - AC1 #52: legacy 2.0084/2.0099/2.0110, pooled 2.0091/2.0120/2.0107 (overlapping).
  - Two more programs (#15, #76) were valid in the original hardware run and invalid under both graders on re-run.
- Edge cases: all 14 agree on validity. A 6 GiB program passes both (the old 4 GiB cap killed such programs: 3/16 ac1
  smoke programs were OOM-killed at 4 GiB); a 12 GiB program is killed by the pooled grader.
| Qwen3.8 remote-only v5p-32 erdos (1983) | PASS: 16 rollouts on farms, 13 graded via pooled math (3 format), 5 valid, best C5 0.3814, train 131 s |
| Qwen3.8 remote-only v5p-32 qubit (1985) | PASS: survived a trainer preemption (restarted on a new worker, re-leased a farm); 14 graded, 5 valid, best 0.537, train 128 s |
| Qwen3.8 remote-only v5p-32 ac1 (1984) | PASS: 16 graded via pooled math, 7 valid, best ac1 1.5859, train 155 s |

## Production rollout (2026-10-06, from merged main 60ff7041, branch agent/rq4-rollout)
41 farms in 5 discovery groups; every farm with a worker passed /health, model, grading-ready (math/routing/placement),
attestation and supervisor visibility (11/11 at 20:26); one farm per group leased + generated correctly.
