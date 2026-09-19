# Full-topology winners: independent replay, 2026-09-19

These are the best saved **combined-objective** policies for each model, selected
across the previous v4/v6e full-suite runs. They are not the newer Q20-only
specialists. `manifest.json` records the original run, state ID, source hash,
reward, and aggregate grade. Each replay uses the exact archived source.

Replay uses the existing isolated CPU routing grader on an allocated v4 host,
with three independent four-CPU, 8 GiB systemd units. Each acquires a normal
CPU grading slot. No TPU allocation, model update, or training cancellation is
part of this analysis. Seed 42; 20 layout trials and 20 routing trials; 72 cases.
The verifier checks operations/dependencies and topology and independently counts
SWAPs in the routed circuit, rather than trusting the candidate's reported count.

`*-replay.json` contains every independently verified case. `summarize.py` checks
source hashes, suite identity/completeness, SWAP/CNOT consistency, total SWAPs,
and the weighted objective before producing `summary.json`.

The inherited SimpleTES per-case weights are Q20 0.2, Willow 0.4, and Heron 0.4.
These are coefficients on absolute added-gate counts, not normalized shares of
the score. The published suite metadata is at:
https://github.com/wq-will/SimpleTES/blob/main/datasets/qubit_routing/swap_reduction/python/benchmarks/sabre_suite.json

Gemma's replay differs from its archived grade (91,205 SWAPs recorded versus
90,556 in this replay). Its candidate SWAP set uses `HashSet::new()` and directly
iterates it when forming the tie list passed to the supplied RNG. Rust HashSet
is independently randomly seeded, so this introduces a source of nondeterminism
that the benchmark RNG seed alone does not control. This is a likely explanation
for the discrepancy, not a controlled isolation of every possible cause.
The candidate is preserved unchanged; these counts are a single verified replay,
not a deterministic guarantee or a repeated-run mean.
https://doc.rust-lang.org/std/collections/struct.HashSet.html

Published comparison, added CNOTs converted to SWAPs by dividing by three:

| Method | Q20 | Willow | Heron | Unweighted total |
| --- | ---: | ---: | ---: | ---: |
| LightSABRE | 20,063 | 36,802 | 45,827 | 102,692 |
| SimpleTES gpt-oss-20b | 14,180 | 32,135 | 43,032 | 89,347 |
| SimpleTES gpt-oss-120b | 15,147 | 32,258 | 42,274 | 89,679 |

Source: Table 2, https://arxiv.org/html/2604.19341v1#S3.T2 . These are published
numbers, not reruns of SimpleTES in this audit. Do not combine different
SimpleTES programs' per-topology minima into one claimed policy result.

Verified replay results (all three passed 72/72 cases):

| Model | Q20 | Willow | Heron | Total SWAPs | Reward |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemma | 14,922 | 32,325 | 43,309 | 90,556 | 0.545904148 |
| qwen | 15,257 | 32,718 | 44,268 | 92,243 | 0.541408437 |
| muse | 16,873 | 37,011 | 45,593 | 99,477 | 0.523187150 |
