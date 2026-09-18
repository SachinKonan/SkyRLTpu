# Q20-only qubit experiments

Three independent Gemma, Muse and Qwen runs, starting from pretrained weights
and their original bootstrap programs rescored on Q20. Existing combined
Q20/Willow/Heron experiments remain separate. No adaptive PWC experiment.

Reward is maximized: `22714/(22714+total_Q20_SWAPS)` (floor 1e-6).
All 24 Q20 circuits must verify. The fixed seed, 20 layout trials and 20 routing
trials, Rust engine, independent Qiskit verifier, and runtime limits are unchanged.
`SCIENCE_ROUTING_SUITE=q20` selects the objective in the prompt, Ray request,
isolated worker and startup reference checks. The default remains all 72 cases.

`python -m tpu.science.q20_seeds --client ... --destination ...` reconstructs
seed admission from the archived bootstrap grade journals. It validates source
hashes, complete case IDs, independently counted SWAPs, and original rewards;
then computes Q20 rewards and rebuilds top-two-per-parent admission plus global
code deduplication. It does not rerun already verified circuits. Full-suite
failures are omitted: they are not assumed invalid on Q20, and their old visits
are not imported. Original case timing is not presented as Q20 execution time.
All seeds were discovered against the combined objective; this is a warm start
for a new Q20 objective, not Q20-prompt bootstrap generation.

| Model | Valid archived submissions | Distinct valid programs | Retained /32 | Best Q20 SWAPs |
|---|---:|---:|---:|---:|
| Qwen | 99 | 26 | 6/32 | 16667 |
| Gemma | 194 | 52 | 7/32 | 16259 |
| Muse | 110 | 30 | 7/32 | 17173 |

Profiles: `science-qubit-q20-v6e-{model}-seeded-001.json`.
V6e-32: four trainer hosts, four TP4 inference engines. Qwen/Muse trainer
TP8/FSDP2; Gemma TP4/FSDP4. Native prompt-plus-thinking budget 16384 and total
sequence 22528. Groups 16 x 32, learning rate 4e-5, mean-baseline advantages,
importance-sampling loss, 15 epochs. Each host admits 16 CPU grading tasks,
each limited to four CPUs and 8 GiB; up to 128 tasks across eight hosts.

Each run has independent client/checkpoint, trainer compilation and inference
compilation destinations. Checkpoint recovery remains enabled for these new
runs; no prior experiment checkpoints or optimizer state are imported.

Local launch records and full seed provenance:
`.science/qubit-q20-20260918/{model}/`.

Validation: 84 tests and 24 subtests passed across Q20 selection/rescoring,
science prompts/rewards, bootstrap, seed handoff, CPU admission, and launcher
commands. The deployed native suite checksum and all 24 input paths were checked
read-only. All three seed pools were uploaded and verified by exact readback.
