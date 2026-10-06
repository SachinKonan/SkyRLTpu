# Gemma attention backward replay

SkyPilot job 807, `gemma-v5p-bwd256-replay-001`, submitted to the existing
`tpuswarm-v5p32-east5a-erdos` pool using Ray v2. The replay passed: four backward calls across both buckets, a finite optimizer update (gradient norm 0.0490723), adapter export, and 64 post-update sampled tokens.

The only trainer kernel change from the failed Gemma sweep is
`sa_block_q_dkv=sa_block_kv_dkv=sa_block_kv_dkv_compute=256` (previously 512).
TP4/FSDP1, DQ reduction 3, full rematerialization, rank 32, token budget 22528,
and sequence buckets 18432/22528 remain unchanged. The trainer compilation
cache has a separate `-bwd256` suffix; model and inference caches are reused.

The opt-in `attention_replay` client restores two real, unchanged AC2 training
datums from failed run `native-v5p-gemma-ac2-grpo-lr15e4-s1-002`, request 20
(datum 3, 15515 tokens) and request 28 (datum 1, 18732 tokens). Their full request
hashes and original inputs are recorded in `gemma_attention_replay.json`.
Each datum exercises one of the two padded trainer shapes, twice, recording
first-pass and repeat timings. Both shapes are attempted even if one fails.
No optimizer request is sent after any backward failure. After four successful
passes the probe applies one optimizer update, requires a finite nonzero gradient,
exports the adapter, and samples 64 tokens from the updated adapter with the
original AC2 prompt. This is a compiler regression test, not a fresh 16x32 RL
batch or a grading-quality evaluation; the short post-update sample does not
retest forced thinking-cap transitions.

Local validation: exact fixture roundtrip through installed Tinker SDK; two
failure-path tests passed; profile/build and immutable overlay hash validation
passed. TPU replay timings: 18432 first/repeat 30.91/27.70 s; 22528 first/repeat 33.72/12.70 s. These include API overhead and are two single-datum probes, not steady-state throughput measurements. Updated-adapter sampling took 4.15 s. Eight replacement sweep jobs 809-816 were submitted only after this proof.

Cancelled active Gemma job 794; jobs 795-801 were already FAILED. Inspected all
28 hosts across their seven former slices. Sixteen idle hosts passed storage,
inode, process and TPU ownership gates. The other twelve hosts run Qwen/Muse
and were preserved. No manual kills or storage deletions were necessary.
The launch YAML also runs the clean-host gate on every assigned host.

Operational records:
`/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/benchmarks/gemma-bwd256/`.
