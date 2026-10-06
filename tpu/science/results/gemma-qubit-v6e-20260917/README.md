# Gemma qubit duplicate on v6e-32

User-approved duplicate submitted 2026-09-17 as job 983 to
`tpuswarm-v6e32-east5b-qwen35`; run `science-qubit-v6e-gemma-runtime-001`.
V4-64 job 981 is unchanged. All executor Python and source-overlay manifest
bytes match the Gemma v4-64 package. Profile changes only topology, accelerator,
zone, namespace/root and TPU-generation-specific compile-cache locations.

Four trainer hosts TP4/FSDP4; four TP4 inference engines, 16 sequences per engine,
80% memory utilization, 1024-token chunks. 16x32 rollouts, 16384 prompt-plus-thinking,
22528 total context. CPU grading 16 slots per host, 4 CPUs/8 GiB per candidate.

Training checkpoint 1 and payload-preserving database were copied to an isolated
GCS namespace and verified by size and CRC32C. Client checkpoints journal,
pools 0/1/2 and life/global-step/metrics state were copied under the new run name
and verified byte-for-byte. Restore requires checkpoint >=1 and preserves the
newer pool without advancing the optimizer batch. Uses original archived verified
recovery DB; does not copy or alter any live v4 database.

Required gcloud and refreshed ADC service account verified; TPU/storage probes
passed. Worker 3906 was idle; all eight hosts passed device/process/storage checks.
Config and native contract checks passed; source-overlay and executor byte parity
verified. No new optimizer step is claimed by this submission record.
