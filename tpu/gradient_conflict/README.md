# Valid versus invalid policy gradients

This diagnostic measures the geometry of evaluator-valid and evaluator-invalid
Qwen3.5-27B completions at a frozen, trained rank-32 LoRA checkpoint. It uses the
real Tunix/MaxText forward/backward path on a SkyPilot pool worker.

**Job 326 SUCCEEDED** at 2026-09-07 01:56:34 UTC: all six groups / 192 samples
completed. All six signed gradient cosines are negative (median -0.340155),
and the independent reconstruction check passed with 0.545% relative error.
See the [results and plots](../results/gradient-conflict-qwen-v4-64/RESULTS.md)
and [completion audit](../results/gradient-conflict-qwen-v4-64/verification.json).

The completed submission used `qwen_v4_64.yaml`, overlay revision 3, in
`tpuswarm-v4-64-central2-qwen35-erdos`. It replaced cancelled v4-32 job 314.
It discovers and validates the production v4-64 four-host training row, keeping
TP8/FSDP2 and the original samples/objective unchanged. The remaining four hosts
do not run inference for this diagnostic. Results record the accelerator and
selected trainer ranks explicitly.

Job 317 completed checkpoint staging and model initialization, then failed when
registration hit SQLite's default five-second lock timeout. Revision 2 registers
the source checkpoint before SDK session/model creation and gives registration
the same 30-second busy timeout as the API. A real SQLite contention regression
test and registration against the live slice both passed.

Job 320 failed earlier in setup because a reused worker retained 54 GB of
canary inference weights outside the standard Hugging Face cache. Revision 3
prunes those reproducible `.safetensors` files on selected trainer hosts and
invalidates their cache-ready marker, preserving canary logs/results, code,
environments, tokenizer metadata, and compilation caches.

Current artifacts:
`gs://sk7524-tinker-tpu-us-central2/gradient-conflict/20260906-qwen-v4-64-v1/`
(`manifest-v3.json`, `progress.json`, `result.json`, `cache-v5/`, and `logs/`).
The frozen input and checkpoint source remain under the original v4-32 prefix.

## Interpretation

The archived trajectories contain prompt/response text, rewards, and binary
`correctness`, but not sampled token IDs, sampling log-probabilities, stop tokens,
or phase-two action masks. This is therefore **text-reconstructed gradient
geometry**, not an exact replay of past optimizer steps. No termination token is
invented. The frozen checkpoint supplies reference log-probabilities, so the
importance-sampling ratio is one at measurement (up to kernel numerical error).
The measured objective is the signed advantage-weighted log-likelihood gradient.

Advantages use each original mixed group's mean reward, computed before any
filtering. Prompt advantages are zero. As in the current production API, token
weights are one and each sequence's loss is divided by its full input length.
The exported arrays are **sums of sequence gradients**, before averaging over
sequences, gradient clipping, or Adam. Both sum and per-sample mean norms are
reported. `correctness`, rather than reward sign, defines validity.

Negative cosine means local opposition between the two signed loss gradients.
It does not establish that suppressing invalid samples hurts task reward.
`combined_dot_left < 0` additionally means the combined SGD direction would
increase the valid-subset loss to first order. Adam may behave differently.
The cancellation ratio is `norm(valid + invalid) / (norm(valid) + norm(invalid))`.
A zero-norm group's cosine is undefined (`null`), not zero.

## Original v4-32 pilot

- Release: `20260906-qwen-v1`; SkyPilot job: **314**, overlay revision 5.
  Attempt 309 stopped before any gradient measurement because the fresh API
  had no source-checkpoint registry entry. The retry stages the source tarball
  on every host and uses production's `reregister_states.py` before loading it.
  Attempt 310 exported its first gradient on a different host because JAX
  runtime ranks differ from logical worker ranks. Revision 4 explicitly writes
  on the API host and checks an initial two-sample export before the full pilot.
  Job 313 passed that check and computed the first group's partitions, but GCP
  maintenance interrupted its independent reconstruction pass. Revision 5 saves
  reference scores and completed gradient partitions to GCS so retries can resume.
- Pool: `tpuswarm-v4-32-central2-smoke`; TP8/FSDP2 across four hosts.
- Base worker bundle: `tpuswarm-skyrl-v4-mixed-v41.tar.gz`, plus the SHA-256
  checked, experiment-specific overlay in `qwen_v4_32.yaml`.
- Checkpoint: `tinker://model_4ee1d2d2/weights/000003`, copied to the experiment
  prefix so production checkpoint retention cannot remove its source mid-run.
- Archive source: the `member_qwen/trajectories/` files for steps 0–2 of
  `v4-64-qwen35-grpo-erdos-tp8-fsdp2-005` in the central2 bucket.
- Six groups selected deterministically across the observed validity range,
  from 15 eligible mixed groups; these are **32-sample groups, not whole
  optimizer batches**. The selection is descriptive and not a random estimate
  of the overall training distribution.
- 192 completions; validity 9.375%, 21.875%, 31.25%, 37.5%, 43.75%, 56.25%.
  No selected completion was excluded or truncated at the 22,528-token limit.
- One matched-size random partition per group, seed `20260906 + group_index`.
  Four intersections of validity and random labels let each sample contribute
  to both comparisons with one backward pass. The first group additionally
  gets an independent full-group backward pass to check reconstruction.

`TUNIX_GRADIENT_PROBE_DIR` enables a special accumulator drain: it requires
zero learning rate and weight decay, gathers the global LoRA gradients, writes
arrays on the API coordinator (marked by `TUNIX_GRADIENT_PROBE_WRITER=1`), and resets only the accumulator. It never updates the
optimizer or model. A parameter SHA-256 must remain identical across all drains.
The client removes gradient arrays after loading them; JSON fingerprints remain.

## Run and results

```bash
sky jobs launch --pool tpuswarm-v4-64-central2-qwen35-erdos \
  tpu/gradient_conflict/qwen_v4_64.yaml -d -y
```

Original v4-32 artifacts live under:

`gs://sk7524-tinker-tpu-us-central2/gradient-conflict/20260906-qwen-v1/`

- `manifest-v5.json`: source commit and hashes of each overlaid file; the first
  attempt's `manifest.json` is retained separately.
- `pilot.json.gz`: selected tokenized groups, baselines, exclusions, and source hashes.
- `checkpoints/model_4ee1d2d2/000003.tar.gz`: frozen checkpoint copy.
- `progress.json`: completed group measurements, published incrementally.
  The first comparison is published with `reconstruction_status=pending` before
  its independent check; only a passed check permits successful completion.
- `result.json`: final successful result, including per-parameter-leaf comparisons.
- `cache-v5/<input-sha256>/`: durable reference scores and gradient partitions.
  Gradient arrays are uploaded before their JSON completion markers. Each retry
  performs a fresh two-sample backward/export check; all cached parameter hashes
  must match that live checkpoint, and loaded gradient checksums are validated.
- `logs/`: startup, API, and client logs.

The existing smoke worker copies its local `v6e-tunix-smoke.json` to this result
prefix; that legacy local filename does not imply v6e execution.

After a successful run, generate the report with:

```bash
python tpu/gradient_conflict/report.py result.json tpu/results/gradient-conflict-qwen-v4-64
```

The report requires NumPy and Matplotlib. The completed run and its operational
history are recorded in `tpu/results/gradient-conflict-qwen-v4-64/STATUS.md`.

Local checks:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q \
  tests/tpu_swarm/test_gradient_conflict.py
```
