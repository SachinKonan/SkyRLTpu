# Superseded: matched GRPO/PWC comparison

All jobs in this plan were cancelled. The current approved plan is
[GRPO only](../grpo-only-20260918/README.md). The initial cleanup check below
missed legacy private Ray processes; the later GRPO-only health audit corrected
that omission. This document records the original launch, not current status.

Approved allocation: 12 v4-64 training jobs (three models, two tasks, two
estimators), with one of 13 machines reserved for recovery. Superseded science
jobs are preserved before cancellation. Unrelated pool workloads are untouched.

PWC uses piecewise_valid_entropic_centered_adaptive, rho=0.5, invalid reward=0.
The unlaunched rho=0.05 proposal is archived separately. Both arms package exactly
the same trainer implementation and differ only in estimator and run/cache paths.
Learning rate is 4e-5, loss importance_sampling, batch 16x32, native prompt plus
thinking 16,384 tokens and total 22,528 tokens. Training uses four hosts; inference
uses four TP4 engines (16 sequences each). Grading is CPU only: 16 admitted tasks
per host, 4 CPUs and 8 GiB per task; compilation caches are capped at 128 GiB.
Qwen/Muse trainer mesh is TP8/FSDP2; Gemma TP4/FSDP4.

Q20 pairs fork the same verified checkpoint, optimizer state, and search pool:
Qwen and Gemma checkpoint 2 from their Q20 runs; Muse starts with the same graded
Q20 seeds and pretrained model. The target is 15 additional iterations, with
completed optimizer updates reported separately from iterations/checkpoints.
Review after five additional updates without changing the paired settings.

Circuit uses one new bootstrap per model: 512 drafts, no repair, all eight hosts
for inference, no trainer weights or optimizer loaded. After full grading and
successful bootstrap shutdown, a checksum-verified pool (normal top-two per
root plus global deduplication, at most 32 programs) is cloned unchanged to both
training namespaces. A handoff validates every draft grade and retained program
before submitting either arm. Both train for 15 iterations from the same base
model and adapter seed. Six training templates remain unbound until bootstrap
completion; zero seed hashes must never be launched directly.

The new prompt supplies Evaluator in the candidate namespace, documents cache
lifecycle, move/undo, full-layout batches, legality, runtime, and JAX limitations.
The old prompt is retained. The objective and independent final grader are
unchanged on ibm01/04/08/18 with the same legal Xplace inputs.

Local validation: the starter ran inside bubblewrap on four CPUs with JAX CPU,
using the injected helper, and passed independent legality/scoring on all four
netlists. See helper-integration.json. This local check did not enforce the 8 GiB
cgroup; every actual host must pass the existing resource/reference startup gate.
The helper's separate 12-layout parity checks and seven API tests are recorded
under fast_proxy/. Attribution remains in source/license, not the model prompt.

## Submitted first wave

| Model | Q20 GRPO | Q20 PWC | Circuit bootstrap |
|---|---:|---:|---:|
| Qwen | 1047 | 1048 | 1049 |
| Gemma | 1050 | 1051 | 1052 |
| Muse | 1053 | 1054 | 1055 |

Submission is not service readiness or optimizer progress. Six circuit training
jobs will be submitted by the recorded handoffs after successful bootstrap
completion and verified full grading. Their templates are already packaged.
All 12 previous science jobs were cancelled after preservation; checks across
56 previously active hosts found no remaining workload processes.

Validation: 98 integration/lifecycle tests and 12 subtests, 36 estimator tests,
and seven helper API tests passed. Actual packaged overlay manifests match
within all six estimator pairs. Bundles were submitted before committing at the
user's request, so their recorded base Git HEAD predates this commit; immutable
bundle hashes in submissions.json identify the deployed bytes precisely.
