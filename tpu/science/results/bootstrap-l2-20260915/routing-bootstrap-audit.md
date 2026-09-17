# Routing bootstrap audit and controller recovery

Audited saved journals on 2026-09-16. Both runs completed 512 drafts and 512 invalid-only repair samples before the controller failed. No optimizer step had completed.

| Model | Valid drafts | Valid repairs | Distinct valid draft / repair code | Retained pool | Best reward | Best weighted cost improvement |
|---|---:|---:|---:|---:|---:|---:|
| qwen | 82/512 (16.0%) | 17/512 (3.3%) | 10 / 16 | 5 | 0.519786 | 7.61% |
| gemma | 152/512 (29.7%) | 42/512 (8.2%) | 19 / 33 | 9 | 0.522521 | 8.62% |

Each valid candidate passed all 72 routing cases. Improvements use the fixed weighted baseline added-CNOT cost; one SWAP is three added CNOTs. These are discovery-suite results, not held-out results. Code uniqueness is exact text equality within each layer; draft and repair counts can overlap. Repair validity is conditional on selected failed parents, not a directly comparable random draft cohort.

Qwen repaired 4 of 16 selected invalid parents into at least one valid child; Gemma repaired 7 of 16. Every selected repair parent was invalid. Missing/empty extracted code is excluded from repair-parent selection by the existing implementation. Qwen had 170 draft and 178 repair output-format failures; Gemma had 104 draft and 98 repair routing timeouts. Other failures include Rust compilation/type/API errors.

Prompt tokens plus allowed thinking tokens equal 16,384 throughout. Qwen draft prompt: 8,574 tokens, 7,810 thinking allowance; repair prompts: 5,877–14,445 tokens. Gemma draft prompt: 9,516 tokens, 6,868 thinking allowance; repair prompts: 7,914–10,785 tokens. No budget changes were made.

The best Qwen repair uses a greedy initial layout and SABRE-style front-layer/lookahead scoring (lookahead weight 0.5, size 20). The best Gemma repair uses component-aware BFS layout construction and front-layer/lookahead scoring (weight 0.8, size 32). These are source-code observations, not causal ablations.

Controller failure: `ray.wait()` received a Ray Serve `DeploymentResponse`. The fix uses its public synchronous `result(timeout_s=...)` API with deadline and interruption polling. A regression test constructs an actual DeploymentResponse and rejects any ray.wait call. Bootstrap tests: 6 passed, 3 subtests passed.

Published replacement archives change only `tpu/swarm/ray_train/controller.py`. Every other archive member, including profiles, bootstrap implementation, prompts, and source overlays, is byte-identical. Both cloud and host promoted pools match their saved SHA256 checksums; config and bootstrap implementation satisfy the resume contract.

Recovery submissions: Gemma 957, Qwen 958. These retain their original run IDs and skip completed bootstrap on resume. At submission, neither had completed an optimizer step.

On idle worker 367, removed only retired Gemma per-candidate Rust `evaluation/target` build directories after checking TPU ownership and workload processes. All journals, source programs, grader outputs and checkpoints were preserved. Free disk increased to 60–64 GiB across all eight hosts. Two verified PID-1 GCS orphans belonging to terminal science runs were stopped. Active Muse worker 366 was not modified.
