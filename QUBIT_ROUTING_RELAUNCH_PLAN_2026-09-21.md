# Qubit routing: parallel evaluation and relaunch plan

Date: 2026-09-21. Status: proposed implementation and rollout; this document does not launch, cancel, or modify any experiment.

## Goal and order

Improve the chance of beating SimpleTES Gemini on all three topology totals while limiting wasted grading compute. Preserve joint optimization and the existing reward. Implement and validate the evaluator, regrade each model's original bootstrap, and restart training in priority order **Gemma -> Qwen -> Muse**. Gemma must demonstrate one completed generation/grading/training cycle before admitting the next model; models may subsequently overlap if capacity permits.

Working repository: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-hybrid-inference-migration`. It contains unrelated active changes: preserve them and do not reset, clean, or deploy an unreviewed moving worktree. Record the exact implementation commit and deployment bundle digest when implementation is complete.

## Preserve the evaluation and learning objective

- One general policy is evaluated on 72 cases: 24 each for Q20, Willow, and Heron.
- Each case retains 20 layout variants x 20 routing trials. Keep the successful trial with the minimum added SWAP count; do not average attempts or increase their number.
- Preserve the pinned suite, case identifiers, per-case seed behavior, trial ordering, candidate state reset, tie handling, and independent circuit verification.
- Require all 72 cases to succeed and verify. Missing, illegal, or timed-out cases make the entire candidate invalid, reward zero. Partial results are diagnostic only.
- For topology totals Q, W, H, cost is C = 0.2Q + 0.4W + 0.4H. The fixed SABRE reference has weighted cost B; reward remains B / (B + C), with the existing validity/floor behavior. The implementation's factor of three CNOTs per SWAP cancels in this ratio.
- Gemini targets are guidance and reporting fields, not a new reward, acceptance filter, or per-case baseline. Beating all three targets is the experiment goal; maximizing the existing scalar reward does not guarantee that outcome.

## 1. Parallelize cases inside each Ray grading task

Current inspected path: Ray distributes candidates across prepared hosts; each Ray task starts a systemd-managed process with hard CPU, memory, and lifetime limits. Within that process the router and verifier currently iterate cases sequentially.

Retain Ray's candidate-level distribution and local process isolation. Compile a candidate once, then run a bounded local process pool with **four concurrent case workers per program**, as requested. Do not create 72 unbounded workers or cross-host Ray child tasks. Each case worker runs that case's full routing search and trusted verification; promptly release large intermediate circuits afterward. Use the same compiled binary read-only, isolated output paths, and trusted verifier boundaries.

Use a supported single-case CLI or a trusted one-case suite manifest. Verify that splitting the suite preserves seed initialization and policy construction exactly. Candidate code cannot modify another case's inputs or outputs. Aggregate verified results in canonical suite order regardless of completion order.

Requested resource profile: **2 vCPU / 4 GiB per case worker**, four workers per program, with a **100 vCPU / 200 GiB total grading budget per worker VM**. This replaces the earlier two-worker proposal and the old 64-vCPU grading partition. Treat these host budgets as inclusive of compilation, coordinators, case routing, and verification. Use GiB consistently in implementation; the user's GB shorthand is interpreted here as GiB, matching the existing cgroup configuration.

Ignoring overhead, each VM has 50 case-worker slots, or 400 across an eight-VM v4-64 or v6e-32 slice. With four workers reserved per program and programs confined to one VM, the packing limit is **12 programs / 48 case workers per VM**, or **96 programs / 384 case workers per eight-VM slice**. Those workers consume 96 vCPU and 192 GiB per VM, leaving 4 vCPU and 8 GiB for all grading overhead. This is a target ceiling, not measured sustainable capacity: reduce concurrent programs if coordinator, compilation, or verifier peaks cannot fit. Keep four workers per admitted program. Fewer workers will be busy during compilation and at the end of a program's case queue.

Benchmark aggregate peak usage and training/inference interference before enabling the maximum host concurrency. Choose disjoint CPU sets from actual host affinity and NUMA topology rather than extending the current hardcoded ranges blindly. Update Ray reservations, cgroup limits, host-wide admission locks, and memory accounting together; version admission so old and new graders cannot independently claim overlapping resources. Do not increase fleet size. The 400 figure is a raw slot ceiling, not a promise of 400 simultaneous case workers under fixed four-worker program allocation.

Bound internal library threads to prevent oversubscription. Compilation may use the full candidate CPU allocation before case workers start. Stop and reap all case children on cancellation, timeout, coordinator failure, or an irrecoverably invalid case.

Persist each completed case's independently verified counts, timings, and failure details incrementally. Only emit a valid aggregate after all 72 exact case IDs pass. A failed candidate must never acquire a full-suite reward from its completed subset.

## 2. Use one 1,900-second evaluation deadline

Define the candidate deadline at admitted evaluation startup, including private workspace preparation, compilation, routing, and verification. Exclude Ray scheduling and host admission queue wait. Every phase and child receives the remaining time on this one monotonic deadline; there is no fresh 1,900-second allowance per phase or case.

Remove the current hidden 900-second compilation-plus-routing cap and separate 900-second verifier cap as unintended restrictions. Keep bounded subprocess termination and result collection.

Set the outer systemd runtime limit to **2,100 seconds**, allowing startup, cleanup, and durable failure reporting outside the 1,900-second candidate budget. Propagate compatible bounds through the Ray subprocess wait and client grading-request timeout; account for queue wait separately. Raising only the evaluator argument is insufficient. Candidate computation must not consume the outer grace period.

Expose the evaluation deadline and resource profile explicitly in the grading request and immutable run contract. Avoid changing unrelated science task budgets.

## 3. Add Gemini targets to prompts and topology feedback

Fixed reference totals, each over 24 cases:

| Topology | SimpleTES Gemini added SWAPs |
|---|---:|
| Q20 | 13,470 |
| Willow | 31,481 |
| Heron | 42,396 |

Reference: SimpleTES paper, Gemini routing discussion and Table 5, <https://arxiv.org/html/2604.19341v2>. These are published reference totals, not locally rerun Gemini measurements. Existing local record: `tpu/science/results/best-solutions-20260921/qubit-routing-baselines-per-topology.csv`.

Suggested task text:

> Produce one general routing policy evaluated on 72 cases: 24 each for Q20, Willow, and Heron. Each case's score is the minimum added SWAP count among its successful routing attempts. Each topology's score is the sum across its 24 cases. Aim to beat the SimpleTES Gemini reference totals: Q20 13,470; Willow 31,481; Heron 42,396. Lower is better. Seek improvements on all three topologies and avoid sacrificing one to improve another. Adapt using circuit structure and hardware graph properties, without hardcoding benchmark names or identities. Every required case must route correctly and pass verification within the shared 1,900-second evaluation budget; otherwise the candidate receives zero reward. The scalar reward remains the existing SABRE-normalized weighted aggregate, so inspect each topology's totals as well.

Use one canonical target mapping. Add `gemini_target_swaps` and `gap_to_gemini = swaps - gemini_target_swaps` to each topology summary; negative means strictly better, zero means tied. Emit these comparisons only for complete verified topology totals, or mark incomplete summaries explicitly without a success claim. Preserve all 72 per-case SWAP/SABRE/difference rows and existing overall feedback. Do not invent per-case Gemini counts from topology totals.

Ensure the task text appears in initial and parent-improvement prompts, including the path that omits starter code. Regenerate parent observations using the new formatter. Check the complete JSON against the 12,000-character routing feedback budget. Record that targets are supplied to the model for paper reporting. Do not additionally inject Gemini's algorithm description in this change.

## 4. Validate before a large regrade

Run focused tests for deadline propagation, bounded concurrency, seed preservation, output ordering, cancellation cleanup, missing/duplicate/invalid cases, and feedback completeness/sign conventions. Confirm unrelated grading tasks retain their settings.

On a CPU allocation, compare serial and parallel evaluation using a small fixed panel: historical valid winners, representative previously timed-out bootstrap programs, and deliberate invalid inputs. Compare identical budgets to isolate parallelization; separately compare the legacy deadline with 1,900 seconds to measure rescued candidates.

Record compile, routing, verification, queue, total wall time, CPU time, peak aggregate memory, per-case latency, completed case count, validity, and all SWAP counts. Some archived programs have non-seeded HashSet iteration and already vary across processes: use paired repeats where needed, and distinguish existing variability from changed RNG or policy state semantics. Do not promise bitwise equality for those programs.

Proceed only after valid full-suite outputs verify, no resource oversubscription or child leaks occur, and the parallel path provides a measured benefit. Longer runtime alone is not evidence that a recovered candidate is competitive.

## 5. Regrade the original bootstrap programs

Freeze each model's original 1,024-draft archive, including formerly invalid and timed-out entries. Do not restrict regrading to the 46/96/48 retained valid seeds. Reuse generated source; do not spend another 1,024 LLM generations or silently replace it with historical winners.

Known source campaigns (verify object generations and completeness at execution time):

| Model | Original campaign | Bucket |
|---|---|---|
| Gemma | science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920 | sk7524-tinker-tpu-us-east5 |
| Qwen | science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920 | sk7524-tinker-tpu-us-east5 |
| Muse | capacity-v6e-muse-qubit-grpo-4e5-20260920 | sk7524-tinker-tpu-us-central1 |

Candidate grades/sources live under `ray-training/<run>/client/bootstrap/drafts/`. Preserve original draft IDs, code hashes, source object generations, old outcomes, and new evaluator/version/resource metadata.

To conserve compute, evaluate each exact duplicate program once under a declared canonical source key, mapping its result to all original occurrences. Report raw draft validity and unique-program validity separately. Do not select the best of duplicate replays; this would add a hidden search budget. Revalidate malformed or noncompiling entries through the normal contract, preserving their failure class rather than treating them as routed candidates.

Recompute all rewards and observations from the new verdicts. Rank unique valid programs by unchanged overall reward, with deterministic tie handling, and retain up to 512. If fewer qualify, retain that number rather than padding duplicates. Never mix old and new evaluator rewards in the new PUCT tree.

Create new immutable seed manifests and run IDs with fresh PUCT statistics, adapters, and optimizer state. Explicitly record that these are historical generations made with the old prompt and regraded under the new evaluator; the new training prompt includes Gemini targets. Implement a provenance-preserving seed import rather than falsely claiming the old bootstrap prompt matches the new prompt contract.

Produce one table per model: drafts, unique programs, valid before/after, timeouts before/after, newly valid programs, retained seed count, best combined-policy Q/W/H totals and reward, and regrading wall/CPU cost. Keep independent per-topology minima separate from one policy's complete score vector.

## 6. Controlled relaunch: Gemma, then Qwen, then Muse

Prepare complete immutable deployment artifacts before replacing any existing run. Inventory live candidate jobs, host allocations, farm leases, and grading load at execution time; do not use historical job IDs as cancellation targets. This plan does not authorize unrelated fleet cancellation or expansion.

Keep the current model-specific training recipe, group sizes, token limits, loss, and learning rates. Proposed run length remains the existing ten-step campaign limit; verify the source configs when building the new profiles. Use fresh run IDs and checkpoints, the regraded model-specific seed pool, and the established Ray v2/farm discovery path. Record the exact configuration diff: evaluator/runtime/resources, Gemini prompt/feedback, and regraded initialization. Preserve compatible model/tokenizer caches and separate versioned evaluator artifacts and writable run state.

Launch Gemma first. Validate a full 16-parent x 32-rollout step, grading completion, one optimizer update, durable checkpoints, and subsequent generation using the updated adapter. Then admit Qwen and finally Muse as capacity permits. Any retirement of an older qubit run must identify the exact replacement and preserve its durable results; leave AC2, circuit optimization, inference farms, and unrelated workloads alone.

Do not deploy by overwriting code used by active graders. Give each deployment an immutable evaluator, runtime dependency, prompt, target mapping, and seed manifest version. On failure, stop or hold only the affected new run and retain its evidence; do not silently fall back to the old evaluator within the same reward history.

## 7. Reporting and success criteria

Report at bootstrap and every completed step, per model: valid fraction, timeout/compile/verification failure fractions, completed case distribution, grading time/CPU cost, training step time, best combined-policy reward, and that same policy's Q20/Willow/Heron totals and Gemini gaps. Also report topology-specific minima with their candidate IDs, clearly identified as potentially different policies.

A win on all three means one frozen policy with verified totals strictly below all three Gemini targets, reproduced across independent full-suite executions with variation reported. Do not construct that claim by combining minima from different policies. Publish evaluator resource/budget changes and target-informed prompting; these runs are not automatically compute-matched to the legacy campaign or SimpleTES.

## Implementation touchpoints and deliverables

- `tpu/science/routing.py`: single build, case pool, shared deadline, trusted aggregation, incremental evidence.
- Pinned router wrapper/CLI: supported single-case invocation with unchanged case/trial semantics.
- `tpu/science/ray_cpu.py`, `worker.py`, CPU admission/cgroup helpers, and grading client: consistent resources, budget propagation, cleanup, and outer timeouts.
- `tpu/science/feedback.py`, task prompt source/rendering, `training_env.py`: canonical targets and complete parent feedback.
- Bootstrap import/preparation and run builders: frozen source provenance, regrade manifests, new contracts and run IDs.
- Deliverables: focused test results, paired timing/correctness report, three bootstrap regrade summaries, reviewed configuration diffs, submission receipts, and first-step evidence per model.

Only this plan was written in the planning turn. Implementation, bootstrap regrading, cancellation, and relaunch remain future actions.
