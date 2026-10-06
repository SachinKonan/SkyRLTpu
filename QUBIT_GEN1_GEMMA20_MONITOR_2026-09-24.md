# Qubit Gen1: Gemma step20 to Qwen and Muse

The user authorized waiting for Gemma's exact completed step20 and then creating
Qwen and Muse Gen1 branches on the existing v5p-32 pool. The configured Gen1
recipe follows the existing circuit Gen1 handoff: full donor program pool,
fresh recipient base/LoRA/optimizer, and ten new optimizer steps. These are
separate run identities; donor and existing qubit continuations are unchanged.
The fresh-weights/ten-step setting was stated as the existing-recipe assumption;
no Gemma parameters are loaded into an incompatible recipient architecture.

Donor: `qubit-v4-gemma-parallel2-pwc-rho05-20260921-r2`.
Recipients:

- `qubit-gen1-qwen-on-gemma20-pwc05-10step-20260924`
- `qubit-gen1-muse-on-gemma20-pwc05-10step-20260924`

Pool: `tpuswarm-v5p32-east5a-erdos`, region us-east5, accelerator v5p-32.
Uses Ray v2, one TP1/FSDP4 trainer host and three TP4 inference hosts; dynamic
farm borrowing remains enabled. Keeps model-specific learning rates, adaptive
PWC rho0.5, importance_sampling, 16 groups x32 rollouts, 22528 context with
16384 phase1, seed1, and eight parallel-v2 grading slots per host. Trainer token
budget45056 retains two full-length examples. RAM caches128GiB, reserve240GiB.
V5p compilation seed caches are used, not the donor's v4 compilation cache.

## Trigger and state transfer

Poll every60 seconds for checkpoint batch20 and successful metrics step20,
then require the matching PUCT step20 snapshot and nonempty training/optimizer
and sampler archives. Step19 or21 is never substituted. Validate every
program's complete72-case feedback and recalculate its saved weighted reward.
Preserve the full pool, including code, constructions, scores, feedback, IDs,
and ancestry. Reset PUCT visit statistics, state timestamps, and the recipient
counter to zero, matching the existing Gen1 recipe. The untouched donor
snapshot and corresponding checkpoint archives are preserved in regional
east5 storage with pinned GCS generations and checksums.

Templates are immutable before the trigger. Binding replaces only their seed
pool hash. Regional runtime/model/cache paths were checked; the CPU grader
bundle was copied and verified in east5. Destination uploads are create-only
and checked on retries. Unique run IDs, a monitor lock, and exclusive
submission-intent files prevent duplicate launches. Ambiguous submissions
require reconciliation; the monitor does not blindly resubmit. Jobs queue for
existing pool capacity; the monitor neither cancels work nor resizes pools.
The packaged clean-host audit runs at startup.

## Monitor

User systemd service: `qubit-gen1-gemma20-20260924.service`.
Enabled for the user manager, Restart=on-failure, RestartSec60.
The unit includes an explicit gcloud PATH; its initial missing-PATH startup
was corrected before the first successful poll.

Implementation: `tpu/science/ops/qubit_gen1_watch.py`.
Pool validation: `tpu/science/qubit_gen1.py`.
Artifacts/status: `.science/routing-relaunch-20260921/gen1-gemma20/`.
Inspect `status.json`, `watch.log`, and each model's submission receipt.
After both submissions, the monitor exits successfully; it does not supervise
training completion or promise a notification in this chat.

Validation on Slurm CPU job14354360: seven tests passed; real step19 fixture
validated684 states, preserving scores/content. Both profiles validated and
packaged. This establishes the gate and preparation, not TPU execution of the
new branches. At installation the donor was at19 saved steps and no Gen1 job
had been submitted.
