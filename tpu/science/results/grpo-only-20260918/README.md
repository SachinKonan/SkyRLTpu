# Q20 GRPO on v4-64; fresh circuit GRPO/PWC on v6e-32

This supersedes the earlier September 18 GRPO-to-PWC continuation experiment.
The six GRPO runs use TTD_ADV_ESTIMATOR=mean_baseline without PWC parameters.
The user subsequently authorized three ADDITIONAL fresh circuit PWC runs on
v6e-32 with rho=0.5. Q20 remains GRPO. No PWC run imports a GRPO checkpoint.
Do not switch estimators when resuming any of these runs.

Q20: three v4-64 jobs in new clean namespaces. Qwen and Gemma continue the
original GRPO checkpoint 2, with optimizer state and saved search pools, from
jobs 1000 and 999 respectively. Muse uses the original seven graded Q20 seeds
from job 1001 and has no previous optimizer checkpoint. Copies originate from
the immutable preserved originals, never a PWC attempt. Archived checkpoints
remain intact. Qwen/Gemma target iteration 17; Muse targets 15.

Circuit: three v6e-32 jobs, each starting from pretrained weights plus the same
adapter initialization convention, with no imported candidate seeds or frozen
bootstrap phase. First-batch generation uses the helper-enabled prompt and its
legal starter; the ordinary training loop grades, admits valid programs to PUCT,
and computes a GRPO update. The fast C Evaluator is injected into candidates.
All three target 15 iterations. Independent scoring uses the four fixed Xplace
inputs and unchanged circuit objective.

Common settings: four trainer hosts, four TP4 inference replicas, 16 sequences
per replica, 16x32 rollouts, learning rate 4e-5, importance_sampling loss, 16,384
prompt-plus-thinking budget, 22,528 total context, and systemd runtime ownership.
Qwen/Muse trainer TP8 FSDP2; Gemma TP4 FSDP4. CPU grading is 16 slots per host,
4 CPUs and 8 GiB per task, on all eight hosts. Every run has separate write
cache, checkpoint and output destinations.

Jobs 1047-1058 were cancelled. Circuit bootstrap handoff processes were stopped;
no circuit PWC job had been submitted at that cancellation point. Earlier worker cleanup missed unmanaged Ray
processes from pre-systemd bundles; their PIDs, process start times and retired
run ownership were audited before targeted termination. A subsequent check on
all 64 hosts of the eight ready workers passed: TPU devices unowned, no private
Ray/model leftovers or active grading units, sufficient disk/inodes/RAM, and
1,018 required ports available per host. Minimum free home storage was 59.62 GiB;
minimum available RAM was 336.86 GiB. These are startup checks, not proof of
completed inference or optimizer steps. Other workers had been preempted and
were not reachable; GCP/SkyPilot snapshots distinguish them from ready workers.

## Additional fresh circuit PWC runs

One per model, with piecewise_valid_entropic_centered_adaptive, rho=0.5, invalid
reward=0. These differ from the fresh circuit GRPO controls only in estimator
and run/cache destinations. They use no imported seeds/checkpoints, no frozen
bootstrap, and the same C evaluator. Sampling is independent between arms.
The expanded v6e health snapshot covered 72/72 hosts of nine ready workers.
Six workers are requested for circuit, leaving three of that snapshot unused.

The obsolete Q20 PWC jobs 1056/1057/1058 were confirmed CANCELLED before the
new GRPO-only Q20 namespaces were submitted. The earlier automatic bootstrap
handoffs remain stopped. All submissions are recorded with immutable bundle
hashes; a submitted/STARTING state does not establish optimizer progress.

## Final submissions

| Model | Q20 GRPO, v4-64 | Circuit GRPO, v6e-32 | Circuit PWC rho=0.5, v6e-32 |
|---|---:|---:|---:|
| Gemma | 1059 | 1060 | 1065 |
| Muse | 1061 | 1062 | 1066 |
| Qwen | 1063 | 1064 | 1067 |

All nine reported STARTING at the recorded post-submit snapshot. Ten focused
profile/initial-prompt tests passed, including checks that Q20 retains GRPO and
that fresh circuit estimator pairs differ only in estimator and destinations.
The prior eight helper/direct-start checks also passed. Changes were committed
after submission, as requested; immutable bundle hashes identify deployed bytes.
