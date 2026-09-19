# Mixture-of-Models status, 2026-09-17 (table refreshed 2026-09-19 17:15Z)

Question: starting from the same qwen tree at step 15, does handing the tree to a
different model (muse or gemma, carrying its own gen-0 weights) produce a lower C₅
than giving qwen the same 15 extra steps?

Conventions used throughout:

- C₅ values are upper bounds, lower is better. Record to beat: **0.380856777** (job 716).
- "Step 15" = the end of a 15-batch gen-0 run (batches are 0-indexed; 15/15 done).
- Improvements are in units of **1e-6**, positive = better, each relative to that
  model's own step-15 value; the takeover columns are relative to **qwen** at step 15.
- Winner = the cell with the lowest absolute C₅ in the row.
- Reference noise: gen-0 seed-to-seed spread at step 15 is 23.5 (qwen), 32.3 (gemma)
  in the same units; muse seeds are unfinished.

## 2026-09-19 19:20Z: Stage K -- paired-roots gen-0 rows (user decision, option A)

**Why.** The seed rows were never paired: each run drew its own 16 initial constructions from an unseeded
generator, and the audit (below) found that draw to be the dominant source of the 24-32e-6 seed-to-seed spread.
"Which model wins seed i" was therefore a lottery over starting basins, not a model comparison. The user's
call: pause the unpaired seed runs; make each ORIGIN's 16 roots a shared row and run the other two models on
them; fix the LoRA init seed; nothing else needs a seed (PUCT selection is deterministic; TPU vLLM sampling
cannot be seeded; the grader is wall-clock-budgeted).

**Design.** Row Q = qwen origin's roots (stageC-pwc-n step 0): qwen origin (existing) + stageK-rootsQ-g +
stageK-rootsQ-m. Row G = gemma origin's roots (stageB2-g-pwc-n): gemma origin + stageK-rootsG-q +
stageK-rootsG-m. Row M = muse origin's roots (stageB-m-pw-n): muse origin + stageK-rootsM-q + stageK-rootsM-g.
Six new gen-0 runs, 15 steps each, per-model config identical to the Stage G replication yamls. Mechanism: the
origin's `puct_sampler_step_000000.json` (16 states = 16 initial_states, empty PUCT stats) copied into the new
GCS_RUN before launch; sampler.py resumes from step 0 when that file exists; META_SEED_ONLY=1 guards against a
cold tree. Fresh base weights, fresh LoRA with a FIXED seed per row: LORA_SEED 1001 (Q), 1002 (G), 1003 (M).
Then per row: best gen-0 parent at 15 -> two takeover children (other two models' step-15 weights on the
parent's top-48) + parent continues alone to 30 -- the same table as before, now paired.

**Code (bundle v30, sha256 ef585218...).** `third_party/discover/ttt_discover/rl/ensemble.py`:
EnsembleMemberConfig.lora_seed, passed as `seed=` to all three create_lora_training_client_async calls;
`tpu/run_ttd_ensemble.py`: LORA_SEED / TTD_M{i}_LORA_SEED env. The tinker SDK, skyrl API (LoRAConfig.seed) and
tunix backend already honour it for the first adapter of a fresh run (later adapters copy the seed-0 template,
which is what a resume overwrites from checkpoint anyway). Caveat: v30 is the current worktree head; the origin
runs were bundles v21/v23 (same cell code per the side check, but not byte-identical bundles).

**Paused 19:22Z (state intact in GCS, torn-row check clean, resumable):** 818 stageG-m-rep1 (7/15), 1074
stageG-m-rep2 (7/15), 925 gen1-g-on-q2 (13/15), 1156 gen1-m-on-q2 (8/15), 1199 cont30-q-q1 (16/30), 1200
cont30-q-q2 (20/30), 1202 cont30-g-rep1 (21/30), 1203 cont30-g-rep2 (18/30). 993/924 (seed-1 qwen-tree children)
were cancelled at 17:34Z. Kept: origin row 941 gen1-g-on-q0, 1198 cont27-q-orig, 1201 cont28-g-orig, 1204
cont26-m-orig; inspiration 1205/1206/1207.

**Launched 19:31Z:** 1209 stageK-rootsG-q, 1210 stageK-rootsM-q, 1211 stageK-rootsQ-g, 1212 stageK-rootsM-g,
1213 stageK-rootsQ-m, 1214 stageK-rootsG-m. Priority now: Stage K six > origin-row jobs > inspiration.
Compute per run to step 15: qwen ~23 h, gemma ~36 h, muse ~38 h before reclaim losses.

**Randomness audit (2026-09-19, agent).** Seedable: initial constructions (env.py:311, unseeded default_rng --
we pair by copying the step-0 file instead), LoRA init (done, v30). Deterministic already: PUCT node selection,
elite slots, pruning (ties break on grading-completion order only). Not seedable: rollout sampling (TPU vLLM
rejects per-request seeds; one engine key split per decode step, batch-composition dependent), grader outcome
(programs run `run(seed=42, budget_s=1000)` on WALL CLOCK -> iterations depend on host load), pipelined fb
accumulation order (TTD_LEAGUE_PIPELINE=1). So: statistically paired rows, never bit-identical reruns.

## The table (snapshot 2026-09-19 17:15Z)

Status key: DONE = all steps banked; RUN = running; WAIT = lost its worker, back in the race.
Step labels are "banked/cap". Improvements in 1e-6, positive = better; "continues alone" columns are
relative to that model's own step-15 value, takeover columns relative to qwen at 15.

| Seed | Qwen at 15 | Gemma at 15 | Muse at 15 | Qwen continues alone | Gemma continues alone | Muse continues alone | Muse takes over Qwen | Gemma takes over Qwen | Winner so far |
|---|---|---|---|---|---|---|---|---|---|
| Origin | 0.380857128 (15/15 DONE, at 15) | 0.380863326 (15/15 DONE) | 0.380859181 (15/15 DONE, at 13) | +0.04 → 0.380857092 (17/27 RUN, 1187) | +2.17 → 0.380861153 (19/28 RUN, 1188) | 0.00 (15/26 RUN, 1193, no extra step yet) | **+0.35 → 0.380856777** (15/15 DONE, best at step 5, 716) | −0.05 → 0.380857178 (6/15 RUN, 941) | **Muse takes over** (record) |
| Seed 1 | 0.380872859 (15/15 DONE, at 15) | 0.380874732 (15/15 DONE) | 0.380870180 (7/15 RUN, 818, at 7) | +0.01 → 0.380872851 (16/30 RUN, 1189) | **+2.96 → 0.380871775** (21/30 RUN, 1191) | not started (gen-0 unfinished) | +0.27 → 0.380872593 (3/15 RUN, 993) | +0.01 → 0.380872847 (6/15 RUN, 924) | Gemma continues alone; **muse gen-0 parent at 7/15 is lower than all of them** |
| Seed 2 | 0.380881108 (15/15 DONE, at 15) | 0.380895593 (15/15 DONE, at 14) | 0.380910491 (7/15 RUN, 1074, at 7) | **+0.08 → 0.380881031** (20/30 RUN, 1190) | 0.00 → 0.380895591 (18/30 WAIT, 1192) | not started (gen-0 unfinished) | 0.00 → 0.380881107 (8/15 RUN, 1156) | 0.00 → 0.380881107 (13/15 RUN, 925) | Qwen continues alone |

Weights and algorithm per cell: qwen = centered piecewise (piecewise_valid_entropic_centered, lr 1.5e-4);
gemma = centered piecewise (lr 4e-5); muse = piecewise LOO (piecewise_valid_entropic, lr 4e-5).
Continuations resume the same run (same weights, tree, algorithm) past step 15. Muse takeovers load
muse ORIGIN weights at step 11 (tinker://model_df2f51fc/weights/000011) onto the top-48 of qwen seed i's
tree; gemma takeovers load gemma seed i's step-15 weights (origin child: gemma origin step 13).

What the table says right now:

- **Origin row is the only row with a finished takeover.** Muse on qwen origin ended 15/15 at
  0.380856777, set at step 5 and never improved. The gain over qwen at 15 is +0.35, not the +0.81
  quoted while qwen origin was at 13/15 (qwen's own steps 14-15 closed part of the gap). Qwen alone
  has gained +0.04 in two extra steps; the gemma takeover is slightly behind qwen at 15.
- **Seed 1 is gemma-alone territory so far**, and the muse gen-0 parent (0.380870180 at 7/15) is lower
  than every post-15 arm. The seed-1 muse takeover (993) is the only child with a real gain (+0.27).
- **Seed 2: nothing has moved.** Both takeovers sit exactly on the parent's best after 8 and 13
  steps; qwen alone leads by +0.08. Gemma at 15 on this seed is 14.5 worse than qwen, muse 29 worse.
- Takeover cells showing 0.00 equal the parent to the last digit because batch 0 loads the parent's
  top-48 nodes; a child counts only when a later step beats them.

## Where the incomplete cells are

Snapshot 18:46Z: nothing running, 6 recovering (worker assigned, cell restarting),
10 pending. The pool was fully reclaimed by GCP three times today (08:00Z, 16:10Z,
18:25Z); 25 steps were banked across the held windows.

| Cell | Job | Needs | State |
|---|---|---|---|
| Origin, Qwen continues alone | 942 | first extension step, 0 of 15 | recovering |
| Origin, Muse continues alone | 946 | finish gen-0 (13/15), then 11 more | recovering |
| Origin, Gemma continues alone | 943 | 13 more of 15 | recovering |
| Origin, Muse takes over | 716 | 5 more of 15 | recovering |
| Origin, Gemma takes over | 941 | first real step, 0 of 15 | pending |
| Seed 1, Qwen continues alone | 926 | 0 of 15 | pending |
| Seed 1, Muse at 15 | 818 | 9 more of 15 | pending |
| Seed 1, Gemma continues alone | 953 | 13 more of 15 | pending |
| Seed 1, Gemma takes over | 924 | 0 of 15 | pending |
| Seed 2, Qwen continues alone | 959 | 13 more of 15 | pending |
| Seed 2, Muse at 15 | 819 | 12 more of 15 | pending |
| Seed 2, Gemma continues alone | 954 | 0 of 15 | pending |
| Seed 2, Gemma takes over | 925 | 14 more of 15 | pending |

**Not queued, and cannot be yet under the current design (4 cells):**

- Seed 1 and seed 2, **Muse takes over Qwen**: need muse seed 1 (818) and muse
  seed 2 (819) to reach 15/15 so the child gets their final weights
  (`META_INIT_STATE_PATH` must be a numbered `weights/000015`, never `weights/final`).
- Seed 1 and seed 2, **Muse continues alone**: the same two runs must finish gen-0,
  then each is resumed with `NUM_EPOCHS` raised to 30, one job each.

The three inspiration arms (772 qwen, 922 gemma, 923 muse) are queued but not in this table.

Pace: muse seed 1 has 9 steps left at ~4 h each on a held slice (~36 h of compute);
today averaged ~6 held hours per 14. Realistically the seed 1 muse cells open in
3–4 days, seed 2 a day or two after.

## Minimum set for a causal claim

The claim: from the same qwen tree at step 15, a different model taking over beats
qwen taking the same 15 steps. A causal test needs, per seed, control and
intervention from an identical starting point with one thing changed, repeated
enough to clear the seed noise.

| Role | Runs | Status |
|---|---|---|
| Control: qwen continues alone, 15 extra | 3, one per seed | all queued (942, 926, 959) |
| Intervention: muse takes over, 15 steps | 3, one per seed | origin queued (716); seeds 1 and 2 not built |
| Confound guard: muse alone for the same steps | 1 | queued (946) |

Seven runs, five already queued. Three paired comparisons give a sign on every seed
and a mean against the 2e-5 seed spread. Gemma is a second intervention that turns
"mixing" into "across models"; its three children (941, 924, 925) are already
queued, so keep them, but they are not required for the claim.

**The change that shortens the critical path:** use **muse origin weights** for all
three children instead of muse seed-i weights. The current design pairs the seed-i
tree with a seed-i child, which changes two things between rows (tree and weights)
and forces the seed 1 and 2 children to wait days for muse gen-0. One fixed set of
muse weights on all three trees changes only the tree, which is the cleaner
experiment, and the two missing children can launch today. The muse-alone confound
then needs only one run (946), because the weights are the same in every row. What
is given up is the "child uses its own seed" symmetry from the original design,
which served the compute-matching argument, not causality.

Dropped from the critical path under that change: muse seed 1 and 2 gen-0 as
prerequisites (818/819 keep running for the seed-spread result, they just stop
gating anything), and the two muse seed 1 and 2 continuations.

Decision needed: muse origin weights at **step 11** (`tinker://model_df2f51fc/weights/000011`,
what 716 carries) or the muse origin's latest checkpoint. Recommendation: step 11, so
all three children carry exactly the weights of the record run. Building the two
children = one `build_meta_seed.py --op winner-top16 --k 48` seed per qwen tree
(stageG-q-rep1 snapshot 14, stageG-q-rep2 snapshot 14) staged as
`puct_sampler_step_000000.json`, plus a yaml each cloned from `stageH-m-on-qpwc.yaml`;
~10 minutes, then they join the placement race.

## Worker priority (user, 2026-09-18 03:00Z)

Grants last 20 min to 3.5 h and the pool rarely holds all 16 jobs; placement is a first-come race,
so the lever is which jobs are in the race. Order:

1. Muse gen-0 seeds 818, 819 (they gate the muse takeover children).
2. Takeover children 716, 941, 924, 925 (and the future muse-on-qwen seed 1/2).
3. Last: continuation controls 942, 926, 959, 943, 953, 954, 946 and inspiration arms 772, 922, 923.

Rule: when a tier-1/2 job has no worker, cancel every WAITING tier-3 job so the next worker goes to
the priority job; if that is not enough, evicting a RUNNING tier-3 job is allowed too (user,
03:1xZ: "yes, but anything you cancel needs to be requeued"). Every cancelled job is relaunched from
the same yaml as soon as all tier-1/2 jobs hold workers (same GCS_RUN resumes, nothing lost).
First application 03:02Z: cancelled waiting 959, 943, 923; 716 took the next worker within 2 min.
Re-queued 03:19Z once every priority job was placed: 959 -> **995** (cont30-q-q2), 943 -> **996**
(cont28-g-orig), 923 -> **997** (stageI-m-insp-n).
Second application 04:10-04:16Z after a full reclaim left 8 priority jobs waiting: cancelled the nine
waiting tier-3 jobs (995, 996, 997, 946, 953, 954, 772, 922, 998) and evicted the one running tier-3
job (942, qwen origin control, ~28 min into a step) so its worker goes to a priority job. All ten
are owed a re-queue: `jobs/f6d76b15/tmp/requeue10.sh`, to run when every tier-1/2 job holds a worker.
Re-queued 06:26Z, all eight priority jobs placed by 06:25Z (a two-hour dry spell, then one wave):
942 -> **1002** (cont27-q-orig), 998 -> **1003** (cont30-q-q1), 995 -> **1004** (cont30-q-q2),
996 -> **1005** (cont28-g-orig), 953 -> **1006** (cont30-g-rep1), 954 -> **1007** (cont30-g-rep2),
946 -> **1008** (cont26-m-orig), 772 -> **1009** (stageI-q-insp-n), 922 -> **1010** (stageI-g-insp-n),
997 -> **1011** (stageI-m-insp-n). Live MoM set: 716 818 819 924 925 941 993 994 + 1002-1011.
Third application 06:41-07:28Z: 993 then 716 and 924 lost workers one at a time (controller SSH
health-probe drops on 965, 960, 975 -- not GCP reclaims); cancelled the waiting 1003-1011 and evicted the
running 1002 (all ten parked again, `requeue10.sh` covers them). Eviction cost: 716 landed on the
vacated worker 961 twelve minutes later and its cell exited 33 on the previous cell's leftovers
("leaving the worker clean for the next job"), so it went back to the race; ~25 min lost. Lesson:
prefer cancelling waiting tier-3 jobs; evict only when a priority job would otherwise wait for a
fresh grant, and expect one failed landing on the freed worker.
Re-queued 08:18Z with all eight priority jobs placed (716 on 979, 993 on 961, 924 on 974):
1002 -> **1012** cont27-q-orig, 1003 -> **1013** cont30-q-q1, 1004 -> **1014** cont30-q-q2, 1005 -> **1015**
cont28-g-orig, 1006 -> **1016** cont30-g-rep1, 1007 -> **1017** cont30-g-rep2, 1008 -> **1018** cont26-m-orig,
1009/1010/1011 -> **1019/1020/1021** stageI q/g/m insp. Live MoM set: 716 818 819 924 925 941 993 994 + 1012-1021.
Fourth application 08:57Z: workers 964/962/967 dropped together at 08:47Z (health-probe, three in 30 s,
i.e. a small reclaim) taking 819, 925, 818; no idle workers and no waiting tier-3 jobs, so evicted the
three inspiration arms 1019/1020/1021 (15 min into their steps, not in the table). All three priority
jobs claimed the freed workers within a minute (818 -> 978, 819 -> 989, 925 -> 991). Inspiration arms
owed a re-queue: `jobs/f6d76b15/tmp/requeue_insp.sh`. Also: `sky jobs queue` default window is 300
rows and 716 fell outside it (read as blank); the watch now queries `--all --limit 800`.
Results so far this grant: 819 banked batch 3 (4/15, 0.380932069); 925 banked batch 2 (3/15, still
0.380881107 = parent, zero gain after two real steps); 994 wrote its seeded baseline row.
**09:07Z: gemma on qwen ORIGIN (941) banked its first real step: 0.380857199 vs parent 0.380857586,
gain +0.39 (1e-6 units), validity 0.71.** First non-zero gemma takeover delta; the origin row now has
both children ahead of the parent (muse +0.81 at 10/15, gemma +0.39 at 2/15) while qwen-alone has
0 extension steps banked.
10:53Z (grant still holding since 08:18Z): 1014 qwen seed 2 continuation banked 18/30, 0.380881031
(+0.03 in extension); 924 gemma on qwen seed 1 banked its first real step, 0.380872859 = parent (zero
gain). 818 recovered onto 978 after the dirty-landing exit 33. Inspiration arms re-queued 10:55Z:
1019/1020/1021 -> **1022/1023/1024**. Live MoM set: 716 818 819 924 925 941 993 994 + 1012-1018 + 1022-1024.
10:55Z: 1016 gemma seed 1 continuation 18/30, 0.380871892 (+2.84 in extension, largest continuation gain).
**11:07Z: qwen ORIGIN continuation (1012) banked its first extension step, 14/27, 0.380857494 (+0.09).**
The origin row is now fully populated: qwen alone +0.09 (1 extra step), gemma takes over +0.39 (1 real
step), muse takes over +0.81 (10 steps). Also 993 muse-on-qwen-seed-1 wrote its baseline row
(0.380872859 = parent) and 994 muse-on-qwen-seed-2 banked its first real step with zero gain
(0.380881107, val 0.92).
11:18Z: 716 11/15 (best unchanged), 941 3/15 (0.380857194), 1015 gemma origin cont 18/28 (0.380861187).
11:30-12:45Z: 925 4/15 (still = parent), 819 5/15 (0.380918606), 1013 qwen seed 1 cont banked its
first extension step (16/30, no gain yet), 1014 19/30, 1016 19/30 (0.380871842, +2.89), 1017 16/30
(0.380895591), 1018 muse origin 14/26 (unchanged). Grant holding since 08:18Z (4.5 h, 21 steps).
Fifth priority application 12:45Z: 818 fell out of its worker (978) again and lost the race to the
just re-queued inspiration arms 1022/1024; cancelled 1023 (waiting) and evicted 1022/1024 (just
started); 818 took 997 within a minute. Lesson: re-queue tier-3 only after the last priority job is
RUNNING, not merely assigned, or it can lose the race on a dirty-landing retry. Inspiration arms
owed another re-queue (`requeue_insp.sh`).
12:58Z: 818 RUNNING on 997 (all eight priority jobs running); inspiration arms re-queued as
**1025/1026/1027**. 924 gemma on qwen seed 1 banked 3/15 at 0.380872853 (+0.006, first non-zero gain on
seed 1). Live MoM set: 716 818 819 924 925 941 993 994 + 1012-1018 + 1025-1027.
13:11Z: all 18 running (inspiration arms on 972/978/996).
**13:23Z: qwen ORIGIN continues alone (1012) banked 15/27 at 0.380857128 (+0.46 in extension after 2
steps), now ahead of the gemma child on the origin row (+0.39 after 2 real steps) and behind the muse
child (+0.81).** 925 gemma on qwen seed 2 at 5/15, still exactly the parent value (four real steps, zero gain).
13:34-13:45Z: 941 4/15 (0.380857178, +0.41); 1016 20/30 (0.380871788, +2.94); 994 3/15 (no gain);
716 12/15 (unchanged since step 5); 993 first real step 0.380872849 (+0.01); 1015 19/28 (0.380861153).

## 2026-09-18 14:00Z: matched-validity ("native" v32) cells relaunched on spare workers (user)

Four idle v5p workers with the whole MoM grid running -> user: "run those immediately", priority
qwen pair, gemma pair, then muse pair. All resume from GCS state (registries checked with
`ckpt_torn.py`; stageC-v32-ttd-n had a torn 000007 row -> dropped, backup `.bak-torn-20260918T135851Z`,
resumes from step 6). Jobs: **1028** stageC-v32-grpo-n (qwen GRPO whitened, 7/15), **1029** stageC-v32-ttd-n
(qwen TTD, 6/15 after the drop), **1030** stageB2-g-v32-grpo-n (gemma GRPO, 13/15), **1031** stageB2-g-v32-ttd-n
(gemma TTD, 6/15), **1032** stageB-m-v32-grpo-n (muse GRPO, 4/15), **1033** stageB-m-v32-ttd-n (muse TTD, 2/15).
Question they answer: with exactly 32 valid rollouts per group, does GRPO still beat TTD? Provisional
reading from 09-08: no, TTD leads once the validity term is removed. Priority: below the MoM tier-3 set
(they are spare-worker fillers); cancel them first when a MoM job needs a worker, re-queue later.
14:02Z: 1028-1031 took the four idle workers (992, 994, 1001, 1003); 1032/1033 waited.
14:12Z: GCP terminated worker 973 under 941 (gemma on qwen origin, 4/15). No idle workers -> cancelled
the waiting 1032/1033 and evicted 1031 (gemma native TTD, ~12 min in) to free a worker for 941.
Native cells owed a re-queue (1031, 1032, 1033): rerun `launch_native.sh` for those three.
Steps 14:02-14:14Z: 819 6/15 (0.380912814); 1013 qwen seed 1 cont banked its first real extension
step (17/30, 0.380872851, +0.008); 1017 17/30 (unchanged).
14:17Z: 1018 muse origin banked batch 14 -> gen-0 complete at 15/15, best 0.380859181 ("Muse at 15"
origin cell is final). 14:28Z: 941 RUNNING on 1003 (the freed worker, clean landing); 1014 20/30
(unchanged). Native cells re-queued 14:30Z: 1031 -> **1034** gemma TTD, 1032 -> **1035** muse GRPO,
1033 -> **1036** muse TTD. Live set (24): 716 818 819 924 925 941 993 994 + 1012-1018 + 1025-1027 +
1028 1029 1030 1034 1035 1036.
14:29Z: GCP terminated four workers together (983, 992, 989, 961) -> 1013 qwen seed 1 cont, 1028 qwen
native GRPO, 819 muse seed 2 and 993 muse-on-qwen-seed-1 lost their slices; the freshly re-queued
1035/1036 grabbed workers ahead of the two priority jobs. Applied the rule: cancelled waiting 1034,
1028, 1013 and evicted just-started 1035, 1036 (14:32-14:33Z). 819 and 993 now the only jobs in the
race. Owed re-queue (five): `jobs/f6d76b15/tmp/requeue_parked5.sh`, once 819 and 993 are RUNNING.
14:57Z: 818 lost worker 997 (health probe). The two workers vacated by 1035/1036 never came back as
idle (they were still STARTING when cancelled), so three priority jobs waited with no idle workers ->
evicted 1029 (qwen native TTD), 1030 (gemma native GRPO) and 1026 (gemma inspired, 2/15). Within a
minute 993 -> 1001, 819 -> 978, 818 -> 994. Owed re-queue is now EIGHT: `requeue_parked8.sh` (1013,
1026, 1028, 1029, 1030, 1034, 1035, 1036), to run once 818/819/993 are RUNNING. 16 workers in use:
8 priority + 6 continuations + 2 inspiration arms (1025 qwen, 1027 muse).
Steps 14:41-14:56Z: 1016 gemma seed 1 cont 21/30 (0.380871775, +2.96).

## 2026-09-18 15:01Z: priority order revised (user) -- natives above inspiration

User: "prioritize the native matched validity runs ... put that ahead of inspiration". Order is now
1 muse seeds, 2 takeovers, 3 continuations, 4 native v32 cells, 5 inspiration arms. Applied at once:
evicted 1025 (qwen inspired, 8/15) and 1027 (muse inspired, 9/15) so their workers go to native cells.
Parked and owed a re-queue, in the new order: cont30-q-q1 (1013); natives qwen GRPO/TTD, gemma
GRPO/TTD, muse GRPO/TTD (1028 1029 1030 1034 1035 1036); inspiration q/g/m (1025 1026 1027).
Script: `jobs/f6d76b15/tmp/requeue_all.sh` (LIST env var to subset), to run once 818/819/993 are RUNNING.
15:08Z: 818/819/993 all RUNNING (994, 978, 1001) -> re-queued the ten: cont30-q-q1 -> **1037**; natives
qwen GRPO **1038**, qwen TTD **1039**, gemma GRPO **1040**, gemma TTD **1041**, muse GRPO **1042**, muse TTD
**1043**; inspiration q/g/m **1044/1045/1046**. Steps 15:07Z: 924 gemma on qwen seed 1 4/15 (0.380872847,
+0.012); 925 gemma on qwen seed 2 6/15 (still = parent). Live set (24): 716 818 819 924 925 941 993 994 +
1012 1014-1018 1037 + 1038-1043 + 1044-1046.
15:11Z: 1012 qwen origin cont 16/27 (0.380857094, +0.49). 15:13Z: GCP terminated worker 988 under 1016
(gemma seed 1 cont, 21/30, +2.96 -- the strongest continuation). 1037 and 1038 had already taken the two
freed workers (972, 996), in order. To give 1016 the next worker, cancelled the eight waiting lower-tier
jobs 1039-1046 (natives qwen TTD, gemma GRPO/TTD, muse GRPO/TTD; inspiration q/g/m). Owed re-queue in
that order once 1016 is RUNNING:
`LIST="stageC-v32-ttd-n stageB2-g-v32-grpo-n stageB2-g-v32-ttd-n stageB-m-v32-grpo-n stageB-m-v32-ttd-n stageI-q-insp-n stageI-g-insp-n stageI-m-insp-n" bash requeue_all.sh`.
Live set now (16 on workers or assigned): 716 818 819 924 925 941 993 994, 1012 1014 1015 1017 1018, 1037 1038, + 1016 waiting.

## 2026-09-18 15:57-16:06Z: self-binding deadlock found, then a reclaim wave

**Deadlock:** 819 showed `PENDING worker=978` for 35 min while 978 was READY and idle. Cause (fork,
`sky/serve/serve_utils.get_free_worker_resources` + `sky/jobs/state.get_nonterminal_job_ids_by_pool`):
a worker is "free" only if no NON-TERMINAL job's `job_info.current_cluster_name` points at it. After an
exit-33 recovery the job keeps its own binding to the worker it just left, so its controller sees
"No idle replicas" on the very worker it owns -> it waits forever (818 sat like this 12:44-14:xxZ,
819 15:19-15:58Z). Fix used: cancel the job and relaunch the same yaml (new id; GCS state resumes):
819 -> **1074**. Symptom to watch for: `PENDING worker=N` for >10 min while pool shows N READY with
that job in USED_BY. Proper fix (fork): clear `current_cluster_name` when a recover_on_exit_codes
recovery starts. (Cancelled 1016 first so 1074 would win 978; 1016 owed a re-queue.)
**Wave 15:59-16:04Z:** GCP terminated 984 (1014), 985 (1015), 994 (818), 981 (1018), 978 (just re-taken
by 818), 974 (924), 968 (994). Applied the rule: cancelled waiting 1014, 1015, 1018; evicted 1038 and
1037 (their workers went to 1074 -> 996 and 994 -> 972 within a minute); evicted 1017 for 818/924.
Kept 1012 (qwen origin control, the key control) running for now.
Owed re-queue (in order): continuations q-seed1, q-seed2, g-orig, g-seed1, g-seed2, m-orig; natives x6;
inspiration x3 -> `LIST="cont30-q-q1-v5p cont30-q-q2-v5p cont28-g-orig-v5p cont30-g-rep1-v5p cont30-g-rep2-v5p cont26-m-orig-v5p stageC-v32-grpo-n stageC-v32-ttd-n stageB2-g-v32-grpo-n stageB2-g-v32-ttd-n stageB-m-v32-grpo-n stageB-m-v32-ttd-n stageI-q-insp-n stageI-g-insp-n stageI-m-insp-n" bash requeue_all.sh`
once all eight priority jobs are RUNNING. 16:06Z: 9 READY all used, 40 provisioning.
16:07Z: wave continued -- 996 (just taken by 1074) and 968 terminated. 818 and 1074 waiting with no
idle worker; evicted 1012 (qwen origin control, 16/27, ~55 min into a step) as the last lower-tier job
holding a worker. Add `cont27-q-orig-v5p` to the front of the owed re-queue list. 16:08Z: 8 READY
(all priority: 716 941 925 993 running; 994 -> 972, 924 -> 987 recovering; 818/1074 racing for the
worker 1012 releases), 41 provisioning. Nothing lower-tier is running on v5p any more.
16:14-16:26Z: 924 -> 987, 994 -> 972, 818 -> 976 running; 1074 got a worker (starting); 716 banked
13/15 (best unchanged, val 0.86); 925 7/15 (still = parent).
**16:38Z: second wave** -- GCP terminated 1003 (941), 979 (716), 987 (924), 1001 (993), 991 (925) within
50 s. Only 994 (w972) and 818 (w976) still running; 716, 924, 925, 941, 993, 1074 waiting. Pool 3 READY
(all used), 46 provisioning. Nothing lower-tier left to cancel; waiting on new grants. Steps banked
today: 36; the 08:18Z grant effectively ended at 15:59Z (7.7 h, the longest of the week).

## 2026-09-18 17:30Z: the six native cells are now running on v6e (another session)

Jobs **1090** stagec-v32-grpo-n-v6e-resume, **1091** stagec-v32-ttd-n-v6e-resume, **1092**
stageb-m-v32-grpo-n-v6e-resume, **1093** stageb-m-v32-ttd-n-v6e-resume, **1094** stageb2-g-v32-grpo-n-v6e-resume,
**1095** stageb2-g-v32-ttd-n-v6e-resume on pool tpuswarm-v6e32-east5b-qwen35, all RUNNING at 17:43Z,
launched by another session (muse included, so that session has a muse v6e layout). They resume the
SAME GCS_RUNs as the parked v5p native yamls -> **the natives are removed from the v5p re-queue list**
(a second copy on v5p would double-run one run directory). The watch now tracks 1090-1095.
v5p owed re-queue is now continuations x7 + inspiration x3:
`LIST="cont27-q-orig-v5p cont30-q-q1-v5p cont30-q-q2-v5p cont28-g-orig-v5p cont30-g-rep1-v5p cont30-g-rep2-v5p cont26-m-orig-v5p stageI-q-insp-n stageI-g-insp-n stageI-m-insp-n" bash requeue_all.sh`.
17:47Z grant: 716 -> 1007, 924 -> 1013, 941 -> 1011, 1074 -> 986 running; 818 -> 982 and 993 -> 1000
recovering; 994 (972) and 925 (990) still running = all eight priority jobs hold workers. One idle
worker -> re-queued the qwen origin control as **1096** (cont27-q-orig) at 17:50Z; the remaining nine
(six continuations, three inspiration arms) wait until 818/993 are RUNNING. v6e natives 1090-1095 all
RUNNING on v6e workers 4511-4527 (no new steps yet).
18:02Z: 818 (982) and 993 (1000) RUNNING -> all eight priority jobs running; 1096 running on 1004.
Re-queued the nine: **1097** cont30-q-q1, **1098** cont30-q-q2, **1099** cont28-g-orig, **1100** cont30-g-rep1,
**1101** cont30-g-rep2, **1102** cont26-m-orig, **1103/1104/1105** stageI q/g/m insp.
Live set (24): 716 818 1074 924 925 941 993 994 | 1096-1102 | 1103-1105 (v5p) | 1090-1095 (v6e, other session).
18:38Z: worker 1015 arrived -> 1101 (gemma seed 2 cont). Then GCP took 1000 (993), 1011 (941) and 1008
(1097) within 3 min. Sixth application 18:40-18:48Z: cancelled waiting 1098 1099 1100 1102 1103 1104
1105 and 1097; evicted running 1101 and 1096 so 941 and 993 get workers. All ten lower-tier v5p jobs
parked again; owed re-queue = the full 10-job LIST (cont27-q-orig first) once all eight priority jobs are
RUNNING. 18:48Z: 9 READY (6 priority running: 716 818 1074 924 925 994; 941/993 racing for the two
released workers), 40 provisioning.
18:40Z v6e: all 16 v6e east5b workers failed their health probes within 15 s and were torn down --
the six natives 1090-1095 (other session) are RECOVERING; v6e pool at 19:00Z = 0 READY, 47 provisioning.
18:59Z v5p: fourth wave of the evening -- 982 (818), 990 (925), 972 (994), 1013 (924), 1007 (716)
terminated within 25 s. 19:00Z: running 941 (1004), 993 (1015), 1074 (986); waiting 716 818 924 925
994; 4 READY + 1 NOT_READY, 40 provisioning. Nothing lower-tier is running anywhere on v5p.
20:40Z: 1074 muse seed 2 banked 7/15 (0.380910491) -- first step since 15:11Z.
**20:52Z grant:** all five waiting priority jobs placed (716 -> 1022, 818 -> 1016, 924 -> 1034, 925 -> 1026,
994 -> 1014); all eight RUNNING. Re-queued the v5p lower tier. MISTAKE: ran `requeue_all.sh` with its
old default LIST, which still held the six native yamls -> 1108-1113 launched on v5p while the same runs
are live on v6e (1090-1095). Cancelled 1108-1113 within 3 min; none reached a worker (no double-run).
Default LIST fixed to exclude natives. Continuations then launched separately.
New v5p ids: **1117** cont27-q-orig, **1107** cont30-q-q1, **1118** cont30-q-q2, **1119** cont28-g-orig,
**1120** cont30-g-rep1, **1121** cont30-g-rep2, **1122** cont26-m-orig; inspiration **1114/1115/1116**.
20:57Z: nine of the ten already STARTING on workers (1005 1006 1009 1018-1024), 1119 assigning.
Live set (24): 716 818 1074 924 925 941 993 994 | 1117 1107 1118 1119 1120 1121 1122 | 1114 1115 1116 (v5p) | 1090-1095 (v6e).
20:56-21:05Z: nine workers terminated (1012 1018 1024 1030 1038 1027 1022 986 + one) but the six idle
spares absorbed them: 716 -> 1025, 1074 -> 1035, 1119 -> 1037 within minutes. 21:11Z: only 1120 (gemma
seed 1 cont, +2.96) left waiting -> evicted 1115 (gemma inspired, tier 5, 3/15) for it. 1115 owed a
re-queue. 1115 had banked 3/15 at 0.380920277 earlier in the day.
21:17-21:23Z wave: GCP took 1026 (925), 1025 (716), 1006 (1117), 1023 (1116), 1016 (818), 1037 (1119),
1005 (1107), 1020 (1122). Seventh application 21:24-21:28Z: cancelled waiting 1122 1117 1116 1119 1107;
evicted running 1114 (qwen insp), 1121 (gemma seed 2 cont), 1118 (qwen seed 2 cont) for 716/818/925.
Still running lower-tier: 1120 (gemma seed 1 cont, +2.96, kept). Owed re-queue: all seven continuations
except cont30-g-rep1 + three inspiration arms -> `LIST="cont27-q-orig-v5p cont30-q-q1-v5p cont30-q-q2-v5p cont28-g-orig-v5p cont30-g-rep2-v5p cont26-m-orig-v5p stageI-q-insp-n stageI-g-insp-n stageI-m-insp-n" bash requeue_all.sh`
once all eight priority jobs are RUNNING. Grant lengths this evening: 17:47Z (~12 min), 20:52Z (~25 min).
21:40Z: 716 -> 1019, 925 -> 1021 placed; 941 banked 5/15 (best unchanged); 1090 recovered on v6e.
**21:41-21:44Z: full reclaim, 10 workers.** v5p = 0 READY, 48 provisioning at 21:52Z. All eight priority
jobs + 1120 waiting; nothing lower-tier left to cancel. Owed re-queue unchanged (nine jobs; 1120 still live).
23:07Z grant (after 75 min dry): 716 -> 1028, 924 -> 1033, 818 -> 1032 (recovering); 1120 (gemma seed 1
cont) also grabbed 1046 while five priority jobs still waited -> evicted 1120 at 23:09Z (owed re-queue;
the owed list is now all ten lower-tier v5p jobs = requeue_all.sh default LIST). 5 READY, 44 provisioning.
23:10-00:18Z: workers trickled in one or two at a time; 924 lost 1033 (health probe) and re-landed on 1036;
941 -> 1053, 993 -> 1047, 1074 -> 1052, 994 -> 1046, 925 -> 1051. **00:18Z: all eight priority jobs RUNNING.**
Eighth re-queue (default LIST, natives excluded): **1123** cont27-q-orig, **1124** cont30-q-q1, **1125** cont30-q-q2,
**1126** cont28-g-orig, **1127** cont30-g-rep1, **1128** cont30-g-rep2, **1129** cont26-m-orig, **1130/1131/1132**
stageI q/g/m. Live set (24): 716 818 1074 924 925 941 993 994 | 1123-1129 | 1130-1132 (v5p) | 1090-1095 (v6e).
00:24-00:29Z: 12 workers terminated (1047 (993), 1053 (941), 1052 (1074), 1048 1049 1055-1058 1062 1064
1065 -- mostly freshly granted, some carrying the just re-queued lower tier). Ninth application
00:34-00:40Z: cancelled waiting 1125 1126 1127 1128 1129 1132; evicted 1130, 1131 (inspiration) and 1124
(qwen seed 1 cont) for 1074/941/993; kept **1123** (qwen origin control) running on 1029. Owed re-queue:
cont30-q-q1, cont30-q-q2, cont28-g-orig, cont30-g-rep1, cont30-g-rep2, cont26-m-orig, stageI q/g/m
(nine; pass LIST without cont27-q-orig while 1123 lives).
00:50Z: 1074 -> 1054, 941 -> 1059, 993 -> 1045 running -> all eight priority RUNNING. Tenth re-queue
(LIST without cont27-q-orig): **1133** cont30-q-q1, **1134** cont30-q-q2, **1135** cont28-g-orig, **1136**
cont30-g-rep1, **1137** cont30-g-rep2, **1138** cont26-m-orig, **1139/1140/1141** stageI q/g/m.
Live set (24): 716 818 1074 924 925 941 993 994 | 1123 1133-1138 | 1139-1141 (v5p) | 1090-1095 (v6e).
00:56Z 1051 (925) and 01:03Z 1028 (716) terminated. Eleventh application 01:06-01:09Z: cancelled the
nine waiting lower-tier jobs 1133-1141 (none had a worker); kept 1123 running (qwen origin control,
~50 min into its step; exemption still pending the user's answer). 01:10Z: 6 priority running (818
1074 924 941 993 994), 716/925 waiting alone, 8 READY used, 41 provisioning. Owed re-queue: the same
nine (LIST without cont27-q-orig).
Steps: 02:05Z 994 muse-on-qwen-seed-2 6/15 (still = parent, five real steps, zero gain); 02:16Z 818
muse seed 1 7/15 at **0.380870180** -- now BELOW qwen seed 1's finished 0.380872859, so muse is the best
parent on the seed 1 row (takeovers there must beat 0.380870, not 0.380873); 02:28Z 924 gemma on qwen
seed 1 5/15 (0.380872847, unchanged). 02:28Z: 716 -> 1070, 925 -> 1067 running -> all eight priority
RUNNING. Twelfth re-queue (LIST without cont27-q-orig): **1142** cont30-q-q1, **1143** cont30-q-q2,
**1144** cont28-g-orig, **1145** cont30-g-rep1, **1146** cont30-g-rep2, **1147** cont26-m-orig,
**1148/1149/1150** stageI q/g/m. Live set (24): 716 818 1074 924 925 941 993 994 | 1123 1142-1147 |
1148-1150 (v5p) | 1090-1095 (v6e).
02:23-02:32Z wave: nine workers terminated (1063 1068 1059 (941) 1072 1036 (924) 1042 1029 (1123) 1046
(994) 1054 (1074) 1060). 924 re-landed on 1061. Twelfth application 02:34-02:41Z: cancelled the nine
waiting 1142-1150 and the now-waiting 1123 (its worker died; ~2 h into a step, one step short of banking).
Owed re-queue: all ten (requeue_all.sh default LIST) once the eight priority jobs are RUNNING.
02:38Z: a grant landed -- all eight priority jobs assigned (716 818 925 993 running; 1074 -> 1029,
994 -> 1054, 924 -> 1046, 941 -> 1071 recovering), 5 READY idle, 25 provisioning. Re-queued the top
five continuations onto the idle workers: **1151** cont27-q-orig, **1152** cont30-q-q1, **1153** cont30-q-q2,
**1154** cont28-g-orig, **1155** cont30-g-rep1. Still owed once the four restarting jobs are RUNNING:
cont30-g-rep2, cont26-m-orig, stageI q/g/m.
Correction: 1123 (qwen origin control) HAD banked 17/27 at 0.380857092 (+0.49 total) before its worker died.
02:53Z: the workers assigned to 924/994/1074 at 02:38Z (1046/1054/1029) were already-terminated replicas,
so those three fell back into the race while 1152/1153 had landed (1078/1079) and 1151/1154/1155 waited.
Thirteenth application 02:54-02:57Z: cancelled 1151 1154 1155, evicted 1152 1153. 8 READY used, 41
provisioning. Owed re-queue: all ten (default LIST).
02:56Z: 924 -> 1078, 994 -> 1079 (the workers vacated by 1152/1153). 03:23-03:28Z: 1067 (925), 1070
(716), 1045 (993) terminated. 03:30Z: 994's cell exited 33 on the dirty 1079 and then DEADLOCKED on its
own binding ("No idle replicas" while bound to 1079) -- second occurrence of the self-binding bug.
Cancelled 994, relaunched gen1-m-on-q2 as **1156** (registry clean, 6/15). Priority set is now
716 818 1074 924 925 941 993 **1156**. 03:34Z: running 818 924 941; waiting 716 925 993 1074 1156.
03:39Z: the last three (1032, 1078, 1071) terminated too -> v5p 0 READY, 48 provisioning at 03:48Z; all
eight priority jobs waiting, all ten lower-tier parked. The 02:38Z grant lasted ~1 h with no step banked.
**05:46Z grant** after ~2 h dry: all eight priority jobs assigned (818 -> 1080 running; 716 -> 1087,
924 -> 1082, 925 -> 1084, 941 -> 1085, 993 -> 1093, 1074 -> 1094, 1156 -> 1077 recovering); 2 READY idle,
2 STARTING, 30 provisioning. Re-queued the top two continuations onto the idle pair: **1163** cont27-q-orig
(17/27), **1164** cont30-q-q1. Still owed once the seven restarts are RUNNING: cont30-q-q2, cont28-g-orig,
cont30-g-rep1, cont30-g-rep2, cont26-m-orig, stageI q/g/m.
05:46-05:57Z: 17 workers terminated as the grant landed (1084 under 925, 1077 under 1156, ...). 06:02Z:
716 818 941 993 1074 running; 924 -> 1100, 1156 -> 1098 recovering; 1163 -> 1103 starting; 1164 running
on 1096; 925 waiting with no idle worker -> evicted 1164 (just started) for it; kept 1163. Owed
re-queue: cont30-q-q1 + the six above + stageI q/g/m (nine).
**06:13Z: all eight priority RUNNING** (716 -> 1096, 818 1080, 1074 1094, 924 1100, 925 1087, 941 1085,
993 1093, 1156 1098) + 1163 on 1103. Fourteenth re-queue (LIST without cont27-q-orig): **1165** cont30-q-q1,
**1166** cont30-q-q2, **1167** cont28-g-orig, **1168** cont30-g-rep1, **1169** cont30-g-rep2, **1170**
cont26-m-orig, **1171/1172/1173** stageI q/g/m. Live set (24): 716 818 1074 924 925 941 993 1156 | 1163
1165-1170 | 1171-1173 (v5p) | 1090-1095 (v6e).
06:27-06:41Z: 1165-1171 all placed; 1170 on 1109; 1172 pending, 1173 starting. 06:40Z: 1163 (qwen origin
control) exited 1 on 1103 and then looped "No idle replicas" (self-binding). Cause: TORN ROW again --
stageC-pwc-n registry row 000017 (written at the 02:31Z termination) has no archive; WEIGHTS-RESET
marker present. Cancelled 1163; will drop the torn row and relaunch (resumes from 000016, redoing one step).
06:44Z: dropped row 000017 (backup `checkpoints.jsonl.bak-torn-20260919T064421Z`), registry verified
clean (16 rows, 000016 archive OK), relaunched as **1174** at 06:45Z. Lesson reinforced: after any
termination that hits a run mid-save, run `ckpt_torn.py` on that run BEFORE re-queuing it; a torn row
costs a failed landing plus a self-binding deadlock. Live lower-tier: 1174 1165-1170 1171-1173.
06:59Z: all 18 v5p jobs placed for the first time since 20:52Z (1172/1174 starting, 1173 on 1103).
**07:16-07:23Z wave:** 1100 (924), 1103 (1173), 1106 (1171), 1105 (1166), 1117 (1168), 1080 (818),
1110 (1169), 1096 (716), 1085 (941) terminated. Fifteenth application 07:22-07:27Z: cancelled waiting
1171 1172 1173 1168 1166 1174; evicted running 1165 1167 1170 for 716/818/924/941. Owed re-queue: all ten
(default LIST) -- and run `ckpt_torn.py stageG-q-rep1 stageG-q-rep2 stageG-g-rep1 stageG-g-rep2 stageB2-g-pwc-n stageB-m-pw-n stageC-pwc-n`
FIRST, since several continuations were killed mid-run.

**Sixteenth cycle, 2026-09-19 13:28-13:45Z:** all eight priority jobs RUNNING again since the
~10:50-11:20Z grants (716 on 1113, 818 on 1114, 1074 on 1119, 924 on 1108, 925 on 1116, 941 on 1111,
993 on 1123, 1156 on 1112); 1169 (cont30-g-rep2) survived the 07:16Z wave via recovery and runs on
1122. Pool 9/48 READY, no idle worker. `ckpt_torn.py` on the six killed continuations: every last
row's archive OK (old WEIGHTS-RESET markers on stageG-q-rep1/q-rep2/stageC-pwc-n are informational).
Re-queued the nine parked jobs 13:40Z: **1175** cont27-q-orig, **1176** cont30-q-q1, **1177**
cont30-q-q2, **1178** cont28-g-orig, **1179** cont30-g-rep1, **1180** cont26-m-orig, **1181/1182/1183**
stageI q/g/m. Natives: the other session replaced 1090-1095 with **1157-1162** (v6e east5b, all
STARTING -- that pool has 1/48 READY). Banked since 08:12Z: nothing new (925 at 9/15 was the last);
716 is mid-step 14, cell up since ~11:20Z. Record unchanged at 0.380856777 (716, 13/15).
Worker2 of 716 logs `/tmp/ray_skypilot ... over 95% full` (5 GB free of 104 GB) -- watch for a disk-full
exit on that cell.


**Seventeenth cycle, 2026-09-19 13:59-14:05Z:** workers 1114 (818) and 1119 (1074) terminated at 13:59Z --
both muse gen-0 seeds (tier 1) RECOVERING, pool down to 7 READY, no idle worker. Applied the rule:
cancelled the nine waiting lower-tier jobs 1175-1183 (free) and evicted running 1169 cont30-g-rep2
(18/30, mid-step 19) so a muse seed gets worker 1122. Expect one dirty landing (exit 33) on 1122 and
check for the `PENDING worker=1122` deadlock symptom. Owed re-queue when 818 and 1074 are RUNNING: all
ten (default LIST of requeue_all.sh) -- run `ckpt_torn.py stageG-g-rep2` first (killed mid-step).
Banked 13:58Z: 924 -> 6/15, 1156 -> 8/15, 1169 -> 18/30; no best changed.


14:20Z: 716 banked 14/15 (record unchanged 0.380856777, one step left). **993 muse-on-qwen-seed-1 banked 3/15 with a new
best 0.380872593** -- first real gain on the seed-1 row: +0.27e-6 vs qwen seed 1 at 15 (0.380872859), up from +0.01
(gemma child 924 at 6/15 is +0.012; gemma alone 21/30 is +2.96 vs its own parent). 1074 landing on 1122 (RECOVERING
worker=1122) 17 min after the eviction; 818 still without a worker (7/48 READY, all bound).


**14:25-14:33Z reclaim wave:** workers 1108 (924), 1111 (941), 1112 (1156), 1122 (1074, mid-landing), 1123 (993)
terminated. Pool 2/48 READY: only 716 (1113, at 14/15) and 925 (1116, 10/15) still run. Six priority jobs waiting
(818 1074 924 941 993 1156); no lower-tier job is live, so nothing to cancel -- waiting on fresh grants.


**Eighteenth cycle, 2026-09-19 16:05-16:15Z:** grant burst -- pool 2 -> 10 READY. All eight priority jobs placed:
716 (1113, RUNNING 14/15), 925 (1116, RUNNING 11/15), 818 -> 1120, 941 -> 1126, 993 -> 1115, 1156 -> 1090,
924 -> 1099, 1074 -> 1127 (all six RECOVERING = landing on fresh workers). Two idle READY workers (1132, 1137), so
per the "as many as idle workers" rule re-queued the two origin-row controls at 16:13Z: **1187** cont27-q-orig
(17/27), **1188** cont28-g-orig (19/28). Torn-row check: stageG-g-rep2 000018 archive OK (evicted mid-step but
clean), stageC-pwc-n / stageB2-g-pwc-n OK. Still owed once the six landings are RUNNING: cont30-q-q1, cont30-q-q2,
cont30-g-rep1, cont30-g-rep2, cont26-m-orig, stageI q/g/m (LIST for requeue_all.sh).


**Nineteenth cycle, 2026-09-19 16:27-16:35Z:** pool 10 -> 17 READY (user: "we have 17 machines!!!"). Landings
done: 818 (1120), 941 (1126), 993 (1115), 1156 (1090) RUNNING; 924 (1099) and 1074 (1127) still RECOVERING on
fresh workers; 1187 -> 1132, 1188 -> 1121 STARTING. Seven idle workers, so re-queued seven of the eight owed at
16:33Z: **1189** cont30-q-q1 (17/30), **1190** cont30-q-q2 (20/30), **1191** cont30-g-rep1 (21/30), **1192**
cont30-g-rep2 (18/30), **1193** cont26-m-orig (15/26), **1194** stageI-q-insp, **1195** stageI-g-insp. Held back:
stageI-m-insp (lowest tier) until 924/1074 are RUNNING and a worker is idle. 925 banked 12/15 at 16:14Z (no change).
Live v5p set: 716 818 1074 924 925 941 993 1156 | 1187 1188 1189 1190 1191 1192 1193 | 1194 1195.


16:45Z: everything placed -- all 8 priority RUNNING (924 on 1099, 1074 on 1127 landed clean) and all nine re-queued
lower-tier jobs RUNNING (1187/1132, 1188/1121, 1189/1095, 1190/1102, 1191/1118, 1192/1124, 1193/1133, 1194/1135,
1195/1137). 17/17 workers busy. Re-queued the last owed job, **1196** stageI-m-insp (waits for the next idle
worker). Nothing is parked; owed list empty. Next reclaim wave: cancel WAITING 1196 first, then evict inspiration
arms 1194/1195, then continuations, lowest tier first.


**16:58Z: 716 (muse takes over qwen ORIGIN) SUCCEEDED 15/15.** Final best 0.380856777 (set at 13/15, unchanged by
steps 14-15). Origin row is now closed on the takeover side: +0.81e-6 vs qwen origin at 15 (0.380857586), +2.40 vs
muse origin at 15 (0.380859181); the origin controls are still running (qwen alone 17/27 at 0.380857092 = +0.49,
gemma alone 19/28 at 0.380861153). Record for the whole effort remains 0.380856777. Worker 1113 is released; 1196
stageI-m-insp landed on 1128 (18 READY). Priority set is now seven jobs: 818 1074 924 925 941 993 1156.


## 2026-09-19 17:30Z: strategy correction (user) -- children branch from the BEST gen-0 parent, not qwen

User: "our algo is that we pick the best at gen0 and continue, i shouldn't have asked to start off qwen for each."
Seed 1's best gen-0 parent is muse (0.380870180 at 7/15, already below qwen's 15-step final 0.380872859), so the
seed-1 children must be qwen-takes-over-muse and gemma-takes-over-muse, seeded from muse seed 1's tree at step 15
(user: wait for 818 to finish; do not branch from step 7). **Cancelled 17:34Z: 993 gen1-m-on-q1 (3/15) and 924
gen1-g-on-q1 (6/15)** -- wrong-parent arms. Their workers went to 1192 and 1195 within a minute.
Seed 2: qwen leads at 15 by 14.5 (gemma) and 29.4 (muse at 7/15); decide the parent when 1074 reaches 15/15.
Gemma-takes-over-qwen seed 2 (925, 13/15, zero gain) config verified: piecewise_valid_entropic_centered, lr 4e-5,
weights model_e619a0df/000015 = stageG-g-rep2 registry row 15 (gemma seed 2 at step 15), seed = top-48 of
stageG-q-rep2 at 15. Not a config error. The zero is a search collapse: from batch 1 on, sampled_value mean ==
sampled_value max == 0.380881107 to nine digits, i.e. every valid rollout reproduces the incumbent exactly; the
muse child on the same tree (1156) shows the identical signature. The qwen seed-2 parent itself was converging to
that state (sampled mean within 2e-8 of max from batch 13; +1.3e-6 total over steps 7-15, +0.08 over 5 extension
steps). A plain rerun of gen1-g-on-q2 would almost certainly reproduce 0.00; only a different seed slice, restarts
(TTD_RESTART_RATIO>0) or a different parent could change it. Muse step cost: 2.5 h compute per step (sampling
96%, 8290 action tokens/turn vs qwen 3747), 6 engines at TP=2 = same 12 chips as qwen's 3 at TP=4.
Pool after the cancels: 15 READY, all bound (818 1074 941 925 1156 | 1187-1193 | 1194-1196), 0 idle.
Prepared next (launch when 818 hits 15/15): gen1-q-on-m1, gen1-g-on-m1 (top-48 of stageG-m-rep1@15; qwen seed-1
step-15 weights model_d6979b8a? -- verify row 15 id; gemma seed-1 step-15 weights model_1062a2d1/000015) and
cont30-m-rep1-v5p (muse seed 1 continues alone to 30).


**17:37-17:56Z: full reclaim.** All 15 workers terminated; pool 0/48 READY (user noticed first). Every live job
RECOVERING/PENDING, none FAILED. Twentieth application 17:59-18:00Z: cancelled all ten waiting lower-tier jobs
(1187-1196) so the next grants go to the five priority jobs 818 1074 941 925 1156. Torn-row sweep over all 15
runs: one torn row, stageI-g-insp-n 000003 (model_67f425b6 archive missing) -- dropped 18:00Z after 1195 was
CANCELLED (backup checkpoints.jsonl.bak-torn-20260919T180034Z); registry now 2 rows, resumes from step 2 (loses one
inspiration step). Owed re-queue when all five priority jobs are RUNNING: cont27-q-orig cont28-g-orig cont30-q-q1
cont30-q-q2 cont30-g-rep1 cont30-g-rep2 cont26-m-orig stageI-q/g/m-insp (default LIST of requeue_all.sh).


**Twenty-first cycle, 18:40-18:47Z: grant burst, 0 -> 22 workers (18 READY + 4 STARTING).** Five priority jobs
landing (818/1165, 1074/1149, 941/1139, 925/1138, 1156/1156); 13 idle READY, so re-queued all ten owed jobs at
18:45Z with no prioritisation needed: **1198** cont27-q-orig, **1199** cont30-q-q1, **1200** cont30-q-q2, **1201**
cont28-g-orig, **1202** cont30-g-rep1, **1203** cont30-g-rep2, **1204** cont26-m-orig, **1205/1206/1207** stageI q/g/m.
Live v5p set: 818 1074 925 941 1156 | 1198-1204 | 1205-1207 (15 jobs, ~7 spare workers). Owed list empty.
Note for the next scarcity: under the best-parent rule the seed-1 qwen/gemma continuations (1199, 1202) rank
below the origin-row and seed-2 controls; evict them before those.


## 2026-09-18 03:1xZ decisions

- **Muse children on qwen seeds 1/2 use the muse ORIGIN weights** (user: "yes i agree thats all we
  need go ahead, idw to wait"). `gen1-m-on-q1.yaml` / `gen1-m-on-q2.yaml` are clones of
  `stageH-m-on-qpwc.yaml` (CELL m-pw-n, LOO piecewise, `META_INIT_STATE_PATH
  tinker://model_df2f51fc/weights/000011`, `META_SEED_ONLY 1`, 15 steps) with the run name changed.
  Tree seeds are the SAME files the gemma children use (top-48 of stageG-q-rep1 snapshot 14 and
  stageG-q-rep2 snapshot 15), staged as `puct_sampler_step_000000.json` under each new GCS_RUN.
  Launched 03:19Z as jobs **993** (seed 1) and **994** (seed 2); both had workers (940, 942) within a
  minute. The "Muse takes over" column for seeds 1 and 2 is therefore live; 818/819 no longer gate
  anything and continue only for the muse seed-spread result. Interpretation: one fixed set of muse
  weights on all three qwen trees, so only the tree varies across rows.
- **API-server capacity restart shelved** (user: "we can pass on this"). Plan stays in
  `tpu/swarm/SKYPILOT_RESTART_PLAN_2026-09-17.md`; nothing was changed on the server.
- Muse-continues-alone for seeds 1/2 stays unqueued (would need 818/819 to finish first).

## 2026-09-18 03:21Z: qwen seed 1 continuation (926) FAILED -- torn registry row

926 (cont30-q-q1-v5p, resumes stageG-q-rep1) went FAILED with return codes [0, 0, 1, 0]: the client
on rank 0 exited 1 about 30 min after every restart, never banking an extension step. Cause: the last
row of `member_qwen/checkpoints.jsonl` (name 000015, `tinker://model_58d62cab/weights/000015`) was
written at the 2026-09-17 07:51Z preemption but its archive `skyrl-checkpoints/model_58d62cab/000015.tar.gz`
never made it to GCS (only an empty `sampler_weights/` prefix). On resume `ensemble.py` cannot
restore that state, records the member as `start_batches=[None]`, appends a `WEIGHTS-RESET` marker
in the run dir, and the strict resume guard (`TTD_RESUME_STRICT=1`) refuses to start with fresh
weights -> rc 1 -> the managed job treats it as a user-program failure (no recovery). Four such
attempts, four WEIGHTS-RESET lines.

Fix: `jobs/f6d76b15/tmp/drop_torn_rows.py stageG-q-rep1 --apply` dropped the one row whose archive is
missing (backup `checkpoints.jsonl.bak-torn-20260918T032823Z`); the run now resumes from
`model_661d0470/000014` and redoes batch 14 (one qwen step, ~2 h). Relaunched as **998**.
`jobs/f6d76b15/tmp/ckpt_torn.py <runs...>` checked all 16 runs: only stageG-q-rep1 had a torn last row
(stageG-q-rep2 and stageI-m-insp-n carry old WEIGHTS-RESET markers from earlier episodes but their
archives are intact and they have banked since). Gotcha for the notes: a FAILED (not RECOVERING)
managed job right after a resume, with rc list [0,0,1,0], means "check the registry's last row's
archive", and the fix must be applied while the job is terminal (a live worker's final rsync would
put the old registry back).

## Gen-0 reference values (best within the first 15 batches)

| Model | Origin | Seed 1 | Seed 2 |
|---|---|---|---|
| qwen | 0.380857128 (15/15) | 0.380872859 (15/15) | 0.380881108 (15/15) |
| gemma | 0.380863326 (15/15) | 0.380874732 (15/15) | 0.380895593 (15/15) |
| muse | 0.380859181 (15/15, best at 13) | 0.380870180 (7/15) | 0.380910491 (7/15) |

Origins are the best of each model's three draws. Nothing in the stack is seeded
("seed replication" = identical yaml, fresh run name; `create_initial_state` uses an
unseeded `default_rng()`, LoRA init draws a random seed, vLLM samples unseeded).

## Continuation caps

Controls were queued as "old batch count + 15", so caps differ: qwen origin 27,
gemma origin 28, muse origin 26, all seed replications 30. If a 45-step column is
wanted later, a run is extended in place by resuming the same `GCS_RUN` with a higher
`NUM_EPOCHS` once it reaches its current cap.
