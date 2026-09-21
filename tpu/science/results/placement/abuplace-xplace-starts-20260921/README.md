# Three AbuPlace Xplace starting layouts, 2026-09-21

51 independent tasks: all 17 IBM cases times off, rudy, rudy_hv.
Uses pinned AbuPlace a24087c45588f1873eb2dc18293e407e5477041d,
its unmodified Xplace Python code and GPConfig defaults (including GP seed 0).
Calls the upstream LEF/DEF conversion, parser, GP and coordinate mapping.
No Adam refinement, congestion cleanup, polish, escape or discrete operator stage.
Each raw output is preserved, legalized with the existing seed-42 legalizer,
and scored by the independent canonical scorer. Nonfinite GP metrics fail closed.

Slurm gpu-test: four A100 GPUs, 32 CPUs, 64 GiB per one-hour allocation.
Local Ray schedules four workers with one GPU and eight CPUs each.
Candidate cap 1800s including legalization; full task envelope 1980s.
These are preprocessing caps, distinct from the generated program's 300s search budget.
Pilot 14241324 precedes the array. Array allocations require all three ibm01
pilot variants to pass independent grading before claiming further work.
Completed invalid results are terminal, preserved and block portfolio publication.

Training publication requires all 51 successful reports. It supplies arrays
starting_layouts[3,M,2], starting_scores[3,4], starting_names and score-column names.
The old initial_positions remain unchanged as a legal fallback. Precomputation
costs and input hashes are recorded; no per-rollout GPU preprocessing is needed.
Existing single-start runs and public baseline reproductions remain unchanged.

## Training destination

User changed the destination to the existing tpuswarm-v6e32-central1b pool.
The three circuit300-v6e-*-central-three-starts-10step-20260921 profiles use
v6e-32 topology from the current v6e model recipes, fresh 1024-draft bootstraps
(target 512 valid distinct seeds), 16 groups x 32 rollouts and ten training steps.
They borrow the inference farm and use 32 four-CPU/four-GiB grading slots per host.
There are eight hosts per v6e-32 slice, so the scheduling ceiling is 256 case
tasks per slice; actual CPU and RAM preflight must pass on each allocated host.
Submission remains gated on complete independently verified Xplace publication.
Publication job 14241687 depends on array 14241497, which depends on pilot 14241324.
No Central training job has been submitted yet.

The pending pilot allocation was shortened to 45 minutes for backfill eligibility;
its 1800s candidate and 1980s task caps remain unchanged. Remaining array
allocations still request one hour. No job was cancelled or replaced.
