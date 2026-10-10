# Deliverable manifest

Primary code: `candidates/diverse_ensemble_jax.py` (three-branch diversity),
`candidates/hybrid_physics_jax.py` (JAX neural+physics),
`candidates/{multiscale_jax,spectral_jax,baseline_jax}.py` (ablations).

Coding-agent prompt: `prompts/READY_TO_PASTE.md` or short split
`SYSTEM_V2.md` + `TASK_V2.md`; automated iteration:
`python scripts/create_agent_prompt.py --previous path/to/development/reward.json`.

Evaluate: `python -m fairdiscover.trial --candidate candidates/diverse_ensemble_jax.py --stage data/staged_full --run-dir results/trial_001 --steps 250 --batch-size 16`.

Codabench predictions: `python -m fairdiscover.submission ...` with the official
noisy test data. The ZIP contains `result.json` with `ood_scores`; it does not
contain training code or JAX weights.

`evidence/` includes synthetic development and locked-holdout score summaries;
no weights or real FAIR dataset are included. `RESEARCH_PLAYBOOK_V2.md`
documents experimental conclusions and future hypotheses.

WARNING: no official FAIR leaderboard evaluation has been performed; even a
high proxy score does not prove state-of-the-art. JAX CPU trials are tested;
actual TPU/GPU runs and full 6.8GB training dataset remain unverified here.
