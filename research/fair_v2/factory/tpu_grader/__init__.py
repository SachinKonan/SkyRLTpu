"""Self-contained TPU grader for FAIR feature candidates (numpy + jax only).

CLI: python -m tpu_grader.grade <candidate.py> --out <dir> --mode rl|full|screen --stack v1|v2
API: from tpu_grader import grade as G; G.init(chip, stack); G.grade_source(src, mode)
(The package import is kept empty so `python -m tpu_grader.grade` loads grade.py exactly once.)
"""
