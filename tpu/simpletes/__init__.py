"""SimpleTES (arXiv 2604.19341) as an inference-only baseline on our problems.

Only the search loop is SimpleTES (vendored unmodified in third_party/simpletes).
Generation goes through our native-thinking ingress with the RL client's renderers
and per-model budgets (backend.py); grading goes through the same discover
Environment._safe_grade path the RL client uses (evaluator.py). The driver runs on
the head of an inference-only Ray v2 job (frozen_benchmark.driver == "simpletes").
"""
