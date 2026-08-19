"""Single source of truth for the rf3 evolution smoke: which cells, which
graded case lists. gen_smoke renders prompts from EXACTLY these names and
grade_smoke validates against the same problems -- prompt/contract drift is
structurally impossible (the lesson of the stale TASK_CASES KeyError that
killed a landed v6e-8 run).

Phase-1 case lists: grouping (GQA/MQA) and asymmetric d_v are day-1 contract
-- they are shape dims. Feature cases (window/soft_cap/sinks) and TP cases
are phase 2; the kernel SIGNATURE already carries the feature kwargs so
phase 2 is not a contract break.
"""

SPLASH_P1 = [
    "probe-h8-s4096", "probe-h4-s2048", "probe-h16-s1024", "probe-h8-s4096-d64",
    "mixtral-8x7b-gqa32x8-s4096", "mqa-h32kv1-s4096",
    "deepseek2-16b-s1024-d192-dv128",
    "probe-holdout-h4-s2049", "mixtral-holdout-gqa32x8-s2049",
]
RGLRU_P1 = [
    "probe-4x2048x2560", "probe-2x1024x2560", "probe-8x512x2560",
    "probe-2x4096x2560", "probe-4x2048x1024", "probe-holdout-2x1500x2560",
]

# (task, variant) -> (graded case names, include minimal pallas example)
CELLS = {
    ("splash_attention", "rf3"): (SPLASH_P1, False),
    ("splash_attention", "rf3e"): (SPLASH_P1, True),
    ("rg_lru", "rf3"): (RGLRU_P1, False),
    ("rg_lru", "rf3e"): (RGLRU_P1, True),
}
