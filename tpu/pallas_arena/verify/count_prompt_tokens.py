"""Token cost of the seeded prompts, under BOTH models' tokenizers.

Budgeting has been done on qwen-token estimates, which already mispredicted
gemma badly enough to 400 every request in an earlier run. This measures the
real thing: the whole prompt, and the seed program inside it, per model.

    python3 count_prompt_tokens.py
"""
import os
import sys

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
sys.path.insert(0, f"{REPO}/tpu")
sys.path.insert(0, f"{REPO}/tpu/pallas_arena/probe")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# RAW tokenizers, not transformers: AutoTokenizer.from_pretrained hung past a
# 20-minute slurm limit on this cluster even with HF_HUB_OFFLINE=1, and
# tokenizer.json is all that is needed to count.
from tokenizers import Tokenizer

from pallas_arena.probe.prompt_ref_first import SEED_IMPROVE_TEMPLATE, build3seed
from pallas_arena.judge.problems import rg_lru, splash_attention

QWEN_JSON = sys.argv[1] if len(sys.argv) > 1 else None
GEMMA_JSON = sys.argv[2] if len(sys.argv) > 2 else None

SEEDS = {
    "rg_lru": f"{REPO}/tpu/pallas_arena/probe/seed_rglru_active.py",
    "splash_attention": f"{REPO}/tpu/pallas_arena/probe/seed_splash_flash.py",
}
OBS = {
    "rg_lru": f"{REPO}/runs/pallas_arena/seed-obs-rglru.txt",
    "splash_attention": f"{REPO}/runs/pallas_arena/seed-obs-splash.txt",
}
REWARD = {"rg_lru": "1.000x", "splash_attention": "0.262x"}
PROBLEMS = {"rg_lru": rg_lru.PROBLEM, "splash_attention": splash_attention.PROBLEM}

toks = {}
for name, path in (("qwen3.5-27B", QWEN_JSON), ("gemma-4-31B", GEMMA_JSON)):
    if not path:
        continue
    try:
        toks[name] = Tokenizer.from_file(path)
        print(f"[ok] {name} tokenizer loaded", flush=True)
    except Exception as e:
        print(f"[warn] {name} tokenizer unavailable: {type(e).__name__}: {e}", flush=True)

CTX = {"qwen3.5-27B": 32768, "gemma-4-31B": 16384}

for task, seed_path in SEEDS.items():
    program = open(seed_path).read()
    obs = open(OBS[task]).read().strip() if os.path.exists(OBS[task]) else "(no observation)"
    cases = [c.name for c in PROBLEMS[task].shape_cases() if not getattr(c, "probe", False)] or None
    base = build3seed(task, cases)
    prompt = SEED_IMPROVE_TEMPLATE.format(
        base=base, reward=REWARD[task], program=program, observation=obs)

    print(f"\n=== {task} ===")
    print(f"  chars: whole prompt {len(prompt):6d} | seed program {len(program):6d} "
          f"| observation {len(obs):5d}")
    for name, tk in toks.items():
        enc = lambda s: len(tk.encode(s, add_special_tokens=False).ids)
        n_prompt = enc(prompt)
        n_prog = enc(program)
        n_base = enc(base)
        n_obs = enc(obs)
        ctx = CTX[name]
        n_chat = n_prompt + 8   # chat wrapper is a handful of control tokens
        print(f"  [{name}] prompt {n_prompt:6d} tok (+~8 chat wrapper)")
        print(f"      of which: task/instructions {n_base:5d} | SEED PROGRAM {n_prog:5d}"
              f" | judge observation {n_obs:4d}")
        print(f"      leaves {ctx - n_chat:6d} of {ctx} context for thinking + answer")
