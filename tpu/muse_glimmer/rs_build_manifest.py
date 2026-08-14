#!/usr/bin/env python
"""Build the reasoning-strength experiment manifest.

Prompts are produced by the SAME machinery the Qwen3.5 sweep used, and that is
the whole point of this file existing separately from the generation client:

    ProblemGroupBuilder.env_thunk()  ->  env_type(renderer, initial_state=..., ...)
    env.initial_observation()        ->  renderer.build_generation_prompt(
                                             [{"role": "user",
                                               "content": env.get_question()}])

Hand-writing a prompt here would turn the experiment into a comparison of
prompts rather than of models. So we construct the real ``DatasetConfig``, the
real env class, and the real renderer, and we call the real
``initial_observation``. The ONLY difference between the two arms is the
``renderer_name`` (``muse_glimmer_high_reasoning`` vs
``muse_glimmer_xhigh_reasoning``), which changes exactly one system-prompt line.

Parent states come from the recorded PUCT pools. We take the EARLIEST step at
which the pool contains states with non-empty code, so the Qwen policy that
produced them had as little task-specific RL as possible -- comparing an
untrained Muse-Glimmer against a step-14 Qwen would be rigged.

Output is one JSON file per problem holding
  * ``prompts`` -- one entry per (state, arm), token ids + rendered text,
  * ``items``   -- one entry per rollout, referencing a prompt key,
  * ``states``  -- the full parent State dicts, so the grader can rebuild them.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
DISCOVER = os.path.join(REPO, "third_party", "discover")
for _p in (DISCOVER, os.path.join(DISCOVER, "examples")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MUSE_DIR = os.environ.get(
    "MUSE_GLIMMER_DIR", "/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b"
)

# problem -> (env import path, class, problem_type, pool runs, earliest timestep
#             whose states carry code, maximize)
PROBLEMS = {
    "erdos": dict(
        module="erdos_min_overlap.env",
        cls="ErdosMinOverlapEnv",
        problem_type="",
        runs=["sweep1-ttd-erdos", "sweep1-grpo-erdos"],
        early_timestep=1,
        maximize=False,
        eval_timeout=1100,
    ),
    "jssp": dict(
        module="frontier_algo.env",
        cls="FrontierAlgoEnv",
        problem_type="46",
        runs=["sweep1-ttd-jssp", "sweep1-grpo-jssp"],
        early_timestep=0,
        maximize=True,
        eval_timeout=1100,
    ),
    "ac1": dict(
        module="ac_inequalities.env",
        cls="AutoCorrInequalityEnv",
        problem_type="ac1",
        runs=["sweep1-ttd-acineq", "sweep1-grpo-acineq"],
        early_timestep=0,
        maximize=True,
        eval_timeout=1100,
    ),
}

ARMS = {
    "high": "muse_glimmer_high_reasoning",
    "xhigh": "muse_glimmer_xhigh_reasoning",
}


class _SamplerStub:
    """``Environment.__init__`` refuses ``sampler=None``; ``get_question`` never
    touches it. A stub keeps us on the real constructor path without dragging a
    PUCT archive (and its disk writes) into a prompt-building script."""

    def update_states(self, *a, **k):  # pragma: no cover - never called here
        raise AssertionError("prompt building must not mutate a sampler")


def load_pool(pool_dir: str, run: str, step: int) -> dict:
    path = os.path.join(pool_dir, run, f"puct_sampler_step_{step:06d}.json")
    with open(path) as f:
        return json.load(f)


def _split(n, k):
    """Spread ``n`` picks over ``k`` runs, remainder to the earlier runs.

    ``n // k`` silently returns 0 for the n=1 start-state case, which would
    build a manifest with no from-scratch arm at all -- the exact thing this
    round is meant to report separately.
    """
    return [n // k + (1 if i < n % k else 0) for i in range(k)]


def pick_states(pool_dir, spec, n_parents, n_seeds, seed):
    """Deterministically choose parent states and start-state seeds.

    Balanced across the ttd and grpo pools so the sample is not an artefact of
    one advantage estimator. Selection is over states with non-empty code and a
    recorded value, sorted by id before sampling so the choice is reproducible
    from the seed alone.
    """
    rng = random.Random(seed)
    parents, seeds, children = [], [], {}
    runs = spec["runs"]
    n_p = _split(n_parents, len(runs))
    n_s = _split(n_seeds, len(runs))
    for i, run in enumerate(runs):
        # Early pool: the step file that first contains `early_timestep` states.
        early = load_pool(pool_dir, run, spec["early_timestep"] + 1)

        cands = [
            s
            for s in early["states"]
            if s["timestep"] == spec["early_timestep"]
            and (s.get("code") or "").strip()
            and s.get("value") is not None
        ]
        cands.sort(key=lambda s: s["id"])
        chosen = rng.sample(cands, min(n_p[i], len(cands)))
        for s in chosen:
            s = dict(s)
            s["_run"] = run
            parents.append(s)

        seed_cands = sorted(early["initial_states"], key=lambda s: s["id"])
        for s in rng.sample(seed_cands, min(n_s[i], len(seed_cands))):
            s = dict(s)
            s["_run"] = run
            seeds.append(s)
        del early

        # The step-15 pool is where Qwen's recorded children live. It is 400 MB
        # for acineq, so it is loaded AFTER the parents are chosen, reduced to
        # just those parents' children, and dropped again -- otherwise two of
        # them are resident at once for no reason.
        want = {s["id"] for s in chosen} | {s["id"] for s in seeds if s["_run"] == run}
        try:
            full = load_pool(pool_dir, run, 15)
        except FileNotFoundError:
            print(f"  [{run}] no step-15 pool; Qwen children unavailable")
            for sid in want:
                children.setdefault(sid, [])
            continue
        for sid in want:
            children.setdefault(sid, [])
        for s in full["states"]:
            ps = s.get("parents") or []
            if ps and ps[0]["id"] in want:
                children[ps[0]["id"]].append(s)
        del full
    return parents, seeds, children


def build(args):
    from ttt_discover.tinker_utils.dataset_builder import DatasetConfig
    from ttt_discover.tinker_utils.state import State
    import importlib

    spec = PROBLEMS[args.problem]
    mod = importlib.import_module(spec["module"])
    env_type = getattr(mod, spec["cls"])

    parents, seeds, qwen_children = pick_states(
        args.pool_dir, spec, args.n_parents, args.n_seeds, args.seed
    )
    print(f"[{args.problem}] parents={len(parents)} seeds={len(seeds)}")

    prompts, items, states_out = {}, [], {}
    for arm, renderer_name in ARMS.items():
        config = DatasetConfig(
            problem_type=spec["problem_type"],
            env_type=env_type,
            batch_size=args.n_parents,
            model_name_for_tokenizer=MUSE_DIR,
            renderer_name=renderer_name,
            group_size=args.rollouts,
            eval_timeout=spec["eval_timeout"],
            log_path=args.log_path,
            convo_prefix=None,
        )
        # Same construction path SingleProblemDataset._make_env_group_builder
        # uses: tokenizer -> renderer by NAME out of the registry.
        from ttt_discover.tinker_utils import renderers
        from ttt_discover.tinker_utils.misc_utils import get_tokenizer

        tokenizer = get_tokenizer(config.model_name_for_tokenizer)
        renderer = renderers.get_renderer(config.renderer_name, tokenizer=tokenizer)
        # Pin the date so the prompt is reproducible and prefix-cacheable.
        renderer.current_date = args.current_date

        # A tiny question rendered through the SAME renderer, for the live
        # channel smoke. It is deliberately something the model can finish in
        # a few hundred tokens: the first attempt at this gate used a real
        # erdos prompt with a 900-token cap, the model was still reasoning at
        # 900, and the gate scored "no answer channel" -- the exact
        # budget-too-small-looks-like-no-effect failure this experiment exists
        # to measure. Not part of `items`; never graded.
        smoke_msg = [{"role": "user",
                      "content": "How much is 1+1? Reply with just the number."}]
        smoke_toks = renderer.build_generation_prompt(smoke_msg).to_ints()
        prompts[f"smoke|{arm}"] = {
            "kind": "smoke", "arm": arm, "state_id": "smoke", "run": "n/a",
            "renderer": renderer_name, "tokens": smoke_toks,
            "prompt_len": len(smoke_toks),
            "text": tokenizer.decode(smoke_toks), "stop": [], "parent_value": None,
        }

        for kind, group, n_roll in (
            ("parent", parents, args.rollouts),
            ("start", seeds, args.start_rollouts),
        ):
            for st in group:
                state = State.from_dict(st)
                env = env_type(
                    renderer,
                    initial_state=state,
                    sampler=_SamplerStub(),
                    config=config,
                )
                obs, stop = asyncio.run(env.initial_observation())
                key = f"{kind}|{st['id']}|{arm}"
                toks = obs.to_ints()
                prompts[key] = {
                    "kind": kind,
                    "arm": arm,
                    "state_id": st["id"],
                    "run": st["_run"],
                    "renderer": renderer_name,
                    "tokens": toks,
                    "prompt_len": len(toks),
                    "text": tokenizer.decode(toks),
                    "stop": list(stop),
                    "parent_value": st["value"],
                }
                states_out[st["id"]] = {
                    k: v for k, v in st.items() if k != "_run"
                } | {"run": st["_run"]}
                for r in range(n_roll):
                    items.append(
                        {
                            "item_id": f"{args.problem}|{kind}|{st['id'][:8]}|{arm}|r{r:02d}",
                            "prompt_key": key,
                            "rollout": r,
                        }
                    )

    # Round-robin by rollout index so a run cut short still has EQUAL coverage
    # of every state and both arms, just fewer rollouts each -- far better than
    # finishing 8 states and never touching the rest.
    items.sort(key=lambda it: (it["rollout"], it["prompt_key"]))

    out = {
        "problem": args.problem,
        "problem_type": spec["problem_type"],
        "maximize": spec["maximize"],
        "eval_timeout": spec["eval_timeout"],
        "env_module": spec["module"],
        "env_class": spec["cls"],
        # The env's OWN code-block contract. frontier_algo is C++17; cueing or
        # extracting it as python silently voids every rollout.
        "code_languages": env_type._get_code_languages(None),
        "keep_code_separators": env_type._should_keep_code_separators(None),
        "arms": ARMS,
        "seed": args.seed,
        "current_date": args.current_date,
        "early_timestep": spec["early_timestep"],
        "rollouts_per_parent": args.rollouts,
        "rollouts_per_seed": args.start_rollouts,
        "prompts": prompts,
        "items": items,
        "states": states_out,
        "qwen_children": {k: v for k, v in qwen_children.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    lens = sorted(p["prompt_len"] for p in prompts.values())
    print(
        f"[{args.problem}] prompts={len(prompts)} items={len(items)} "
        f"prompt_len min={lens[0]} med={lens[len(lens)//2]} max={lens[-1]}"
    )
    # The knob must change exactly one line and nothing else.
    for kind, group in (("parent", parents), ("start", seeds)):
        for st in group:
            a = prompts[f"{kind}|{st['id']}|high"]["text"]
            b = prompts[f"{kind}|{st['id']}|xhigh"]["text"]
            assert a.replace("strength: high", "strength: xhigh") == b, (
                f"arms differ by more than the reasoning line for {st['id']}"
            )
    print(f"[{args.problem}] ARM-DIFF-IS-ONE-LINE OK -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=sorted(PROBLEMS))
    ap.add_argument("--pool-dir", default="/n/fs/vision-mix/sk7524/muse-rs2/pools")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-parents", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=1)
    ap.add_argument("--rollouts", type=int, default=32)
    ap.add_argument("--start-rollouts", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260813)
    ap.add_argument("--current-date", default="2026-08-13")
    ap.add_argument("--log-path", default="/n/fs/vision-mix/sk7524/muse-rs2/envlog")
    build(ap.parse_args())
