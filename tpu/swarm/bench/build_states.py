#!/usr/bin/env python3
"""Select a reproducible PUCT batch per problem and render the env questions.

Generalized from tpuswarm-state/benchmarks/realistic-improvement-v4-32/
build_states.py for the v5p-32 serving sweep: the snapshot sources are fetched
from GCS when absent, the per-problem count is a flag (default 6 -> 18 groups =
576 rollouts >= one production step of 512), and the output carries the
selection provenance so every benchmark job replays the SAME prompts.

Runs on a CPU login node:
  .venv/bin/python tpu/swarm/bench/build_states.py --out states.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

REPO_DEFAULT = Path(__file__).resolve().parents[3]
BUCKET = "gs://sk7524-tinker-tpu-us-east5"


class State:
    def __init__(self, **values):
        defaults = {
            "id": None,
            "timestep": 0,
            "value": None,
            "code": "",
            "construction": [],
            "parent_values": [],
            "parents": [],
            "observation": "",
            "origin": None,
        }
        defaults.update(values)
        self.__dict__.update(defaults)

    @classmethod
    def from_dict(cls, value: dict) -> "State":
        return cls(**value)

    def to_prompt(self, target, metric_name: str = "value", maximize: bool = True, language: str = "") -> str:
        value_ctx = f"You are iteratively optimizing {metric_name}."
        direction = "higher" if maximize else "lower"
        if self.code and self.code.strip():
            value_ctx += "\nHere is the last code we ran:\n"
            value_ctx += f"```{language}\n{self.code}\n```" if language else self.code
        else:
            value_ctx += "\nNo previous code available."
        if self.parent_values and self.value is not None and self.construction:
            before = self.parent_values[0] if maximize else -self.parent_values[0]
            after = self.value if maximize else -self.value
            gap = target - after if maximize else after - target
            value_ctx += (
                f"\nHere is the {metric_name} before and after running the code "
                f"above ({direction} is better): {before:.6f} -> {after:.6f}"
            )
            value_ctx += f"\nTarget: {target}. Current gap: {gap:.6f}. Further improvements will also be generously rewarded."
        elif self.value is not None:
            after = self.value if maximize else -self.value
            gap = target - after if maximize else after - target
            value_ctx += f"\nCurrent {metric_name} (higher is better): {after:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {gap:.6f}. Further improvements will also be generously rewarded."
        else:
            value_ctx += f"\nTarget {metric_name}: {target}"
        if self.observation and self.observation.strip():
            stdout = self.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        return value_ctx


class Environment:
    def __init__(self, renderer, initial_state, sampler, config):
        self.renderer = renderer
        self.initial_state = initial_state
        self.sampler = sampler
        self.config = config
        self.problem_type = config.problem_type
        self.state = initial_state


class BaseRewardEvaluator:
    pass


class SandboxRewardEvaluator:
    pass


def install_import_stubs() -> None:
    """The env modules import the discover package for types only; stub it so
    the builder runs without the RL client's heavy dependencies."""
    module = types.ModuleType("ttt_discover")
    for name, value in {
        "Environment": Environment,
        "BaseRewardEvaluator": BaseRewardEvaluator,
        "SandboxRewardEvaluator": SandboxRewardEvaluator,
        "State": State,
        "DiscoverConfig": object,
        "discover": lambda *args, **kwargs: None,
    }.items():
        setattr(module, name, value)
    sys.modules["ttt_discover"] = module


def full_lineage(state: dict, children: dict[str, set[str]]) -> set[str]:
    lineage = {state["id"]}
    lineage.update(str(p["id"]) for p in state.get("parents") or [] if p.get("id"))
    pending = deque([state["id"]])
    visited = {state["id"]}
    while pending:
        sid = pending.popleft()
        for child in children.get(sid, ()):
            if child not in visited:
                visited.add(child)
                lineage.add(child)
                pending.append(child)
    return lineage


def select_states(store: dict, count: int, elite_slots: int) -> list[tuple[dict, dict]]:
    """Reproduce the sampler's PUCT ranking (saved score) and pick `count`
    lineage-disjoint states, elites (by value) first -- the production batch."""
    states = store.get("states", [])
    initial_ids = {s["id"] for s in store.get("initial_states", [])}
    values = np.array([float(s["value"]) if s.get("value") is not None else float("-inf") for s in states])
    non_initial = np.array([s["id"] not in initial_ids for s in states])
    scale_values = values[non_initial] if non_initial.any() else values
    scale = float(max(np.max(scale_values) - np.min(scale_values), 1e-6)) if scale_values.size else 1.0
    ranks = np.argsort(np.argsort(-values))
    priors = (len(values) - ranks).astype(np.float64)
    priors /= priors.sum()
    total_visits = int(store.get("puct_T", 0) or 0)
    visits = store.get("puct_n", {}) or {}
    maxima = store.get("puct_m", {}) or {}

    scored = []
    for i, s in enumerate(states):
        n = int(visits.get(s["id"], 0))
        q = float(maxima.get(s["id"], values[i])) if n > 0 else float(values[i])
        bonus = scale * float(priors[i]) * math.sqrt(1.0 + total_visits) / (1.0 + n)
        scored.append((q + bonus, float(values[i]), s,
                       {"n": n, "Q": q, "P": float(priors[i]), "bonus": bonus, "score": q + bonus,
                        "scale": scale, "T": total_visits}))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    children: dict[str, set[str]] = defaultdict(set)
    for s in states:
        for p in s.get("parents") or []:
            if p.get("id"):
                children[str(p["id"])].add(s["id"])

    selected, blocked = [], set()
    if elite_slots:
        by_value = sorted((it for it in scored if it[2]["id"] not in initial_ids and (it[2].get("code") or "").strip()),
                          key=lambda it: it[1], reverse=True)
        for it in by_value:
            if len(selected) >= min(elite_slots, count):
                break
            if it[2]["id"] in blocked:
                continue
            selected.append((it[2], it[3]))
            blocked.update(full_lineage(it[2], children))
    for it in scored:
        if it[2]["id"] in blocked:
            continue
        selected.append((it[2], it[3]))
        blocked.update(full_lineage(it[2], children))
        if len(selected) >= count:
            break
    if len(selected) != count:
        raise RuntimeError(f"requested {count} states, selected {len(selected)}")
    return selected


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_url(run: str, step: int) -> str:
    return f"{BUCKET}/skyrl-runs/{run}/tinker_log/{run}/puct_sampler_step_{step:06d}.json"


def ensure_source(source_dir: Path, run: str, url: str) -> Path:
    path = source_dir / f"{run}.json"
    if not path.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        print(f"fetching {url}", flush=True)
        subprocess.run(["gcloud", "storage", "cp", url, str(path)], check=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(REPO_DEFAULT))
    parser.add_argument("--source-dir", default=str(REPO_DEFAULT / ".cache" / "bench-sources"),
                        help="local copies of the PUCT snapshots (downloaded from GCS when absent)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--states-per-problem", type=int, default=6,
                        help="6 -> 18 groups x 32 = 576 rollouts, i.e. >= one production step")
    parser.add_argument("--erdos-run", default="stageC-lr-n")
    parser.add_argument("--jssp-run", default="sweep1-ttd-jssp")
    parser.add_argument("--ac1-run", default="sweep1-ttd-acineq",
                        help="fallback: stageB-g-ttd-n-a (gemma ac1, step 15)")
    parser.add_argument("--step", type=int, default=15)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    source_dir = Path(args.source_dir).resolve()
    sys.path.insert(0, str(repo / "third_party" / "discover"))
    install_import_stubs()

    from types import SimpleNamespace
    from examples.ac_inequalities.env import AutoCorrInequalityEnv
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from examples.frontier_algo.env import FrontierAlgoEnv

    specs = {
        "erdos": {"run": args.erdos_run, "env": ErdosMinOverlapEnv, "problem_type": "",
                  "code_language": "python", "elite_slots": 2},
        "jssp": {"run": args.jssp_run, "env": FrontierAlgoEnv, "problem_type": "46",
                 "code_language": "cpp", "elite_slots": 0},
        "ac1": {"run": args.ac1_run, "env": AutoCorrInequalityEnv, "problem_type": "ac1",
                "code_language": "python", "elite_slots": 0},
    }

    output = {
        "schema_version": 1,
        "selection": "saved PUCT score with full-lineage diversity",
        "states_per_problem": args.states_per_problem,
        "rollouts_per_state": 32,
        "problems": {},
    }
    for problem, spec in specs.items():
        url = snapshot_url(spec["run"], args.step)
        path = ensure_source(source_dir, spec["run"], url)
        with path.open() as stream:
            store = json.load(stream)
        selected = select_states(store, args.states_per_problem, spec["elite_slots"])
        rows = []
        for state_dict, puct in selected:
            state = State.from_dict(state_dict)
            env = spec["env"](None, state, object(), SimpleNamespace(problem_type=spec["problem_type"]))
            question = env.get_question()
            rows.append({
                "state_id": state.id,
                "timestep": state.timestep,
                "value": state.value,
                "parent_values": state.parent_values,
                "code_chars": len(state.code or ""),
                "construction_len": len(state.construction or []),
                "question_chars": len(question),
                "question": question,
                "puct": puct,
            })
        output["problems"][problem] = {
            "run": spec["run"],
            "source": url,
            "source_sha256": sha256(path),
            "problem_type": spec["problem_type"],
            "code_language": spec["code_language"],
            "elite_slots": spec["elite_slots"],
            "states": rows,
        }
        del store

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"wrote {out}")
    for problem, spec in output["problems"].items():
        chars = [r["question_chars"] for r in spec["states"]]
        print(f"{problem}: {len(chars)} states  question_chars min/median/max = "
              f"{min(chars)}/{sorted(chars)[len(chars)//2]}/{max(chars)}")
        for row in spec["states"]:
            print(f"  {row['state_id']} t={row['timestep']} value={row['value']:.12g} chars={row['question_chars']}")


if __name__ == "__main__":
    main()
