"""Shared harness for the distillation-ablation experiment.

Offline measurement of the contrastive context-distillation objective, decoupled
from the RL/PUCT confounds: freeze RL (fine-tune BASE gpt-oss-20b on a distill
corpus, cross-entropy only), then measure a held-out improver eval.

Everything here reuses the discover library; nothing in `third_party/discover`
is modified. See tpu/distill_ablation/README or the approved plan
(/u/sk7524/.claude/plans/gleaming-swimming-moon.md) for the design.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from functools import partial
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DISCOVER_ROOT = REPO_ROOT / "third_party" / "discover"
if str(DISCOVER_ROOT) not in sys.path:
    sys.path.insert(0, str(DISCOVER_ROOT))

STUDENT_MODEL = "openai/gpt-oss-20b"          # fixed student for every arm
RENDERER_NAME = "gpt_oss_high_reasoning"
CONTEXT_WINDOW = 32768
# 20b and 120b share the o200k tokenizer, so the student tokenizer is correct for
# corpora written by either teacher.
TOKENIZER_MODEL = STUDENT_MODEL
# gpt-oss trainable-vocab ceiling; sampled reserved harmony ids above it poison the
# TrainingClient (the live prod regression), so they must be dropped from CE too.
MAX_TRAIN_TOKEN_ID = 200018


def env_bits(log_path: str, eval_timeout: int = 1100, num_cpus_per_task: int = 1):
    """(renderer, tokenizer, DatasetConfig) for the Erdos env, local grading."""
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover.tinker_utils import renderers
    from ttt_discover.tinker_utils.dataset_builder import DatasetConfig
    from ttt_discover.tinker_utils.misc_utils import get_tokenizer

    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer=tokenizer)
    Path(log_path).mkdir(parents=True, exist_ok=True)
    cfg = DatasetConfig(
        problem_type="",
        env_type=ErdosMinOverlapEnv,
        batch_size=1,
        model_name_for_tokenizer=TOKENIZER_MODEL,
        renderer_name=RENDERER_NAME,
        group_size=1,
        num_cpus_per_task=num_cpus_per_task,
        eval_backend="local",
        eval_timeout=eval_timeout,
        log_path=log_path,
        convo_prefix=None,
    )
    return renderer, tokenizer, cfg


def load_pool_sampler(snapshot_path: str, scratch_dir: str):
    """Load a puct_sampler_step_*.json snapshot into a PUCTSampler WITHOUT touching
    the original run: the file is copied to `scratch_dir` first, so any accidental
    _save writes to scratch. Returns the sampler (populated `_states`, `_lock`)."""
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover.tinker_utils.sampler import PUCTSampler

    snap = Path(snapshot_path)
    step = int(snap.stem.split("_step_")[1])
    Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    dst = Path(scratch_dir) / f"pool_step_{step:06d}.json"
    shutil.copyfile(snap, dst)
    base = str(Path(scratch_dir) / "pool")  # _sampler_file_for_step appends _step_NNNNNN.json
    return PUCTSampler(file_path=base, env_type=ErdosMinOverlapEnv, resume_step=step)


def make_seed_builder(sampler, renderer, cfg, seed_state, num_envs: int):
    """A ProblemGroupBuilder that runs `num_envs` improver rollouts on one held-out seed.
    Mirrors SingleProblemDataset._make_env_group_builder (dataset_builder.py:82)."""
    from examples.erdos_min_overlap.env import ErdosMinOverlapEnv
    from ttt_discover.rl.types import ProblemGroupBuilder

    return ProblemGroupBuilder(
        env_thunk=partial(
            ErdosMinOverlapEnv,
            renderer,
            initial_state=seed_state,
            sampler=sampler,
            config=cfg,
        ),
        num_envs=num_envs,
        logging_name="erdos_heldout",
    )


# ---- datum (de)serialization: persist token ids + 0/1 mask; re-normalize at
# fine-tune time so each corpus subset contributes exactly distill_weight*meanNLL ----

def datum_to_record(datum, mask01) -> dict:
    """Serialize a distill datum to a plain dict of ints (JSON-safe)."""
    return {
        "model_input_ids": [int(x) for x in datum.model_input.to_ints()],
        "target_tokens": [int(x) for x in datum.loss_fn_inputs["target_tokens"].to_torch().tolist()],
        "mask01": [float(x) for x in mask01.tolist()],
    }


def record_max_token_id(rec: dict) -> int:
    ids = list(rec["model_input_ids"]) + list(rec["target_tokens"])
    return max(ids) if ids else 0


def record_to_datum(rec: dict, weight_scale: float):
    """Rebuild a tinker.Datum from a record; weights = mask01 * weight_scale.
    Pass weight_scale = distill_weight / (total trained tokens in the chosen subset)
    so CE = distill_weight * meanNLL over the subset."""
    import tinker
    import torch
    from tinker import TensorData

    targets = torch.tensor(rec["target_tokens"], dtype=torch.int64)
    weights = torch.tensor(rec["mask01"], dtype=torch.float32) * weight_scale
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(rec["model_input_ids"]),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(targets),
            "weights": TensorData.from_torch(weights),
        },
    )


def load_dotenv_key():
    """Load TINKER_API_KEY/WANDB from third_party/discover/.env (discover never does)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(DISCOVER_ROOT / ".env")
    except Exception as exc:  # pragma: no cover
        print(f"WARNING: could not load .env: {exc}", file=sys.stderr)
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit("TINKER_API_KEY not set (third_party/discover/.env)")


def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def write_json(path: str, obj: Any):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
