#!/usr/bin/env python3
"""Tier-(a) dry check for the ensemble run.

Builds both members' renderers, renders the REAL Erdos generator prompt for a
real initial State through the exact builder/env path, and:
  - asserts Nemotron's <|im_end|> encodes to exactly 1 token;
  - asserts the qwen3 renderer's generation prompt byte-matches Nemotron's own
    HF chat template (empty system turn + forced <think>\n);
  - with --live: creates the Nemotron LoRA training client on Tinker prod,
    exports sampler weights, and samples a few tokens end-to-end.
"""

from __future__ import annotations

import ssl
import sys
from functools import partial
from pathlib import Path

ssl.create_default_context()

repo_root = Path(__file__).resolve().parents[1]
discover_root = repo_root / "third_party" / "discover"
sys.path.insert(0, str(discover_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(discover_root / ".env")

from examples.erdos_min_overlap.env import ErdosMinOverlapEnv  # noqa: E402
from ttt_discover.rl.types import ProblemGroupBuilder  # noqa: E402
from ttt_discover.tinker_utils import renderers  # noqa: E402
from ttt_discover.tinker_utils.dataset_builder import DatasetConfig  # noqa: E402
from ttt_discover.tinker_utils.misc_utils import get_tokenizer  # noqa: E402

GPTOSS = "openai/gpt-oss-20b"
NEMOTRON = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"

MEMBERS = [
    (GPTOSS, "gpt_oss_high_reasoning", None, "gptoss"),
    (NEMOTRON, "qwen3", [{"role": "system", "content": ""}], "nemotron"),
]


def main() -> None:
    state = ErdosMinOverlapEnv.create_initial_state("")
    import threading
    import types

    sampler = types.SimpleNamespace(_states=[state], _lock=threading.Lock())

    prompts = {}
    for model, renderer_name, convo_prefix, tag in MEMBERS:
        tok = get_tokenizer(model)
        renderer = renderers.get_renderer(renderer_name, tokenizer=tok)
        cfg = DatasetConfig(
            problem_type="", env_type=ErdosMinOverlapEnv, batch_size=1,
            model_name_for_tokenizer=model, renderer_name=renderer_name,
            group_size=1, num_cpus_per_task=1, eval_backend="local",
            eval_timeout=120, log_path=str(repo_root / "runs" / "_ensemble_check"),
            convo_prefix=convo_prefix, member_tag=tag,
        )
        builder = ProblemGroupBuilder(
            env_thunk=partial(ErdosMinOverlapEnv, renderer, initial_state=state,
                              sampler=sampler, config=cfg),
            num_envs=1, logging_name=tag,
        )
        env = builder.env_thunk()
        convo = env.convo_prefix + [{"role": "user", "content": env.get_question()}]
        mi = renderer.build_generation_prompt(convo)
        text = tok.decode([t for ch in mi.chunks for t in ch.tokens])
        prompts[tag] = (mi, text, tok, renderer, convo)
        print(f"\n=== {tag} ({model}) prompt: {mi.length} tokens ===")
        print("--- head ---")
        print(text[:500])
        print("--- tail ---")
        print(text[-260:])

    # Nemotron-specific assertions
    mi, text, tok, renderer, convo = prompts["nemotron"]
    end_tokens = tok.encode("<|im_end|>", add_special_tokens=False)
    assert len(end_tokens) == 1, f"<|im_end|> must be 1 token, got {end_tokens}"
    hf = tok.apply_chat_template(
        [{"role": "system", "content": ""},
         {"role": "user", "content": convo[-1]["content"]}],
        tokenize=False, add_generation_prompt=True,
    )
    assert text == hf, (
        "qwen3 renderer output != Nemotron HF chat template:\n"
        f"renderer tail: {text[-160:]!r}\nHF tail:       {hf[-160:]!r}"
    )
    print("\nASSERT OK: <|im_end|> single token; renderer byte-matches HF chat template")

    if "--live" in sys.argv:
        import tinker

        print("\n[live] creating Nemotron LoRA training client on Tinker prod ...")
        sc = tinker.ServiceClient()
        tc = sc.create_lora_training_client(NEMOTRON, rank=32)
        print("[live] training client OK; exporting sampler weights ...")
        sampling = tc.save_weights_and_get_sampling_client()
        print("[live] sampling a few tokens ...")
        fut = sampling.sample(
            prompt=mi, num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=24, temperature=1.0,
                stop=renderer.get_stop_sequences()),
        )
        seq = fut.result().sequences[0]
        print("[live] sampled:", repr(tok.decode(seq.tokens))[:240])
        print("[live] LIVE CHECK PASSED")

    print("\nDRY CHECK PASSED")


if __name__ == "__main__":
    main()
