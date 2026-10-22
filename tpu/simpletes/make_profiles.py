"""Scaffold experiment (2026-10-19): local no-train run profiles and SimpleTES specs for one problem.

Runs ``scaf2-<problem>-<arm>-<start>-<model>-s<seed>``: arm ours (TTT-Discover PUCT, the RQ4 frozen
client) or stes (SimpleTES rpucg), start boot (1024-draft bootstrap) or none, model qwen38 or muse,
seeds 1 and 2. Every run: 32k context, fixed 15.5k thinking + 6k answer (21,644 tokens), 25 steps of
16 x 32, the RQ4 no-train Muse profile otherwise. That is the largest budget that leaves our arm's
longest prompts (circuit, ~10.8k) inside 32k, so whichever arm wins can be trained at 32k; SimpleTES
drops its lowest-scored inspirations to fit (logged in prompt_fit.jsonl). (``scaf-`` runs,
2026-10-19: 12k / 17k; Qwen3.8 hit the thinking cap on most samples and lost answers to the total.)

Usage: python -m tpu.simpletes.make_profiles erdos --bundle <gs://...tar.gz> --sha256 <hex>
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROFILES = HERE.parents[0] / "swarm/ray_train/profiles"
SPECS = HERE / "specs"
ROOT = "/scratch/gpfs/ZHUANGL/sk7524/rq4-local/scaffold"
TEMPLATE = "rq4-notrain-local-muse-{problem}-s1-20261007.json"
CONTEXT, THINK, TOTAL, STEPS, CHAINS, K = 32768, 15500, 15500 + 6144, 25, 16, 32
EAGER = 0.95
# Groups per engine: about one engine's sequence slots (Muse v6e TP8 128, Qwen3.8 v6e TP4 32 + v5p TP2 64).
MODELS = {
    # target_leases: farms per run at 2026-10-19 capacity (8 Muse v5p-32 farms for 8 runs; ~15 Qwen3.8 farms).
    "muse": dict(model_preset="muse-glimmer-30b", family="muse", requests_per_engine=4, target_leases=1,
                 hf="gs://sk7524-tinker-tpu-us-east5/hf-cache-muse-glimmer-v1"),
    "qwen38": dict(model_preset="qwen3.8-27b", family="qwen", requests_per_engine=2, target_leases=2,
                   hf="gs://sk7524-tinker-tpu-us-east5/hf-cache-qwen38-1d4bf0f2"),    # Control (2026-10-22): Qwen3.5 under the scaf2 settings, ours without bootstrap, seeds 1-2.
    "qwen35": dict(model_preset="qwen3.5-27b", family="qwen", requests_per_engine=2, target_leases=2,
                   hf="gs://sk7524-tinker-tpu-us-east5/hf-cache-qwen35-v1"),
}
PROBLEMS = {"erdos": dict(env=dict(TTD_ERDOS_PROMPT="v2")),
            # The agreed circuit prompt (ibm17-v3 wording, CSV feedback, standard structure) plus record/goal/gap.
            "circuit": dict(env=dict(SCIENCE_PLACEMENT_PROMPT="ibm17-v3", SCIENCE_FEEDBACK_FORMAT="csv",
                                     SCIENCE_PROMPT_STYLE="standard", TTD_CIRCUIT_PROMPT="v2"))}


def run_id(problem, arm, start, model, seed):
    return f"scaf2-{problem}-{arm}-{start}-{model}-s{seed}"


def profile(problem, arm, start, model, seed, port, bundle, sha256):
    doc = json.loads((PROFILES / TEMPLATE.format(problem=problem)).read_text())
    rid, spec = run_id(problem, arm, start, model, seed), MODELS[model]
    doc.update(run_id=rid, root=f"{ROOT}/{rid}", base_bundle=bundle, base_bundle_sha256=sha256,
               model_preset=spec["model_preset"], client_context_window=CONTEXT,
               ports=dict(doc["ports"], inference=port))
    if (arm, start) == ("ours", "boot"):
        doc["bootstrap_seed"] = seed
    else:
        # SimpleTES runs never bootstrap (a boot run reads the ours boot run's drafts); the seed is in its spec.
        doc.update(bootstrap_max_drafts=0, bootstrap_layers=0, bootstrap_seed=None, bootstrap_fixed_budget=False,
                   bootstrap_require_full_pool=False)
    doc["trainer"] = dict(doc["trainer"], sequence_length=CONTEXT, max_target_length=CONTEXT,
                          token_budget=CONTEXT, seq_buckets=str(CONTEXT))
    doc["inference"] = dict(doc["inference"], max_model_length=CONTEXT, external_pool_max_concurrent_requests=16,
                            external_pool_target_leases=spec["target_leases"],
                            external_pool_requests_per_engine=spec["requests_per_engine"])
    doc["cache"] = dict(doc["cache"], hf=spec["hf"])
    doc["client_env"] = dict(doc["client_env"], NUM_EPOCHS=str(STEPS), GROUPS_PER_BATCH=str(CHAINS),
                             GROUP_SIZE=str(K), TTD_FIXED_THINK_TOKENS=str(THINK),
                             TTD_FIXED_TOTAL_TOKENS=str(TOTAL), TTD_ANSWER_MODEL_FAMILY=spec["family"],
                             TTD_SICK_MARKER=f"{ROOT}/{rid}/runs/{rid}/ENGINE-SICK", **PROBLEMS[problem]["env"])
    return doc


def simpletes_spec(problem, start, model, seed):
    spec = dict(task=problem, prompt="scaffold", start=start, seed=seed, num_chains=CHAINS, k_candidates=K,
                max_generations=STEPS * CHAINS * K, eval_concurrency=CHAINS * K, selector="rpucg",
                num_inspirations=3, reflection=True, eager=EAGER, log_interval=CHAINS * K, init_eval_repeats=1)
    if start == "boot":
        ours = run_id(problem, "ours", "boot", model, seed)
        spec["bootstrap_from"] = f"{ROOT}/{ours}/runs/{ours}/client/bootstrap"
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", choices=sorted(PROBLEMS))
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--port-base", type=int, default=27000)
    args = parser.parse_args()
    port = args.port_base
    for arm in ("ours", "stes"):
        for start in ("boot", "none"):
            for model in MODELS:
                for seed in (1, 2):
                    rid = run_id(args.problem, arm, start, model, seed)
                    doc = profile(args.problem, arm, start, model, seed, port, args.bundle, args.sha256)
                    (PROFILES / f"{rid}.json").write_text(json.dumps(doc, indent=2) + "\n")
                    if arm == "stes":
                        (SPECS / f"{rid}.json").write_text(
                            json.dumps(simpletes_spec(args.problem, start, model, seed), indent=1) + "\n")
                    print(rid, port)
                    port += 10


if __name__ == "__main__":
    main()
