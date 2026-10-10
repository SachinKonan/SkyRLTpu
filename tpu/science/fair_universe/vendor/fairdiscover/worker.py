"""Single-phase worker. Run via `python -m fairdiscover.trial`, not directly, normally."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from .data import MapStore, DataView
from .metric import low_fpr_tpr, per_family_scores
from .shifts import proxy_shift, FAMILIES


def load_candidate(candidate_path):
    path = Path(candidate_path).resolve()
    if not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(f"Expected Python candidate file: {path}")
    spec = importlib.util.spec_from_file_location("candidate_program", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["candidate_program"] = mod
    spec.loader.exec_module(mod)
    for name in ("fit", "load_model", "predict"):
        if not callable(getattr(mod, name, None)):
            raise TypeError(f"candidate.py must define callable {name}")
    return mod


def code_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_predict(mod, model, maps, mask, cfg):
    scores = np.asarray(mod.predict(model, maps, mask, cfg), dtype=np.float64)
    if scores.shape != (len(maps),):
        raise ValueError(f"predict returned {scores.shape}, expected ({len(maps)},)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("predict returned non-finite scores")
    return scores


def phase_train(args):
    store = MapStore(args.stage)
    cfg = json.loads(Path(args.run_dir, "trial_config.json").read_text())
    mod = load_candidate(args.candidate)
    train_view = DataView(store, "train", cfg)
    val_view = DataView(store, "val", cfg)
    artifacts = Path(args.run_dir, "artifacts")
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    info = mod.fit(train_view, val_view, cfg, artifacts)
    elapsed = time.perf_counter() - started
    meta = {"candidate_sha256": code_hash(args.candidate),
            "train_seconds": elapsed, "fit_info": info,
            "train_count": len(train_view), "val_count": len(val_view)}
    (Path(args.run_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps(meta, default=str))


def phase_reward(args):
    store = MapStore(args.stage)
    run = Path(args.run_dir)
    cfg = json.loads((run / "trial_config.json").read_text())
    mod = load_candidate(args.candidate)
    meta = json.loads((run / "train_meta.json").read_text())
    if meta["candidate_sha256"] != code_hash(args.candidate):
        raise ValueError("Candidate changed after training. Re-run the trial.")
    model = mod.load_model(run / "artifacts", cfg)
    rng = np.random.default_rng(int(cfg["seed"]) + 90210)
    reward_ids = store.ids("reward")
    max_ind = int(cfg.get("reward_max_ind", 2048))
    if max_ind > 0 and len(reward_ids) > max_ind:
        reward_ids = reward_ids[np.sort(rng.choice(len(reward_ids), max_ind, replace=False))]
    if len(reward_ids) < 1000:
        print("WARNING: fewer than 1000 InD samples => FPR=0.001 is poorly resolved", file=sys.stderr)
    batch_size = int(cfg.get("infer_batch_size", 32))
    if batch_size <= 0:
        raise ValueError("infer_batch_size must be positive")
    families = list(cfg["reward_families"])
    if not families or not set(families).issubset(set(FAMILIES)):
        raise ValueError("Unknown/empty reward families")
    ind_scores = []
    ood_scores = {f: [] for f in families}
    started = time.perf_counter()
    for start in range(0, len(reward_ids), batch_size):
        batch_ids = reward_ids[start:start+batch_size]
        clean = store.get_clean(batch_ids)
        noisy_ind = store.apply_noise(clean, rng, cfg)
        ind_scores.append(checked_predict(mod, model, noisy_ind, store.mask, cfg))
        for fi, family in enumerate(families):
            altered = np.empty_like(clean)
            # Each family uses a fixed seed per sample, independent of candidate code.
            for j, pair in enumerate(batch_ids):
                individual_rng = np.random.default_rng(np.random.SeedSequence([
                    int(cfg["seed"]), 79 + fi, int(pair[0]), int(pair[1])]))
                altered[j] = proxy_shift(clean[j], store.mask, family, individual_rng,
                                         float(cfg.get("reward_shift_strength", 1.)))
            noisy_ood = store.apply_noise(altered, rng, cfg)
            ood_scores[family].append(checked_predict(mod, model, noisy_ood, store.mask, cfg))
    ind = np.concatenate(ind_scores)
    by_family = {k: np.concatenate(v) for k, v in ood_scores.items()}
    per_family = per_family_scores(ind, by_family)
    # Equal weight for each shift family, rather than overweighting its number of samples.
    reward = float(np.mean(list(per_family.values())))
    y = np.r_[np.zeros(len(ind)), np.ones(sum(len(v) for v in by_family.values()))]
    all_scores = np.r_[ind, *by_family.values()]
    report = {"reward": reward, "higher_is_better": True,
              "official_metric_formula": "mean TPR on 100 log-FPRs from 0.001 to 0.05",
              "reward_is_proxy_not_official": True,
              "pooled_proxy_score": low_fpr_tpr(y, all_scores),
              "per_family": per_family,
              "n_ind": len(ind), "n_ood": sum(len(v) for v in by_family.values()),
              "reward_seconds": time.perf_counter() - started,
              "train_seconds": meta["train_seconds"],
              "candidate_sha256": meta["candidate_sha256"]}
    (run / "reward.json").write_text(json.dumps(report, indent=2))
    (run / "reward.txt").write_text(f"{reward:.9f}\n")
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description="FAIR Discover single worker")
    p.add_argument("phase", choices=("train", "reward"))
    p.add_argument("--candidate", required=True)
    p.add_argument("--stage", required=True)
    p.add_argument("--run-dir", required=True)
    a = p.parse_args()
    if a.phase == "train":
        phase_train(a)
    else:
        phase_reward(a)


if __name__ == "__main__":
    main()
