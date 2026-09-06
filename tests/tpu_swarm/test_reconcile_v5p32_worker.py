import os
import subprocess
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tpu/swarm/reconcile_v5p32_worker.sh"


def _fake_worker(home: Path, bundle_gen: str = "200") -> dict:
    hub = home / ".cache/huggingface/hub"
    orbax = home / "skyrl-maxtext-ckpts-local"
    bundles = home / ".cache/tpuswarm/bundles"
    paths = {
        "hf_gemma": hub / "models--google--gemma-4-31B-it",
        "hf_muse": hub / "models--meta-models--Muse-Glimmer-30B",
        "hf_qwen": hub / "models--Qwen--Qwen3.5-27B",
        "orbax_gemma": orbax / "gemma4-31b",
        "orbax_muse": orbax / "muse-glimmer-30b",
        "bundle_old": bundles / "100",
        "bundle_cur": bundles / bundle_gen,
        "run_dir": home / "skyrl-runs/stageB-m-grpo-n/tinker_log",
        "ckpts": home / "gcs/skyrl-checkpoints/model_780cc984",
        "xla_own": home / "vllm-xla-cache-local",
        "xla_bench": home / "vllm-xla-cache-bench-qwen35-tp4",
    }
    for p in paths.values():
        p.mkdir(parents=True)
        (p / "blob").write_bytes(b"x" * 1024)
    repo_link = home / "SkyRLTpu-tpuswarm"
    repo_link.symlink_to(paths["bundle_cur"])
    paths["repo_link"] = repo_link
    return paths


def _run(home: Path, cell: str, rank: int) -> str:
    env = dict(os.environ, HOME=str(home), CELL=cell, SKYPILOT_NODE_RANK=str(rank),
               SKYRL_REPO_DIR=str(home / "SkyRLTpu-tpuswarm"))
    env.pop("TUNIX_MAXTEXT_MODEL_NAME", None)
    env.pop("REMOTE_HF_HOME", None)
    env.pop("TUNIX_MAXTEXT_CKPT_CACHE", None)
    out = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
    return out.stdout


def test_trainer_host_keeps_only_its_own_model_caches(tmp_path):
    p = _fake_worker(tmp_path)
    out = _run(tmp_path, "g-meta-ttd", rank=0)
    assert p["hf_gemma"].exists() and p["orbax_gemma"].exists()
    assert not p["hf_muse"].exists() and not p["hf_qwen"].exists()
    assert not p["orbax_muse"].exists()
    assert "foreign orbax checkpoint" in out and "foreign HF snapshot" in out


def test_engine_host_keeps_its_hf_snapshot_and_no_orbax(tmp_path):
    p = _fake_worker(tmp_path)
    _run(tmp_path, "m-pw-n", rank=2)
    assert p["hf_muse"].exists()
    assert not p["hf_gemma"].exists() and not p["hf_qwen"].exists()
    assert not p["orbax_muse"].exists() and not p["orbax_gemma"].exists()


def test_reconcile_never_touches_run_state_or_the_current_bundle(tmp_path):
    p = _fake_worker(tmp_path)
    _run(tmp_path, "pw-n", rank=0)
    assert p["run_dir"].exists() and (p["run_dir"] / "blob").exists()
    assert p["ckpts"].exists() and (p["ckpts"] / "blob").exists()
    assert p["bundle_cur"].exists()
    assert not p["bundle_old"].exists()
    assert p["repo_link"].resolve() == p["bundle_cur"].resolve()
    # the engine's own compile cache stays; benchmark caches go
    assert p["xla_own"].exists()
    assert not p["xla_bench"].exists()


def test_compile_caches_are_wiped_on_a_model_switch_and_kept_otherwise(tmp_path):
    p = _fake_worker(tmp_path)
    marker = p["xla_own"] / ".tpuswarm-model"
    # a cache with no marker was filled by pre-reconcile jobs of unknown models: wiped
    _run(tmp_path, "pw-n", rank=1)
    assert not (p["xla_own"] / "blob").exists()
    assert marker.read_text().strip() == "qwen3.5-27b"
    # same model again: kept
    (p["xla_own"] / "blob").write_bytes(b"x")
    _run(tmp_path, "pw-n", rank=1)
    assert (p["xla_own"] / "blob").exists()
    # model switch: wiped and re-marked
    _run(tmp_path, "m-pw-n", rank=1)
    assert not (p["xla_own"] / "blob").exists()
    assert marker.read_text().strip() == "muse-glimmer-30b"
    # the trainer's jax cache follows the same rule on the head
    jax = tmp_path / "jax_cache"
    jax.mkdir(exist_ok=True)
    (jax / "blob").write_bytes(b"x")
    _run(tmp_path, "g-x", rank=0)
    assert not (jax / "blob").exists()
    assert (jax / ".tpuswarm-model").read_text().strip() == "gemma4-31b"


def test_wipe_all_drops_every_model_cache_but_not_run_state(tmp_path):
    p = _fake_worker(tmp_path)
    env = dict(os.environ, HOME=str(tmp_path), CELL="x", SKYPILOT_NODE_RANK="0",
               SKYRL_REPO_DIR=str(tmp_path / "SkyRLTpu-tpuswarm"), RECONCILE_WIPE_ALL="1")
    subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=True)
    for k in ("hf_gemma", "hf_muse", "hf_qwen", "orbax_gemma", "orbax_muse", "xla_own", "xla_bench", "bundle_old"):
        assert not p[k].exists(), k
    assert p["run_dir"].exists() and p["ckpts"].exists() and p["bundle_cur"].exists()


def test_reconcile_is_a_noop_on_a_clean_worker(tmp_path):
    (tmp_path / "SkyRLTpu-tpuswarm").mkdir()
    out = _run(tmp_path, "g-ttd-n", rank=1)
    assert "removing" not in out
    assert "keeps gemma4-31b / models--google--gemma-4-31B-it" in out


def test_run_v5p32_cell_reconciles_every_rank_before_the_head_gate():
    source = (REPO / "tpu/swarm/run_v5p32_cell.sh").read_text()
    reconcile = source.index("reconcile_v5p32_worker.sh")
    gate = source.index('if [ "$JOBMAN_WORKER_ID" != "0" ]; then')
    orbax = source.index("ensure_orbax_ckpt.sh")
    assert reconcile < gate < orbax


def test_reconcile_ships_in_the_bundle():
    manifest = (REPO / "tpu/swarm/build_skyrl_bundle.sh").read_text()
    assert "tpu/swarm/reconcile_v5p32_worker.sh" in manifest
