import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tpu/swarm/cleanup_v5p32_worker.sh"


def _dirty_host(home: Path) -> dict:
    hub = home / ".cache/huggingface/hub/models--google--gemma-4-31B-it"
    (hub / "snapshots/rev").mkdir(parents=True)
    (hub / "blobs").mkdir(parents=True)
    partial = hub / "blobs/abc_.gstmp"
    partial.write_bytes(b"x")
    trackers = home / ".config/gcloud/surface_data/storage/tracker_files"
    trackers.mkdir(parents=True)
    (trackers / "t1").write_text("x")
    foreign = home / ".cache/huggingface/hub/models--Qwen--Qwen3.5-27B"
    foreign.mkdir(parents=True)
    (foreign / "blob").write_bytes(b"x" * 1024)
    orbax_foreign = home / "skyrl-maxtext-ckpts-local/muse-glimmer-30b"
    orbax_foreign.mkdir(parents=True)
    (orbax_foreign / "blob").write_bytes(b"x" * 1024)
    run_state = home / "skyrl-runs/stageB2-g-ttd-n/tinker_log/x"
    run_state.mkdir(parents=True)
    (run_state / "metrics.jsonl").write_text("{}\n")
    (home / "ENGINE-SICK").write_text("sick")
    repo = home / "SkyRLTpu-tpuswarm"
    (repo / "tpu/swarm").mkdir(parents=True)
    # the bundle install: reconcile is invoked from there
    (repo / "tpu/swarm/reconcile_v5p32_worker.sh").write_text(
        (REPO / "tpu/swarm/reconcile_v5p32_worker.sh").read_text())
    (repo / "tpu").mkdir(exist_ok=True)
    (repo / "tpu/dedupe_hf_snapshot.sh").write_text((REPO / "tpu/dedupe_hf_snapshot.sh").read_text())
    return {"partial": partial, "trackers": trackers, "foreign": foreign, "orbax_foreign": orbax_foreign,
            "run_state": run_state, "sick": home / "ENGINE-SICK", "hub": hub}


def test_failed_cell_leaves_the_head_clean_without_touching_run_state(tmp_path):
    p = _dirty_host(tmp_path)
    env = dict(os.environ, HOME=str(tmp_path), CELL="g-ttd-n",
               JOBMAN_TPU_INTERNAL_IPS="10.0.0.1",  # head only: no engine ssh
               SKYRL_REPO_DIR=str(tmp_path / "SkyRLTpu-tpuswarm"),
               SSH_KEY_FILE=str(tmp_path / "no-such-key"))
    out = subprocess.run(["bash", str(SCRIPT), "33"], env=env, capture_output=True, text=True, check=True).stdout
    assert "cell g-ttd-n exited 33" in out
    assert not p["partial"].exists()
    assert not p["trackers"].exists()
    assert not p["sick"].exists()
    assert not p["foreign"].exists() and not p["orbax_foreign"].exists()
    assert p["hub"].exists()  # own model kept
    assert (p["run_state"] / "metrics.jsonl").exists()
    assert "cleanup[rank 0]: free" in out and "cleanup: done" in out


def test_engine_hosts_are_reported_when_the_key_is_missing(tmp_path):
    _dirty_host(tmp_path)
    env = dict(os.environ, HOME=str(tmp_path), CELL="m-pw-n",
               JOBMAN_TPU_INTERNAL_IPS="10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4",
               SKYRL_REPO_DIR=str(tmp_path / "SkyRLTpu-tpuswarm"),
               SSH_KEY_FILE=str(tmp_path / "no-such-key"))
    out = subprocess.run(["bash", str(SCRIPT), "1"], env=env, capture_output=True, text=True, check=True).stdout
    assert out.count("engine host not cleaned") == 3


def test_run_v5p32_cell_traps_every_nonzero_exit_and_no_longer_execs_the_monitor():
    source = (REPO / "tpu/swarm/run_v5p32_cell.sh").read_text()
    trap = source.index("trap cleanup_on_failure EXIT")
    orbax = source.index("ensure_orbax_ckpt.sh", trap)
    worker = source.index("cell_worker.sh", orbax)
    monitor = source.index('bash "$REPO/tpu/jobman/cell_monitor.sh" || monitor_rc=$?', worker)
    assert trap < orbax < worker < monitor
    assert "exec bash" not in source[trap:]
    assert 'cleanup_v5p32_worker.sh" "$rc"' in source
    manifest = (REPO / "tpu/swarm/build_skyrl_bundle.sh").read_text()
    assert "tpu/swarm/cleanup_v5p32_worker.sh" in manifest


def test_cleanup_ssh_ignores_the_heads_ssh_config():
    # a corrupted ~/.ssh/config on a head made every default ssh fail (rc 255)
    # and left the engine hosts uncleaned (job 280, 2026-09-06)
    source = (REPO / "tpu/swarm/cleanup_v5p32_worker.sh").read_text()
    assert "SSHO=(-F /dev/null -i" in source
    lib = (REPO / "tpu/tpu_ssh_lib.sh").read_text()
    assert "-F /dev/null" in lib
