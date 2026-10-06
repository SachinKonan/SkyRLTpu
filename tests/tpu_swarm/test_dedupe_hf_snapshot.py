import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tpu/dedupe_hf_snapshot.sh"


def _restored_cache(root: Path) -> dict:
    # what `gcloud storage cp --recursive` leaves behind: shards as real files
    # under BOTH snapshots/<rev>/ and blobs/<sha>, plus small metadata files
    model = root / "hub/models--google--gemma-4-31B-it"
    snap = model / "snapshots/842da379"
    blobs = model / "blobs"
    snap.mkdir(parents=True)
    blobs.mkdir(parents=True)
    shard = os.urandom(2 * 1024 * 1024)
    (snap / "model-00002-of-00002.safetensors").write_bytes(shard)
    (blobs / "018912220f").write_bytes(shard)
    (snap / "model-00001-of-00002.safetensors").write_bytes(os.urandom(3 * 1024 * 1024))
    (snap / "config.json").write_text("{}")
    (blobs / "aa11").write_text("{}")  # small blob, same size as config.json: left alone
    other = os.urandom(2 * 1024 * 1024)  # same size as the shard, different bytes, no snapshot twin
    return {"model": model, "snap": snap, "blobs": blobs, "shard": shard, "other": other}


def test_duplicate_shard_becomes_a_hardlink_and_bytes_are_unchanged(tmp_path):
    c = _restored_cache(tmp_path)
    out = subprocess.run(["bash", str(SCRIPT), str(c["model"])], capture_output=True, text=True, check=True).stdout
    blob = c["blobs"] / "018912220f"
    snap_file = c["snap"] / "model-00002-of-00002.safetensors"
    assert blob.stat().st_ino == snap_file.stat().st_ino
    assert blob.read_bytes() == c["shard"]
    assert "reclaimed 2 MB" in out
    # small metadata blobs and the unmatched shard are untouched
    assert (c["blobs"] / "aa11").stat().st_ino != (c["snap"] / "config.json").stat().st_ino
    assert (c["snap"] / "model-00001-of-00002.safetensors").stat().st_nlink == 1


def test_dedupe_is_idempotent_and_tolerates_missing_dirs(tmp_path):
    c = _restored_cache(tmp_path)
    subprocess.run(["bash", str(SCRIPT), str(c["model"])], capture_output=True, text=True, check=True)
    again = subprocess.run(["bash", str(SCRIPT), str(c["model"]), str(tmp_path / "nope")],
                           capture_output=True, text=True, check=True).stdout
    assert "reclaimed 0 MB" in again


def test_vllm_runner_dedupes_right_after_a_successful_restore():
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()
    manifest = source.index("materialize_hf_tree_manifest \\\\\n      || echo")
    dedupe = source.index('bash "\\$HOME/dedupe_hf_snapshot.sh" "${REMOTE_HF_HOME}/hub/${HF_MODEL_DIR}"', manifest)
    ready = source.index('if ! hf_snapshot_ready && [[ "\\${HF_HUB_OFFLINE}" == "1" ]]', dedupe)
    assert manifest < dedupe < ready
    assert 'tpu_vm_scp "$worker" "${repo_root}/tpu/dedupe_hf_snapshot.sh" "~/dedupe_hf_snapshot.sh"' in source


def test_reconcile_dedupes_the_kept_snapshot_and_bundle_ships_the_script():
    reconcile = (REPO / "tpu/swarm/reconcile_v5p32_worker.sh").read_text()
    assert 'dedupe_hf_snapshot.sh" "$HUB/$HF"' in reconcile
    manifest = (REPO / "tpu/swarm/build_skyrl_bundle.sh").read_text()
    assert "tpu/dedupe_hf_snapshot.sh" in manifest


def test_vllm_restore_prelinks_duplicate_blobs_before_the_full_copy():
    # Copying the whole model dir needs the duplicated size on disk before the
    # post-copy dedupe can run: 71 GB transient for gemma on an engine with
    # 76 GB free hit 0 bytes mid-restore (job 275, 2026-09-06). The restore
    # must copy snapshots/ first, hardlink size-matched blobs, then --no-clobber.
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()
    loop = source.index("for _try in 1 2 3; do")
    subdirs = source.index("for _sub in refs trees snapshots; do", loop)
    listing = source.index('gcloud storage ls -l "${HF_CACHE_GCS}/${HF_MODEL_DIR}/blobs/"', subdirs)
    link = source.index('ln "\\$_match" "\\${_hub_model}/blobs/\\${_name}"', listing)
    full_copy = source.index("if gcloud storage cp --recursive --no-clobber \\\n        \"${HF_CACHE_GCS}/${HF_MODEL_DIR}\" \"${REMOTE_HF_HOME}/hub\"", link)
    assert loop < subdirs < listing < link < full_copy


def test_prelink_pipeline_can_never_fail_the_runner():
    # muse has no blobs/ and qwen's blobs have no snapshot twins: the while-read
    # pipeline's last iteration returned 1, pipefail + set -e killed every qwen
    # and muse engine runner on v17/v18 right after the XLA restore (jobs 278/280).
    source = (REPO / "tpu/start_vllm_tpu.sh").read_text()
    listing = source.index('gcloud storage ls -l "${HF_CACHE_GCS}/${HF_MODEL_DIR}/blobs/"')
    loop_end = source.index("done || true", listing)
    full_copy = source.index("if gcloud storage cp --recursive --no-clobber", listing)
    assert listing < loop_end < full_copy
    body = source[listing:loop_end]
    assert '[ -n "\\$_match" ] || continue' in body
