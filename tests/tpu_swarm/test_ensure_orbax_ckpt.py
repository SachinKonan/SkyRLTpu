from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def test_orbax_restore_evicts_sibling_model_checkpoints_before_its_own_hf_weights():
    # Pool workers are reused across models. A gemma cell on an ex-muse head
    # found 40 GB of muse orbax in the cache, could not fit its 44 GB restore
    # on the 97 GB disk, and wedged in bring-up (jobs 200/212, 2026-09-05).
    source = (REPO / "tpu/jobman/ensure_orbax_ckpt.sh").read_text()
    need = source.index("need_kb=$(( (want / 1024) + (MARGIN_GB * 1024 * 1024) ))")
    evict = source.index('for _other in "$CACHE"/*/; do', need)
    keep_own = source.index('[ "$_other" != "$DST" ] || continue', evict)
    remove = source.index('rm -rf "$_other"', keep_own)
    recheck = source.index('free_kb=$(df -Pk "$DST" | awk \'NR==2 {print $4}\')', remove)
    hf_drop = source.index('hf_dir="$HOME/.cache/huggingface/hub/models--${HF_MODEL//\\//--}"', recheck)
    copy = source.index('"$GCS_CLI" storage rsync -r "$SRC" "$DST"', hf_drop)
    assert need < evict < keep_own < remove < recheck < hf_drop < copy


def test_orbax_restore_retries_clear_partials_and_fall_back_to_unsliced_copy():
    # gcloud's sliced download lost a component of one 40 GB composite object;
    # every retry resumed from the leftover _.gstmp + tracker files and failed
    # the same way, and the cell died (job 243, 2026-09-05).
    source = (REPO / "tpu/jobman/ensure_orbax_ckpt.sh").read_text()
    loop = source.index("for try in 1 2 3 4; do")
    incomplete = source.index('echo "ckpt: attempt $try incomplete', loop)
    clear_partials = source.index("find \"$DST\" \\( -name '*_.gstmp' -o -name '*.gstmp' \\) -delete", incomplete)
    clear_trackers = source.index("surface_data/storage/tracker_files", clear_partials)
    unsliced = source.index("export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD=0", clear_trackers)
    sequential = source.index("export CLOUDSDK_STORAGE_PROCESS_COUNT=1", unsliced)
    loop_end = source.index("\ndone\n", sequential)
    assert loop < incomplete < clear_partials < clear_trackers < unsliced < sequential < loop_end
