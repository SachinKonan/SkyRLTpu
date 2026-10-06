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
    # The gpt-oss marker-aware restore threads rsync_extra (CHECKPOINT_COMPLETE
    # exclusion) between -r and the source; the ordering contract is unchanged.
    copy = source.index('"$GCS_CLI" storage rsync -r "${rsync_extra[@]}" "$SRC" "$DST"', hf_drop)
    assert need < evict < keep_own < remove < recheck < hf_drop < copy


def test_orbax_restore_uses_unsliced_copy_from_attempt_one_and_clears_partials():
    # gcloud's sliced download lost a component of one 40 GB composite object;
    # every retry resumed from the leftover _.gstmp + tracker files and failed
    # the same way, and the cell died (job 243, 2026-09-05).
    source = (REPO / "tpu/jobman/ensure_orbax_ckpt.sh").read_text()
    lock = source.index("flock -w")
    unsliced = source.index('export CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD="${CACHE_DOWNLOAD_SLICED_THRESHOLD:-0}"', lock)
    sequential = source.index('export CLOUDSDK_STORAGE_PROCESS_COUNT="${CACHE_DOWNLOAD_PROCESSES:-1}"', unsliced)
    loop = source.index("for try in 1 2 3 4; do")
    refuse = source.index('echo "ckpt: refusing restore:', loop)
    incomplete = source.index('echo "ckpt: attempt $try incomplete', loop)
    clear_partials = source.index("find \"$DST\" \\( -name '*_.gstmp' -o -name '*.gstmp' \\) -delete", incomplete)
    clear_failed = source.index('find "$DST" -mindepth 1 -delete', clear_partials)
    loop_end = source.index("\ndone\n", clear_failed)
    assert lock < unsliced < sequential < loop < refuse < incomplete
    assert incomplete < clear_partials < clear_failed < loop_end
