from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_launch_cell_points_reregister_at_the_running_servers_db():
    # Pool workers run the tinker server from the bundle (CLIENT_ROOT), where
    # skyrl/tinker/config.py creates tinker.db; reregister_states.py's own
    # default is the jobman checkout path and failed with "unable to open
    # database file" on the first pool resume (job 159, 2026-09-04).
    source = (REPO / "tpu/launch_cell.sh").read_text()

    assert '"$CLIENT_ROOT/skyrl/tinker/tinker.db"' in source
    assert '--db "$TINKER_DB" --ckpt-root "$CKPT_ROOT"' in source
    assert "--db '$TINKER_DB' --ckpt-root '$CKPT_ROOT'" in source


def test_launch_cell_restores_and_publishes_checkpoints_off_gcsfuse():
    source = (REPO / "tpu/launch_cell.sh").read_text()

    # restore happens before re-register, and only for local (non-mount) roots
    restore = source.index("_restore_ckpts() {")
    assert "ckpt_root_is_mount && return 0" in source[restore:]
    assert 'gcloud storage cp "$SKYRL_CKPT_GCS/$_model/$_rel" "$_dst"' in source
    assert source.index('_restore_ckpts "$_jsonl"') < source.index('_rereg "$_jsonl"')
    # the sidecar publishes the checkpoint tree additively (no -d) when local
    assert 'if [ "CKPTWRITEBACKPLACEHOLDER" = "1" ]; then' in source
    assert '"CKPTROOTPLACEHOLDER" "CKPTGCSPLACEHOLDER"' in source
    assert "_ckpt_writeback=0; ckpt_root_is_mount || _ckpt_writeback=1" in source

    # gcloud-based rsync: gsutil fell back to pure-Python CRC32C on the multi-GB
    # tarballs and the sidecar sat for many minutes per cycle.
    assert 'bash "GCSRSYNCPLACEHOLDER" -r --exclude=' in source
    assert "s|GCSRSYNCPLACEHOLDER|$CLIENT_ROOT/tpu/gcs_rsync.sh|" in source
    assert "gsutil -m rsync -r -x '.*\\.tmp$|.*\\.gstmp$|.*\\.partial$'" not in source

    sync = (REPO / "tpu/jobman/cell_sync.sh").read_text()
    assert '"$CKPT_ROOT" "$SKYRL_CKPT_GCS"' in sync
    assert 'mountpoint -q "$CKPT_ROOT"' in sync
    assert 'bash "$(dirname "$0")/../gcs_rsync.sh" -r --exclude=' in sync
    assert "rsync -r -d" not in sync


def test_cell_monitor_recreates_a_missing_sidecar():
    source = (REPO / "tpu/jobman/cell_monitor.sh").read_text()

    # exact tmux target: a plain "cell-backup" would prefix-match nothing else,
    # but "cell" prefix-matched "cell-backup" (852cc354), so every probe uses "=".
    heal = source.index('if ! tmux has-session -t "=${SESSION}-backup"')
    assert 'tmux new-session -d -s "${SESSION}-backup" "bash $HOME/sidecar_${RUN}.sh"' in source[heal:]
    # inside the monitoring loop, after the client-death exit and before the
    # engine health check
    assert source.index("while true; do") < heal < source.index("if engines_healthy 0; then")


def test_engine_failures_use_unlimited_recovery_exit_codes():
    worker = (REPO / "tpu/jobman/cell_worker.sh").read_text()
    monitor = (REPO / "tpu/jobman/cell_monitor.sh").read_text()

    assert 'SETUP_RETRY_EXIT_CODE="${SETUP_RETRY_EXIT_CODE:-33}"' in worker
    assert 'exit "$SETUP_RETRY_EXIT_CODE"' in worker
    assert 'SETUP_RETRY_EXIT_CODE="${SETUP_RETRY_EXIT_CODE:-33}"' in monitor
    assert 'RUNTIME_RECOVERY_EXIT_CODE="${RUNTIME_RECOVERY_EXIT_CODE:-34}"' in monitor
    assert 'exit "$SETUP_RETRY_EXIT_CODE"' in monitor
    assert 'exit "$RUNTIME_RECOVERY_EXIT_CODE"' in monitor
    assert "vLLM health check failed: worker=$worker ip=$ip port=$port" in monitor
