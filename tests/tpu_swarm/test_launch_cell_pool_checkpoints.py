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


def test_published_local_checkpoints_are_pruned(tmp_path):
    # Local disk is not the durable store on pool workers: each muse step leaves
    # ~3 GB of tarballs that filled a 97 GB boot disk holding the 41 GB orbax
    # checkpoint before step 15 (jobs 188/161, 2026-09-05).
    source = (REPO / "tpu/launch_cell.sh").read_text()
    assert 's|CKPTPRUNEPLACEHOLDER|$CLIENT_ROOT/tpu/jobman/prune_local_checkpoints.sh|' in source
    writeback = source.index('echo "ckpt-writeback-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/sidecar.log"')
    assert 'bash "CKPTPRUNEPLACEHOLDER" "CKPTROOTPLACEHOLDER" "CKPTGCSPLACEHOLDER"' in source[writeback:]
    sync = (REPO / "tpu/jobman/cell_sync.sh").read_text()
    assert 'prune_local_checkpoints.sh" "$CKPT_ROOT" "$SKYRL_CKPT_GCS"' in sync
    assert sync.index("prune_local_checkpoints.sh") > sync.index("ckpt-writeback-rc")

    # Functional check with a stubbed gcloud: only published, same-size, older
    # tarballs go; the newest two, "final", seeds, and unpublished files stay.
    root = tmp_path / "ckpt"
    published = {}
    for model, steps in (("model_a", ["000001", "000002", "000003", "000004"]), ("model_b", ["000007"])):
        for family in ("", "sampler_weights/"):
            for step in steps:
                p = root / model / family / f"{step}.tar.gz"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x" * (100 + len(step)))
                if not (model == "model_a" and step == "000002" and family == ""):
                    published[f"{model}/{family}{step}.tar.gz"] = p.stat().st_size
    (root / "model_a" / "sampler_weights" / "ss0_seq1.tar.gz").write_bytes(b"seed")
    (root / "model_a" / "final.tar.gz").write_bytes(b"final")
    published["model_a/sampler_weights/000001.tar.gz"] = 1  # wrong size -> keep
    stub = tmp_path / "gcloud"
    stub.write_text(
        "#!/bin/bash\n"
        "# args: storage ls -l gs://bkt/ckpts/<rel>\n"
        'rel="${4#gs://bkt/ckpts/}"\n'
        "case \"$rel\" in\n"
        + "".join(f'  "{k}") echo "{v} 2026-09-05T00:00:00Z gs://bkt/ckpts/{k}";;\n' for k, v in published.items())
        + "  *) exit 1;;\nesac\n"
    )
    stub.chmod(0o755)
    import os, subprocess
    env = dict(os.environ, GCLOUD_STORAGE_CLI=str(stub))
    out = subprocess.run(
        ["bash", str(REPO / "tpu/jobman/prune_local_checkpoints.sh"), str(root), "gs://bkt/ckpts"],
        capture_output=True, text=True, env=env, check=True,
    ).stdout
    remaining = sorted(str(p.relative_to(root)) for p in root.rglob("*.tar.gz"))
    assert remaining == [
        "model_a/000002.tar.gz",  # unpublished -> kept
        "model_a/000004.tar.gz",  # newest two of the digit names + final
        "model_a/final.tar.gz",
        "model_a/sampler_weights/000001.tar.gz",  # size mismatch -> kept
        "model_a/sampler_weights/000003.tar.gz",
        "model_a/sampler_weights/000004.tar.gz",
        "model_a/sampler_weights/ss0_seq1.tar.gz",
        "model_b/000007.tar.gz",
        "model_b/sampler_weights/000007.tar.gz",
    ]
    assert "removed=3" in out


def test_cell_monitor_recreates_a_missing_sidecar():
    source = (REPO / "tpu/jobman/cell_monitor.sh").read_text()

    # exact tmux target: a plain "cell-backup" would prefix-match nothing else,
    # but "cell" prefix-matched "cell-backup" (852cc354), so every probe uses "=".
    heal = source.index('if ! tmux has-session -t "=${SESSION}-backup"')
    assert 'tmux new-session -d -s "${SESSION}-backup" "bash $HOME/sidecar_${RUN}.sh"' in source[heal:]
    # Inside the monitoring loop and before the split trainer/vLLM checks.
    split_health = source.index('engine_failure_kind=""', heal)
    assert source.index("while true; do") < heal < split_health


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


def test_cell_monitor_restarts_only_a_failed_no_ray_vllm_host():
    source = (REPO / "tpu/jobman/cell_monitor.sh").read_text()

    assert 'VLLM_INPLACE_RESTART_LIMIT="${VLLM_INPLACE_RESTART_LIMIT:-2}"' in source
    assert 'if [[ "$VLLM_RAY_EXECUTOR" != "0" ]]' in source
    assert 'VLLM_RELATIVE_WORKER_ID=0 VLLM_USE_RAY_EXECUTOR=0' in source
    assert 'VLLM_START_SERVER=1 VLLM_CLEANUP=1' in source
    assert 'capture_vllm_diagnostics "$worker" "$ip" "$port"' in source
    assert 'restart_vllm_worker "$UNHEALTHY_VLLM_WORKER"' in source


def test_v4_recovery_fences_the_previous_client_attempt():
    wrapper = (REPO / "tpu/swarm/run_qwen35_v4_64_grpo.sh").read_text()
    monitor = (REPO / "tpu/jobman/cell_monitor.sh").read_text()

    fence = wrapper.index('if [[ -n "${SKYPILOT_INTERNAL_JOB_ID:-}" ]]')
    reconcile = wrapper.index("reconcile_v4_64_role_caches.sh")
    assert fence < reconcile
    assert 'tmux kill-session -t "=$session"' in wrapper[fence:reconcile]
    assert 'rm -f "$HOME/ENGINE-SICK"' in wrapper[fence:reconcile]
    assert '${SKYPILOT_INTERNAL_JOB_ID:-standalone}' in monitor


def test_tunix_checkpoint_write_through_runs_on_multihost_owner():
    backend = (REPO / "skyrl/backends/tunix_backend.py").read_text()
    engine = (REPO / "skyrl/tinker/engine.py").read_text()
    launcher = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()

    assert "checkpoint_mirror_gcs: str | None" in backend
    mirror = (REPO / "skyrl/utils/checkpoint_mirror.py").read_text()
    assert '["gcloud", "storage", "cp", str(local_path), destination]' in mirror
    assert '"objects",\n            "describe"' in mirror
    assert "remote_size != str(local_size)" in mirror
    assert 'self._mirror_checkpoint(output_path, model_id, "sampler_weights")' in backend
    assert "request_data.sampling_session_seq_id is None or bool(checkpoint_mirror)" in engine
    assert '"checkpoint_mirror_gcs": "${SKYRL_CKPT_GCS}"' in launcher


def test_launch_cell_carries_a_foreign_init_state_on_pool_workers():
    # Meta weights-carry: the init state lives under another run's model id, so
    # nothing in this run's checkpoints.jsonl restores or registers it, and pool
    # hosts have no gcsfuse. The launcher must restore + register the named
    # state before the client starts, then pass it as TTD_INIT_STATE_PATH_<TAG>.
    source = (REPO / "tpu/launch_cell.sh").read_text()
    own_rereg = source.index('_rereg "$_jsonl"')
    carry = source.index('if [ -n "${META_INIT_STATE_PATH:-}" ]; then', own_rereg)
    restore = source.index('_restore_ckpts "$_carry_jsonl"', carry)
    rereg = source.index('_rereg "$_carry_jsonl"', restore)
    env = source.index('TTD_INIT_STATE_PATH_${_carry_tag}=$META_INIT_STATE_PATH', rereg)
    client = source.index('tmux new-session -d -s "$SESSION"', env)
    assert own_rereg < carry < restore < rereg < env < client
    # the tag is the member metric prefix (member_gemma -> GEMMA), matching ensemble.py
    assert '${MEMBER_DIR#member_}' in source[carry:client]
    # the carried path must reach the client through the EXTRA_TTD_ENV passthrough
    assert '${EXTRA_TTD_ENV:-}' in source[client:client + 6000]
