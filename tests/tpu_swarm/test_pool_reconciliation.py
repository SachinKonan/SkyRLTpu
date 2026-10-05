"""Reservation-preserving controller recovery regression tests."""
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from sky.serve import replica_managers as rm
from sky.server.requests import requests as api_requests
from sky.skylet import job_lib


def test_recovery_attaches_original_request(tmp_path):
    with mock.patch.object(rm.task_lib.Task, 'from_yaml_str'), \
            mock.patch.object(rm.sdk, 'launch') as launch, \
            mock.patch.object(rm.sdk, 'stream_and_get') as wait:
        rm.launch_cluster(1, 'unused', 'pool-1', str(tmp_path / 'launch.log'),
                          {}, {}, existing_request_id='original')
    launch.assert_not_called()
    wait.assert_called_once_with('original')


def test_recovery_prefers_running_request():
    m = rm.SkyPilotReplicaManager.__new__(rm.SkyPilotReplicaManager)
    m._is_pool = True
    status = api_requests.RequestStatus
    rows = [SimpleNamespace(request_id='queued', status=status.PENDING,
                            created_at=1),
            SimpleNamespace(request_id='running', status=status.RUNNING,
                            created_at=2)]
    with mock.patch.object(rm.controller_utils, '_is_consolidation_mode',
                           return_value=True), \
            mock.patch.object(api_requests, 'get_request_tasks',
                              return_value=rows):
        assert m._inflight_pool_launch('pool-1') == 'running'


@pytest.mark.parametrize('error', [requests.ReadTimeout('busy API'),
                                 RuntimeError('database is locked')])
def test_probe_observation_error_is_unknown(error):
    info = SimpleNamespace(cluster_name='pool-1')
    with mock.patch.object(rm.backend_utils, 'check_cluster_available',
                           side_effect=error):
        assert rm.ReplicaInfo.probe_pool(info)[1] is None


def test_positive_setup_failure_is_still_failure():
    backend = mock.Mock()
    backend.get_job_status.return_value = {1: job_lib.JobStatus.FAILED}
    with mock.patch.object(rm.backend_utils, 'check_cluster_available'), \
            mock.patch.object(rm.backend_utils, 'get_backend_from_handle',
                              return_value=backend):
        assert rm.ReplicaInfo.probe_pool(
            SimpleNamespace(cluster_name='pool-1'))[1] is False


def test_missing_handle_does_not_abort_other_workers():
    manager = rm.SkyPilotReplicaManager.__new__(rm.SkyPilotReplicaManager)
    manager._service_name = 'pool'
    manager._is_pool = True
    bad, good = mock.Mock(), mock.Mock()
    bad.handle.return_value = None
    good.handle.return_value = 'live-handle'
    backend = mock.Mock()
    backend.get_job_status.return_value = {1: job_lib.JobStatus.SUCCEEDED}
    with mock.patch.object(rm.serve_state, 'get_replica_infos',
                           return_value=[bad, good]), \
            mock.patch.object(rm.serve_state, 'add_or_update_replica'), \
            mock.patch.object(rm.backends, 'CloudVmRayBackend',
                              return_value=backend):
        manager._fetch_job_status.__wrapped__(manager)
    assert bad.status_property.service_ready_now is False
    backend.get_job_status.assert_called_once_with('live-handle', [1],
                                                   stream_logs=False)


def test_unknown_probe_clears_failure_history_without_teardown():
    manager = rm.SkyPilotReplicaManager.__new__(rm.SkyPilotReplicaManager)
    manager._service_name = 'pool'
    manager._is_pool = True
    manager._terminate_replica = mock.Mock()
    info = mock.Mock()
    info.first_not_ready_time = 1
    info.consecutive_failure_times = [1, 2]
    with mock.patch.object(rm.serve_state, 'get_replica_infos',
                           return_value=[info]), \
            mock.patch.object(rm.serve_state, 'add_or_update_replica'), \
            mock.patch.object(rm.mp_pool, 'ThreadPool') as pool:
        pool.return_value.__enter__.return_value.apply_async.return_value.get.return_value = (
            info, None, 10)
        manager._probe_all_replicas.__wrapped__(manager)
    assert info.first_not_ready_time is None
    assert info.consecutive_failure_times == []
    assert info.status_property.service_ready_now is False
    manager._terminate_replica.assert_not_called()


def test_missing_launch_log_does_not_block_absent_cluster_cleanup(tmp_path):
    manager = rm.SkyPilotReplicaManager.__new__(rm.SkyPilotReplicaManager)
    manager._service_name = 'pool'
    manager._is_pool = True
    manager._launch_thread_pool = {}
    manager._down_thread_pool = {}
    manager._handle_sky_down_finish = mock.Mock()
    info = mock.Mock()
    saved = tmp_path / 'replica.log'
    saved.write_text('saved evidence')
    with mock.patch.object(rm.serve_state, 'get_replica_info_from_id',
                           return_value=info), \
            mock.patch.object(rm.serve_utils, 'generate_replica_log_file_name',
                              return_value=str(saved)), \
            mock.patch.object(rm.serve_utils,
                              'generate_replica_launch_log_file_name',
                              return_value=str(tmp_path / 'missing.log')), \
            mock.patch.object(rm.global_user_state, 'get_handle_from_cluster_name',
                              return_value=None), \
            mock.patch.object(rm.global_user_state, 'cluster_with_name_exists',
                              return_value=False):
        manager._terminate_replica(1, sync_down_logs=True,
                                   replica_drain_delay_seconds=0)
    manager._handle_sky_down_finish.assert_called_once_with(info, format_exc=None)
    assert saved.read_text() == 'saved evidence'
