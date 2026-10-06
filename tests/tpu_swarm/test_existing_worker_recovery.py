import importlib.util
from pathlib import Path
import pickle
import sqlite3
from types import SimpleNamespace

import pytest


@pytest.fixture
def recovery():
    path = Path(__file__).parents[2] / 'tpu/swarm/recover_v5p_existing_workers.py'
    spec = importlib.util.spec_from_file_location('existing_worker_recovery', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample(module):
    name = module.NODES[189]
    node = dict(name=f'projects/vision-mix/locations/us-east5-a/nodes/{name}',
                state='READY', health='HEALTHY', acceleratorType='v5p-32',
                labels={'skypilot-user': 'sk7524', 'ray-cluster-name': 'cloud-id'},
                networkEndpoints=[{}] * 4, createTime='original')
    queued = dict(name=f'projects/vision-mix/locations/us-east5-a/queuedResources/{name}-q',
                  state={'state': 'ACTIVE'}, tpu={'nodeSpec': [{'nodeId': name}]},
                  createTime='original-queue')
    return node, queued


@pytest.mark.parametrize('field,value', [('state', 'PREEMPTED'),
                                        ('health', 'UNHEALTHY'),
                                        ('acceleratorType', 'v5p-64')])
def test_unhealthy_or_wrong_topology_refused(recovery, field, value):
    node, queued = sample(recovery)
    node[field] = value
    with pytest.raises(recovery.RecoveryRefused):
        recovery.validate_snapshot(189, node, queued)


def test_suspending_reservation_refused(recovery):
    node, queued = sample(recovery)
    queued['state']['state'] = 'SUSPENDING'
    with pytest.raises(recovery.RecoveryRefused):
        recovery.validate_snapshot(189, node, queued)


def test_wrong_owner_refused(recovery):
    node, queued = sample(recovery)
    node['labels']['skypilot-user'] = 'someone-else'
    with pytest.raises(recovery.RecoveryRefused):
        recovery.validate_snapshot(189, node, queued)


def test_existing_only_never_creates_or_deletes(recovery, monkeypatch):
    from sky.provision import gcp, provisioner
    from sky.provision.gcp.instance_utils import GCPTPUVMInstance
    node, queued = sample(recovery)
    monkeypatch.setattr(recovery, 'snapshot', lambda worker: (node, queued))
    original = gcp.run_instances
    config = SimpleNamespace(count=1, provider_config={
        'project_id': 'vision-mix', 'availability_zone': 'us-east5-a'})
    with recovery.existing_only(189, node):
        record = gcp.run_instances('us-east5', recovery.POOL + '-189',
                                   'cloud-id', config)
        assert record.created_instance_ids == []
        assert record.head_instance_id == recovery.NODES[189]
        for method in [GCPTPUVMInstance.create_instances,
                       GCPTPUVMInstance.delete_queued_resource,
                       GCPTPUVMInstance.stop, GCPTPUVMInstance.terminate,
                       gcp.stop_instances, gcp.terminate_instances,
                       provisioner.teardown_cluster]:
            with pytest.raises(recovery.RecoveryRefused):
                method()
        with pytest.raises(recovery.RecoveryRefused):
            gcp.run_instances('us-east5', 'wrong-cluster', 'cloud-id',
                              config)
    assert gcp.run_instances is original


def test_replacement_during_recovery_refused(recovery, monkeypatch):
    from sky.provision import gcp
    node, queued = sample(recovery)
    replaced = dict(node, createTime='replacement')
    monkeypatch.setattr(recovery, 'snapshot', lambda worker: (replaced, queued))
    config = SimpleNamespace(count=1, provider_config={
        'project_id': 'vision-mix', 'availability_zone': 'us-east5-a'})
    with recovery.existing_only(189, node):
        with pytest.raises(recovery.RecoveryRefused):
            gcp.run_instances('us-east5', recovery.POOL + '-189', 'cloud-id', config)


def test_activation_requires_probe_and_unchanged_record(recovery, monkeypatch, tmp_path):
    from sky.serve.replica_managers import ReplicaInfo
    from sky.utils.common_utils import ProcessStatus
    from sky.serve.serve_state import ReplicaStatus
    monkeypatch.setattr(recovery, 'HOME', tmp_path)
    directory = tmp_path / '.sky/serve'
    directory.mkdir(parents=True)
    with sqlite3.connect(directory / 'services.db') as db:
        db.execute('CREATE TABLE replicas (service_name TEXT, replica_id INTEGER, replica_info BLOB)')
        info = ReplicaInfo(189, recovery.POOL + '-189', '-', True, None, 3, None)
        info.status_property.sky_launch_status = ProcessStatus.FAILED
        info.status_property.sky_down_status = ProcessStatus.FAILED
        before = pickle.dumps(info)
        db.execute('INSERT INTO replicas VALUES (?,?,?)', (recovery.POOL, 189, before))
    monkeypatch.setattr(recovery, 'check_no_operation', lambda worker: None)
    recovery.activate(189, before)
    result = pickle.loads(recovery.replica_blob(189))
    assert result.status != ReplicaStatus.READY
    assert result.status_property.should_track_service_status()
    with pytest.raises(recovery.RecoveryRefused):
        recovery.activate(189, before)
