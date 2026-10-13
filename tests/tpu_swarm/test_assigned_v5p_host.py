import copy
import pytest
from tpu.swarm.ray_train.config import Config

BASE='tpu/swarm/ray_train/profiles/qcmp-qwen38-qubit-s1-east5-continue15-20261010.json'

def raw():
    p=Config.load(BASE).to_dict()
    p.update(accelerator='tpu-v5p-64',zone='us-east5-a',hosts=1,physical_node='existing-node',assigned_host_ips=['10.128.1.95'])
    p['trainer'].update(hosts=1,tp=1,fsdp=4,process_bounds='1,1,1',chip_bounds='2,2,1')
    return p

def test_single_host_remote_trainer():
    c=Config.from_dict(raw())
    assert c.inference_hosts==0 and c.chips_per_host==4
    assert c.trainer.tp*c.trainer.fsdp==4

@pytest.mark.parametrize('change',[
    dict(hosts=8),dict(assigned_host_ips=['10.128.1.95','10.128.1.94']),
    dict(assigned_host_ips=['not-an-ip']),dict(physical_node=''),dict(single_bucket=False),
    dict(accelerator='tpu-v6e-8'),dict(assigned_host_ips=[]),
])
def test_reject_unsafe_assignment(change):
    p=raw();p.update(change)
    with pytest.raises(ValueError):Config.from_dict(p)

def test_reject_distributed_mesh():
    p=raw();p['trainer']['chip_bounds']='2,4,1'
    with pytest.raises(ValueError,match='four local chips'):Config.from_dict(p)


def test_two_host_remote_trainer_and_process_identity(tmp_path):
    from tpu.swarm.ray_train.commands import trainer_environment
    p=raw();p.update(hosts=2,assigned_host_ips=['10.128.1.94','10.128.1.95'])
    p['trainer'].update(hosts=2,tp=4,fsdp=2,process_bounds='1,2,1')
    c=Config.from_dict(p)
    assert c.inference_hosts == 0 and c.trainer.tp*c.trainer.fsdp == 8
    env=trainer_environment(c,tmp_path,tmp_path/'run',c.assigned_host_ips,1)
    assert env['CLOUD_TPU_TASK_ID']=='1'
    assert env['TPU_PROCESS_BOUNDS']=='1,2,1'
    assert env['TUNIX_ROW_SHARD']=='2'
    assert env['TPU_PROCESS_ADDRESSES'].split(',')[1].startswith('10.128.1.95:')
    p['assigned_host_ips']=['10.128.1.94']*2
    with pytest.raises(ValueError):Config.from_dict(p)


def test_assigned_partition_accepts_physical_permutation_but_rejects_overlap():
    from tpu.swarm.ray_train.assigned_topology import verify_partition
    rows=[dict(process_id=r,coords=[[x,y,0] for x in range(2) for y in range(r*2,r*2+2)]) for r in range(2)]
    assert verify_partition(rows,'1,2,1','2,2,1')
    with pytest.raises(ValueError,match='wrong chip coordinates'):
        verify_partition([rows[0],dict(rows[1],coords=rows[0]['coords'])],'1,2,1','2,2,1')
    assert verify_partition([dict(rows[0],process_id=1),dict(rows[1],process_id=0)],'1,2,1','2,2,1')
    with pytest.raises(ValueError,match='wrong chip coordinates'):
        verify_partition([rows[0],dict(rows[1],coords=[[x,y+2,z] for x,y,z in rows[1]['coords']])],'1,2,1','2,2,1')
    with pytest.raises(ValueError,match='missing or duplicate'):
        verify_partition([rows[0],rows[0]],'1,2,1','2,2,1')

def test_cannot_provision_assigned_host(tmp_path):
    import json
    from tpu.swarm.ray_train.build import build
    f=tmp_path/'profile.json';f.write_text(json.dumps(raw()))
    with pytest.raises(ValueError,match='direct deployment'):build(f,tmp_path/'out')

@pytest.mark.parametrize('engines,cap',[(1,4),(4,16),(8,16)])
def test_mixed_farm_capacity(engines,cap):
    from types import SimpleNamespace
    from tpu.swarm.ray_train.borrowing import Borrower
    p=raw();p['inference'].update(external_pool_max_concurrent_requests=16,external_pool_requests_per_engine=4)
    c=Config.from_dict(p)
    fake=SimpleNamespace(settings=c.inference,_attestation=lambda *a:True,_farm_engines=lambda x:engines)
    lease=SimpleNamespace()
    Borrower._verify(fake,lease,{})
    assert lease.max_requests==cap
