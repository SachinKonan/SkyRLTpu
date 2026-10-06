import io
import json
import sqlite3
import tarfile
from tpu.science.circuit_weight_handoff import strip_optimizer, minimal_registry


def test_strip_preserves_weights_and_removes_moments(tmp_path):
    source, dest = tmp_path/'source.tar.gz', tmp_path/'weights.tar.gz'
    with tarfile.open(source, 'w:gz') as t:
        for name, data in [('lora_weights.npz', b'exact-lora-tensor-bytes'),
                           ('optimizer_state.npz', b'adam-moments'),
                           ('tunix_checkpoint_meta.json', json.dumps(dict(lora_config={'rank':32}, lora_layouts={'a': []}, optimizer_layouts={'b':[]})).encode())]:
            m=tarfile.TarInfo(name);m.size=len(data);t.addfile(m,io.BytesIO(data))
    strip_optimizer(source,dest)
    with tarfile.open(dest) as t:
        assert 'optimizer_state.npz' not in t.getnames()
        assert t.extractfile('lora_weights.npz').read()==b'exact-lora-tensor-bytes'
        meta=json.load(t.extractfile('tunix_checkpoint_meta.json'))
        assert 'optimizer_layouts' not in meta and meta['lora_layouts']=={'a': []}


def test_registry_carries_only_selected_checkpoint(tmp_path):
    src,dst=tmp_path/'source.db',tmp_path/'registry.db'
    with sqlite3.connect(src) as db:
        db.executescript('CREATE TABLE sessions(session_id TEXT); CREATE TABLE models(model_id TEXT, session_id TEXT, base_model TEXT, lora_config TEXT); CREATE TABLE checkpoints(model_id TEXT, checkpoint_id TEXT, checkpoint_type TEXT, status TEXT); CREATE TABLE futures(request_id INTEGER);')
        db.execute('INSERT INTO sessions VALUES (?)',('s',))
        db.execute('INSERT INTO models VALUES (?,?,?,?)',('m','s','Qwen',json.dumps({'rank':32})))
        db.executemany('INSERT INTO checkpoints VALUES (?,?,?,?)',[('m','15','TRAINING','COMPLETED'),('m','14','TRAINING','COMPLETED')])
        db.execute('INSERT INTO futures VALUES (9)')
    result=minimal_registry(src,dst,'m','15')
    assert result['lora_config']['rank']==32
    with sqlite3.connect(dst) as db:
        assert db.execute('SELECT checkpoint_id FROM checkpoints').fetchall()==[('15',)]
        assert db.execute('SELECT COUNT(*) FROM futures').fetchone()==(0,)
