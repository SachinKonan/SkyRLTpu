"""Prepare isolated weights-only circuit continuations without changing donors."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import tarfile


def strip_optimizer(source, destination):
    """Keep exact LoRA tensor bytes, omit optimizer payload and its layout metadata."""
    kept = {}
    removed = []
    with tarfile.open(source, 'r:gz') as src, tarfile.open(destination, 'w:gz', compresslevel=1) as dst:
        for member in src:
            name = Path(member.name).name
            if name == 'optimizer_state.npz':
                removed.append(member.name)
                continue
            if not member.isfile():
                dst.addfile(member)
                continue
            data = src.extractfile(member).read()
            if name == 'tunix_checkpoint_meta.json':
                meta = json.loads(data)
                meta.pop('optimizer_layouts', None)
                if 'lora_mix' in meta:
                    raise ValueError('mixed adapters require a separate handoff')
                data = json.dumps(meta).encode()
            if name == 'lora_weights.npz':
                kept[name] = hashlib.sha256(data).hexdigest()
            member.size = len(data)
            dst.addfile(member, io.BytesIO(data))
    if len(removed) != 1 or 'lora_weights.npz' not in kept:
        raise ValueError('expected one optimizer payload and LoRA weights')
    with tarfile.open(destination) as archive:
        for member in archive:
            if Path(member.name).name == 'optimizer_state.npz':
                raise ValueError('optimizer survived stripping')
            if Path(member.name).name == 'lora_weights.npz':
                assert hashlib.sha256(archive.extractfile(member).read()).hexdigest() == kept['lora_weights.npz']
    return dict(lora_weights_sha256=kept['lora_weights.npz'], removed=removed,
                archive_sha256=hashlib.sha256(Path(destination).read_bytes()).hexdigest())


def minimal_registry(source, destination, model_id, checkpoint_id):
    """Keep only the source model, its session, and the requested training checkpoint."""
    if Path(destination).exists():
        raise ValueError('destination registry exists')
    with sqlite3.connect('file:'+str(Path(source).resolve())+'?mode=ro', uri=True) as src, sqlite3.connect(destination) as dst:
        assert src.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        for (sql,) in src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"):
            dst.execute(sql)
        src.row_factory = sqlite3.Row
        model = dict(src.execute('SELECT * FROM models WHERE model_id=?', (model_id,)).fetchone())
        ckpt = dict(src.execute("SELECT * FROM checkpoints WHERE model_id=? AND checkpoint_id=? AND checkpoint_type='TRAINING'", (model_id, checkpoint_id)).fetchone())
        if ckpt['status'] != 'COMPLETED':
            raise ValueError('source checkpoint incomplete')
        session = dict(src.execute('SELECT * FROM sessions WHERE session_id=?', (model['session_id'],)).fetchone())
        for table, row in [('sessions', session), ('models', model), ('checkpoints', ckpt)]:
            cols = ','.join('"'+k+'"' for k in row)
            dst.execute(f'INSERT INTO {table} ({cols}) VALUES ({",".join("?" for _ in row)})', tuple(row.values()))
        dst.commit()
        assert dst.execute('SELECT COUNT(*) FROM futures').fetchone()[0] == 0
        assert dst.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
        return dict(model_id=model_id, checkpoint_id=checkpoint_id, base_model=model['base_model'], lora_config=json.loads(model['lora_config']))
