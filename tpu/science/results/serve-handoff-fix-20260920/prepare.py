"""Patch immutable submitted artifacts without changing run configs or journals."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import yaml

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
IDS = set(range(1227, 1239)) | {1259, 1260, 1261} | set(range(1273, 1281))
TARGET = 'tpu/swarm/ray_train/serving.py'
old_code = subprocess.check_output(['git', 'show', 'HEAD:' + TARGET], cwd=ROOT)
new_code = (ROOT / TARGET).read_bytes()
found = {}
for p in (ROOT / '.science/packages').rglob('submission.json'):
    receipt = json.loads(p.read_text())
    if receipt.get('job_id') in IDS:
        found[receipt['job_id']] = p.parent
if set(found) != IDS:
    raise RuntimeError('Missing original package receipts')
rows = []
for old_job in sorted(found):
    original = found[old_job]
    row = json.loads((original / 'submission-manifest.json').read_text())
    source, source_task = ROOT / row['archive'], ROOT / row['task_yaml']
    if hashlib.sha256(source.read_bytes()).hexdigest() != row['archive_sha256']:
        raise RuntimeError(f'Original artifact changed: {old_job}')
    if hashlib.sha256(source_task.read_bytes()).hexdigest() != row['yaml_sha256']:
        raise RuntimeError(f'Original task changed: {old_job}')
    out = ROOT / f'.science/packages/serve-handoff-fix-20260920/{old_job}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'submission.json').exists():
        raise RuntimeError('Never repackage submitted replacements')
    archive = out / source.name
    originals = {}
    with tarfile.open(source) as src, tarfile.open(archive, 'w:gz') as dst:
        changed = 0
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            if data is not None:
                originals[member.name] = hashlib.sha256(data).hexdigest()
            if member.name == TARGET:
                if data != old_code:
                    raise RuntimeError(f'Unexpected serving implementation: {old_job}')
                data = new_code; member.size = len(data); changed += 1
            dst.addfile(member, io.BytesIO(data) if data is not None else None)
    assert changed == 1
    with tarfile.open(archive) as check:
        patched = {m.name: hashlib.sha256(check.extractfile(m).read()).hexdigest()
                   for m in check.getmembers() if m.isfile()}
    assert set(patched) == set(originals)
    assert [k for k in patched if patched[k] != originals[k]] == [TARGET]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    uri = row['code_uri'].rsplit('/', 1)[0] + '/serve-handoff-' + digest + '.tar.gz'
    doc = yaml.safe_load(source_task.read_text())
    doc['name'] = row['run_id'] + '-handofffix'
    doc['envs']['RAY_TRAIN_CODE'] = uri
    doc['envs']['RAY_TRAIN_CODE_SHA256'] = digest
    task = out / (doc['name'] + '.yaml')
    task.write_text(yaml.safe_dump(doc, sort_keys=False))
    row.update(old_job_id=old_job, original_archive_sha256=row['archive_sha256'],
               original_package_dir=row['package_dir'], package_dir=str(out.relative_to(ROOT)),
               archive=str(archive.relative_to(ROOT)), task_yaml=str(task.relative_to(ROOT)),
               code_uri=uri, archive_sha256=digest,
               yaml_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),
               changed_files=[TARGET], preserved_files=len(originals)-1)
    (out / 'submission-manifest.json').write_text(json.dumps(row, indent=2)+'\n')
    rows.append(row)
(HERE / 'jobs.json').write_text(json.dumps({'jobs': rows}, indent=2)+'\n')
print(json.dumps({'packages':len(rows),'changed_files':[TARGET],'config_changes':0}))
