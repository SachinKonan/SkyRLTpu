"""Build an immutable CPU-worker artifact, excluding final-test data."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def build(simpletes,cargo_home,rustup_home,data,output):
    repo=Path(__file__).resolve().parents[2];output=Path(output).resolve()
    if output.exists():raise ValueError('refusing to overwrite an existing artifact')
    output.parent.mkdir(parents=True,exist_ok=True)
    simpletes=Path(simpletes).resolve();cargo_home=Path(cargo_home).resolve();rustup_home=Path(rustup_home).resolve()
    source_pins=json.loads((repo/'tpu/science/sources.json').read_text())
    revision=subprocess.check_output(['git','-C',str(simpletes),'rev-parse','HEAD'],text=True).strip()
    if revision!=source_pins['SimpleTES']['commit']:raise ValueError('SimpleTES revision does not match the pin')
    data=Path(data).resolve();manifest=json.loads((data/'manifest.json').read_text())
    for split in ('train','valid','discovery'):
        if hashlib.sha256((data/(split+'.npz')).read_bytes()).hexdigest()!=manifest['splits'][split]['sha256']:
            raise ValueError('portfolio data checksum mismatch')
    def keep(info):
        if '__pycache__' in info.name or '/share/doc/' in info.name or '/share/man/' in info.name:return None
        return info
    with tarfile.open(output,'w:gz',compresslevel=1) as tar:
        pairs=[(repo/'tpu/science','tpu/science'),(repo/'tpu/swarm/ray_train','tpu/swarm/ray_train'),
            (repo/'tpu/swarm/bench','tpu/swarm/bench'),(simpletes,'.science/routing-task'),
            (rustup_home,'.science/rustup')]
        pairs += [(cargo_home/name,'.science/cargo/'+name) for name in ('bin','registry','git') if (cargo_home/name).exists()]
        pairs += [(data/name,'.science/data/portfolio/'+name) for name in ('train.npz','valid.npz','discovery.npz','manifest.json')]
        for source,dest in pairs:tar.add(source,arcname=dest,filter=keep)
    hasher=hashlib.sha256()
    with output.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):hasher.update(block)
    digest=hasher.hexdigest()
    record=dict(artifact=str(output),sha256=digest,bytes=output.stat().st_size,
        final_test_included=False,source_pins=source_pins)
    output.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--simpletes',required=True);p.add_argument('--cargo-home',required=True)
    p.add_argument('--rustup-home',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    print(json.dumps(build(**vars(p.parse_args())),indent=2))
