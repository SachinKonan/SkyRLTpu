"""Deploy the canonical Ray-v2 serving path on the explicitly assigned lab slice.

Requires the separately started, private lab Ray cluster and staged data.
Does not submit SkyPilot jobs, restore GCS caches, or write cloud objects.
"""
import json
from pathlib import Path
import time
import ray
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.serving import Catalog, deploy
from .lab_access import ROOT, IPS

OLD_ROOT = Path('/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-002')
SOURCE = OLD_ROOT/'sources/native-3bba6f994de00589cd35348494e88ca5c8dfa9f712104a56ac65322f9c966dcd'


def main():
    root = Path(ROOT)
    config = Config.from_dict(json.loads((root/'code-v8/lab-profile.json').read_text()))
    if config.root != ROOT or config.inference_only_ranks != [0, 1] or config.grading.tpu_ranks != [2, 3]:
        raise RuntimeError('lab profile does not match the assigned host roles')
    ray.init(address=IPS[0]+':'+str(config.ports.ray), namespace=config.run_id)
    nodes = {n['NodeManagerAddress'] for n in ray.nodes() if n['Alive']}
    if nodes != set(IPS):
        raise RuntimeError(f'lab Ray membership mismatch: {nodes}')
    prepared = {}
    for rank, ip in enumerate(IPS):
        prepared[ip] = dict(role='inference' if rank < 2 else 'grading', root=ROOT,
            source=str(SOURCE), snapshot=str(root/'ram/hf/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'))
    catalog = Catalog.options(name='fair-lab-catalog').remote(IPS[:2], config.inference.restart_limit)
    deploy(config, prepared, catalog, IPS[0])
    while True:
        time.sleep(30)


if __name__ == '__main__':
    main()
