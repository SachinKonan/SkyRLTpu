"""CPU-only Ray Serve regression: removing a neighbor must preserve a replica."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import uuid

import ray
from ray import serve


def main():
    p = argparse.ArgumentParser(); p.add_argument('--serving-file', required=True)
    args = p.parse_args()
    spec = importlib.util.spec_from_file_location('tpu.swarm.ray_train.serving_fix_probe', args.serving_file)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

    @serve.deployment(num_replicas=1, ray_actor_options={'num_cpus': 0.1})
    class Engine:
        def __init__(self): self.identity = uuid.uuid4().hex
        def identity_value(self): return [self.identity, os.getpid()]

    @serve.deployment(ray_actor_options={'num_cpus': 0.1})
    class Gateway:
        def __init__(self, engines): self.engines = engines
        async def identities(self):
            return [await e.identity_value.remote() for e in self.engines]

    with tempfile.TemporaryDirectory(prefix='serve-handoff-') as tmp:
        ray.init(address='local', num_cpus=2, include_dashboard=False,
                 _node_ip_address='127.0.0.1', _temp_dir=tmp, log_to_driver=False)
        try:
            serve.start(http_options={'host': '127.0.0.1', 'port': 39781})
            records = []
            for stable in (False, True):
                name = 'fixed' if stable else 'unversioned'
                def deploy(keys):
                    engines = []
                    for key in keys:
                        opts = dict(name=name+'-'+key)
                        if stable:
                            version = mod.engine_version({'run_id': 'cpu-probe'}, {key: {'role': 'inference'}}, 'head')
                            deployment = mod.pin_deployment_version(Engine.options(**opts), version)
                        else:
                            deployment = Engine.options(**opts)
                        engines.append(deployment.bind())
                    return serve.run(Gateway.bind(engines), name=name, route_prefix='/'+name)
                first = deploy(['a','b']).identities.remote().result(timeout_s=30)
                second = deploy(['b']).identities.remote().result(timeout_s=30)
                preserved = first[1] == second[0]
                assert preserved == stable, (name,first,second)
                records.append(dict(mode=name, before=first, after=second, survivor_preserved=preserved))
                serve.delete(name)
            print(json.dumps(dict(ray_version=ray.__version__, results=records)), flush=True)
        finally:
            serve.shutdown(); ray.shutdown()

if __name__ == '__main__': main()
