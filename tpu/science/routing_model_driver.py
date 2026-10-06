"""Native Muse sampling with the pinned CPU routing grader on every Ray host."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import httpx
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from .ray_cpu import grade
from .rewards import invalid
from tpu.swarm.ray_train.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True)
    profile = str(Path(parser.parse_args().profile).resolve())
    config = Config.load(profile)
    root = Path.cwd()
    native = Path(config.root).expanduser()
    out = root / ('results-' + config.run_id)
    out.mkdir(exist_ok=False)
    head = os.environ['SKYPILOT_NODE_IPS'].split()[0]
    base = f'http://{head}:{config.ports.inference}'
    status = None
    pending = []
    try:
        ray.init(address=f'{head}:{config.ports.ray}', namespace=config.run_id)
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            try:
                status = ray.get_actor('runtime-status', namespace=config.run_id)
                nodes = [n for n in ray.nodes() if n['Alive']]
                if len(nodes) == config.hosts:
                    break
            except ValueError:
                pass
            time.sleep(5)
        else:
            raise TimeoutError('routing Ray hosts did not register')
        seed = (root / 'tpu/science/seed_routing.py').read_text()
        pending = [grade.options(scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'], soft=False))
                   .remote('routing', seed, str(root)) for n in nodes]
        references = ray.get(pending, timeout=2100)
        pending = []
        (out / 'references.json').write_text(json.dumps(references, indent=2))
        if not all(r['correctness'] == 1 and r['metrics'].get('case_count') == 72 for r in references):
            raise RuntimeError('72-case routing reference failed on one or more hosts')
        print(json.dumps(dict(event='routing_references_passed', hosts=len(references),
                              rewards=[r['reward'] for r in references])), flush=True)
        snapshot = None
        while time.monotonic() < deadline:
            log = native / 'runs' / config.run_id / 'host-1.jsonl'
            if log.exists():
                for line in log.read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get('event') == 'host_prepared' and row.get('role') == 'inference':
                        snapshot = row['snapshot']
            try:
                catalog = ray.get_actor('inference-catalog', namespace=config.run_id)
                state = ray.get(catalog.snapshot.remote(), timeout=10)
                if (snapshot and len(state['replicas']) == config.inference_hosts * config.engines_per_host
                        and httpx.get(base + '/health', timeout=5).status_code == 200):
                    break
            except (ValueError, ray.exceptions.RayError, OSError, httpx.HTTPError):
                pass
            time.sleep(10)
        else:
            raise TimeoutError('native Muse inference did not become ready')
        with (out / 'sampling.log').open('wb') as log:
            subprocess.run([str(native / 'envs/serving/bin/python'), '-m', 'tpu.science.sample',
                            '--profile', profile, '--tokenizer', snapshot,
                            '--prompt', str(root / 'tpu/science/prompts/rendered/routing.txt'),
                            '--base', base, '--output', str(out / 'samples'), '--samples', '4'],
                           check=True, timeout=7500, stdout=log, stderr=subprocess.STDOUT)
        summary = json.loads((out / 'samples/summary.json').read_text())
        verdicts = {}
        indices = []
        for row in summary['candidates']:
            index = row['index']
            if not row['has_code']:
                verdicts[index] = invalid(row.get('format_error') or 'missing code', phase='format')
            else:
                source = (out / 'samples' / f'candidate-{index:03d}.py').read_text()
                pending.append(grade.options(scheduling_strategy='SPREAD').remote('routing', source, str(root)))
                indices.append(index)
        for index, result in zip(indices, ray.get(pending, timeout=3900)):
            verdicts[index] = result
        pending = []
        for index, result in verdicts.items():
            (out / f'candidate-{index:03d}-verdict.json').write_text(json.dumps(result, indent=2))
            print(json.dumps(dict(event='routing_candidate_graded', candidate=index,
                                  reward=result['reward'], correctness=result['correctness'])), flush=True)
        (out / 'summary.json').write_text(json.dumps(dict(run_id=config.run_id, references=references,
                                                          candidates=verdicts), indent=2))
    except BaseException as exc:
        (out / 'failure.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')))
        raise
    finally:
        try:
            archive = root / (config.run_id + '-results.tar.gz')
            subprocess.run(['tar', '-czf', str(archive), '-C', str(root), out.name], check=True, timeout=120)
            subprocess.run(['gcloud', 'storage', 'cp', str(archive), os.environ['SCIENCE_RESULT_URI']],
                           check=True, timeout=120)
        finally:
            for ref in pending:
                ray.cancel(ref, force=True)
            if status:
                ray.get(status.request_stop.remote(), timeout=30)
            ray.shutdown()


if __name__ == '__main__':
    main()
