"""Write the scaffold-experiment farm profiles (32k context, prefix caching on v5p/v6e).

Each profile starts from the RQ4 Muse v33 farm profile for the same hardware, so grading contracts and the base
bundle stay identical to RQ4; only serving settings, model and names change.

  python -m tpu.swarm.scaffold_exp.make_farm_profiles muse-v5p32:6 muse-v6e8:10 qwen38-v6e8:15

Hardware v6e8c1 = v6e-8 spot in us-central1-b (tpuswarm-v6e8-central1b): the v6e-8 layout with every GCS
path in the us-central1 bucket (no cross-region reads) and its zone.
"""
import copy
import json
from pathlib import Path
import sys

P = Path(__file__).resolve().parents[1] / 'ray_train/profiles'
B = 'gs://sk7524-tinker-tpu-us-east5'
TEMPLATE = {'v5p32': 'inference-farm-rq4-muse-east5a-v5p32-004.json', 'v6e8': 'inference-farm-rq4-muse-east5-v6e8-001.json'}
TEMPLATE['v6e8c1'] = TEMPLATE['v6e8']
# Hardware -> (name infix, zone, bucket).
SITE = {'v5p32': ('east5a', None, B), 'v6e8': ('east5b', None, B),
        'v6e8c1': ('central1b', 'us-central1-b', 'gs://sk7524-tinker-tpu-us-central1')}
# Per model and hardware: serving layout (user, 2026-10-19). Qwen3.8 v5p-32 = shu's v5p settings on a 4-host farm.
SERVING = {
    ('muse', 'v5p32'): dict(tp=2, max_sequences=64, memory_utilization=0.85),
    ('muse', 'v6e8'): dict(tp=8, max_sequences=128, memory_utilization=0.85, batched_rpa_kernel=False),
    # TP2 is not validated for Qwen3.8 on SkyPilot v5p-32; use the RQ4 Qwen3.8 v5p-32 layout (TP4, 4 engines).
    ('qwen38', 'v5p32'): dict(tp=4, max_sequences=64, memory_utilization=0.85, ragged_conv1d=True),
    ('qwen38', 'v6e8'): dict(tp=4, max_sequences=32, memory_utilization=0.85, ragged_conv1d=True),
}
# Qwen3.5 (scaf2 control, 2026-10-22): the RQ4 Qwen3.5 layouts (v5p-32 TP2 x 64, v6e-8 TP4 x 32).
SERVING[('qwen35', 'v5p32')] = dict(tp=2, max_sequences=64, memory_utilization=0.85, ragged_conv1d=True)
SERVING[('qwen35', 'v6e8')] = dict(tp=4, max_sequences=32, memory_utilization=0.85, ragged_conv1d=True)
SERVING[('muse', 'v6e8c1')] = SERVING[('muse', 'v6e8')]
SERVING[('qwen38', 'v6e8c1')] = SERVING[('qwen38', 'v6e8')]
MODEL = {
    'muse': dict(model_preset='muse-glimmer-30b', hf=f'{B}/hf-cache-muse-glimmer-v1', inference_gib=80, reserve_gib=264),
    'qwen35': dict(model_preset='qwen3.5-27b', hf=f'{B}/hf-cache-qwen35-v1', inference_gib=80, reserve_gib=264),
    'qwen38': dict(model_preset='qwen3.8-27b', hf=f'{B}/hf-cache-qwen38-1d4bf0f2', inference_gib=128, reserve_gib=280),
}


def profile(model, hw, index):
    site, zone, bucket = SITE[hw]
    run = f'inference-farm-scaf-{model}-{site}-{hw.replace("c1", "")}-{index:03d}'
    d = json.loads((P / TEMPLATE[hw]).read_text())
    d['run_id'] = run
    d['root'] = f'~/.cache/{run}'
    d['model_preset'] = MODEL[model]['model_preset']
    d.pop('model', None)
    cache = d['cache']
    seed = cache.get('inference_compile_seed') if model == 'muse' else None
    cache.update(hf=MODEL[model]['hf'], inference_gib=MODEL[model]['inference_gib'], reserve_gib=MODEL[model]['reserve_gib'],
                 inference_compile=f'{B}/scaffold-exp-farms/{run}/compile-cache',
                 trainer_compile=f'{B}/scaffold-exp-farms/{run}/trainer-compile-cache')
    cache['inference_compile_seed'] = seed or ''
    inf = d['inference']
    for key in ('batched_rpa_kernel', 'ragged_conv1d'):
        inf.pop(key, None)
    inf.update(SERVING[(model, hw)], max_model_length=32768, prefix_caching=True)
    if bucket != B:
        d = json.loads(json.dumps(d).replace(B, bucket))
    if zone:
        d['zone'] = zone
    return run, d


def main(specs):
    written = []
    for spec in specs:
        name, count = spec.split(':')
        model, hw = name.split('-')
        for i in range(1, int(count) + 1):
            run, d = profile(model, hw, i)
            (P / f'{run}.json').write_text(json.dumps(d, indent=2) + '\n')
            written.append(run)
    print('\n'.join(written))


if __name__ == '__main__':
    main(sys.argv[1:])
