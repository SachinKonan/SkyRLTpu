"""Worker robustness self-test: bad candidates must return ok=False and never break the worker.

  python -m tpu_grader.selftest [--chip k] [--stack v1] [--repeat 12]

Grades a valid candidate, a set of broken ones, then the valid one again (same reward expected), then
--repeat more valid grades while logging host RSS and device memory.
"""
import argparse
import json
import time
from pathlib import Path

from . import grade as G

BAD = {
    'import_error': 'raise ValueError("boom at import")\n',
    'no_features': 'x = 1\n',
    'trace_error': 'def features(maps, mask):\n    raise RuntimeError("boom in features")\n',
    'nan': 'import jax.numpy as jnp\ndef features(maps, mask):\n    return jnp.full((maps.shape[0], 4), jnp.nan)\n',
    'one_d': 'import jax.numpy as jnp\ndef features(maps, mask):\n    return maps.mean((1, 2))\n',
    'too_wide': 'import jax.numpy as jnp\ndef features(maps, mask):\n    return jnp.tile(maps[:, :1, :1].reshape(-1, 1), (1, 5000))\n',
    'python_loop': 'while True:\n    pass\n',
    'config_change': ('import jax, jax.numpy as jnp\njax.config.update("jax_default_matmul_precision", "default")\n'
                      'def features(maps, mask):\n    return jnp.stack([maps.mean((1, 2)), (maps ** 2).mean((1, 2))], 1)\n'),
    'slow': ('import jax, jax.numpy as jnp\ndef features(maps, mask):\n    x = maps\n'
             '    for _ in range(400):\n        x = jnp.fft.irfft2(jnp.fft.rfft2(x) * 0.999, s=maps.shape[1:])\n'
             '    return jnp.stack([x.mean((1, 2)), (x ** 2).mean((1, 2))], 1)\n'),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chip', type=int, default=None)
    ap.add_argument('--stack', default='v2')
    ap.add_argument('--mode', default='rl')
    ap.add_argument('--repeat', type=int, default=12)
    args = ap.parse_args()
    G.init(args.chip, args.stack, verbose=False)
    good = (Path(__file__).parent / 'examples' / 'dirpower.py').read_text()
    rows = []

    def run(name, src, **kw):
        t = time.time()
        r = G.grade_source(src, args.mode, **kw)
        rows.append(dict(name=name, ok=r['ok'], reward=r['reward'], fail=(r.get('contract') or {}).get('fail'),
                         wall=round(time.time() - t, 1), private_gb=r['runtime'].get('host_private_gb'),
                         shared_gb=r['runtime'].get('host_shared_gb'),
                         dev=r['runtime'].get('device_gb_in_use')))
        print(json.dumps(rows[-1]), flush=True)
    run('valid', good)
    for name, src in BAD.items():
        run(name, src, timeout=60)
    run('valid_again', good)
    for i in range(args.repeat):
        run(f'valid_{i}', good)


if __name__ == '__main__':
    main()
