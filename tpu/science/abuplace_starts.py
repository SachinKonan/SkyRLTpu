"""Extract the pinned AbuPlace global-placement stage, without refinement."""
from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path

VARIANTS = ('off', 'rudy', 'rudy_hv')
ABUPLACE_COMMIT = 'a24087c45588f1873eb2dc18293e407e5477041d'


def variant_environment(variant):
    if variant not in VARIANTS:
        raise ValueError(f'unknown AbuPlace Xplace variant: {variant}')
    return {'XP_CONG_LOSS': '0' if variant == 'off' else '1',
            'XP_CONG_LOSS_KERNEL': 'rudy' if variant == 'off' else variant}


def run(module, benchmark, plc, output, variant):
    import numpy as np
    import torch
    # Preserve upstream defaults, including GPConfig.seed=0, rather than
    # silently substituting the outer pipeline's seed=42 for the GP seed.
    unexpected = [k for k in os.environ if k.startswith('XP_')]
    if unexpected:
        raise ValueError(f'ambient AbuPlace overrides: {unexpected}')
    os.environ.update(variant_environment(variant))
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    output = Path(output)
    lef, deff = output / 'input.lef', output / 'input.def'
    module._generate_lefdef(benchmark, plc, str(lef), str(deff))
    module._ensure_xplace()
    placer = module.XplacePlacer()
    logger = logging.getLogger('xplace')
    logger.setLevel(logging.CRITICAL)
    parsed = placer._parse_lefdef(benchmark, str(lef), str(deff))
    result = placer._run_gp(benchmark, parsed, torch.device('cuda:0'), logger)
    if result is None:
        raise RuntimeError('AbuPlace Xplace GP failed; fallback coordinates are not a successful variant')
    positions, hpwl, overflow, seconds, iterations = result
    if not np.isfinite([hpwl, overflow, seconds]).all():
        raise ValueError('nonfinite AbuPlace GP metrics')
    metadata = dict(variant=variant, source_commit=ABUPLACE_COMMIT,
                    stage='global_placement_only', hpwl=hpwl, overflow=overflow,
                    gp_seconds=seconds, iterations=iterations,
                    gp_config=asdict(module._gp_config_for_design(benchmark.num_macros)),
                    environment=variant_environment(variant))
    metadata['source_sha256'] = {
        name: hashlib.sha256((Path(module.__file__).parent / name).read_bytes()).hexdigest()
        for name in ('placer.py', 'Xplace/src/calculator.py', 'Xplace/src/run_placement_nesterov.py')}
    (output / 'gp.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return placer._map_positions(positions, benchmark)
