"""Validate explicit reuse of completed seeds across operational-only migrations."""
import hashlib
import json

from .config import Config


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_reuse(config, document, summary):
    contract = document['contract']
    expected = config.bootstrap_reuse_contract_sha256
    if (identity(contract) != expected or document.get('sha256') != expected
            or summary.get('contract_sha256') != expected):
        raise RuntimeError('bootstrap reuse contract checksum mismatch')
    if summary.get('pool_sha256') != config.bootstrap_reuse_pool_sha256:
        raise RuntimeError('bootstrap reuse pool checksum mismatch')
    if summary.get('optimizer_steps') != 0:
        raise RuntimeError('bootstrap reuse requires a pre-optimizer seed pool')
    old, new = Config.from_dict(contract['config']).to_dict(), config.to_dict()
    # The pinned contract records how the seeds were generated. Only the reviewed
    # runtime migration controls and final training step cap may differ. Sampling,
    # grading, model, seed, optimizer settings, run identity and bootstrap limits
    # remain subject to exact comparison.
    # Placement concurrency changes admission only, never per-case budgets or scoring.
    if old['science_placement_runtime'] == new['science_placement_runtime'] == 'cpu300-4g-v1':
        old.pop('science_placement_slots_per_host')
        new.pop('science_placement_slots_per_host')
        old['cache'].pop('reserve_gib')
        new['cache'].pop('reserve_gib')
        # Cache capacity is operational; model revisions and grading remain pinned.
        for key in ('trainer_gib', 'inference_gib'):
            old['cache'].pop(key)
            new['cache'].pop(key)
    for settings in (old, new):
        settings.pop('resume_min_checkpoint_step', None)
        for key in ('bootstrap_reuse_contract_sha256', 'bootstrap_reuse_pool_sha256'):
            settings.pop(key)
        for key in ('trainer_compile', 'inference_compile'):
            settings['cache'].pop(key, None)
        settings['client_env'].pop('NUM_EPOCHS', None)
        settings['inference'] = {k: v for k, v in settings['inference'].items()
                                 if not k.startswith('external_pool_')}
    if old != new:
        raise RuntimeError('bootstrap reuse changes settings outside the allowed operational migration')
