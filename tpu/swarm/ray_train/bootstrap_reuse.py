"""Validate explicit reuse of completed seeds across operational-only migrations."""
import hashlib
import json

from .config import Config


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _assigned_kv_layout(old, new, *, shared=False):
    # TP8 injects these overrides in commands.maxtext_kwargs. A resumed TP1/TP4
    # trainer must state them explicitly to retain its checkpoint tensor shapes.
    kwargs = new['trainer']['maxtext_kwargs']
    if (new.get('assigned_host_ips') and new['model'] == 'Qwen/Qwen3.8-27B'
            and new['trainer']['tp'] in (1, 4)
            and old['trainer']['logical_kv_heads'] == new['trainer']['logical_kv_heads'] == 8
            and (shared or old['trainer']['tp'] == 8)
            and kwargs.get('base_num_kv_heads') == 8 and kwargs.get('override_model_config') is True):
        for settings in (old, new):
            values = settings['trainer']['maxtext_kwargs']
            if values.get('base_num_kv_heads') == 8 and values.get('override_model_config') is True:
                values.pop('base_num_kv_heads')
                values.pop('override_model_config')


def validate_assigned_migration(original, target):
    """Same-run Qwen3.8 migration to assigned v5p hosts, scientific recipe fixed.

    Used for both pinned completed pools and unfinished draft journals. Checkpoint
    and journal bytes are separately checked by the deployment handoff.
    """
    original.validate()
    target.validate()
    if not (original.run_id == target.run_id and target.assigned_host_ips
            and original.model == target.model == 'Qwen/Qwen3.8-27B'
            and (original.accelerator == 'tpu-v6e-8'
                 or (original.accelerator == 'tpu-v5p-64' and original.assigned_host_ips))
            and target.accelerator == 'tpu-v5p-64'
            and original.hosts in (1, 2) and target.hosts in (1, 2)
            and original.inference.remote_only and target.inference.remote_only
            and target.checkpoint_resume):
        raise RuntimeError('unsupported assigned-host bootstrap migration')
    old, new = original.to_dict(), target.to_dict()
    _assigned_kv_layout(old, new)
    for settings in (old, new):
        bucket = settings['bucket'].rstrip('/')
        settings['base_bundle'] = settings['base_bundle'].removeprefix(bucket + '/')
        for key in ('hf', 'orbax'):
            settings['cache'][key] = settings['cache'][key].removeprefix(bucket + '/')
        for key in ('accelerator', 'hosts', 'zone', 'bucket', 'assigned_host_ips', 'physical_node',
                    'resume_min_checkpoint_step', 'bootstrap_reuse_contract_sha256',
                    'bootstrap_reuse_pool_sha256'):
            settings.pop(key, None)
        for key in ('trainer_compile', 'inference_compile', 'trainer_compile_seed',
                    'inference_compile_seed', 'trainer_gib', 'inference_gib', 'reserve_gib'):
            settings['cache'].pop(key, None)
        for key in ('hosts', 'tp', 'fsdp', 'process_bounds', 'chip_bounds', 'request_timeout'):
            settings['trainer'].pop(key, None)
        settings['inference'] = {k: v for k, v in settings['inference'].items()
                                 if not k.startswith('external_pool_') and k != 'request_timeout'}
    if old['client_env'].get('NUM_EPOCHS') != new['client_env'].get('NUM_EPOCHS'):
        if not (target.resume_min_checkpoint_step > 0
                and int(new['client_env']['NUM_EPOCHS']) > int(old['client_env']['NUM_EPOCHS'])
                and int(new['client_env']['NUM_EPOCHS']) > target.resume_min_checkpoint_step):
            raise RuntimeError('assigned-host extension requires a saved checkpoint')
        old['client_env']['NUM_EPOCHS'] = new['client_env']['NUM_EPOCHS']
    if old != new:
        raise RuntimeError('assigned-host migration changes scientific settings')


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
    if config.assigned_host_ips and not config.bootstrap_reuse_source_run_id:
        validate_assigned_migration(Config.from_dict(contract['config']), config)
        return
    old, new = Config.from_dict(contract['config']).to_dict(), config.to_dict()
    _assigned_kv_layout(old, new, shared=bool(config.bootstrap_reuse_source_run_id))
    # A validated host partition changes placement, not the frozen seed recipe.
    for settings in (old, new):
        settings.pop('assigned_host_ips', None)
        settings.pop('physical_node', None)
    if config.bootstrap_reuse_source_run_id:
        # Extending an existing training run does not change its frozen seeds.
        # Require a checkpoint floor so this cannot silently change a fresh arm.
        old_cap = old['client_env'].get('NUM_EPOCHS')
        new_cap = new['client_env'].get('NUM_EPOCHS')
        if old_cap != new_cap:
            try:
                extension = (config.checkpoint_resume
                             and config.resume_min_checkpoint_step > 0
                             and int(new_cap) > int(old_cap)
                             and int(new_cap) > config.resume_min_checkpoint_step)
            except (TypeError, ValueError):
                extension = False
            if not extension:
                raise RuntimeError('shared bootstrap step-cap change requires a checkpointed continuation')
            old['client_env']['NUM_EPOCHS'] = new_cap
        if old['run_id'] != config.bootstrap_reuse_source_run_id:
            raise RuntimeError('shared bootstrap source run mismatch')
        if (summary.get('drafted') != config.bootstrap_max_drafts
                or summary.get('total') != config.bootstrap_max_drafts
                or type(summary.get('retained')) is not int
                or not 1 <= summary['retained'] <= config.bootstrap_target_valid
                or (config.bootstrap_require_full_pool and summary['retained'] != config.bootstrap_target_valid)
                or not config.bootstrap_fixed_budget):
            raise RuntimeError('shared bootstrap must contain the complete fixed budget and retained pool')
        # The source generated frozen-base seeds. Training gets a new identity,
        # fresh LoRA/optimizer and possibly a different hardware/region layout.
        # Preserve every other recipe, grading and sampling setting.
        for settings in (old, new):
            bucket = settings['bucket'].rstrip('/')
            settings['base_bundle'] = settings['base_bundle'].removeprefix(bucket + '/')
            settings['cache'] = {key: settings['cache'][key].removeprefix(bucket + '/')
                                 for key in ('hf', 'orbax')}
            for key in ('run_id', 'root', 'accelerator', 'hosts', 'zone', 'bucket',
                        'bootstrap_only', 'bootstrap_all_hosts', 'bootstrap_max_groups'):
                settings.pop(key)
            settings['client_env'].pop('TTD_SICK_MARKER', None)
            settings['client_env'].pop('TTD_M0_LORA_SEED', None)
            for key in ('hosts', 'tp', 'fsdp', 'process_bounds', 'chip_bounds', 'token_budget', 'request_timeout'):
                settings['trainer'].pop(key, None)
            # The opt-in diagnostic replays identical examples without an
            # optimizer update, then restores the original accumulation state.
            for key in ('TUNIX_PACKING_PROBE_REPORT', 'TUNIX_PACKING_PROBE_BASE_BUDGET'):
                settings['trainer_env'].pop(key, None)
            for key in ('remote_only', 'memory_utilization', 'request_timeout'):
                settings['inference'].pop(key, None)
        # Shared-pool comparisons below retain all non-operational fields.
        for settings in (old, new):
            settings.pop('bootstrap_reuse_source_run_id', None)
            settings.pop('resume_min_checkpoint_step', None)
            settings.pop('bootstrap_reuse_contract_sha256')
            settings.pop('bootstrap_reuse_pool_sha256')
            settings['inference'] = {k:v for k,v in settings['inference'].items()
                                     if not k.startswith('external_pool_')}
        # A capped pool changes only whether a completed journal may proceed.
        # It does not change the frozen sampling budget, grading or ranking.
        if not new['bootstrap_require_full_pool']:
            old['bootstrap_require_full_pool'] = False
        # Host size changes placement concurrency, not an individual case's
        # CPU, memory, time budget, helper, or scoring contract.
        if old['science_placement_runtime'] == new['science_placement_runtime'] == 'cpu300-4g-v1':
            for settings in (old, new):
                settings.pop('science_placement_slots_per_host')
                placement = settings['grading']['families'].get('placement', {})
                placement.pop('slots_per_host', None)
        if old != new:
            raise RuntimeError('shared bootstrap changes scientific settings')
        return
    for settings in (old, new):
        settings.pop('bootstrap_reuse_source_run_id', None)
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
    # Reviewed v4-64 -> v6e-32 placement continuation: preserve the run and
    # scientific settings while relocating durable storage and physical topology.
    if (old['science_placement_runtime'] == new['science_placement_runtime'] == 'cpu300-4g-v1'
            and (old['accelerator'], new['accelerator']) == ('tpu-v4-64', 'tpu-v6e-32')
            and (old['zone'], new['zone']) == ('us-central2-b', 'us-central1-b')
            and (old['bucket'], new['bucket']) == (
                'gs://sk7524-tinker-tpu-us-central2', 'gs://sk7524-tinker-tpu-us-central1')
            and (old['trainer']['process_bounds'], new['trainer']['process_bounds']) == ('1,1,4', '2,2,1')):
        for settings in (old, new):
            for key in ('accelerator', 'zone', 'bucket'):
                settings.pop(key)
            settings['trainer'].pop('process_bounds')
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
