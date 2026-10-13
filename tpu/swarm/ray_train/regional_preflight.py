"""Read-only launch checks for regional storage; HTTP farm URLs are unrestricted."""
import json
import re


def storage_uris(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from storage_uris(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from storage_uris(item)
    elif isinstance(value, str):
        yield from re.findall(r"gs://[^\s\"'<>]+", value)


def check_paths(profile, location, *, zone=None, task=None):
    """Check actual bucket metadata, not just the bucket's region-looking name."""
    zone = zone or profile['zone']
    region = zone.rsplit('-', 1)[0]
    if location.lower() != region:
        raise ValueError(f'bucket location {location} does not match job region {region}')
    bucket = profile['bucket'].rstrip('/')
    if profile.get('single_bucket') is not True:
        raise ValueError('regional launch requires single_bucket=true')
    for uri in storage_uris([profile, task]):
        if uri != bucket and not uri.startswith(bucket + '/'):
            raise ValueError(f'cross-region or unapproved bucket URI: {uri}')
    return region


def check_qwen38(config, storage, *, task=None):
    """Validate regional Qwen3.8 assets immediately before submission.

    No copying and no model downloads: only metadata and completion markers.
    Inference-only farms do not need a converted trainer checkpoint.
    """
    config.validate()
    if config.model != 'Qwen/Qwen3.8-27B':
        raise ValueError('this asset preflight is scoped to Qwen3.8')
    profile = config.to_dict()
    bucket = storage.get_bucket(config.bucket.removeprefix('gs://').rstrip('/'))
    region = check_paths(profile, bucket.location, zone=config.effective_zone, task=task)
    revision = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'

    def key(uri):
        return uri.removeprefix(config.bucket.rstrip('/') + '/')

    def marker(uri):
        blob = bucket.get_blob(key(uri))
        if blob is None:
            raise ValueError(f'missing completion marker: {uri}')
        return json.loads(blob.download_as_bytes(if_generation_match=blob.generation))

    hf = marker(config.cache.hf.rstrip('/') + '/HF_CACHE_COMPLETE')
    if hf.get('model') != config.model or hf.get('revision') != revision:
        raise ValueError('HF completion marker names a different model or revision')
    blobs = list(storage.list_blobs(bucket, prefix=key(config.cache.hf.rstrip('/') + '/')))
    shards = [b for b in blobs if b.name.endswith('.safetensors')]
    if len(shards) != hf['shard_count'] or sum(b.size for b in shards) != hf['shard_bytes']:
        raise ValueError('HF shard inventory does not match its completion marker')
    checkpoint = None
    if not config.inference_only and not config.bootstrap_only:
        prefix = config.cache.orbax.rstrip('/') + '/' + config.trainer.maxtext_model + '/'
        checkpoint = marker(prefix + 'CHECKPOINT_COMPLETE')
        if checkpoint.get('hf_model') != config.model or checkpoint.get('hf_revision') != revision:
            raise ValueError('trainer completion marker names a different model or revision')
        arrays = list(storage.list_blobs(bucket, prefix=key(prefix)))
        if sum(b.size for b in arrays if not b.name.endswith('/CHECKPOINT_COMPLETE')) != checkpoint['checkpoint_bytes']:
            raise ValueError('trainer checkpoint inventory does not match its completion marker')
    if bucket.get_blob(key(config.base_bundle)) is None:
        raise ValueError('missing region-local base bundle')
    return dict(region=region, bucket=bucket.name, revision=revision,
                hf_shards=len(shards), hf_bytes=sum(b.size for b in shards),
                trainer_checkpoint_verified=checkpoint is not None)
