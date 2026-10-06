"""Download the pinned Qwen3.8 checkpoint once to a dedicated staging cache.

Run on a host with Hugging Face network access. Publishing regional mirrors is a separate step;
the JSON receipt is written only after every indexed tensor shard exists.
"""
import argparse
import json
from pathlib import Path

MODEL = 'Qwen/Qwen3.8-27B'
REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage-root', type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    hub = args.stage_root / 'hf' / 'hub'
    snapshot = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=str(hub),
        max_workers=4, allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model']))
    index = json.loads((snapshot / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if any(not (snapshot / name).is_file() for name in shards):
        raise RuntimeError('Incomplete checkpoint download')
    size = sum((snapshot / name).stat().st_size for name in shards)
    if size < int(index['metadata']['total_size']):
        raise RuntimeError('Checkpoint shards are smaller than indexed tensor data')
    repo = hub / 'models--Qwen--Qwen3.8-27B'
    (repo / 'refs').mkdir(exist_ok=True)
    (repo / 'refs/main').write_text(REVISION)
    receipt = dict(model=MODEL, revision=REVISION, snapshot=str(snapshot),
                   shard_count=len(shards), shard_bytes=size, tensor_count=len(index['weight_map']))
    path = args.stage_root / 'download-complete.json'
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(receipt, indent=2) + '\n')
    temporary.replace(path)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
