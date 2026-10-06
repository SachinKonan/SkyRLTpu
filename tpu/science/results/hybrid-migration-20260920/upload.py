"""Upload prepared immutable bundles/seeds, verify downloaded SHA-256, no launches."""
import hashlib
import json
from pathlib import Path
from google.cloud import storage

HERE = Path(__file__).parent
ROOT = HERE.parents[3]


def main():
    manifest = json.loads((HERE / 'prepared.json').read_text())
    client = storage.Client(project='vision-mix')
    records = []
    for row in manifest['jobs']:
        objects = [(row['archive'], row['code_uri'], row['archive_sha256'])]
        if row['kind'] == 'training':
            objects.append((row['seed_file'], row['seed_destination'], row['seed_file_sha256']))
        for relative, uri, expected in objects:
            path = ROOT / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
            bucket, name = uri.removeprefix('gs://').split('/', 1)
            blob = client.bucket(bucket).blob(name)
            if not blob.exists():
                blob.upload_from_filename(str(path), if_generation_match=0, timeout=300)
            blob.reload()
            actual = hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation, timeout=300)).hexdigest()
            if actual != expected:
                raise RuntimeError('immutable upload mismatch: ' + uri)
            records.append(dict(uri=uri, sha256=expected, generation=blob.generation, bytes=blob.size))
            print(row['kind'], row['model'], 'verified', path.name, flush=True)
    (HERE / 'uploads.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
