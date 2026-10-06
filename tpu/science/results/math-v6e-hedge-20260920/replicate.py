"""Copy only missing cache assets; generation-pin sources and verify CRC32C."""
import concurrent.futures
import datetime
import json
from pathlib import Path
from google.cloud import storage
HERE=Path(__file__).resolve().parent
client=storage.Client(project='vision-mix')
SRC='sk7524-tinker-tpu-us-central2'
DST='sk7524-tinker-tpu-us-central1'
items=[]
for model, slug in (('gemma','gemma4-31b'),('muse','muse-glimmer-30b')):
    hf='hf-cache-gemma4-v1' if model=='gemma' else 'hf-cache-muse-glimmer-v1'
    for prefix in (hf+'/', 'skyrl-maxtext-ckpts/'+slug+'/'):
        items += [(b,DST,b.name) for b in client.list_blobs(SRC,prefix=prefix)]
    for phase in ('trainer','inference'):
        prefix=f'science-q20-v6e-{model}-grpo-hedge-20260918-{phase}_compile-v1/' if phase=='trainer' else f'fresh-v6e-{model}-rglru-grpo-lr4e5-s1-20260919-fix1-inference_compile-v1/'
        blobs=list(client.list_blobs('sk7524-tinker-tpu-us-east5',prefix=prefix))
        items += [(b,DST,f'math-hedge-20260920/cache-seeds/{model}/{phase}/'+b.name[len(prefix):]) for b in blobs]

def copy(item):
    source,bucket,name=item
    dest=client.bucket(bucket).blob(name)
    exists=dest.exists()
    if exists:dest.reload()
    if exists and (source.size,source.crc32c)!=(dest.size,dest.crc32c):
        raise RuntimeError('Existing destination differs: '+dest.name)
    if not exists:
        token=None
        while True:
            token,_,_=dest.rewrite(source,token=token,if_generation_match=0,if_source_generation_match=int(source.generation),timeout=180)
            if token is None:break
        dest.reload()
    if (source.size,source.crc32c)!=(dest.size,dest.crc32c):raise RuntimeError('Copy checksum mismatch')
    return dict(source='gs://'+source.bucket.name+'/'+source.name,source_generation=source.generation,
                destination='gs://'+bucket+'/'+name,generation=dest.generation,bytes=source.size,crc32c=source.crc32c,copied=not exists)

results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
    for r in pool.map(copy,items):results.append(r)
(HERE/'cache-copies.json').write_text(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),objects=results),indent=2)+'\n')
print(json.dumps(dict(objects=len(results),copied=sum(r['copied'] for r in results),total_gib=round(sum(r['bytes'] for r in results)/2**30,2))),flush=True)
