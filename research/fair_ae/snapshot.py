"""Capture a coherent best checkpoint while a training process continues."""
import argparse,hashlib,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('dest');a=p.parse_args()
s=Path(a.source);d=Path(a.dest)
for attempt in range(5):
 before=(s/'best.json').read_bytes();stat=(s/'best.msgpack').stat();model=(s/'best.msgpack').read_bytes();after=(s/'best.json').read_bytes();stat2=(s/'best.msgpack').stat()
 if before==after and stat.st_mtime_ns==stat2.st_mtime_ns and len(model)==stat.st_size==stat2.st_size and (s/'best.json').stat().st_mtime_ns>=stat2.st_mtime_ns:break
 time.sleep(.1)
else:raise RuntimeError('Checkpoint changed repeatedly; try again at next interval')
d.mkdir(exist_ok=False)
for n in ('config.json','normalization.json','source.py'):(d/n).write_bytes((s/n).read_bytes())
(d/'best.msgpack').write_bytes(model);(d/'best.json').write_bytes(before)
(d/'snapshot.json').write_text(json.dumps(dict(source=str(s),step=json.loads(before)['step'],sha256=hashlib.sha256(model).hexdigest(),captured_at=time.time()),indent=2))
print(d)
