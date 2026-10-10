"""Create the normal-only reference/calibration cache for R6."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r2'))
from common import splits,write_json,sha


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    with np.load(Path(a.run)/'normalization.npz',allow_pickle=False) as f:
        count=f['count'];support=f['support'].astype(bool);noise=float(f['noise_sigma'])
    index=splits();clean=np.load(Path(a.clean)/'clean.npy',mmap_mode='r')
    old=np.load(Path(a.benchmark)/'observed.npy',mmap_mode='r')
    manifest=json.loads((Path(a.benchmark)/'manifest.json').read_text())
    assert manifest['groups']['calibration']==dict(offset=0,count=3232)
    values=np.lib.format.open_memmap(out/'observed.npy',mode='w+',dtype=np.float32,
                                   shape=(6464,*support.shape))
    noise_scale=noise/np.sqrt(np.maximum(count,1))
    for lo in range(0,3232,16):
        ids=index['val'][lo:lo+16]
        normal=np.stack([np.random.default_rng(978000+int(i)).normal(size=support.shape) for i in ids])
        values[lo:lo+len(ids)]=(clean[ids]+normal*noise_scale)*support
    values[3232:]=old[:3232];values.flush()
    assert np.array_equal(values[3232:],old[:3232])
    ids=np.r_[index['val'],index['calibration']];np.save(out/'source_ids.npy',ids)
    result=dict(kind='NORMAL_REFERENCE_AND_EXISTING_CALIBRATION_ONLY',public_test_read=False,
                groups=dict(val=dict(offset=0,count=3232),calibration=dict(offset=3232,count=3232)),
                shape=list(values.shape),observations_sha256=file_sha(out/'observed.npy'),
                source_ids_sha256=sha(out/'source_ids.npy'),source_sha256=sha(__file__),
                normal_validation_noise_seed='978000 + source ID',noise_sigma=noise,
                calibration_bit_identical=True)
    write_json(out/'manifest.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','clean','benchmark','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
