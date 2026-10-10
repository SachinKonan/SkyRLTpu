"""Freeze normal-only scales; combine independently scored maps without test fitting."""
import argparse,datetime,hashlib,json,zipfile
from pathlib import Path
import numpy as np

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def transform(scores,manifest):
    center=np.array([c['normal_median'] for c in manifest['components']])
    scale=np.array([c['normal_q90_minus_median'] for c in manifest['components']])
    return ((np.asarray(scores)-center)/scale)@np.asarray(manifest['weights'])

def freeze(a):
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    components=[]
    for value in a.components:
        run=Path(value);path=run/'density/report.json';report=json.loads(path.read_text())
        cal=np.load(run/'density/calibration_scores.npy',allow_pickle=False)
        if cal.shape!=(3232,) or not np.isfinite(cal).all():raise ValueError('Unexpected calibration set')
        median,q90=np.quantile(cal,[.5,.9]);scale=float(q90-median)
        if scale<=0:raise ValueError('Degenerate normal calibration')
        components.append(dict(name=run.name,density_report_sha256=digest(path),calibration_sha256=digest(run/'density/calibration_scores.npy'),checkpoint_sha256=report['checkpoint_sha256'],normal_median=float(median),normal_q90_minus_median=scale))
    manifest=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),components=components,weights=[1/len(components)]*len(components),rule='Equal mean of (individual score - normal calibration median) / (normal calibration q90 - median)',selection='Fixed complementary physical, joint-summary, and reconstruction representations. No test values, labels, ranks, or distributions used. Development diagnostics examined before freezing; novel families remain held out.')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest),flush=True)

def combine(a):
    manifest=json.loads(Path(a.manifest).read_text());arrays=[];receipts=[]
    if len(a.zips)!=len(manifest['components']):raise ValueError('Component count mismatch')
    for value,c in zip(a.zips,manifest['components']):
        path=Path(value);receipt=json.loads(path.with_suffix('.receipt.json').read_text())
        if digest(path)!=receipt['sha256'] or c['density_report_sha256']!=receipt['density_report_sha256']:raise ValueError('Export does not match frozen component')
        if not receipt['independence_verified']:raise ValueError('Component independence was not verified')
        with zipfile.ZipFile(path) as z:
            if z.namelist()!=['result.json']:raise ValueError('Unexpected archive structure')
            values=np.asarray(json.loads(z.read('result.json'))['ood_scores'],dtype=float)
        if values.shape!=(10000,) or not np.isfinite(values).all():raise ValueError('Invalid component predictions')
        arrays.append(values);receipts.append(receipt)
    matrix=np.column_stack(arrays);scores=transform(matrix,manifest)
    # Fixed pointwise function: these checks cannot fit or modify it.
    if not np.allclose(scores[:1],transform(matrix[:1],manifest),rtol=1e-12,atol=1e-12):raise ValueError('Singleton mismatch')
    if not np.allclose(scores,transform(matrix[::-1],manifest)[::-1],rtol=1e-12,atol=1e-12):raise ValueError('Permutation mismatch')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('result.json',json.dumps({'ood_scores':scores.tolist()},allow_nan=False))
    receipt=dict(count=len(scores),sha256=digest(out),manifest_sha256=digest(a.manifest),independence_verified=True,component_receipts=receipts)
    out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)

def audit(a):
    from diagnostics import low_fpr_metric
    manifest=json.loads(Path(a.manifest).read_text());paths=[Path(p) for p in a.components]
    for p,c in zip(paths,manifest['components']):
        if digest(p/'density/report.json')!=c['density_report_sha256']:raise ValueError('Changed component')
    cal=transform(np.column_stack([np.load(p/'density/calibration_scores.npy') for p in paths]),manifest)
    normal=transform(np.column_stack([np.load(p/'density/audit_scores.npy') for p in paths]),manifest)
    report={'kind':'ensemble_normal_and_synthetic_NOT_OFFICIAL_OOD','normal_audit_count':len(normal),'normal_audit_exceedance_counts':[int((normal>t).sum()) for t in np.quantile(cal,[.95,.99,.999])]}
    for phase in ['development','novel']:
        blobs=[np.load(p/f'{phase}_diagnostic_scores.npz') for p in paths]
        report[phase]={key:low_fpr_metric(cal,transform(np.column_stack([b[key] for b in blobs]),manifest)) for key in blobs[0].files}
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freeze','combine','audit']);p.add_argument('--components',nargs='+');p.add_argument('--zips',nargs='+');p.add_argument('--manifest');p.add_argument('--out',required=True);a=p.parse_args()
    {'freeze':freeze,'combine':combine,'audit':audit}[a.mode](a)
