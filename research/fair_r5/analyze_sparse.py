"""Bounded comparison of dense, sparse and angular latent scores."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scores import fit_moments,score

sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz,average,normalized,inspect,paired_bootstrap
from check_noise_hybrid import groups
from common import load_labels,write_json


def load_latents(root):
    result={k:read_npz(root/(k+'.npz'))['latent'] for k in ('val','calibration','audit')}
    for phase in ('development','r1_novel_now_reused','morphology_r2'):
        result.update({phase+'/'+k.removesuffix('__latent'):v for k,v in read_npz(root/(phase+'.npz')).items()
                       if k.endswith('__latent')})
    return result


def main(a):
    root,r4,out=map(Path,(a.root,a.r4,a.out));out.mkdir(parents=True,exist_ok=False)
    all_scores={};seed_reports={};latent_diagnostics={}
    for seed in range(3):
        run=root/f'latents-s{seed}';latents=load_latents(run);m=fit_moments(latents['val'])
        np.savez(out/f'moments-s{seed}.npz',**m)
        scored={phase:score(z,m) for phase,z in latents.items()}
        per_score={name:{phase:value[name] for phase,value in scored.items()} for name in scored['val']}
        for name,part in per_score.items():all_scores.setdefault(name,[]).append(part)
        seed_reports[str(seed)]={name:inspect(part) for name,part in per_score.items()}
        audit=(latents['audit'].astype(np.float64)-m['mean'])/m['std']
        normal=read_npz(run/'audit.npz');truth=normal['true_latent'].astype(np.float64)
        corr=np.corrcoef(audit,rowvar=False);off=corr[~np.eye(72,dtype=bool)]
        latent_diagnostics[str(seed)]=dict(
            validation_mean_norm=float(np.linalg.norm(m['mean'])),
            validation_std_range=[float(m['std'].min()),float(m['std'].max())],
            audit_coordinate_3sigma_fraction=float(np.mean(abs(audit)>3)),
            gaussian_reference_3sigma_fraction=.0026997960632601866,
            audit_offdiagonal_correlation_rms=float(np.sqrt(np.mean(off**2))),
            true_cosmology_audit_energy_mean=float(np.mean((truth**2).sum(1))))
    methods={name:average(parts) for name,parts in all_scores.items()}
    original=read_npz(r4/'comparison-001/continuous-summary.npz')
    errors={k:float(np.max(abs(v-methods['raw_energy'][k]))) for k,v in original.items()}
    assert all(np.allclose(v,methods['raw_energy'][k],rtol=3e-6,atol=2e-4) for k,v in original.items())
    base=read_npz(r4/'score-calibration-001/scaled-hybrid.npz');methods['08']=base
    base_n=normalized(base)
    for name in ('diagonal_max','diagonal_top4','diagonal_bj','angular_max'):
        sn=normalized(methods[name])
        methods['08+'+name+'-mean']={k:.5*(v+sn[k]) for k,v in base_n.items()}
        methods['08+'+name+'-max']={k:np.maximum(v,sn[k]) for k,v in base_n.items()}
    reports={name:inspect(part) for name,part in methods.items()};labels=load_labels(a.raw)
    bins={name:groups(part,labels) for name,part in methods.items()}
    for name,part in methods.items():np.savez(out/(name+'.npz'),**part)
    rows=[dict(probe=k,**{name:r['probes'].get(k) for name,r in reports.items()})
          for k in reports['raw_energy']['probes']]
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    differences={name:{k:paired_bootstrap(base,part,k,a.bootstrap) for k in base if '/' in k and not k.endswith(('/normal','/second'))}
                 for name,part in methods.items() if name.startswith('08+')}
    report=dict(kind='R5_REUSED_SYNTHETIC_LATENT_DIAGNOSTICS_NOT_OFFICIAL_OOD',public_test_read=False,
        fit_source='Only 3232 normal validation latents per seed; moments fitted at inferred cosmology.',
        latent_diagnostics=latent_diagnostics,energy_reproduction_errors=errors,
        results=reports,per_seed=seed_reports,known_parameter_false_positives=bins,
        paired_differences_from_08=differences,
        limitations=['Approximate learned Gaussianization and fitted physical coordinates invalidate naive independent-Gaussian p-value claims.',
                     'All probe families and clean audit maps are reused diagnostics.',
                     'Ensemble comparisons are exploratory; paired bootstrap does not correct selection bias.',
                     'Latent axes are architecture-dependent, and separate initialization seeds are not Bayesian posterior draws.'])
    write_json(out/'report.json',report)
    show=('development/spectral_low:0.2','development/spectral_high:0.2','development/quadratic_peak:0.2',
          'morphology_r2/phase_iaaft:1.0','morphology_r2/negative_peak:0.2','r1_novel_now_reused/spatial_modulation:0.5')
    print(json.dumps({name:dict(normal1pct=r['normal']['exceedances'][1],
                               probes={p:r['probes'].get(p) for p in show}) for name,r in reports.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','r4','raw','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--bootstrap',type=int,default=500);main(p.parse_args())
