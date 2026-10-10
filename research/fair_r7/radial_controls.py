"""Post-result control for the score transform used by joint-score ensembles.

Squaring changes a score's influence when averaging, despite preserving its
standalone ranking. Report Gaussian radii alongside the registered squared
distances; do not silently replace the original negative results.
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz,normalized,paired_bootstrap
from common import low_fpr_metric,write_json


def main(a):
    root,out=Path(a.root),Path(a.out);out.mkdir(parents=True,exist_ok=False)
    summary=normalized(read_npz(root/'summary_only.npz'))
    scores={'08':read_npz(root/'08.npz')}
    for ref in ('oof','in-sample'):
        for method in ('diagonal','gaussian_joint'):
            key=ref+'/'+method
            field=read_npz(root/('field__'+key.replace('/','__')+'.npz'))
            assert all(np.min(v)>=0 for v in field.values())
            radius={g:np.sqrt(v) for g,v in field.items()}
            rank_metric_error=max(abs(low_fpr_metric(field['calibration'],field[g])-
                                      low_fpr_metric(radius['calibration'],radius[g]))
                                  for g in field if ':' in g)
            assert rank_metric_error<1e-12
            radius=normalized(radius)
            scores['hybrid/'+key]={g:.5*(v+summary[g]) for g,v in radius.items()}
    results={k:dict(probes={g:low_fpr_metric(v['calibration'],vv) for g,vv in v.items() if ':' in g},
                    normal={g:[int((v[g]>t).sum()) for t in np.quantile(v['calibration'],[.95,.99,.999])]
                            for g in ('audit','normal_float32','normal_float16')}) for k,v in scores.items()}
    for k,v in scores.items():np.savez(out/(k.replace('/','__')+'.npz'),**v)
    with (out/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['probe']+list(scores));w.writeheader()
        w.writerows(dict(probe=g,**{k:v['probes'][g] for k,v in results.items()}) for g in results['08']['probes'])
    differences={k:{g:paired_bootstrap(scores['08'],v,g,a.bootstrap) for g in results[k]['probes'] if g.startswith('closure/')}
                 for k,v in scores.items() if k.startswith('hybrid/oof/')}
    joint={g:paired_bootstrap(scores['hybrid/oof/diagonal'],scores['hybrid/oof/gaussian_joint'],g,a.bootstrap)
           for g in results['08']['probes'] if g.startswith('closure/')}
    write_json(out/'report.json',dict(kind='POST_RESULT_SCORE_TRANSFORM_CONTROL_NOT_OFFICIAL_OOD',
        registered_before_results=False,public_test_read=False,neural_retraining=False,
        rationale='Use Gaussian radius rather than squared radius before averaging with the summary score. This preserves standalone rankings but controls one source of unequal tail weighting.',
        results=results,paired_differences_from08=differences,joint_minus_diagonal=joint,
        limitations=['Families and maps reused, and this control was motivated after inspecting the registered R7 result.',
                     'Equal median/q90 scaling still does not make complete marginal score distributions identical.',
                     'No Gaussian-reference score is an exact conditional p-value.']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True)
    p.add_argument('--bootstrap',type=int,default=500);main(p.parse_args())
