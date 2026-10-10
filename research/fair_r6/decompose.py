"""Post-result paired decomposition of the existing field likelihood."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r3'))
from compare import read_npz
from common import write_json,sha


def main(a):
    old,new,out=map(Path,(a.old,a.new,a.out));out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((old/'closure-cache/manifest.json').read_text());group=manifest['groups']
    ids=np.load(old/'closure-cache/source_ids.npy');n=len(ids);models=[]
    for seed in range(2):
        full={k:np.empty(n) for k in ('nll','base_energy','integrated_divergence')};seen=np.zeros(n,int)
        for shard in range(8):
            run=old/f'field-s{seed}-{shard}'
            record=json.loads((run/'complete.json').read_text())
            assert sha(run/'scores.npz')==record['scores_sha256']
            values=read_npz(run/'scores.npz');row=values.pop('indices')
            for k,v in values.items():full[k][row]=v
            seen[row]+=1
        assert np.all(seen==1);models.append(full)
    full={k:.5*(models[0][k]+models[1][k]) for k in models[0]}
    baseline=group['normal_float16'];ni=np.arange(baseline['offset'],baseline['offset']+baseline['count'])
    lookup={int(i):int(row) for i,row in zip(ids[ni],ni)};reports={}
    for name,g in group.items():
        if ':' not in name:continue
        rows=np.arange(g['offset'],g['offset']+g['count']);normal=np.array([lookup[int(i)] for i in ids[rows]])
        delta={k:v[rows]-v[normal] for k,v in full.items()}
        energy=.5*delta['base_energy'];volume=-delta['integrated_divergence']
        error=float(np.max(abs(delta['nll']-(energy+volume))))
        assert np.allclose(delta['nll'],energy+volume,rtol=1e-4,atol=.05)
        reports[name]=dict(count=len(rows),max_decomposition_error=error,
                          median_absolute_nll_change=float(np.median(abs(delta['nll']))),
                          median_absolute_energy_contribution=float(np.median(abs(energy))),
                          median_absolute_volume_contribution=float(np.median(abs(volume))),
                          spearman_nll_with_energy_change=float(spearmanr(delta['nll'],energy).statistic),
                          spearman_nll_with_volume_change=float(spearmanr(delta['nll'],volume).statistic))
    e64=read_npz(new/'mean-statistics.npz')['energy'][6464:]
    report=dict(kind='POST_RESULT_PAIRED_MECHANISM_DIAGNOSTIC',public_test_read=False,
                likelihood_resolution=32,new_terminal_statistic_resolution=64,
                paired_decomposition=reports,
                energy32_64=dict(spearman=float(spearmanr(full['base_energy'],e64).statistic),
                                  absolute_difference_quantiles=np.quantile(abs(full['base_energy']-e64),[.5,.9,.99,1]).tolist()),
                limitations=['This analysis was added after seeing R6 underperformance.',
                             'Likelihood shifts include ordinary input-complexity changes, so a large volume contribution alone is not evidence of useful OOD discrimination.',
                             'The algebraic decomposition uses paired, same-resolution 32-step scores; the separate 64-step comparison is numerical only.',
                             'Weak AD/spectrum/correlation tests do not establish that the full latent distributions are identical or that all latent-space methods fail.'])
    write_json(out/'report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('old','new','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
