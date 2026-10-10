"""(d) Is the normal tail of the standardized score homogeneous across parameters?"""
import numpy as np
from scipy import stats as st
import lib

zS, zC = lib.r1_z('Sd'), lib.r1_z('C')
lab = {'calib': np.repeat(lib.labels('calib'), 2, 0), 'audit': lib.labels('audit')}
cos = {}
for name in ['calib', 'audit']:
    idx = lib.load('stats', name, 'index')
    cos[name] = idx[:, 0] if name == 'audit' else np.repeat(idx[:, 0], 2)
names = ['Om', 'S8', 'T_AGN', 'f0', 'dz']
for tag, z in [('Sd', zS), ('C', zC), ('mean:Sd+C', {k: (zS[k] + zC[k]) / 2 for k in zS})]:
    print(f'===== {tag}')
    for setn in ['calib', 'audit', 'both']:
        if setn == 'both':
            zz = np.r_[z['calib'], z['audit']]; L = np.r_[lab['calib'], lab['audit']]
        else:
            zz = z[setn]; L = lab[setn]
        thr99, thr995 = np.quantile(zz, [.99, .995])
        print(f'-- {setn} n={len(zz)} q99={thr99:.2f} q995={thr995:.2f} q999={np.quantile(zz,.999):.2f}')
        for j, nm in enumerate(names):
            qs = np.quantile(L[:, j], [0, .2, .4, .6, .8, 1])
            b = np.clip(np.searchsorted(qs, L[:, j], side='right') - 1, 0, 4)
            r99 = [np.mean(zz[b == i] > thr99) * 100 for i in range(5)]
            r995 = [np.mean(zz[b == i] > thr995) * 100 for i in range(5)]
            sds = [zz[b == i].std() for i in range(5)]
            q90 = [np.quantile(zz[b == i], .9) for i in range(5)]
            # chi2 homogeneity of the 99% exceedances
            cnt = np.array([[np.sum(zz[b == i] > thr99), np.sum(zz[b == i] <= thr99)] for i in range(5)])
            p = st.chi2_contingency(cnt)[1]
            print(f'  {nm:6s} exc>q99 % ' + ' '.join(f'{v:4.1f}' for v in r99) + '  >q995 % ' +
                  ' '.join(f'{v:4.1f}' for v in r995) + f'  p={p:.3f}  sd ' + ' '.join(f'{v:.2f}' for v in sds) +
                  ' q90 ' + ' '.join(f'{v:.2f}' for v in q90))
