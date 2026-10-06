"""Causal TradeMaster DJ30 replay data; never trust upstream derived features."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from .resources import PORTFOLIO_LIBRARIES

FEATURES = ['close_return_1', 'close_return_5', 'close_return_20', 'open_to_close', 'high_to_close', 'low_to_close']
SPLITS = {'train': ('2012-01-01','2019-01-01'), 'valid': ('2019-01-01','2020-01-01'),
          'discovery': ('2020-01-01','2021-01-01'), 'final': ('2021-01-01','2100-01-01')}


def build(source, output):
    source, output = Path(source), Path(output)
    frame = pd.read_csv(source)
    if frame.duplicated(['date','tic']).any():
        raise ValueError('duplicate date/asset records')
    dates = sorted(frame.date.unique())
    assets = sorted(frame.tic.unique())
    arrays = {key: frame.pivot(index='date', columns='tic', values=key).reindex(index=dates,columns=assets).to_numpy(dtype=float)
              for key in ('open','high','low','close','adjcp')}
    if any(not np.isfinite(x).all() or (x <= 0).any() for x in arrays.values()):
        raise ValueError('requires a complete positive-price panel; no future-dependent imputation')
    close = arrays['adjcp']
    adjusted_open = arrays['open'] * arrays['adjcp'] / arrays['close']
    features = []
    # First 20 observations are burn-in; each later sample also needs 19 causal context rows.
    for t in range(20,len(dates)-2):
        features.append(np.stack([close[t]/close[t-k]-1 for k in (1,5,20)] +
            [arrays['close'][t]/arrays['open'][t]-1, arrays['high'][t]/arrays['close'][t]-1,
             arrays['low'][t]/arrays['close'][t]-1],axis=-1))
    features = np.asarray(features,dtype=np.float32)
    returns = (adjusted_open[22:]/adjusted_open[21:-1]-1).astype(np.float64)
    decision_dates = np.asarray(dates[20:-2])
    end_dates = np.asarray(dates[22:])
    output.mkdir(parents=True,exist_ok=True)
    manifest = dict(version='portfolio_v1', source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        feature_names=FEATURES, asset_names=assets, hardware='cpu', allowed_libraries=list(PORTFOLIO_LIBRARIES),
        execution_lag='features through close[t]; trade open[t+1]; mark holdings open[t+2]',
        price_convention='adjusted open = open * adjcp / close; fixed historical constituents',
        fee_rate=0.001, fee_rule='wealth fraction = 0.001 * sum(abs(target_assets - drifted_assets)); deducted before next return; no cash leg fee',
        cash_return=0., initial_weights='all cash', seeds=[1], sample_std_ddof=1,
        std_floor=1e-8, annualization=252, context_length=20, splits={})
    for name,(start,end) in SPLITS.items():
        # Purge labels crossing the split boundary, not just decision dates.
        idx = np.flatnonzero((decision_dates >= start)&(end_dates < end))
        idx = idx[idx >= 19]
        if len(idx)<2 or not np.all(np.diff(idx)==1):
            raise ValueError(f'insufficient/noncontiguous {name} split')
        data = dict(features=features[idx],returns=returns[idx],cash_returns=np.zeros(len(idx)),
                    context_features=features[idx[0]-19:idx[0]])
        dest = output/(name+'.npz')
        np.savez_compressed(dest,**data)
        manifest['splits'][name] = dict(first_decision=str(decision_dates[idx[0]]),
            last_decision=str(decision_dates[idx[-1]]), last_return_date=str(end_dates[idx[-1]]),
            observations=len(idx), sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args()
    print(json.dumps(build(a.source,a.output),indent=2))
