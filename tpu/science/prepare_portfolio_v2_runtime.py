"""Create CPU-only fit/replay artifacts with reviewed corporate-action flows.

Final-test files are deliberately not emitted by this discovery-worker builder.
The original verified dataframe is immutable; corrected return features/labels
and accounting factors are a separately hashed runtime derivative.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .portfolio_v2 import UNIVERSE, OBSERVATION_COLUMNS, FEATURES, fit_frame


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def write_frame(frame,path):
    text=frame.to_json(orient='split',date_format='iso',double_precision=15,index=False)
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:f.write(text.encode())


def reconstruct_quotes(prices,split_events):
    """Convert vendor adjusted prices/dividends to date-specific share units."""
    result={}
    for symbol,group in prices.groupby('symbol'):
        p=group.copy().set_index('session').sort_index()
        if p.index.duplicated().any():raise ValueError('duplicate source quotes: '+symbol)
        scale=np.ones(len(p))
        for event in split_events[split_events.symbol.eq(symbol)].itertuples():
            num,den=map(float,event.split_factor.split(':'))
            if not np.isfinite(num/den) or num/den<=0:raise ValueError('invalid vendor factor')
            scale[p.index<pd.Timestamp(event.report_date)]*=num/den
        p['unit_scale']=scale
        for col in ['open','close']:
            p[col]=pd.to_numeric(p[col],errors='raise').astype(float)*scale
        p['volume']=pd.to_numeric(p.volume,errors='raise').astype(float)
        result[symbol]=p
    return result


def accounting_panel(panel,quotes,actions,fee_rate=.001):
    days=pd.DatetimeIndex(sorted(panel.session.unique()))
    arrays={k:np.zeros((len(days),len(UNIVERSE))) for k in
            ['parent_overnight','cash_distribution','intraday','tradable',
             'right_units_per_dollar','forced_sale_per_dollar']}
    corrected=[];events=[]
    for j,asset in enumerate(UNIVERSE):
        vendor='RTX' if asset=='UTX' else asset
        rows=panel[panel.asset_id.eq(asset)].set_index('session').reindex(days).copy()
        q=quotes[vendor].reindex(days)
        active=q[['open','close']].notna().all(axis=1)&q.volume.gt(0)
        terminal=next((x for x in actions['terminal'] if x['asset']==asset),None)
        expected=days>=pd.Timestamp(actions.get('first_vendor_quote',{}).get(asset,str(days[0].date())))
        if terminal:
            end=pd.Timestamp(terminal['date'])
            expected &= days<end
            if active.loc[days>=end].any():raise ValueError('quote after reviewed termination: '+asset)
        if (~active.to_numpy()&expected).any():raise ValueError('unexplained missing price: '+asset)
        raw_open=q.open.where(active);raw_close=q.close.where(active)
        if (raw_open.dropna()<=0).any() or (raw_close.dropna()<=0).any():raise ValueError('bad quote')
        share_ratio=pd.Series(1.,index=days)
        for event in actions['splits']:
            if event['asset']==asset and pd.Timestamp(event['date']) in days:
                share_ratio.loc[event['date']]=event['shares']
        cash=rows.dividend.astype(float)*q.unit_scale
        cash=cash.fillna(0)*share_ratio
        forced=pd.Series(0.,index=days)
        rights=pd.Series(0.,index=days)
        for event in actions['spinoffs']:
            if event['asset']!=asset:continue
            day=pd.Timestamp(event['date']);proceeds=0.
            if day not in days:continue
            for child,ratio in event['children'].items():
                child_quote=quotes[child].loc[day]
                if not child_quote.volume>0 or not np.isfinite(child_quote.open):raise ValueError('missing spinoff open')
                proceeds += ratio*float(child_quote.open)
            cash.loc[day]+=proceeds*(1-fee_rate)
            forced.loc[day]=proceeds
            events.append(dict(asset=asset,date=str(day.date()),type='spinoff',gross_child_cash=proceeds))
        parent_open=raw_open*share_ratio;parent_close=raw_close*share_ratio
        if terminal and pd.Timestamp(terminal['date']) in days:
            day=pd.Timestamp(terminal['date'])
            cash.loc[day]=terminal['cash_per_share']+terminal['right_units_per_share']*terminal['right_value']
            rights.loc[day]=terminal['right_units_per_share']
            parent_open.loc[day]=0.;parent_close.loc[day]=0.
            events.append(dict(asset=asset,date=str(day.date()),type='terminal',cash=cash.loc[day],right_value=terminal['right_value']))
        previous=raw_close.shift(1)
        overnight=(parent_open/previous).fillna(0)
        distributions=(cash/previous).fillna(0)
        intraday=(raw_close/raw_open).fillna(1.)
        arrays['parent_overnight'][:,j]=overnight
        arrays['cash_distribution'][:,j]=distributions
        arrays['intraday'][:,j]=intraday
        arrays['tradable'][:,j]=active
        arrays['right_units_per_dollar'][:,j]=(rights/previous).fillna(0)
        arrays['forced_sale_per_dollar'][:,j]=(forced/previous).fillna(0)
        # Only today's resolved distributions and quotes contribute to inputs.
        daily=(parent_close+cash)/previous-1
        rows['return_1d']=daily
        for k in [5,20]:rows[f'return_{k}d']=(1+daily).rolling(k,min_periods=k).apply(np.prod,raw=True)-1
        rows['volatility_20d']=daily.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)
        rows['feature_ready']=rows[FEATURES[:9]].notna().all(axis=1)
        # Learning labels remain next-open to following-open returns, including
        # the reviewed liquidation. They are not the daily portfolio PnL series.
        rows['target_return']=(parent_open.shift(-2)+cash.shift(-2))/raw_open.shift(-1)-1
        rows['target_valid']=rows.target_return.notna()&active.shift(-1,fill_value=False)
        rows['label_released']=rows.target_valid&rows.split.eq(rows.target_split)
        rows.index.name='session';corrected.append(rows.reset_index())
    arrays['tradable']=arrays['tradable'].astype(bool)
    return pd.concat(corrected).sort_values(['session','asset_id']).reset_index(drop=True),days,arrays,events


def build(data,raw,output):
    data=Path(data);raw=Path(raw);output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    actions_path=Path(__file__).with_name('manifests')/'portfolio-v2-actions.json'
    actions=json.loads(actions_path.read_text())
    base=json.loads((data/'manifest.json').read_text())
    for name in ['joined.parquet']:
        if sha(data/name)!=base['files'][name]['sha256']:raise ValueError('base artifact hash mismatch')
    for name in ['stock_prices.parquet','stock_split_events.parquet']:
        record=next(x for x in base['sources'] if x['file']==name)
        if sha(raw/name)!=record['sha256']:raise ValueError('source artifact hash mismatch')
    panel=pd.read_parquet(data/'joined.parquet')
    symbols=list(UNIVERSE)+['RTX']+list({c for e in actions['spinoffs'] for c in e['children']})
    connection=duckdb.connect();connection.execute("SET threads=4; SET memory_limit='4GB'")
    prices=connection.execute('SELECT * FROM read_parquet(?) WHERE symbol IN (SELECT unnest(?)) AND report_date>=?',
        [str(raw/'stock_prices.parquet'),symbols,'2019-01-01']).df()
    prices['session']=pd.to_datetime(prices.report_date)
    splits=pd.read_parquet(raw/'stock_split_events.parquet')
    reviewed={(('RTX' if e['asset']=='UTX' else e['asset']),e['date']):e.get('vendor_factor',e.get('shares'))
              for e in actions['splits']+actions['spinoffs']}
    selected=splits[splits.symbol.isin(list(UNIVERSE)+['RTX']) & (splits.report_date>='2019-01-01')]
    for e in selected.itertuples():
        a,b=map(float,e.split_factor.split(':'))
        if not np.isclose(reviewed.pop((e.symbol,e.report_date),np.nan),a/b):raise ValueError('unreviewed vendor event')
    if reviewed:raise ValueError('reviewed event missing from source')
    quotes=reconstruct_quotes(prices,splits)
    panel,days,market,events=accounting_panel(panel,quotes,actions)
    for split in ['train','valid']:write_frame(fit_frame(panel,split),output/(split+'.json.gz'))
    # Evaluate every 2025 session, from preceding close to current close. The
    # first observation is 2024-12-31, which is permitted inner-validation data.
    index=np.flatnonzero((days>='2025-01-01')&(days<'2026-01-01'))
    if (index==0).any():raise ValueError('missing preceding close')
    observation_days=days[index-1]
    obs=panel[panel.session.isin(observation_days)][OBSERVATION_COLUMNS]
    write_frame(obs,output/'discovery_observations.json.gz')
    np.savez_compressed(output/'discovery_market.npz',**{k:v[index] for k,v in market.items()},
                        sessions=days[index].strftime('%Y-%m-%d').to_numpy(dtype='U10'))
    # Retain corrected research inputs for audit; explicitly exclude 2026.
    panel[panel.session<'2026-01-01'].to_parquet(output/'corrected_pre_final.parquet',index=False)
    manifest=dict(version='portfolio_v2_runtime_001',hardware='cpu',assets=list(UNIVERSE),fee_rate=.001,
        cash_return=0.,decision='previous NYSE close + 15min',execution='next NYSE open',
        valuation='daily close-to-close; observation holdings are valued at decision close',
        fit_label='next-open to following-open total return; split-crossing labels purged',
        actions=actions,action_checks=events,source_dataset_sha256=sha(data/'manifest.json'),
        actions_sha256=sha(actions_path),builder_sha256=sha(__file__),
        observation_columns=OBSERVATION_COLUMNS,evaluation_days=len(index),
        files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()},
        final_test_included=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),days=len(index),actions=events)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--raw',required=True);p.add_argument('--output',required=True)
    build(**vars(p.parse_args()))
