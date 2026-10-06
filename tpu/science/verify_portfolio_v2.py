"""Independent artifact checks, including full-data versus prefix reconstruction."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from .portfolio_v2 import (FEATURES,OBSERVATION_COLUMNS,market_features,join_news,
                          join_treasury,observation,fit_frame,audit)
from .build_portfolio_v2 import load_sources,sha256
import exchange_calendars as xc


def verify(raw,output):
    start=time.monotonic()
    manifest=json.loads((output/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        if sha256(output/name)!=record['sha256']:
            raise AssertionError('artifact checksum mismatch: '+name)
    panel=pd.read_parquet(output/'joined.parquet')
    panel['news_ids']=panel.news_ids.map(list)
    news=pd.read_parquet(output/'news.parquet')
    result=audit(panel,news)
    for split in ['train','valid']:
        saved=pd.read_parquet(output/'candidate'/f'{split}.parquet')
        expected=fit_frame(panel,split).reset_index(drop=True)
        # Arrow restores nested lists as arrays; compare list values explicitly.
        for frame in [saved,expected]:frame['news_ids']=frame.news_ids.map(list)
        assert_frame_equal(saved,expected,check_dtype=False)
        assert set(saved)==set(OBSERVATION_COLUMNS+['target_return'])
    for split in ['discovery','final']:
        saved=pd.read_parquet(output/'trusted'/f'{split}_observations.parquet')
        assert set(saved)==set(OBSERVATION_COLUMNS)
        assert not any(c.startswith(('target','label','execution')) for c in saved)
    prices,dividends,treasury,original_news,_=load_sources(raw)
    calendar=xc.get_calendar('XNYS',start='2019-01-01',end='2026-09-30')
    prefix_checks=[]
    # These checks compare inputs only, never use final outcomes to select code.
    for date in ['2020-12-31','2023-12-29','2024-12-31','2025-12-31']:
        day=pd.Timestamp(date)
        schedule=calendar.schedule.loc['2019-01-02':date]
        partial=market_features(prices[prices.session<=day],dividends[dividends.session<=day],schedule)
        partial=join_treasury(partial,treasury[treasury.report_date<=date])
        # Only the final 25 days need text materialization for the comparison;
        # price history was recomputed from the full prefix above.
        cut=partial.session.max()-pd.Timedelta(days=35)
        partial=partial[partial.session>=cut].copy()
        partial=join_news(partial,original_news[original_news.available_at<=partial.decision_at.max()])
        actual=partial[OBSERVATION_COLUMNS].reset_index(drop=True)
        expected=panel.loc[(panel.session>=cut)&(panel.session<=day),OBSERVATION_COLUMNS].reset_index(drop=True)
        for frame in [actual,expected]:frame['news_ids']=frame.news_ids.map(list)
        assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-12,atol=1e-12)
        prefix_checks.append(dict(cutoff=date,rows=len(actual)))
        print(json.dumps(dict(event='prefix_verified',**prefix_checks[-1])),flush=True)
    # Check exact calendar cutoffs on early-close and DST transition sessions.
    timing=[]
    for day in ['2020-03-06','2020-03-09','2024-11-29']:
        row=panel[(panel.session==pd.Timestamp(day))&(panel.asset_id=='AAPL')].iloc[0]
        expected=calendar.schedule.loc[day,'close']+pd.Timedelta(minutes=15)
        assert row.decision_at==expected
        timing.append(dict(session=day,decision_at=str(expected)))
    numeric=panel[FEATURES].to_numpy(dtype=float)
    assert not np.isinf(numeric).any()
    result.update(prefix_checks=prefix_checks,calendar_checks=timing,
                  exported_observation_allowlist_verified=True,artifact_hashes_verified=True,
                  no_infinite_numeric_features=True,seconds=time.monotonic()-start)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify(args.raw,args.output)
