"""Project historical headlines from pinned FNSPID Parquet without article bodies."""
import argparse
import json
from pathlib import Path
import duckdb

SYMBOLS = 'AAPL AXP BA CAT CSCO CVX DIS DOW GS HD IBM INTC JNJ JPM KO MCD MMM MRK MSFT NKE PFE PG TRV UNH UTX RTX V VZ WBA WMT XOM'.split()
REPO = 'beachside1234/FNSPID'
REVISION = '749c61187fb353c62c91a36f775a909de8c1ecb2'


def extract(output, local=None):
    output.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads=4; SET memory_limit='6GB'")
    if local is None:
        connection.execute('INSTALL httpfs; LOAD httpfs')
    for name in ['nasdaq_exteral_data', 'All_external']:
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/Stock_news/{name}.parquet'
        print(json.dumps(dict(event='extracting', url=url)), flush=True)
        path = str(local / (name + '.parquet')) if local else url
        frame = connection.execute('''SELECT Date AS report_date, Article_title AS title,
            Stock_symbol AS symbol, Url AS link, Publisher AS publisher
            FROM read_parquet(?) WHERE Stock_symbol IN (SELECT unnest(?))
            AND substr(Date,1,10)>='2019-01-01' AND substr(Date,1,10)<'2026-01-01'
            ''', [path, SYMBOLS]).fetch_arrow_table()
        import pyarrow.parquet as pq
        pq.write_table(frame, output / (name + '.parquet'), compression='zstd')
        print(json.dumps(dict(event='extracted', source=name, rows=frame.num_rows)), flush=True)
    (output/'source.json').write_text(json.dumps(dict(repository=REPO, revision=REVISION,
        columns=['Date','Article_title','Stock_symbol','Url','Publisher'], symbols=SYMBOLS),indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--local', type=Path)
    args = parser.parse_args()
    extract(args.output, args.local)
