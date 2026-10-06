"""Download pinned public source files for trusted portfolio dataset preparation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

REPO = 'defeatbeta/yahoo-finance-data'
REVISION = '9ab94c890bf72f574bccd10fe89ee10a4227d26d'
FILES = ['stock_prices.parquet', 'stock_news.parquet',
         'stock_dividend_events.parquet', 'stock_split_events.parquet',
         'daily_treasury_yield.parquet']


def fetch(output, news_history=False):
    output.mkdir(parents=True, exist_ok=True)
    def one(name):
        path = output / name
        url = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/data/{name}'
        if not path.exists():
            temporary = path.with_suffix('.partial')
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                while block := response.read(4 * 1024**2):
                    stream.write(block)
            temporary.replace(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            while block := stream.read(4 * 1024**2):
                digest.update(block)
        row = dict(file=name, url=url, bytes=path.stat().st_size, sha256=digest.hexdigest())
        print(json.dumps(row), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(one, FILES))
    manifest = dict(repository=REPO, revision=REVISION, files=rows)
    (output / 'sources.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if news_history:
        extra=[]
        for name in ['nasdaq_exteral_data','All_external']:
            extra.append((f'fnspid-downloads/{name}.parquet',
                'https://huggingface.co/datasets/beachside1234/FNSPID/resolve/'
                f'749c61187fb353c62c91a36f775a909de8c1ecb2/Stock_news/{name}.parquet'))
        for author,dataset in [('rdolphin','financial-news-with-ticker-level-sentiment'),
                               ('frankossai','apple-stock-aapl-historical-financial-news-data')]:
            extra.append((f'kaggle/{author}.zip',
                f'https://www.kaggle.com/api/v1/datasets/download/{author}/{dataset}?datasetVersionNumber=1'))
        for name,url in extra:
            path=output/name;path.parent.mkdir(exist_ok=True)
            if not path.exists():
                temporary=path.with_suffix('.partial')
                with urllib.request.urlopen(url,timeout=120) as response,temporary.open('wb') as stream:
                    while block:=response.read(4*1024**2):stream.write(block)
                temporary.replace(path)
            if path.suffix=='.zip':
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        if not member.is_dir():
                            # Flatten archive paths; never extract traversal paths.
                            dest=path.parent/(path.stem+'-'+Path(member.filename).name)
                            with archive.open(member) as source,dest.open('wb') as target:
                                while block:=source.read(4*1024**2):target.write(block)
            print(json.dumps(dict(file=name,url=url,bytes=path.stat().st_size)),flush=True)
        (output/'additional_sources.json').write_text(json.dumps(extra,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--news-history',action='store_true')
    args=parser.parse_args()
    fetch(args.output,args.news_history)
