"""Wait for native serving readiness and sample both CPU science tasks."""
import asyncio
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import httpx
from .sample import run


async def main(args):
    start=time.monotonic()
    async with httpx.AsyncClient(timeout=10) as http:
        while True:
            try:ready=(await http.get(args.base.rstrip('/')+'/health')).status_code==200
            except httpx.HTTPError:ready=False
            if ready:break
            if time.monotonic()-start>5400:raise TimeoutError('native serving did not become ready')
            print(json.dumps(dict(event='waiting_for_serving',seconds=time.monotonic()-start)),flush=True)
            await asyncio.sleep(30)
    await asyncio.gather(*[run(SimpleNamespace(profile=args.profile,tokenizer=args.tokenizer,
        prompt=str(Path(args.prompts)/(task+'.txt')),base=args.base,
        output=str(Path(args.output)/task),samples=args.samples)) for task in ('portfolio','routing')])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--prompts',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=4)
    asyncio.run(main(p.parse_args()))
