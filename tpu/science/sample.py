"""Native Qwen sampling pilot using exactly the launched grid's token convention."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import time
import httpx
from transformers import AutoTokenizer
from tpu.swarm.bench.realistic_bench import render_prompt
from tpu.swarm.ray_train.frozen_benchmark import payload,check_choice
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.arena_sampling import extract_code


async def run(args):
    config=Config.load(args.profile)
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    prompt=Path(args.prompt).read_text()
    renderer=config.client_member_spec.split(':')[1]
    rendered=render_prompt(tokenizer,renderer,prompt,'2026-09-13')
    request=payload(config,rendered,n=args.samples)
    record=dict(run_id=config.run_id,model=config.model,model_preset=config.model_preset,renderer=renderer,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),prompt_tokens=len(rendered.prompt),
        native_thinking_allowance=request['thinking_token_budget'],total_generation_allowance=request['max_tokens'],
        context_window=config.client_context_window,phase1_prompt_plus_thinking=config.client_phase1_max_tokens,
        temperature=request['temperature'],top_p=request['top_p'],top_k=request['top_k'],
        sampling_seed='engine default; no per-request seed, matching native grid',samples=args.samples)
    (output/'request-summary.json').write_text(json.dumps(record,indent=2))
    (output/'prompt.txt').write_text(prompt)
    async with httpx.AsyncClient(timeout=config.inference.request_timeout) as http:
        # Check enforcement on the deployed service before spending the full budget.
        warm=dict(request,n=2,max_tokens=64,thinking_token_budget=8)
        rsp=await http.post(args.base.rstrip('/')+'/v1/completions',json=warm)
        if rsp.is_error:raise RuntimeError(f'native preflight HTTP {rsp.status_code}: {rsp.text[:2000]}')
        data=rsp.json()
        if len(data.get('choices',[]))!=2:raise RuntimeError('incomplete native preflight')
        for choice in data['choices']:check_choice(choice,warm)
        (output/'preflight.json').write_text(json.dumps(data))
        print(json.dumps(dict(event='native_preflight_passed',task=Path(args.prompt).stem)),flush=True)
        start=time.monotonic()
        rsp=await http.post(args.base.rstrip('/')+'/v1/completions',json=request)
        if rsp.is_error:raise RuntimeError(f'generation HTTP {rsp.status_code}: {rsp.text[:2000]}')
        data=rsp.json()
        (output/'response.json').write_text(json.dumps(data))
        if len(data.get('choices',[]))!=args.samples:raise RuntimeError('incomplete sample group')
        rows=[]
        for index,choice in enumerate(data['choices']):
            check_choice(choice,request)
            text=choice.get('text','')
            # Require a completed answer and parse only the text after </think>.
            format_error=None
            try:
                if choice.get('finish_reason')=='length':
                    raise ValueError('truncated answer')
                code=extract_code(text,preset=config.model_preset)
            except ValueError as exc:
                code='';format_error=str(exc)
            (output/f'candidate-{index:03d}.txt').write_text(text)
            if code:(output/f'candidate-{index:03d}.py').write_text(code)
            rows.append(dict(index=index,has_code=bool(code),format_error=format_error,finish_reason=choice.get('finish_reason'),
                generated_tokens=len(choice['token_ids']),thinking_budget=choice['thinking_budget']))
        record.update(generation_seconds=time.monotonic()-start,candidates=rows)
        (output/'summary.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--profile',required=True);p.add_argument('--tokenizer',required=True)
    p.add_argument('--prompt',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
    p.add_argument('--samples',type=int,default=4)
    asyncio.run(run(p.parse_args()))
