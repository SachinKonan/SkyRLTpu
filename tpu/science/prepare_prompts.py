from pathlib import Path
import argparse,json
from .routing import START,END
from .resources import portfolio_prompt_resources


def build(root,portfolio_manifest,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    templates=Path(__file__).with_name('prompts')
    scaffold=(root/'rust/router_core/src/candidate.rs').read_text()
    seed=(root/'init_program.rs').read_text().split(START,1)[1].split(END,1)[0]
    fixed=scaffold.split(START,1)[0]+scaffold.split(END,1)[1]
    (output/'routing.txt').write_text((templates/'routing.txt').read_text()+'\nFixed Rust scaffold (read-only):\n```rust\n'+fixed+'\n```\nValid initial policy block:\n```rust\n'+seed+'\n```\n')
    manifest=json.loads(Path(portfolio_manifest).read_text())
    seed=(templates.parent/'seed_portfolio.py').read_text()
    (output/'portfolio.txt').write_text((templates/'portfolio.txt').read_text().replace('{resource_contract}',portfolio_prompt_resources())+'\nPinned manifest:\n'+json.dumps(manifest,indent=2)+
        '\nWorking equal-weight reference illustrating the exact interface:\n```python\n'+seed+'\n```\n'+
        'Improve the allocation algorithm under the same contract. Your final answer must contain one complete fenced Python module with fit and act, all imports and helpers. Close the Python code fence.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--simpletes-task',required=True);p.add_argument('--portfolio-manifest',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();build(a.simpletes_task,a.portfolio_manifest,a.output)
