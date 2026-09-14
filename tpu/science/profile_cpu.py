"""Run trusted reference candidates with measured aggregate cgroup memory."""
import json
from pathlib import Path
from .portfolio import evaluate


def main():
    root=Path(__file__).resolve().parents[2]
    out=root/'.science/profiles';out.mkdir(parents=True,exist_ok=True)
    for seed in ('seed_portfolio','seed_portfolio_libraries'):
        for trial in range(2):
            work=out/f'{seed}-{trial}'
            if work.exists():raise RuntimeError(f'existing profile output {work}; do not overwrite evidence')
            result=evaluate((root/'tpu/science'/f'{seed}.py').read_text(),data=root/'.science/data/portfolio',work=work)
            print(json.dumps(dict(candidate=seed,trial=trial,**result)),flush=True)


if __name__=='__main__':main()
