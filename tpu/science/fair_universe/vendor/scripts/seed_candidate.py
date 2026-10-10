"""Make an editable one-file coding-agent work target without touching baselines."""
import argparse,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='generated_candidates/candidate.py')
    p.add_argument('--source',choices=('hybrid','ensemble'),default='hybrid')
    p.add_argument('--overwrite',action='store_true')
    a=p.parse_args()
    src=ROOT/'candidates'/('hybrid_physics_jax.py' if a.source=='hybrid'
                           else 'diverse_ensemble_jax.py')
    dest=Path(a.out)
    if dest.exists() and not a.overwrite:
        raise FileExistsError(f'{dest} exists; supply --overwrite explicitly')
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(src,dest)
    print(f'Seeded {dest} from {src}')

if __name__=='__main__':main()
