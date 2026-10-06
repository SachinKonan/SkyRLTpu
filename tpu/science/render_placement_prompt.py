"""Render the tested CPU contract plus its complete seed implementation."""
from pathlib import Path

here = Path(__file__).resolve().parent
out = here.parents[1]/'.science/rendered/placement.txt'
out.parent.mkdir(parents=True, exist_ok=True)
prompt = (here/'PLACEMENT_PROMPT.md').read_text()
seed = (here/'challenge_seed.py').read_text()
out.write_text(prompt + '\n\nStarting implementation (replace with your improved algorithm):\n\n'
               + '```python\n' + seed + '```\n')
print(out)
