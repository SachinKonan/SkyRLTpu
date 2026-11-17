"""v2 token budget through the client's own _fixed_budget: prompt + thinking + answer <= 32,768 and the answer always
keeps 10,000 (thinking absorbs prompt growth). Uses the env values from launch_run.sh.
  python tests/test_budget_v2.py   (with the venv python; imports the deployed/working discover checkout)"""
import os, re, sys
DISC = os.environ.get('DISC', '/scratch/gpfs/ZHUANGL/sk7524/discover-tinker-inkling')
sys.path.insert(0, DISC)
launch = open(os.path.join(os.path.dirname(__file__), '..', 'launch_run.sh')).read()
for k in ('CONTEXT_WINDOW', 'TTD_FIXED_TOTAL_TOKENS', 'TTD_FIXED_THINK_TOKENS', 'TTD_FIXED_BUDGET_FIT'):
    os.environ[k] = re.search(rf'\b{k}=(\d+)', launch).group(1)
from ttt_discover.tinker_utils.completers import QwenTwoPhaseTokenCompleter as C

c = object.__new__(C)
c.context_window, c.context_buffer = int(os.environ['CONTEXT_WINDOW']), 50
THINK, TOTAL = int(os.environ['TTD_FIXED_THINK_TOKENS']), int(os.environ['TTD_FIXED_TOTAL_TOKENS'])
room0 = c.context_window - 50 - TOTAL          # prompt size up to which the full budget fits
for prompt in (703, 4000, 9000, room0, room0 + 3000):
    think, total = c._fixed_budget(prompt)
    answer = total - think
    print(f'prompt {prompt:6d}: thinking {think:6d} + answer {answer:6d} = {total:6d}; prompt+total = {prompt + total}')
    assert answer == TOTAL - THINK, (think, total)
    if prompt <= room0:
        assert (think, total) == (THINK, TOTAL), (think, total)
    else:
        assert prompt + total == c.context_window - 50 and think < THINK
print(f'BUDGET OK: thinking {THINK} + answer {TOTAL - THINK} for prompts up to {room0}; longer prompts shrink thinking only')
