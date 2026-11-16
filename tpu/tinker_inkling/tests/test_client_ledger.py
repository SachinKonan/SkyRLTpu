"""Client ledger: a torn last line does not block resume, and the next append lands on its own line."""
import json, os, sys, tempfile, logging
sys.path.insert(0, '/scratch/gpfs/ZHUANGL/sk7524/discover-tinker-inkling')
logging.basicConfig(level=logging.WARNING)
from ttt_discover.tinker_utils import misc_utils as mu

d = tempfile.mkdtemp(dir=os.environ.get('TMPDIR'))
p = os.path.join(d, mu.CHECKPOINTS_BASE_NAME)
with open(p, 'w') as f:
    f.write(json.dumps(dict(name='000001', batch=1, state_path='tinker://a/1')) + '\n')
    f.write(json.dumps(dict(name='000002', batch=2, state_path='tinker://a/2')) + '\n')
    f.write('{"name": "000003", "batch": 3, "state_pa')            # reboot mid-append
last = mu.get_last_checkpoint(d)
assert last['batch'] == 2, last
mu.append_jsonl_row(p, dict(name='000003', batch=3, state_path='tinker://a/3'))
rows = mu.load_checkpoints_file(d)
assert [r['batch'] for r in rows] == [1, 2, 3], rows
assert mu.get_last_checkpoint(d)['state_path'] == 'tinker://a/3'
mu.append_jsonl_row(os.path.join(d, 'new.jsonl'), dict(a=1))
assert open(os.path.join(d, 'new.jsonl')).read() == '{"a": 1}\n'
print('LEDGER TEST PASSED: torn row skipped (resume from step 2), next row appended cleanly, new file fine')
