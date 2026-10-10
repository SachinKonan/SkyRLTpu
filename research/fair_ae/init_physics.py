"""Initialize the fixed physical-feature comparator for score.py fit."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
(out/'config.json').write_text(json.dumps({'architecture':'physics','definition':'72 fixed multiscale physical summaries'},indent=2))
(out/'normalization.json').write_text('{}')
(out/'best.msgpack').write_bytes(b'fixed-physics-summary-v1-no-neural-weights')
(out/'source.py').write_bytes((Path(__file__).parent/'features.py').read_bytes())
