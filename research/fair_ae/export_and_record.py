"""Use a component's exact frozen runtime and preserve its export receipt."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

p=argparse.ArgumentParser()
for name in ['code','run','raw','test','out']:p.add_argument('--'+name,required=True)
a=p.parse_args();run=Path(a.run);out=Path(a.out)
result=subprocess.run([sys.executable,'-u',str(Path(a.code)/'score.py'),'export','--run',a.run,'--raw',a.raw,'--test',a.test,'--out',a.out],text=True,capture_output=True)
if result.returncode:
    print(result.stdout);print(result.stderr,file=sys.stderr);raise SystemExit(result.returncode)
receipt=json.loads(result.stdout.strip().splitlines()[-1])
report=run/'density/report.json';fit=json.loads(report.read_text())
receipt.update(density_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),checkpoint_sha256=fit['checkpoint_sha256'],score_source_sha256=fit['score_source_sha256'],test_filename=Path(a.test).name)
out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt),flush=True)
