"""Run normal density fitting, a common label probe, then development diagnostics."""
import argparse,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--run',required=True);p.add_argument('--count',default='512');a=p.parse_args();code=Path(__file__).parent
for script,extra in [('score.py',['fit']),('probe.py',[]),('diagnostics.py',[])]:
    cmd=[sys.executable,'-u',str(code/script),*extra,'--raw',a.raw,'--run',a.run]
    if script=='diagnostics.py':cmd+=['--count',a.count]
    subprocess.run(cmd,check=True)
