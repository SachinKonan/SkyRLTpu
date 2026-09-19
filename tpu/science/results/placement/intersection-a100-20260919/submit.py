"""Submit two array waves within gpu-test's 25-submission/3-running limits."""
import fcntl,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parents[5]
out=Path(__file__).resolve().parent
os.chdir(root)
lock=(out/'submit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(d):
 t=out/'submissions.tmp';t.write_text(json.dumps(d,indent=2));t.replace(out/'submissions.json')
def submit(indices):
 cmd=['sbatch','--parsable','--job-name=circuit-intersection','--array='+indices+'%3',
      '--output='+str(out/'logs/case-%A_%a.log'),str(out/'run.sh')]
 p=subprocess.run(cmd,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stderr.strip())
 return {'id':p.stdout.strip(),'indices':indices,'submitted_unix':time.time()}
assert (out/'build-ready').exists()
record={'pid':os.getpid(),'waves':[]};save(record)
record['waves'].append(submit('0-15'));save(record);print(record['waves'][-1],flush=True)
while True:
 p=subprocess.run(['squeue','-r','-h','-u','sk7524','-o','%q'],capture_output=True,text=True,check=True)
 if sum(q.strip()=='gpu-test' for q in p.stdout.splitlines())<=8:
  try:record['waves'].append(submit('16-31'));save(record);print(record['waves'][-1],flush=True);break
  except RuntimeError as e:record['last_submit_error']=str(e);save(record)
 time.sleep(60)
