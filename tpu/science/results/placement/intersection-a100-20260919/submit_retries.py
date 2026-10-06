"""Restore the eight interrupted/cancelled tasks within the QoS submit limit."""
import fcntl,json,subprocess,time,os
from pathlib import Path
out=Path(__file__).resolve().parent;os.chdir(out.parents[4])
lock=(out/'retry-submit.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
indices='2,4,5,6,8,10,12,14'
while True:
 r=subprocess.run(['squeue','-r','-h','-u','sk7524','-o','%q'],capture_output=True,text=True,check=True)
 if sum(q.strip()=='gpu-test' for q in r.stdout.splitlines())<=17:
  cmd=['sbatch','--parsable','--job-name=circuit-portable-retry','--array='+indices+'%3',
       '--output='+str(out/'logs/case-%A_%a.log'),str(out/'run.sh')]
  r=subprocess.run(cmd,capture_output=True,text=True)
  if r.returncode==0:
   (out/'retry-submission.json').write_text(json.dumps({'id':r.stdout.strip(),'indices':indices,'submitted_unix':time.time()},indent=2))
   print(r.stdout.strip(),flush=True);break
  print(r.stderr.strip(),flush=True)
 time.sleep(60)
