"""Read-only paired-trainer timing recorder. STOP via latency/STOP; never mutates TPUs."""
import argparse, collections, fcntl, json, os, time
from concurrent.futures import ThreadPoolExecutor
from od_deploy import OUT, ssh
DEST=OUT/'latency'; DEST.mkdir(exist_ok=True)
REMOTE=r'''
import collections,datetime,json,re,shutil,sqlite3,subprocess,time,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
r=ROW; cursors=CURSORS; root=Path.home()/'.cache'/r['run_id']/'runs'/r['run_id']
out={'time':time.time(),'problem':r['problem'],'run_id':r['run_id'],'hosts':r['physical_ranks'],'errors':[],'cursors':{}}
def get(url):
 with urllib.request.urlopen(url,timeout=6) as h:return json.load(h)
def read(name,cap=6000000):
 p=root/name
 if not p.exists():return ''
 st=p.stat(); old=cursors.get(name,{})
 start=old.get('offset',0) if old.get('inode')==st.st_ino else 0
 if start>st.st_size:start=0
 truncated=st.st_size-start>cap
 if truncated:start=st.st_size-cap
 with p.open('rb') as h:
  h.seek(start); b=h.read(cap);end=b.rfind(b'\n')+1
 out['cursors'][name]={'inode':st.st_ino,'offset':start+end}
 if truncated:out['errors'].append(name+': initial/backlog tail truncated')
 if start and truncated:b=b[b.find(b'\n')+1:end]
 else:b=b[:end]
 return b.decode(errors='replace')
try:
 c=sqlite3.connect('file:'+str(root/'tinker.db')+'?mode=ro',uri=True,timeout=2);c.row_factory=sqlite3.Row
 model=c.execute("select model_id,created_at from futures where request_type='CREATE_MODEL' order by request_id desc limit 1").fetchone()
 out['attempt']=dict(model)
 out['futures']=[dict(x) for x in c.execute('select request_id,request_type,status,created_at,completed_at from futures where model_id=?',(model['model_id'],))]
 c.close()
except Exception as e:out['errors'].append('db: '+str(e))
try:
 out['unit']=subprocess.check_output(['systemctl','show',r['unit'],'-p','ActiveState','-p','MainPID','-p','NRestarts'],text=True)
 out['free_gib']=round(shutil.disk_usage(Path.home()).free/2**30,2)
 cp=root/'client/tinker_log'/r['run_id']/'member_qwen/checkpoints.jsonl'
 if cp.exists():out['checkpoint']=json.loads(cp.read_text().splitlines()[-1])
except Exception as e:out['errors'].append('host: '+str(e))
allowed={'event','time','request_id','service','phase','engines','seconds','pool','task','attempt','admission_wait_seconds','route','n','output_tokens','load_seconds','queued','waited_seconds','retries','reason','pid','version'}
out['events']=[]
for line in read('inference-events.jsonl').splitlines():
 try:
  e=json.loads(line)
  if e.get('event')!='borrow_reconcile':out['events'].append({k:v for k,v in e.items() if k in allowed})
 except ValueError:pass
# Wrapped log messages: reconstruct records, keep timing/progress/compilation only.
s=read('trainer.log',12000000)
records=re.split(r'(?m)(?=^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - )',s)
out['training_logs']=[]
for rec in records:
 if any(k in rec for k in ['(timing)','[microbatch','elapsed ', 'Finished XLA compilation','Compiling jit(', 'ERROR -']):
  line=' '.join(rec.split())
  if re.match(r'\d{4}-',line):out['training_logs'].append(line[:3500])
mp='client/tinker_log/'+r['run_id']+'/metrics.jsonl'
out['step_metrics']=[]
for line in read(mp).splitlines():
 try:
  e=json.loads(line);out['step_metrics'].append({k:v for k,v in e.items() if any(x in k.lower() for x in ('time','step','batch','token','pipelined'))})
 except ValueError:pass
try:
 s=get('http://127.0.0.1:24800/status');b=s.get('borrowing') or {}
 out['borrower']={k:b.get(k) for k in ('held','engines','active','ready','target')}
 out['scheduler']=s.get('scheduling');out['ingress_active']=s.get('active')
 urls=b.get('held',[])
 def farm(url):
  d={'url':url,'time':time.time()}
  try:
   s=get(url+'/status');d.update({k:s.get(k) for k in ('owner_run','state','active','instance','ready_engines','grading')});d['engines']=[]
   for replica in s.get('replicas',[]):
    ip=replica['ip'];addr=ip if ':' in ip else ip+':24801';e={'address':addr,'instance':replica.get('instance'),'time':time.time()}
    try:
     raw=urllib.request.urlopen('http://'+addr+'/metrics',timeout=6).read().decode();m={}
     for line in raw.splitlines():
      if not line.startswith('vllm:') or '_bucket{' in line or '_created{' in line:continue
      name=line.split('{',1)[0].split()[0]
      if not any(k in name for k in ('tokens_total','request_success_total','num_requests','_seconds_sum','_seconds_count','cache_usage')):continue
      key,val=line.rsplit(' ',1);m[key]=float(val)
     e['metrics']=m
    except Exception as exc:e['error']=str(exc)
    d['engines'].append(e)
  except Exception as exc:d['error']=str(exc)
  return d
 with ThreadPoolExecutor(max_workers=4) as ex:out['farms']=list(ex.map(farm,urls))
except Exception as e:out['errors'].append('status: '+str(e))
print(json.dumps(out))
'''
def epoch(s):
 from datetime import datetime,timezone
 return datetime.fromisoformat(s).replace(tzinfo=timezone.utc).timestamp()
def aggregate(metrics,name):return sum(v for k,v in metrics.items() if k.split('{',1)[0]==name)
def summarize(d,old,events,logs):
 cutoff=epoch(d['attempt']['created_at']);now=d['time'];ev=[x for x in events if x.get('time',0)>=cutoff]
 counts=collections.Counter(x.get('event') for x in ev)
 fut=d.get('futures',[]);pending=[x for x in fut if x['status'] not in ('COMPLETED','FAILED')]
 bytype=collections.Counter(x['request_type']+':'+x['status'] for x in fut)
 group=[x for x in ev if x.get('event')=='hybrid_generated'];grades=[x for x in ev if x.get('event')=='grading_completed']
 timings=[]
 import re
 for line in logs:
  try:t=epoch(line[:23])
  except ValueError:continue
  m=re.search(r'\(timing\) (.+?) took ([\d.]+)s',line)
  if m and t>=cutoff:timings.append({'end':t,'start':t-float(m[2]),'operation':m[1],'seconds':float(m[2])})
 oldeng={e['address']:(f,e) for f in (old or {}).get('farms',[]) for e in f.get('engines',[])}
 farmrows=[]
 for f in d.get('farms',[]):
  for e in f.get('engines',[]):
   m=e.get('metrics',{});p=oldeng.get(e['address']);a={'farm':f['url'],'address':e['address'],'owner_run':f.get('owner_run'),'running':aggregate(m,'vllm:num_requests_running'),'waiting':aggregate(m,'vllm:num_requests_waiting'),'error':e.get('error'),'tokens_per_second':None}
   if p and p[1].get('instance')==e.get('instance') and p[0].get('owner_run')==f.get('owner_run') and m and p[1].get('metrics'):
    pm=p[1]['metrics'];dt=e['time']-p[1]['time'];delta=aggregate(m,'vllm:generation_tokens_total')-aggregate(pm,'vllm:generation_tokens_total')
    if dt>0 and delta>=0:a.update(interval_seconds=round(dt,1),tokens_per_second=round(delta/dt,1),completed_sequences=aggregate(m,'vllm:request_success_total')-aggregate(pm,'vllm:request_success_total'))
    a['completed_observation_means_seconds']={}
    for name in ('time_to_first_token','inter_token_latency','e2e_request_latency','request_queue_time','request_prefill_time','request_decode_time'):
     n=aggregate(m,'vllm:'+name+'_seconds_count')-aggregate(pm,'vllm:'+name+'_seconds_count');v=aggregate(m,'vllm:'+name+'_seconds_sum')-aggregate(pm,'vllm:'+name+'_seconds_sum')
     if n>0 and v>=0:a['completed_observation_means_seconds'][name]={'n':n,'mean':round(v/n,4)}
   farmrows.append(a)
 first=min((epoch(x['created_at']) for x in fut if x['request_type']=='EXTERNAL'),default=None)
 gradepools={}
 for g in grades:gradepools.setdefault(g.get('pool','unknown'),[]).append(g['seconds'])
 def stats(vals):
  vals=sorted(vals)
  return None if not vals else {'n':len(vals),'mean_seconds':round(sum(vals)/len(vals),2),'p50_seconds':vals[len(vals)//2],'p95_seconds':vals[min(len(vals)-1,int(len(vals)*.95))]}
 result={'time':now,'attempt':d['attempt'],'saved_checkpoint':d.get('checkpoint'),'unit':d.get('unit'),'free_gib':d.get('free_gib'),'scheduler':d.get('scheduler'),'farms':farmrows,'future_counts':dict(bytype),'event_counts':dict(counts),'first_sampling_time':first,'sampling_age_seconds':None if first is None else round(now-first,1),'completed_group_latency':stats([x['seconds'] for x in group if 'seconds' in x]),'grading_per_pool':{k:stats(v) for k,v in gradepools.items()},'trainer_operations':timings,'oldest_pending_seconds':max((now-epoch(x['created_at']) for x in pending),default=0),'errors':d.get('errors')}
 # Intervals use backend wall timings, not sum of the two ranks. Endpoints are
 # actual operation completions; unfinished operations remain pending/censored.
 fb=[x for x in timings if 'forward_backward' in x['operation']]
 result['backward_wall_seconds']=sum(x['seconds'] for x in fb)
 grade_open={}
 for e in ev:
  rid=e.get('request_id')
  if e.get('event')=='grading_dispatched':grade_open[rid]=e
  elif e.get('event') in ('grading_completed','grading_infra_failure','grading_retry'):grade_open.pop(rid,None)
 result['grading_outstanding']={}
 for e in grade_open.values():
  entry=result['grading_outstanding'].setdefault(e.get('pool','unknown'),{'requests':0,'oldest_seconds':0})
  entry['requests']+=1;entry['oldest_seconds']=max(entry['oldest_seconds'],round(now-e['time'],1))
 result['pending_backward_requests']=sum(x['request_type']=='FORWARD_BACKWARD' for x in pending)
 result['recent_training_tiles']=[l for l in logs if 'fb tile ' in l or 'fwd tile ' in l][-4:]
 fbrequests=[x for x in fut if x['request_type']=='FORWARD_BACKWARD']
 result['first_backward_enqueue_seconds_after_sampling']=None if not first or not fbrequests else min(epoch(x['created_at']) for x in fbrequests)-first
 if group or grades:
  last_data=max(x['time'] for x in group+grades)
  result['backward_overlapping_observed_sampling_grading_seconds']=sum(max(0,min(x['end'],last_data)-max(x['start'],first or cutoff)) for x in fb)
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--once',action='store_true');ap.add_argument('--interval',type=int,default=60);a=ap.parse_args()
 lock=(DEST/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (DEST/'monitor.pid').write_text(str(os.getpid()))
 statefile=DEST/'state.json';state=json.loads(statefile.read_text()) if statefile.exists() else {}
 while not (DEST/'STOP').exists():
  started=time.time();rows=[r for r in json.loads((OUT/'deployment.json').read_text())['rows'] if r['role']=='trainer' and r.get('problem') in ('circuit','ac1')]
  def collect(r):
   try:return json.loads(ssh(r['external_ip'],REMOTE.replace('ROW',repr(r)).replace('CURSORS',repr(state.get(r['problem'],{}).get('cursors',{}))),timeout=50))
   except Exception as e:return {'problem':r['problem'],'time':time.time(),'probe_error':str(e)}
  with ThreadPoolExecutor(2) as ex:results=list(ex.map(collect,rows))
  report={'time':time.time(),'rows':{}}
  for d in results:
   key=d['problem'];old=state.get(key,{})
   with (DEST/(key+'-samples.jsonl')).open('a') as h:h.write(json.dumps(d)+'\n')
   if 'probe_error' in d or 'attempt' not in d:report['rows'][key]=d;continue
   events=old.get('events',[])+d.pop('events',[]);logs=old.get('training_logs',[])+d.pop('training_logs',[])
   # Keep only current attempt in working state; raw histories remain on disk.
   cutoff=epoch(d['attempt']['created_at']);events=[e for e in events if e.get('time',0)>=cutoff]
   logs=[l for l in logs if l[:19]>=d['attempt']['created_at'][:19]]
   report['rows'][key]=summarize(d,old,events,logs)
   state[key]=dict(d,events=events,training_logs=logs)
  for path,value in ((statefile,state),(DEST/'latest.json',report)):
   tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
  print(json.dumps({'time':report['time'],'rows':{k:{field:v.get(field) for field in ('future_counts','grading_per_pool','completed_group_latency','backward_wall_seconds','errors','probe_error')} for k,v in report['rows'].items()}}),flush=True)
  if a.once:break
  if len(report['rows'])==2 and all(int((x.get('saved_checkpoint') or {}).get('batch',0))>=15 for x in report['rows'].values()):break
  time.sleep(max(1,a.interval-(time.time()-started)))
if __name__=='__main__':main()
