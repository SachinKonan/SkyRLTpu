import json, os, socket, subprocess
from pathlib import Path
root = Path.home()/'skyrl-systemd-probe-v5'
def live(pid, start):
    try:
        s=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return s[0]!='Z' and s[19]==start
    except FileNotFoundError: return False
results=[json.loads(p.read_text()) for p in root.glob('*/verification.json')]
ready=(root/'sky-cancel/awaiting-cancel.json').exists()
units=[]
for p in [*root.glob('*/systemd-*/unit.json'), *root.glob('*/grader-unit.json')]:
    unit=json.loads(p.read_text())['unit']
    state=subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','-p','ControlGroup','-p','Result'],text=True)
    d=dict(line.split('=',1) for line in state.splitlines() if '=' in line)
    d['unit']=unit
    units.append(d)
processes=[]
for p in [*root.glob('*/daemon.json'), *root.glob('*/grader-daemon.json')]:
    d=json.loads(p.read_text()); processes.append({'artifact':str(p.relative_to(root)), 'pid':d['pid'], 'live':live(d['pid'],d['start'])})
ports={}
for port in (24899,24900,25479,25480,25481,25482,25483,25484,25485,25486,25490,25491,25492,*range(25500,25532)):
    sock=socket.socket();sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    try: sock.bind(('0.0.0.0',port));ports[port]=True
    except OSError: ports[port]=False
    finally:sock.close()
result={'hostname':socket.gethostname(),'ready_for_cancel':ready,'stages':results,'units':units,'processes':processes,'ports_free':ports}
result['clean']=all(d['ActiveState'] in ('inactive','failed') and not d['ControlGroup'] for d in units) and all(not p['live'] for p in processes) and all(ports.values())
print(json.dumps(result))
