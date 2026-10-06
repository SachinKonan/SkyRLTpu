import json, os, signal, socket, subprocess, sys, threading, time
from pathlib import Path
from types import SimpleNamespace
import runtime_service as rs

ROOT = Path.home() / 'skyrl-systemd-probe-v5'
ROOT.mkdir(exist_ok=True)
IPS = os.environ['SKYPILOT_NODE_IPS'].split()
RANK = int(os.environ['SKYPILOT_NODE_RANK'])
TOKEN = os.environ['SKYPILOT_TASK_ID']
CASE = os.environ.get('PROBE_CASE', '')

def report(path, data):
    rs.save(path, data)
    print(json.dumps(data), flush=True)

def wait_file(path, timeout=120):
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(.2)

if sys.argv[1] in ('daemon', 'grader-daemon'):
    if os.getsid(0) != os.getpid(): os.setsid()
    if CASE not in ('normal', 'reuse'): signal.signal(signal.SIGTERM, signal.SIG_IGN)
    sock = socket.socket(); sock.bind(('0.0.0.0', 25492 if sys.argv[1] == 'grader-daemon' else 25490)); sock.listen()
    rs.save(ROOT / CASE / (sys.argv[1] + '.json'), {'pid': os.getpid(), 'start': rs.process_start(os.getpid()), 'cgroup': Path('/proc/self/cgroup').read_text()})
    while True: time.sleep(5)

elif sys.argv[1] == 'workload':
    if CASE == 'preflight-failure' and RANK == 3:
        raise RuntimeError('intentional preflight failure before Ray starts')
    rs.wait_preflight(SimpleNamespace(systemd_runtime=True, setup_timeout=120), lambda: False)
    (ROOT / CASE / 'released').touch()
    subprocess.Popen([sys.executable, __file__, 'daemon'])
    from cgroup_limits import runtime_owner_properties
    import uuid
    properties = runtime_owner_properties()
    assert properties, 'grader must discover its runtime owner'
    unit = 'skyrl-probe-grade-' + uuid.uuid4().hex
    rs.save(ROOT / CASE / 'grader-unit.json', {'unit': unit, 'properties': properties})
    # Match the actual grading ownership and resource wrapper.
    subprocess.Popen(['sudo','-n','systemd-run','--quiet','--wait','--unit='+unit,
        '--uid='+str(os.getuid()),'--gid='+str(os.getgid()),
        '--property=KillMode=control-group','--property=TimeoutStopSec=2',
        '--property=MemoryMax=8G','--property=CPUQuota=400%',*properties,
        '--setenv=SKYPILOT_NODE_IPS='+os.environ['SKYPILOT_NODE_IPS'],
        '--setenv=SKYPILOT_NODE_RANK='+str(RANK),'--setenv=SKYPILOT_TASK_ID='+TOKEN,
        '--setenv=PROBE_CASE='+CASE,sys.executable,__file__,'grader-daemon'])
    wait_file(ROOT / CASE / 'grader-daemon.json')
    ray = str(Path(sys.executable).with_name('ray'))
    args = [ray, 'start', '--node-ip-address=' + IPS[RANK], '--num-cpus=1', '--object-store-memory=134217728',
            '--node-manager-port=25480', '--object-manager-port=25481', '--runtime-env-agent-port=25482',
            '--dashboard-agent-grpc-port=25483', '--dashboard-agent-listen-port=25484', '--metrics-export-port=25485',
            '--min-worker-port=25500', '--max-worker-port=25531', '--disable-usage-stats']
    if RANK == 0:
        args += ['--head', '--port=25479', '--include-dashboard=false', '--dashboard-port=25486', '--temp-dir=/tmp/skyrl-systemd-probe-ray']
    else:
        # Wait for the owned head, never connect to the production port.
        deadline = time.monotonic() + 90
        while True:
            try:
                with socket.create_connection((IPS[0],25479),timeout=1): break
            except OSError:
                if time.monotonic() > deadline: raise
                time.sleep(1)
        args += ['--address=' + IPS[0] + ':25479']
    subprocess.run(args, check=True, timeout=100)
    import ray as r
    r.init(address=IPS[0]+':25479', log_to_driver=False)
    deadline = time.monotonic()+90
    while len([n for n in r.nodes() if n['Alive']]) != len(IPS):
        if time.monotonic()>deadline: raise TimeoutError('Ray membership')
        time.sleep(1)
    @r.remote
    def square(x): return x*x
    assert r.get(square.remote(RANK)) == RANK*RANK
    r.shutdown()
    (ROOT / CASE / 'ray-ready').touch()
    if CASE in ('normal', 'reuse'):
        time.sleep(3)
        # Model the production bootstrap's graceful stop_ray before exit.
        # Select only processes inside this exact service cgroup.
        import psutil
        own = Path('/proc/self/cgroup').read_text()
        targets = []
        for p in psutil.process_iter(['pid','cmdline']):
            try:
                if (Path(f'/proc/{p.pid}/cgroup').read_text() == own
                        and any('/ray/' in a for a in (p.info['cmdline'] or []))):
                    targets.append(p); targets.extend(p.children(recursive=True))
            except (OSError, psutil.Error): pass
        for p in targets:
            try: p.terminate()
            except psutil.Error: pass
        _, alive = psutil.wait_procs(targets, timeout=2)
        for p in alive:
            try: p.kill()
            except psutil.Error: pass
        psutil.wait_procs(alive, timeout=3)
    else:
        while True: time.sleep(1)

elif sys.argv[1] == 'launch':
    raise SystemExit(rs.launch([sys.executable,__file__,'workload'], ROOT / CASE, IPS, RANK,
        TOKEN + '-' + CASE, peer_timeout=8,shutdown_grace=3,setup_timeout=120))

elif sys.argv[1] == 'matrix':
    class Barriers:
        def __init__(self): self.seen={}; self.lock=threading.Lock()
        def update(self,m):
            with self.lock:
                key=m['key']; self.seen.setdefault(key,set()).add(m['rank'])
                return {'complete':len(self.seen[key])==len(IPS)}
    server = rs.start_server((IPS[0],24899),Barriers()) if RANK == 0 else None
    def barrier(key):
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            try:
                if rs.exchange((IPS[0],24899),{'key':key,'rank':RANK})['complete']: return
            except OSError: pass
            time.sleep(.3)
        raise TimeoutError(key)
    sentinel=socket.socket(); sentinel.bind(('0.0.0.0',25491)); sentinel.listen()
    import psutil
    original={p.pid:rs.process_start(p.pid) for p in psutil.process_iter() if p.name() in ('raylet','gcs_server')}
    try:
        for case in ('normal','launcher-kill','supervisor-kill','preflight-failure','reuse','sky-cancel'):
            directory=ROOT / case; directory.mkdir(exist_ok=True)
            barrier(case+'-start')
            env=dict(os.environ,PROBE_CASE=case)
            proc=subprocess.Popen([sys.executable,__file__,'launch'],env=env)
            if case in ('launcher-kill','supervisor-kill','sky-cancel'):
                wait_file(directory/'ray-ready',180)
                barrier(case+'-ray-ready')
                if case=='launcher-kill' and RANK==1: proc.kill()
                if case=='supervisor-kill' and RANK==2:
                    info=json.loads(next(directory.glob('systemd-*/supervisor.json')).read_text())
                    os.kill(info['pid'],signal.SIGKILL)
                if case=='sky-cancel':
                    report(directory/'awaiting-cancel.json', {'rank':RANK,'case':case,'state':'ready for real SkyPilot cancellation'})
            code=proc.wait(timeout=210)
            # A SIGKILLed launcher cannot wait for its service; poll its owned unit.
            unit=json.loads(next(directory.glob('systemd-*/unit.json')).read_text())['unit']
            deadline=time.monotonic()+30
            while True:
                state=subprocess.check_output(['sudo','-n','systemctl','show',unit,'-p','ActiveState','-p','ControlGroup'],text=True)
                active=any('ActiveState='+v in state for v in ('active','activating','deactivating'))
                if not active: break
                if time.monotonic()>deadline: raise RuntimeError('unit remains active: '+state)
                time.sleep(1)
            # Process identity, listeners and unrelated control processes.
            daemon=directory/'daemon.json'
            if daemon.exists():
                data=json.loads(daemon.read_text()); assert rs.process_start(data['pid'])!=data['start'],data
            grader = directory/'grader-daemon.json'
            if grader.exists():
                data=json.loads(grader.read_text()); assert rs.process_start(data['pid'])!=data['start'],data
            for port in (25490,25492,25479,25480,25481):
                sock=socket.socket(); sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('0.0.0.0',port)); sock.close()
            assert all(rs.process_start(pid)==start for pid,start in original.items()),'unrelated Ray was killed'
            assert sentinel.fileno()>=0
            if case=='preflight-failure': assert not (directory/'released').exists()
            if case in ('normal','reuse'): assert code==0,code
            else: assert code!=0,code
            report(directory/'verification.json', {'rank':RANK,'case':case,'exit_code':code,'cleanup_verified':True,'original_ray_preserved':True,'unit_state':state.strip()})
            barrier(case+'-verified')
    finally:
        sentinel.close()
        if server: server.shutdown(); server.server_close()
