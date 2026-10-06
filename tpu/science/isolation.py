"""Minimal filesystem namespaces and hard process limits for CPU candidates.

No fallback to unsandboxed execution. Candidate-visible mounts are explicit;
parent environment, credentials, labels, and grader code are never inherited.
"""
from dataclasses import dataclass
from pathlib import Path
import os
import resource
import signal
import subprocess
import time


@dataclass(frozen=True)
class Limits:
    seconds: float
    memory_gib: int
    cpus: int = 4
    output_bytes: int = 128*1024*1024


def command(argv, *, readonly=(), writable=(), env=None, cwd='/work'):
    cmd = ['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
           '--clearenv','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/work']
    for name in ('/usr','/lib','/lib64','/bin'):
        if Path(name).is_symlink():cmd += ['--symlink',os.readlink(name),name]
        elif Path(name).exists():cmd += ['--ro-bind',name,name]
    for source,target in readonly:
        cmd += ['--ro-bind',str(Path(source).resolve()),str(target)]
    for source,target in writable:
        cmd += ['--bind',str(Path(source).resolve()),str(target)]
    environment=dict(PATH='/usr/bin:/bin',LANG='C.UTF-8',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',
                     MKL_NUM_THREADS='4',NUMEXPR_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
    environment.update(env or {})
    for key,value in environment.items():cmd += ['--setenv',key,str(value)]
    return cmd+['--chdir',cwd,'--',*map(str,argv)]


def run(argv, *, limits, log, readonly=(), writable=(), env=None, cwd='/work'):
    cpus = sorted(os.sched_getaffinity(0))[:limits.cpus]
    if len(cpus)<limits.cpus:raise RuntimeError('insufficient reserved CPU affinity')
    def constrain():
        os.sched_setaffinity(0,cpus)
        memory = limits.memory_gib*1024**3
        resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
        resource.setrlimit(resource.RLIMIT_FSIZE,(limits.output_bytes,limits.output_bytes))
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
        resource.setrlimit(resource.RLIMIT_CPU,(int(limits.seconds*limits.cpus)+1,)*2)
    started=time.monotonic()
    with open(log,'wb') as output:
        proc=subprocess.Popen(command(argv,readonly=readonly,writable=writable,env=env,cwd=cwd),
            stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,
            start_new_session=True,preexec_fn=constrain,env={'PATH':'/usr/bin:/bin'})
        try:
            code=proc.wait(timeout=limits.seconds)
        except BaseException:
            os.killpg(proc.pid,signal.SIGKILL);proc.wait();raise
    if code:raise RuntimeError(f'isolated process exited {code}: {Path(log).read_text(errors="replace")[-2500:]}')
    return time.monotonic()-started


def python_mounts(python):
    python=Path(python).absolute()
    import json
    paths=json.loads(subprocess.check_output([str(python),'-c',
        'import sys,json;print(json.dumps([sys.prefix,sys.base_prefix]))'],text=True))
    return [(p,p) for p in dict.fromkeys(paths)]


class Session:
    """Bounded JSON-lines exchange with one isolated CPU program."""
    def __init__(self, argv, *, limits, log, readonly=(), writable=(), env=None):
        import selectors
        self.deadline=time.monotonic()+limits.seconds
        self.buffer=b'';self.selector=selectors.DefaultSelector()
        self.log=Path(log)
        self.stderr=self.log.open('wb')
        cpus=sorted(os.sched_getaffinity(0))[:limits.cpus]
        if len(cpus)<limits.cpus:raise RuntimeError('insufficient CPU affinity')
        def constrain():
            os.sched_setaffinity(0,cpus)
            memory=limits.memory_gib*1024**3
            resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
            resource.setrlimit(resource.RLIMIT_FSIZE,(limits.output_bytes,)*2)
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
            resource.setrlimit(resource.RLIMIT_NOFILE,(256,256))
        cmd=command(argv,readonly=readonly,writable=writable,env=env)
        self.proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,
            start_new_session=True,preexec_fn=constrain,env={'PATH':'/usr/bin:/bin'},bufsize=0)
        self.selector.register(self.proc.stdout,selectors.EVENT_READ)

    def send(self,value):
        import json
        data=(json.dumps(value,allow_nan=False)+'\n').encode()
        # Observations are small (20*29*6 floats); writes must still obey the
        # phase deadline if malicious code stops reading the protocol.
        import selectors
        fd=self.proc.stdin.fileno();os.set_blocking(fd,False)
        selector=selectors.DefaultSelector();selector.register(fd,selectors.EVENT_WRITE)
        try:
            while data:
                if not selector.select(max(0,self.deadline-time.monotonic())):raise TimeoutError('candidate input timeout')
                try:data=data[os.write(fd,data):]
                except BlockingIOError:continue
        finally:selector.close()

    def receive(self):
        import json
        while b'\n' not in self.buffer:
            if len(self.buffer)>65536:raise ValueError('oversized candidate reply')
            if not self.selector.select(max(0,self.deadline-time.monotonic())):raise TimeoutError('candidate response timeout')
            data=os.read(self.proc.stdout.fileno(),65537)
            if not data:raise RuntimeError('candidate exited before replying: '+self.log.read_text(errors='replace')[-1500:])
            self.buffer+=data
        line,self.buffer=self.buffer.split(b'\n',1)
        if len(line)>65536:raise ValueError('oversized candidate reply')
        return json.loads(line)

    def finish(self):
        self.proc.stdin.close()
        code=self.proc.wait(timeout=max(.01,self.deadline-time.monotonic()))
        if code:raise RuntimeError(f'candidate process exited {code}: '+self.log.read_text(errors='replace')[-1500:])
        # Memory accounting belongs to the enclosing cgroup, not bwrap's RSS.
        return {}

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait()
        self.selector.close();self.stderr.close()
        for f in (self.proc.stdin,self.proc.stdout):
            if not f.closed:f.close()
