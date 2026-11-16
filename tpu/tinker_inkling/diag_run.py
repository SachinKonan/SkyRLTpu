"""Run tpu/run_ttd_ensemble.py unchanged, with crash/exit forensics written to $DIAG_DIR (append-only, flushed):
- faulthandler: all-thread stacks on SIGSEGV/SIGFPE/SIGABRT/SIGBUS/SIGILL and on SIGTERM/SIGHUP/SIGINT/SIGQUIT/SIGUSR1/2
  (chained, so the signal still has its normal effect) -> faults.log
- audit hook: os.kill/os.killpg/signal sends, subprocess/os.exec*/os.system/os.posix_spawn/os.fork, with stacks
  -> audit.log; os._exit and sys.exit are wrapped to log their caller before exiting
- heartbeat every 30 s: rss, threads, fds, and the main thread's current stack -> heartbeat.log
- atexit / threading.excepthook / unraisable hook -> events.log
"""
import atexit, faulthandler, os, runpy, signal, sys, threading, time, traceback

D = os.environ.get('DIAG_DIR', '.')
os.makedirs(D, exist_ok=True)


def _w(name, text):
    with open(os.path.join(D, name), 'a') as f:
        f.write(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {text}\n')
        f.flush()


_faults = open(os.path.join(D, 'faults.log'), 'a', buffering=1)
faulthandler.enable(file=_faults, all_threads=True)
for _s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2):
    faulthandler.register(_s, file=_faults, all_threads=True, chain=True)
_w('events.log', f'start pid={os.getpid()} ppid={os.getppid()} pgid={os.getpgid(0)} sid={os.getsid(0)} '
                 f'sighup={signal.getsignal(signal.SIGHUP)}')

_WATCH = ('os.kill', 'os.killpg', 'signal.pthread_kill', 'subprocess.Popen', 'os.exec', 'os.system', 'os.posix_spawn',
          'os.fork', 'os.forkpty')


def _audit(event, args):
    if event.startswith(_WATCH):
        _w('audit.log', f'{event} {str(args)[:300]}\n' + ''.join(traceback.format_stack(limit=25)))


sys.addaudithook(_audit)

_real_exit = os._exit


def _logged_os_exit(code):
    _w('events.log', f'os._exit({code}) called from\n' + ''.join(traceback.format_stack(limit=30)))
    _real_exit(code)


os._exit = _logged_os_exit
_real_sys_exit = sys.exit


def _logged_sys_exit(code=None):
    _w('events.log', f'sys.exit({code}) called from\n' + ''.join(traceback.format_stack(limit=30)))
    _real_sys_exit(code)


sys.exit = _logged_sys_exit
atexit.register(lambda: _w('events.log', 'atexit: interpreter exiting normally'))
threading.excepthook = lambda a: _w('events.log', f'thread exception in {a.thread}: '
                                    + ''.join(traceback.format_exception(a.exc_type, a.exc_value, a.exc_traceback)))
sys.unraisablehook = lambda u: _w('events.log', f'unraisable: {u.exc_type} {u.exc_value} {u.err_msg} {u.object!r}'[:2000])

_main = threading.main_thread()


def _heartbeat():
    while True:
        try:
            st = dict(l.split(':', 1) for l in open('/proc/self/status') if ':' in l)
            nfd = len(os.listdir('/proc/self/fd'))
            frame = sys._current_frames().get(_main.ident)
            where = ''.join(traceback.format_stack(frame, limit=8)) if frame else '(no main frame)'
            _w('heartbeat.log', f'rss={st["VmRSS"].strip()} hwm={st["VmHWM"].strip()} threads={st["Threads"].strip()} '
                                f'fds={nfd}\n{where}')
        except Exception as e:  # never disturb the run
            _w('heartbeat.log', f'heartbeat error {e!r}')
        time.sleep(30)


threading.Thread(target=_heartbeat, name='diag-heartbeat', daemon=True).start()
sys.argv = ['tpu/run_ttd_ensemble.py']
runpy.run_path('tpu/run_ttd_ensemble.py', run_name='__main__')
