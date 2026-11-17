"""Live dashboard for the Inkling Erdős runs (della-vis1, localhost only; view via `ssh -N -L PORT:localhost:PORT`).

One process: a collector thread re-reads only files whose mtime changed (trajectories per step, metrics, grading
journal tail, client log tail) every VIZ_INTERVAL seconds and keeps a JSON snapshot in memory; a tiny HTTP server
serves the page (/) and the snapshot (/data.json). The page polls every 30 s and draws SVG charts in the browser.
Env: VIZ_PORT (8765), VIZ_INTERVAL (60), VIZ_LIVE (comma runs, e.g. g2-s1), VIZ_REF (comma reference runs).
"""
import glob, gzip, json, os, re, statistics, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RUNS = os.environ.get('VIZ_RUNS_DIR', '/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs')
PORT = int(os.environ.get('VIZ_PORT', '8765'))
INTERVAL = int(os.environ.get('VIZ_INTERVAL', '60'))
LIVE = [r for r in os.environ.get('VIZ_LIVE', 'g2-s1').split(',') if r]
REF = [r for r in os.environ.get('VIZ_REF', 'g1-s1,g1-s2,g1-s3,g1-f1').split(',') if r]
X0 = 0.38085
C5 = re.compile(r'C5 bound:\s*([0-9.eE+-]+)')
TOOL = re.compile(r'<\|end_message\|><\|message_model\|>(?!<\|content_)')
SNAP = {'updated': None, 'runs': {}}
_cache = {}            # trajectory file -> (mtime, step stats)


def outcome(d):
    m = d.get('message') or ''
    if d.get('correctness', 0) > 0 and C5.search(m):
        return 'valid'
    if 'timed out' in m:
        return 'timeout'
    if m.startswith('Invalid solution'):
        return 'verifier'
    if m.startswith('Invalid code') or not (d.get('response') or '').strip():
        return 'tool_call' if TOOL.search(d.get('response_raw') or '') else 'cut_off'
    if 'Program execution failed' in m:
        return 'crash'
    return 'other'


def step_stats(fn):
    st = os.stat(fn).st_mtime
    hit = _cache.get(fn)
    if hit and hit[0] == st:
        return hit[1]
    rows = [json.loads(l) for l in gzip.open(fn, 'rt')]
    oc = {}
    parents, best_child, beat, lines = {}, None, 0, []
    for d in rows:
        o = outcome(d)
        oc[o] = oc.get(o, 0) + 1
        pv = -float(d['parent_value']) if d.get('parent_value') is not None else None
        if pv is not None:
            parents[d.get('parent_id')] = pv
        if o == 'valid':
            c = float(C5.search(d['message']).group(1))
            best_child = c if best_child is None else min(best_child, c)
            if pv is not None and c < pv - 1e-9:
                beat += 1
        if d.get('parsed_code'):
            lines.append(len(d['parsed_code'].splitlines()))
    top5 = sorted(parents, key=parents.get)[:5]
    gain = 0.0
    for d in rows:
        if outcome(d) == 'valid' and d.get('parent_id') in top5:
            gain = max(gain, parents[d['parent_id']] - float(C5.search(d['message']).group(1)))
    s = dict(n=len(rows), outcomes=oc, beat=beat, best_child=best_child, start_best=min(parents.values()) if parents else None,
             gain_e6=gain * 1e6, code_p50=statistics.median(lines) if lines else 0)
    _cache[fn] = (st, s)
    return s


def tail_count(path, start_time):
    """(submitted, graded) grading events after start_time (journal lines end with "time": <epoch>})."""
    sub = res = 0
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4_000_000))
            for line in f.read().decode(errors='ignore').splitlines():
                m = re.search(r'"time": ([0-9.]+)\}?\s*$', line)
                if not m or float(m.group(1)) < start_time:
                    continue
                if line.startswith('{"event": "submitted"'):
                    sub += 1
                elif line.startswith('{"event": "result"'):
                    res += 1
    except OSError:
        pass
    return sub, res


def live_sampling(run):
    """Rollouts sampled so far in the newest sampling round (client log)."""
    try:
        with open(f'{RUNS}/{run}.log', 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 3_000_000))
            text = f.read().decode(errors='ignore')
    except OSError:
        return None, 0
    i = text.rfind('[ensemble] step ')
    m = re.search(r'\[ensemble\] step (\d+): sampling', text[i:]) if i >= 0 else None
    return (int(m.group(1)) if m else None), text[i:].count('two_phase outcome') if i >= 0 else 0


def unit_state(run):
    try:
        return subprocess.run(['systemctl', '--user', 'is-active', f'tinker-run-{run}.service'],
                              capture_output=True, text=True, timeout=20).stdout.strip() or 'unknown'
    except Exception:
        return 'unknown'


def collect_run(run, live):
    lg = f'{RUNS}/{run}/tinker_log/{run}'
    files = sorted(glob.glob(f'{lg}/member_inkling/trajectories/inkling_step_*.jsonl.gz'))
    try:
        metrics = [json.loads(l) for l in open(f'{lg}/metrics.jsonl')]
    except OSError:
        metrics = []
    frozen = run.split('-')[-1].startswith('f')
    steps, best = [], None
    for fn in files:
        k = int(re.search(r'step_(\d+)', fn).group(1))
        try:
            s = step_stats(fn)
        except (EOFError, OSError, ValueError):
            continue                        # file still being written
        if best is None and s['start_best'] is not None:
            best = s['start_best']
        if s['best_child'] is not None:
            best = min(best, s['best_child']) if best is not None else s['best_child']
        m = metrics[k] if k < len(metrics) else {}
        g = m.get('inkling/infer/generated_tokens', 0) / 1e6
        p = m.get('inkling/infer/prompt_tokens', 0) / 1e6
        n = max(1, s['n'])
        steps.append(dict(step=k + 1, best_e6=(best - X0) * 1e6 if best is not None else None,
                          valid=100 * s['outcomes'].get('valid', 0) / n, beat=s['beat'], gain_e6=s['gain_e6'],
                          cut_off=s['outcomes'].get('cut_off', 0), timeout=s['outcomes'].get('timeout', 0),
                          crash=s['outcomes'].get('crash', 0), code_p50=s['code_p50'],
                          gen_k=round(1000 * g / n, 2), prompt_k=round(1000 * p / n, 2),
                          usd=round(g + 0.3 * p + (0 if frozen else 1.5 * (g + p)), 1) if m else None,
                          minutes=round(m.get('time/total', 0) / 60, 1) if m else None))
    out = dict(steps=steps, frozen=frozen)
    if live:
        k, sampled = live_sampling(run)
        t0 = os.stat(files[-1]).st_mtime if files else 0
        sub, res = tail_count(f'{RUNS}/{run}/grading-journal.jsonl', t0)
        try:
            gstep = json.load(open(f'{lg}/global_step.json'))['global_step']
        except (OSError, ValueError, KeyError):
            gstep = 0
        out.update(state=unit_state(run), global_step=gstep, sampling_round=(k + 1) if k is not None else None,
                   sampled=sampled, submitted=sub, graded=res)
    return out


def collect():
    runs = {}
    for r in LIVE + REF:
        try:
            runs[r] = collect_run(r, r in LIVE)
            runs[r]['live'] = r in LIVE
        except Exception as e:      # one bad run never blanks the page
            runs[r] = dict(error=repr(e), live=r in LIVE, steps=[])
    spend = None
    try:
        lines = [l for l in open(f'{RUNS}/watch.log', errors='ignore') if 'tinker spend since' in l]
        spend = lines[-1].strip() if lines else None
    except OSError:
        pass
    grid = {}
    for g in glob.glob(f'{RUNS}/grid-*'):
        name = os.path.basename(g)[5:]
        grid[name] = dict(held=os.path.exists(f'{g}/service.args') and not os.path.exists(f'{g}/RELEASE'),
                          stopped=os.path.exists(f'{g}/STOPPED'))
    samples, timings = {}, {}
    for r in LIVE:
        try:
            samples[r] = pick_samples(r)
        except Exception as e:
            samples[r] = dict(error=repr(e), items=[])
        try:
            timings[r] = timing(r)
        except Exception as e:
            timings[r] = dict(error=repr(e))
    return dict(updated=time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime()), runs=runs, spend=spend, grids=grid,
                live=LIVE, ref=REF, samples=samples, timing=timings)


_samples_cache = {}    # run -> (newest trajectory file, mtime, picks)


def _parts(raw):
    """(thinking, answer) text from the rendered completion (Inkling: to=self thinking, to=user answer)."""
    i = raw.find('<|start|>assistant to=user')
    clean = lambda x: re.sub(r'<\|[a-z_]+\|>|assistant to=(self|user)', '', x).strip()
    return (clean(raw[:i]), clean(raw[i:])) if i >= 0 else (clean(raw), '')


def pick_samples(run, k_best=3, k_fail=3, k_mid=2):
    """A small fixed set from the newest finished step: best valid, a median valid, and failures of different kinds.
    Summaries only (the page fetches full text per rollout on click)."""
    files = sorted(glob.glob(f'{RUNS}/{run}/tinker_log/{run}/member_inkling/trajectories/inkling_step_*.jsonl.gz'))
    if not files:
        return dict(step=None, items=[])
    fn = files[-1]
    mt = os.stat(fn).st_mtime
    hit = _samples_cache.get(run)
    if hit and hit[0] == fn and hit[1] == mt:
        return hit[2]
    try:
        rows = [json.loads(l) for l in gzip.open(fn, 'rt')]
    except (EOFError, OSError, ValueError):
        return hit[2] if hit else dict(step=None, items=[])
    for i, d in enumerate(rows):
        d['_i'], d['_o'] = i, outcome(d)
        m = C5.search(d.get('message') or '')
        d['_c5'] = float(m.group(1)) if d['_o'] == 'valid' and m else None
    valid = sorted((d for d in rows if d['_c5'] is not None), key=lambda d: d['_c5'])
    chosen = [('best', d) for d in valid[:k_best]]
    if valid:
        mid = len(valid) // 2
        chosen += [('median', d) for d in valid[mid:mid + k_mid]]
    seen = set()
    for kind in ('cut_off', 'timeout', 'crash', 'verifier', 'tool_call'):
        for d in rows:
            if d['_o'] == kind and kind not in seen and len(seen) < k_fail:
                chosen.append(('failure', d))
                seen.add(kind)
                break
    items = []
    for tag, d in chosen:
        think, ans = _parts(d.get('response_raw') or '')
        pv = -float(d['parent_value']) if d.get('parent_value') is not None else None
        items.append(dict(i=d['_i'], tag=tag, outcome=d['_o'], c5=d['_c5'], parent_c5=pv,
                          gain_e6=((pv - d['_c5']) * 1e6) if (pv is not None and d['_c5'] is not None) else None,
                          msg=(d.get('message') or '')[:160], think_chars=len(think), answer_chars=len(ans),
                          code_lines=len((d.get('parsed_code') or '').splitlines()),
                          think_head=think[:400], think_tail=think[-400:] if len(think) > 800 else ''))
    out = dict(step=int(re.search(r'step_(\d+)', fn).group(1)) + 1, file=os.path.basename(fn), items=items)
    _samples_cache[run] = (fn, mt, out)
    return out


def timing(run):
    """Per-step phase timeline from group_timeline.jsonl (per group: started, sampled, graded; train_enqueued) and
    grading-events.jsonl (per request: dispatched, running, done). Returns an overview for every step plus a Gantt
    (per-group bars) and a grading-concurrency series for the newest step."""
    lg = f'{RUNS}/{run}/tinker_log/{run}'
    steps = {}
    try:
        for line in open(f'{lg}/group_timeline.jsonl'):
            e = json.loads(line)
            s = steps.setdefault(e['step'], dict(groups={}, train=[]))
            if e['event'] == 'train_enqueued':
                s['train'].append(e['time'])
            else:
                s['groups'].setdefault(e.get('group'), {})[e['event']] = e['time']
    except (OSError, ValueError):
        return None
    if not steps:
        return None
    order = sorted(steps)
    start = {k: min(g.get('started', 1e20) for g in steps[k]['groups'].values()) for k in order}
    # grading requests: first dispatch, running start, end
    req = {}
    try:
        with open(f'{RUNS}/{run}/grading-events.jsonl', 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 8_000_000))
            for line in f.read().decode(errors='ignore').splitlines():
                if '"request_id"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                r = req.setdefault(e['request_id'], {})
                if e['event'] == 'grading_dispatched':
                    r.setdefault('dispatch', e['time'])
                elif e['event'] == 'request_state' and e.get('to') == 'running':
                    r.setdefault('run', e['time'])
                elif e['event'] == 'request_state' and e.get('to') in ('done', 'failed_candidate'):
                    r['end'] = e['time']
    except OSError:
        pass
    now = time.time()
    overview = []
    for i, k in enumerate(order):
        gs = steps[k]['groups'].values()
        t0 = start[k]
        sampled = [g['sampled'] for g in gs if 'sampled' in g]
        graded = [g['graded'] for g in gs if 'graded' in g]
        end = start[order[i + 1]] if i + 1 < len(order) else None
        done = len(graded) == len(steps[k]['groups']) and graded
        t_s = max(sampled) if len(sampled) == len(steps[k]['groups']) else None
        t_g = max(graded) if done else None
        overview.append(dict(step=k + 1, sampling=round(((t_s or now) - t0) / 60, 1),
                             grading_tail=round(((t_g or now) - t_s) / 60, 1) if t_s else None,
                             train_tail=round((end - t_g) / 60, 1) if (end and t_g) else None,
                             complete=bool(end)))
    k = order[-1]
    t0 = start[k]
    gantt = [dict(group=g, s=round((v.get('started', t0) - t0) / 60, 2),
                  sampled=round((v['sampled'] - t0) / 60, 2) if 'sampled' in v else None,
                  graded=round((v['graded'] - t0) / 60, 2) if 'graded' in v else None)
             for g, v in sorted(steps[k]['groups'].items(), key=lambda kv: kv[0] if kv[0] is not None else -1)]
    t_end = now
    rs = [r for r in req.values() if r.get('dispatch', 0) >= t0]
    series = []
    horizon = max([t_end] + [r.get('end', 0) for r in rs])
    t = t0
    while t <= min(horizon, now) and len(series) < 400:
        running = sum(1 for r in rs if r.get('run', 1e20) <= t < r.get('end', 1e20))
        queued = sum(1 for r in rs if r['dispatch'] <= t < r.get('run', r.get('end', 1e20)))
        series.append([round((t - t0) / 60, 2), running, queued])
        t += 30
    runs_s = [(r['end'] - r['run']) for r in rs if 'run' in r and 'end' in r]
    waits = [(r['run'] - r['dispatch']) for r in rs if 'run' in r and 'dispatch' in r]
    med = lambda a: round(statistics.median(a), 1) if a else None
    return dict(overview=overview, step=k + 1, gantt=gantt, train=[round((x - t0) / 60, 2) for x in steps[k]['train']],
                series=series, grade_run_p50_s=med(runs_s), grade_wait_p50_s=med(waits), requests=len(rs))


def rollout_full(run, step, i):
    """Full thinking, answer and grader message of one rollout (read on demand)."""
    if run not in LIVE + REF or not re.fullmatch(r'\d{1,3}', str(step)) or not re.fullmatch(r'\d{1,4}', str(i)):
        return None
    fn = f'{RUNS}/{run}/tinker_log/{run}/member_inkling/trajectories/inkling_step_{int(step) - 1:06d}.jsonl.gz'
    try:
        with gzip.open(fn, 'rt') as f:
            for n, line in enumerate(f):
                if n == int(i):
                    d = json.loads(line)
                    think, ans = _parts(d.get('response_raw') or '')
                    return dict(thinking=think, answer=ans, message=d.get('message'), reward=d.get('reward'))
    except (OSError, EOFError, ValueError):
        return None
    return None


def loop():
    global SNAP
    while True:
        try:
            SNAP = collect()
        except Exception as e:
            SNAP = dict(SNAP, error=repr(e))
        time.sleep(INTERVAL)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Inkling Erdős — live</title>
<style>
:root{--ink:#2b2a27;--muted:#4a4945;--grid:#ecebe7;--surface:#fcfcfb;--live:#2a78d6;--ref:#b8b6b0}
body{font:14px/1.4 -apple-system,system-ui,sans-serif;color:var(--ink);background:var(--surface);margin:20px}
h1{font-size:18px;margin:0 0 4px} .muted{color:var(--muted)} .row{display:flex;gap:16px;flex-wrap:wrap}
.card{background:#fff;border:1px solid #e7e5df;border-radius:8px;padding:12px 14px}
.chart{width:420px} .chart h3{font-size:13px;margin:0 0 6px;font-weight:600}
.bar{height:8px;background:var(--grid);border-radius:4px;overflow:hidden;width:220px;display:inline-block;vertical-align:middle}
.bar>div{height:100%;background:var(--live)} table{border-collapse:collapse;font-size:12px}
td,th{padding:3px 8px;text-align:right;border-bottom:1px solid var(--grid)} th{color:var(--muted);font-weight:500}
#tip{position:fixed;pointer-events:none;background:#fff;border:1px solid #d9d7d0;border-radius:6px;padding:6px 8px;font-size:12px;display:none;box-shadow:0 2px 6px rgba(0,0,0,.08)}
.legend span{display:inline-flex;align-items:center;gap:6px;margin-right:14px;font-size:12px}
.sw{width:18px;height:0;border-top:2px solid}
</style></head><body>
<h1>Inkling-2-Small · Erdős · live</h1><div class="muted" id="upd"></div>
<div class="row" style="margin:12px 0" id="status"></div>
<div class="legend card" style="margin-bottom:12px" id="legend"></div>
<div class="row" id="charts"></div>
<div class="card" style="margin-top:14px"><h3 style="font-size:13px;margin:0 0 6px">Per step (live runs)</h3><div id="table"></div></div>
<div class="card" style="margin-top:14px"><h3 style="font-size:13px;margin:0 0 6px">Where the time goes</h3><div id="timing"></div></div>
<div class="card" style="margin-top:14px"><h3 style="font-size:13px;margin:0 0 6px">Generations — newest finished step (best, median, one of each failure kind; click to load the full text)</h3><div id="gens"></div></div>
<div id="tip"></div>
<script>
const CH=[["best_e6","Best C5 so far, (C5−0.38085)×1e6 · lower is better",true],["valid","Valid programs (%)",false],
["beat","Rollouts beating their parent (of 512)",false],["cut_off","Answers cut off at the budget",false],
["timeout","Programs killed after 1100 s",false],["code_p50","Program length, median lines",false],
["gen_k","Generated tokens per rollout (k)",false],["prompt_k","Prompt tokens per rollout (k)",false],
["usd","Cost per step, list price ($)",false]];
const W=420,H=210,P={l:44,r:12,t:8,b:26};const tip=document.getElementById('tip');
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
function chart(key,title,logy,runs){
 let pts=[];for(const [name,r] of Object.entries(runs)){for(const s of (r.steps||[])){const v=s[key];if(v!=null&&!(logy&&v<=0))pts.push(v)}}
 if(!pts.length)return `<div class="card chart"><h3>${title}</h3><div class="muted">no data yet</div></div>`;
 let lo=Math.min(...pts),hi=Math.max(...pts);if(!logy){lo=Math.min(0,lo);hi=hi*1.08||1}
 const f=logy?(v=>Math.log10(v)):(v=>v);let a=f(logy?lo*0.9:lo),b=f(logy?hi*1.1:hi);if(b===a)b=a+1;
 let xmax=1;for(const r of Object.values(runs))for(const s of (r.steps||[]))xmax=Math.max(xmax,s.step);
 const X=s=>P.l+(s-1)/(Math.max(1,xmax-1))*(W-P.l-P.r),Y=v=>H-P.b-(f(v)-a)/(b-a)*(H-P.t-P.b);
 let g=`<svg width="${W}" height="${H}">`;const ticks=5;
 for(let i=0;i<=ticks;i++){const tv=a+(b-a)*i/ticks,y=H-P.b-(i/ticks)*(H-P.t-P.b),lab=logy?Math.pow(10,tv):tv;
  g+=`<line x1="${P.l}" x2="${W-P.r}" y1="${y}" y2="${y}" stroke="#ecebe7"/><text x="${P.l-6}" y="${y+4}" font-size="10" text-anchor="end" fill="#4a4945">${lab>=100?lab.toFixed(0):lab.toFixed(1)}</text>`}
 for(let s=1;s<=xmax;s++)g+=`<text x="${X(s)}" y="${H-8}" font-size="10" text-anchor="middle" fill="#4a4945">${s}</text>`;
 const order=Object.entries(runs).sort((x,y)=>(x[1].live?1:0)-(y[1].live?1:0));
 for(const [name,r] of order){const ss=(r.steps||[]).filter(s=>s[key]!=null&&!(logy&&s[key]<=0));if(!ss.length)continue;
  const live=r.live,col=live?'var(--live)':'var(--ref)',dash=r.frozen?'4 3':'';
  g+=`<polyline fill="none" stroke="${col}" stroke-width="${live?2:1.2}" stroke-dasharray="${dash}" points="${ss.map(s=>X(s.step)+','+Y(s[key])).join(' ')}"/>`;
  for(const s of ss){g+=`<circle cx="${X(s.step)}" cy="${Y(s[key])}" r="${live?4:2.5}" fill="${r.frozen?'#fff':col}" stroke="${col}" stroke-width="1.5"/>`;
   g+=`<circle cx="${X(s.step)}" cy="${Y(s[key])}" r="10" fill="transparent" data-t="${esc(name)} · step ${s.step}<br><b>${(+s[key]).toFixed(key==='best_e6'?2:1)}</b>"/>`}}
 return `<div class="card chart"><h3>${title}</h3>${g}</svg></div>`}
function render(d){
 document.getElementById('upd').textContent=`updated ${d.updated} · refreshes every 30 s · ${d.spend?d.spend.replace(/^\S+\s/,''):''}`;
 let st='';for(const n of d.live){const r=d.runs[n]||{};const s=r.sampled||0,gr=r.graded||0,sub=r.submitted||0;
  st+=`<div class="card"><b>${n}</b> <span class="muted">· ${r.state||'?'} · committed step ${r.global_step??'?'} · sampling round ${r.sampling_round??'–'}</span><br>
  sampled <span class="bar"><div style="width:${Math.min(100,s/512*100)}%"></div></span> ${s}/512<br>
  graded&nbsp; <span class="bar"><div style="width:${sub?Math.min(100,gr/sub*100):0}%"></div></span> ${gr}/${sub}</div>`}
 for(const [g,v] of Object.entries(d.grids||{}))if(v.held&&!v.stopped)st+=`<div class="card">grid <b>${g}</b>: held at the step-1 gate (waiting for RELEASE)</div>`;
 document.getElementById('status').innerHTML=st;
 document.getElementById('legend').innerHTML=`<span><i class="sw" style="border-color:var(--live)"></i>live: ${d.live.join(', ')}</span><span><i class="sw" style="border-color:var(--ref)"></i>pilot (effort 0.8, lr 2e-5, 40k): ${d.ref.filter(x=>!x.includes('-f')).join(', ')}</span><span><i class="sw" style="border-color:var(--ref);border-top-style:dashed"></i>pilot no-training control</span>`;
 document.getElementById('charts').innerHTML=CH.map(([k,t,l])=>chart(k,t,l,d.runs)).join('');
 let tb='<table><tr><th>run</th><th>step</th><th>best</th><th>valid%</th><th>beat</th><th>cut off</th><th>timeout</th><th>crash</th><th>lines</th><th>gen k</th><th>prompt k</th><th>$</th><th>min</th></tr>';
 for(const n of d.live)for(const s of (d.runs[n]?.steps||[]))tb+=`<tr><td>${n}</td><td>${s.step}</td><td>${s.best_e6==null?'–':s.best_e6.toFixed(2)}</td><td>${s.valid.toFixed(1)}</td><td>${s.beat}</td><td>${s.cut_off}</td><td>${s.timeout}</td><td>${s.crash}</td><td>${s.code_p50}</td><td>${s.gen_k}</td><td>${s.prompt_k}</td><td>${s.usd??'–'}</td><td>${s.minutes??'–'}</td></tr>`;
 document.getElementById('table').innerHTML=tb+'</table>';
 document.getElementById('timing').innerHTML=Object.entries(d.timing||{}).map(([r,T])=>timingHTML(r,T)).join('');
 const gk=JSON.stringify(Object.fromEntries(Object.entries(d.samples||{}).map(([k,v])=>[k,v.file])));
 if(gk!==lastGens){lastGens=gk;let gh='';
  for(const [run,s] of Object.entries(d.samples||{})){gh+=`<div class="muted" style="margin:6px 0">${run} · step ${s.step??'–'}</div>`;
   for(const it of (s.items||[])){const head=it.outcome==='valid'?`C5 ${it.c5.toFixed(7)} (parent ${it.parent_c5?.toFixed(7)}, gain ${it.gain_e6?.toFixed(2)}e-6)`:esc(it.msg);
    gh+=`<details data-run="${run}" data-step="${s.step}" data-i="${it.i}" style="margin:4px 0;border-top:1px solid var(--grid);padding-top:4px">
    <summary><b>${it.tag}</b> · ${it.outcome} · ${head} · <span class="muted">thinking ${(it.think_chars/1000).toFixed(1)}k chars · answer ${(it.answer_chars/1000).toFixed(1)}k · ${it.code_lines} lines · #${it.i}</span></summary>
    <div style="font-size:12.5px;color:var(--ink);white-space:pre-wrap;margin:6px 0;background:#f6f5f2;padding:8px;border-radius:6px">${esc(it.think_head)}${it.think_tail?' … '+esc(it.think_tail):''}</div><div class="full" style="color:var(--ink)">loading…</div></details>`}}
  document.getElementById('gens').innerHTML=gh;
  for(const el of document.querySelectorAll('#gens details'))el.addEventListener('toggle',async()=>{if(!el.open||el.dataset.loaded)return;el.dataset.loaded=1;
   const box=el.querySelector('.full');try{const r=await (await fetch(`rollout?run=${el.dataset.run}&step=${el.dataset.step}&i=${el.dataset.i}`)).json();
   box.innerHTML=`<div style="font-size:12px;margin:4px 0"><b>grader:</b> ${esc(r.message)} · reward ${r.reward}</div><details><summary>full thinking (${r.thinking.length} chars)</summary><pre style="white-space:pre-wrap;font-size:12px;color:var(--ink);max-height:420px;overflow:auto;background:#f6f5f2;padding:8px;border-radius:6px">${esc(r.thinking)}</pre></details><details open><summary>answer (${r.answer.length} chars)</summary><pre style="white-space:pre-wrap;font-size:12px;color:#1f1e1c;max-height:520px;overflow:auto;background:#f6f5f2;padding:8px;border-radius:6px">${esc(r.answer)}</pre></details>`}catch(e){box.textContent='failed: '+e}})}}
let lastGens='';
const PH={s:'#2a78d6',g:'#eb6834',t:'#1baf7a'};
function timingHTML(run,T){
 if(!T||T.error)return `<div class="muted">${run}: ${T?esc(T.error):'no timeline yet'}</div>`;
 const W=860,LH=13,P={l:56,r:12,t:6,b:22},G=T.gantt.length,H=P.t+P.b+G*LH;
 let xmax=1;for(const g of T.gantt)xmax=Math.max(xmax,g.graded??0,g.sampled??0);for(const x of T.train)xmax=Math.max(xmax,x);
 for(const p of T.series)xmax=Math.max(xmax,p[0]);xmax=Math.ceil(xmax);
 const X=v=>P.l+v/xmax*(W-P.l-P.r);
 let g=`<div class="muted" style="margin:4px 0">${run} · step ${T.step} (in progress or newest) · ${T.requests} grading requests · grading run p50 ${T.grade_run_p50_s??'–'} s, queue wait p50 ${T.grade_wait_p50_s??'–'} s</div>`;
 g+=`<div class="legend" style="margin:4px 0"><span><i class="sw" style="border-color:${PH.s};border-top-width:6px"></i>sampling (Tinker)</span><span><i class="sw" style="border-color:${PH.g};border-top-width:6px"></i>grading (sampled → all graded)</span><span><i class="sw" style="border-color:${PH.t}"></i>training request sent (4 groups each)</span></div>`;
 g+=`<svg width="${W}" height="${H}">`;
 for(let m=0;m<=xmax;m+=Math.max(1,Math.ceil(xmax/12))){g+=`<line x1="${X(m)}" x2="${X(m)}" y1="${P.t}" y2="${H-P.b}" stroke="#ecebe7"/><text x="${X(m)}" y="${H-6}" font-size="10" text-anchor="middle" fill="#4a4945">${m} min</text>`}
 T.gantt.forEach((r,i)=>{const y=P.t+i*LH;g+=`<text x="${P.l-6}" y="${y+9}" font-size="10" text-anchor="end" fill="#4a4945">group ${r.group}</text>`;
  const se=r.sampled??null;g+=`<rect x="${X(r.s)}" y="${y+1}" width="${Math.max(1,X(se??xmax)-X(r.s))}" height="${LH-3}" fill="${PH.s}" opacity="${se==null?0.45:1}" data-t="group ${r.group}<br>sampling ${r.s.toFixed(1)}–${se==null?'…':se.toFixed(1)} min"/>`;
  if(se!=null){const ge=r.graded;g+=`<rect x="${X(se)}" y="${y+1}" width="${Math.max(1,X(ge??xmax)-X(se))}" height="${LH-3}" fill="${PH.g}" opacity="${ge==null?0.45:1}" data-t="group ${r.group}<br>grading ${se.toFixed(1)}–${ge==null?'…':ge.toFixed(1)} min"/>`}});
 for(const x of T.train)g+=`<line x1="${X(x)}" x2="${X(x)}" y1="${P.t}" y2="${H-P.b}" stroke="${PH.t}" stroke-width="2"><title>training request ${x.toFixed(1)} min</title></line><circle cx="${X(x)}" cy="${P.t+4}" r="6" fill="transparent" data-t="training request sent at ${x.toFixed(1)} min"/>`;
 g+='</svg>';
 // grading concurrency
 const H2=120,P2={l:56,r:12,t:6,b:20};let ymax=1;for(const p of T.series)ymax=Math.max(ymax,p[1],p[2]);
 const Y2=v=>H2-P2.b-v/ymax*(H2-P2.t-P2.b);
 g+=`<div class="muted" style="margin:8px 0 2px">programs running (blue) and waiting for a grading slot (orange)</div><svg width="${W}" height="${H2}">`;
 for(const f of [0,0.5,1]){const v=Math.round(ymax*f);g+=`<line x1="${P2.l}" x2="${W-P2.r}" y1="${Y2(v)}" y2="${Y2(v)}" stroke="#ecebe7"/><text x="${P2.l-6}" y="${Y2(v)+4}" font-size="10" text-anchor="end" fill="#4a4945">${v}</text>`}
 for(const [k,c] of [[1,PH.s],[2,PH.g]])g+=`<polyline fill="none" stroke="${c}" stroke-width="2" points="${T.series.map(p=>X(p[0])+','+Y2(p[k])).join(' ')}"/>`;
 for(const p of T.series)g+=`<circle cx="${X(p[0])}" cy="${Y2(p[1])}" r="7" fill="transparent" data-t="${p[0].toFixed(1)} min<br>running ${p[1]} · waiting ${p[2]}"/>`;
 g+='</svg>';
 // per-step overview (stacked minutes)
 const ov=T.overview,BW=860,RH=16;let mx=1;for(const o of ov)mx=Math.max(mx,(o.sampling||0)+(o.grading_tail||0)+(o.train_tail||0));
 g+=`<div class="muted" style="margin:10px 0 2px">minutes per step: sampling → grading tail (after the last rollout is sampled) → training tail (before the next step starts)</div><svg width="${BW}" height="${ov.length*RH+6}">`;
 ov.forEach((o,i)=>{let x=56;const y=i*RH+2;g+=`<text x="50" y="${y+11}" font-size="10" text-anchor="end" fill="#4a4945">step ${o.step}</text>`;
  for(const [k,c] of [['sampling',PH.s],['grading_tail',PH.g],['train_tail',PH.t]]){const v=o[k];if(v==null||v<=0)continue;const w=v/mx*(BW-70);
   g+=`<rect x="${x}" y="${y}" width="${Math.max(1,w-2)}" height="${RH-4}" rx="2" fill="${c}" data-t="step ${o.step}<br>${k.replace('_',' ')} ${v} min"/>`;x+=w}
  g+=`<text x="${x+4}" y="${y+11}" font-size="10" fill="#2b2a27">${((o.sampling||0)+(o.grading_tail||0)+(o.train_tail||0)).toFixed(0)} min${o.complete?'':' (running)'}</text>`});
 g+='</svg><table style="margin-top:6px"><tr><th>step</th><th>sampling min</th><th>grading tail min</th><th>training tail min</th></tr>'+
  ov.map(o=>`<tr><td>${o.step}</td><td>${o.sampling??'–'}</td><td>${o.grading_tail??'–'}</td><td>${o.train_tail??'–'}</td></tr>`).join('')+'</table>';
 return g}
document.addEventListener('mousemove',e=>{const t=e.target.dataset&&e.target.dataset.t;if(t){tip.innerHTML=t;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px'}else tip.style.display='none'});
async function tick(){try{render(await (await fetch('data.json',{cache:'no-store'})).json())}catch(e){document.getElementById('upd').textContent='update failed: '+e}}
tick();setInterval(tick,30000);
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/rollout?'):
            from urllib.parse import parse_qs, urlparse
            q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            d = rollout_full(q.get('run', ''), q.get('step', ''), q.get('i', ''))
            if d is None:
                self.send_error(404)
                return
            body, ctype = json.dumps(d).encode(), 'application/json'
        elif self.path.startswith('/data.json'):
            body, ctype = json.dumps(SNAP).encode(), 'application/json'
        elif self.path in ('/', '/index.html'):
            body, ctype = PAGE.encode(), 'text/html; charset=utf-8'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    SNAP = collect()
    threading.Thread(target=loop, daemon=True).start()
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
