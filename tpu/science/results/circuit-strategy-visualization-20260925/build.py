import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32')
out=root/'tpu/science/results/circuit-strategy-visualization-20260925'
names=['muse0','qwen0','gemma0','muse1','qwen1'];labels=['Muse Gen-0','Qwen Gen-0','Gemma Gen-0','Muse Gen-1','Qwen Gen-1']
data=[]
for n,label in zip(names,labels):
 s=json.loads((root/f'.science/strategy-audit/{n}.json').read_text());cases=json.loads(s['observation'])['metrics']['cases']
 data.append(dict(name=label,id=s['id'],cost=1/s['value']-1,cases=cases,components=[np.mean([c[k] for c in cases])*w for k,w in [('wirelength_cost',1),('density_cost',.5),('congestion_cost',.5)]]))
(out/'data.json').write_text(json.dumps(data,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(14,5),gridspec_kw={'width_ratios':[1,1.2]})
colors=['#a7a5bb','#9bbad1','#aa83d1','#43aa9d','#327dca']
y=np.arange(5)
axs[0].barh(y,[d['cost'] for d in data],color=colors,height=.6)
axs[0].set_yticks(y,labels);axs[0].invert_yaxis();axs[0].set_xlim(.90,.975);axs[0].set_xlabel('Mean placement cost (lower is better)')
for i,d in enumerate(data):axs[0].text(d['cost']+.0007,i,f"{d['cost']:.6f}",va='center',fontsize=10)
axs[0].axvline(.95326663466,color='#a65732',ls='--',label='AbuPlace: 0.953267')
axs[0].axvline(.9611068438,color='#666',ls=':',label='ArchGen*: 0.961107')
axs[0].legend(loc='lower right',fontsize=8);axs[0].set_title('Best discovered programs',loc='left',fontweight='bold')
g=np.array(data[2]['components']);x=np.arange(3);width=.34
for i,idx in enumerate([3,4]):
 delta=np.array(data[idx]['components'])-g
 axs[1].bar(x+(i-.5)*width,delta,width,color=colors[idx],label=labels[idx])
 for xx,v in zip(x+(i-.5)*width,delta):axs[1].annotate(f'{v:+.4f}',(xx,v),xytext=(0,5 if v>=0 else -14),textcoords='offset points',ha='center',fontsize=9)
axs[1].axhline(0,color='#777',lw=.8);axs[1].set_xticks(x,['Wirelength','½ Density','½ Congestion']);axs[1].set_ylabel('Change in cost contribution vs Gemma Gen-0');axs[1].legend();axs[1].set_title('Where the Gen-1 improvement comes from',loc='left',fontweight='bold');axs[1].set_ylim(-.025,.013)
fig.suptitle('Circuit placement: cross-model strategy evolution',fontsize=18,fontweight='bold',y=1.02)
fig.text(.01,-.035,'17-case means • Snapshot: Qwen Gen-1 step 5, Muse Gen-1 step 3 • Bars start at 0.90 in left panel.\n*ArchGen is diagnostic: 11 strict boundary failures. Cached baselines use different compute budgets. Component values are rounded in source feedback.',fontsize=9,color='#555')
fig.tight_layout();fig.savefig(out/'strategy-evolution.png',dpi=180,bbox_inches='tight');fig.savefig(out/'strategy-evolution.svg',bbox_inches='tight');plt.close(fig)
html='''<!doctype html><meta charset="utf-8"><title>Circuit strategy evolution</title><style>body{font:16px system-ui;background:#f5f7fb;color:#182336;max-width:1150px;margin:40px auto;padding:0 24px}h1{font-size:32px}section{background:white;padding:24px;border-radius:16px;margin:20px 0;box-shadow:0 3px 15px #1525450a}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.card{border:1px solid #ddd;border-radius:12px;padding:16px}small,.note{color:#617084}img{width:100%}button{padding:10px 16px;margin:4px;border:1px solid #bbb;border-radius:8px;background:white;cursor:pointer}button.active{background:#182336;color:white}td,th{padding:8px;text-align:right;border-bottom:1px solid #eee}td:first-child,th:first-child{text-align:left}table{width:100%;border-collapse:collapse}.bar{height:12px;display:inline-block;background:#327dca}.neg{background:#db7e64}</style>
<h1>How the placement strategy evolved</h1><p>Three independent Gen-0 searches. Two branches evolving Gemma’s program pool.</p>
<section><div class="cards"><div class="card"><b>Muse Gen-0 · 0.952511</b><p>Random local moves; separate hard/soft macro sampling; shrinking radius and batch sizes.</p></div><div class="card"><b>Qwen Gen-0 · 0.940756</b><p>Staged annealing; shrinking candidate batches; perturb-and-restart when stalled.</p></div><div class="card"><b>Gemma Gen-0 · 0.931767</b><p>Probe starting layouts; connectivity-guided moves; increasingly favor congestion reduction.</p></div></div><p style="text-align:center">Gemma’s 567-program pool ↓ branches into two independent searches</p><div class="cards" style="grid-template-columns:1fr 1fr"><div class="card"><b>Muse Gen-1 · 0.929114</b><p>Direct child of Gemma’s final winner. Same-net pin proposals; up to 132 candidates; revised probing and temperature.</p><small>Muse’s own Gen-0 weights + fresh optimizer.</small></div><div class="card"><b>Qwen Gen-1 · 0.925049</b><p>Descends from another Gemma-pool branch. Ten proposals per batch; stronger congestion-weighted search acceptance; no start-probing phase.</p><small>Qwen’s own Gen-0 weights + fresh optimizer.</small></div></div></section>
<section><img src="strategy-evolution.svg" alt="Placement cost comparison and component changes"></section>
<section><h2>Which circuits improved?</h2><p>Select a program. Positive savings mean lower placement cost than Gemma Gen-0.</p><div id="buttons"></div><table><thead><tr><th>Case</th><th>Gemma Gen-0</th><th>Selected program</th><th>Cost saved</th><th>Improvement</th></tr></thead><tbody id="rows"></tbody></table></section>
<p class="note">Snapshot, not live telemetry. One seed per case, 300-second candidate placement budget with cached XPlace starts. These charts describe complete winning programs; they do not isolate the causal contribution of individual code edits. AbuPlace and ArchGen baseline compute budgets differ.</p>
<script>const data=DATA;const base=data[2];function draw(idx){document.querySelectorAll('button').forEach((b,i)=>b.className=i===idx?'active':'');const d=data[idx];document.getElementById('rows').innerHTML=base.cases.map(c=>{const v=d.cases.find(x=>x.case===c.case).proxy_cost;const delta=c.proxy_cost-v;return `<tr><td>${c.case}</td><td>${c.proxy_cost.toFixed(5)}</td><td>${v.toFixed(5)}</td><td>${delta.toFixed(5)}</td><td><span class="bar ${delta<0?'neg':''}" style="width:${Math.min(140,Math.abs(delta)*1200)}px"></span> ${(100*delta/c.proxy_cost).toFixed(2)}%</td></tr>`}).join('')}data.forEach((d,i)=>{const b=document.createElement('button');b.textContent=d.name;b.onclick=()=>draw(i);document.getElementById('buttons').append(b)});draw(4);</script>'''
(out/'index.html').write_text(html.replace('const data=DATA;', 'const data='+json.dumps(data)+';'))
print(out)
