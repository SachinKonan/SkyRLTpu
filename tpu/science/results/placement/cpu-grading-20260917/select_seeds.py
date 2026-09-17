import json,hashlib
from pathlib import Path
from ttt_discover import State
from ttt_discover.tinker_utils.sampler import PUCTSampler
from tpu.science.training_env import PlacementTrainingEnv
from tpu.science.bootstrap import commit_rows,save
p=Path('.science/placement-cpu-20260917');selected={};report={}
for model in ['qwen','gemma','muse']:
 root=p/'source-seeds'/model if model!='muse' else Path('.science/placement-assessment-20260917/muse-final/bootstrap')
 roots={s.id:s for s in map(State.from_dict,json.loads((p/'source-seeds'/(model+'-roots.json')).read_text()))}
 rows=[json.loads(f.read_text()) for f in sorted(root.glob('layer-*/group-*/grade-*.json'))]
 out=p/'selected'/model;out.mkdir(parents=True,exist_ok=True)
 save(out/'puct_sampler_step_000000.json',dict(step=0,states=[s.to_dict() for s in roots.values()],initial_states=[s.to_dict() for s in roots.values()],puct_n={},puct_m={},puct_T=0))
 sampler=PUCTSampler(str(out/'puct_sampler.json'),env_type=PlacementTrainingEnv,problem_type='placement',batch_size=16,resume_step=0)
 commit_rows(sampler,roots,rows)
 pool=json.loads((out/'puct_sampler_step_000000.json').read_text());seeds=[s for s in pool['states'] if s.get('code')]
 selected[model]=seeds
 report[model]=dict(completed_grade_records=len(rows),valid_records=sum(r['correctness']==1 for r in rows),selected=len(seeds),maximum=32)
 print(model,report[model])
(p/'selected-seeds.json').write_text(json.dumps(selected,indent=2)+'\n');(p/'selection-report.json').write_text(json.dumps(report,indent=2)+'\n')
