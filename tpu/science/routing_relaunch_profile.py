"""Build the three matched ten-step v4-64 profiles from reviewed source recipes."""
import argparse
import json
from pathlib import Path
from .routing_regrade import save

SOURCES={
 'gemma':'science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920-10step-deployment.json',
 'qwen':'science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920-10step-deployment.json',
 'muse':'capacity-v6e-muse-qubit-grpo-4e5-20260920-10step-deployment.json',
}


def build(model,seed_import,output):
    from tpu.swarm.ray_train.config import Config
    from .seed_pool import verify_pool
    root=Path(__file__).resolve().parents[2];folder=root/'tpu/swarm/ray_train/profiles'
    original=json.loads((folder/SOURCES[model]).read_text())
    report=json.loads((Path(seed_import)/'seed-import.json').read_text())
    if report['model']!=model:raise ValueError('model seed import mismatch')
    verify_pool(Path(seed_import)/'puct_sampler_step_000000.json',report['pool_sha256'])
    run=report['target_run'];config=json.loads(json.dumps(original))
    config.update(run_id=run,root='~/.cache/'+run,bucket='gs://sk7524-tinker-tpu-us-central2',
        science_routing_evaluator='parallel-v2',science_routing_slots_per_host=8,
        seed_pool_sha256=report['pool_sha256'],bootstrap_layers=0,bootstrap_only=False,
        bootstrap_all_hosts=False,checkpoint_resume=True,resume_min_checkpoint_step=0)
    # Keep compatible cache locations and all model/training/generation settings.
    # Eight program slots leave additional host memory for trainer initialization.
    config['cache']['reserve_gib']=240
    farm=json.loads((folder/'next-v6e-gemma-qubit-10step-20260921.json').read_text())['inference']
    config['inference'].update({k:v for k,v in farm.items() if k.startswith('external_pool_')})
    config['client_env']['SCIENCE_ROUTING_EVALUATOR']='parallel-v2'
    config['client_env']['NUM_EPOCHS']='10'
    config['client_env']['TTD_SICK_MARKER']='/home/gcpuser/.cache/'+run+'/runs/'+run+'/ENGINE-SICK'
    Config.from_dict(config).validate()
    def differences(before,after,path=''):
        if isinstance(before,dict) and isinstance(after,dict):
            result=[]
            for key in sorted(set(before)|set(after)):
                result.extend(differences(before.get(key),after.get(key),path+'.'+key if path else key))
            return result
        return [] if before==after else [dict(field=path,old=before,new=after)]
    save(output,config)
    save(str(output)+'.changes.json',dict(source_profile=SOURCES[model],target_run=run,
        source_manifest_sha256=report['source_manifest_sha256'],evaluator_sha256=report['evaluator_sha256'],
        seed_pool_sha256=report['pool_sha256'],differences=differences(original,config)))
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=tuple(SOURCES),required=True)
    p.add_argument('--seed-import',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    build(a.model,a.seed_import,a.output)
