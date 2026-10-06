"""TTT-Discover environments backed by the workload's distributed CPU graders."""
import asyncio
import os
from pathlib import Path
import ray
from ttt_discover import Environment, State
from ttt_discover.environments.base_reward_evaluator import BaseRewardEvaluator
from ttt_discover.tinker_utils.dataset_builder import VerifyResult
from .ray_cpu import grade
from .rewards import invalid


TASKS={'portfolio_v1':'portfolio','portfolio_v2':'portfolio_v2','qubit_routing_v1':'routing'}


def submit(problem_type,code):
    if not ray.is_initialized():
        raise RuntimeError('initialize the existing workload Ray cluster before science discovery')
    root=os.environ['SCIENCE_WORKER_ROOT']
    return grade.options(runtime_env={'env_vars':{'PYTHONPATH':root}},
        scheduling_strategy='SPREAD').remote(TASKS[problem_type],code,root)


class ScienceReward(BaseRewardEvaluator):
    def __init__(self,problem_type,log_dir,eval_timeout,num_cpus_per_task,eval_backend='ray'):
        if eval_backend!='ray' or num_cpus_per_task!=4:
            raise ValueError('science v1 requires the Ray backend and four CPUs per candidate')
        expected=1800 if problem_type=='qubit_routing_v1' else 300
        if eval_timeout!=expected:raise ValueError(f'this task requires eval_timeout={expected}')
        self.problem_type=problem_type

    def get_reward(self,code,state):
        ref=submit(self.problem_type,code)
        try:return ray.get(ref,timeout=7800)
        finally:ray.cancel(ref,force=False)


class ScienceEnv(Environment):
    reward_function=ScienceReward
    state_type=State

    def get_question(self):
        task=TASKS[self.problem_type]
        prompt=(Path(__file__).with_name('prompts')/'rendered'/f'{task}.txt').read_text()
        if self.initial_state.code:
            prompt+='\nPrevious valid program (normalized reward '+str(self.initial_state.value)+'):\n```python\n'+self.initial_state.code+'\n```\nImprove this program under the same rules.\n'
        return prompt

    def _should_keep_code_separators(self):return False

    def _build_metrics(self,outs,correct_format,message,parsed_code):
        metrics=super()._build_metrics(outs,correct_format,message,parsed_code)
        metrics.update({'science/'+k:v for k,v in outs.metrics.items()})
        return metrics

    async def _safe_grade(self,given_answer,step):
        # Cancel the actual Ray task when the coroutine times out or is cancelled;
        # an asyncio timeout around a background grading thread would leave it alive.
        ScienceReward(self.problem_type,self.log_path,self.eval_timeout,self.num_cpus_per_task,self.eval_backend)
        ref=submit(self.problem_type,given_answer)
        try:
            result=await asyncio.wait_for(asyncio.wrap_future(ref.future()),timeout=self.timeout)
        except asyncio.CancelledError:raise
        except Exception as exc:result=invalid(f'{type(exc).__name__}: {exc}',phase='ray')
        finally:ray.cancel(ref,force=False)
        return VerifyResult(**{k:result[k] for k in ('reward','correctness','raw_score','msg','result_construction','stdout','metrics')})


class PortfolioAllocationEnv(ScienceEnv):pass


class QubitRoutingEnv(ScienceEnv):pass
