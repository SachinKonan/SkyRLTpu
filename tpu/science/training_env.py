"""Science rewards in the unchanged native Discover training loop."""
import asyncio
import json
import os
from pathlib import Path
import threading

import ray
from ttt_discover import Environment, State
from ttt_discover.environments.base_reward_evaluator import BaseRewardEvaluator
from ttt_discover.tinker_utils.dataset_builder import VerifyResult


class ScienceInfrastructureError(RuntimeError):
    abort_training_step = True


_init_lock = threading.Lock()


def connect():
    with _init_lock:
        if not ray.is_initialized():
            ray.init(address=os.environ['RAY_ADDRESS'], namespace=os.environ['RAY_NAMESPACE'])


def placement_group_name(run_id):
    return 'science-placement-' + run_id


def placement_accelerator():
    accelerator = os.environ['SCIENCE_ACCELERATOR']
    if accelerator not in ('tpu-v4-64', 'tpu-v6e-32'):
        raise ValueError('unsupported science placement accelerator')
    return accelerator


def task_prompt(task):
    name = 'prompts/rendered/routing.txt' if task == 'routing' else 'prompts/placement-jax-v6e.txt'
    prompt = (Path(__file__).parent / name).read_text()
    if task == 'placement' and placement_accelerator() == 'tpu-v4-64':
        prompt = prompt.replace('TPU v6e chip', 'TPU v4 chip')
    return prompt


async def evaluate(task, source, timeout):
    connect()
    root = os.environ['SCIENCE_WORKER_ROOT']
    refs = []
    try:
        if task == 'routing':
            from .ray_cpu import grade
            refs.append(grade.options(scheduling_strategy='SPREAD').remote(
                'routing', source, root, admission_timeout_s=timeout))
        elif task == 'placement':
            from ray.util.placement_group import get_placement_group
            from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
            from .placement_ray import grade_case
            from .placement_task import CASES
            group = get_placement_group(placement_group_name(os.environ['RAY_NAMESPACE']))
            for case in CASES:
                refs.append(grade_case.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=group, placement_group_bundle_index=-1,
                    placement_group_capture_child_tasks=False)).remote(
                        source, case, root, accelerator=placement_accelerator()))
        else:
            raise ValueError('unsupported science task')
        results = await asyncio.wait_for(asyncio.gather(
            *(asyncio.wrap_future(ref.future()) for ref in refs)), timeout=timeout)
        if task == 'routing':
            result = results[0]
        else:
            from .challenge_contract import aggregate
            result = aggregate(results)
        if result['reward'] != result['raw_score']:
            raise ValueError('science state ranking must match reward direction')
        return result
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        raise ScienceInfrastructureError(f'{task} Ray grading failed: {type(exc).__name__}: {exc}') from exc
    finally:
        for ref in refs:
            ray.cancel(ref, force=True)


class ScienceTrainingReward(BaseRewardEvaluator):
    def __init__(self, problem_type, log_dir, eval_timeout, num_cpus_per_task, eval_backend='local'):
        if problem_type not in ('routing', 'placement') or num_cpus_per_task != 4 or eval_backend != 'local':
            raise ValueError('Science training uses four-CPU Ray tasks inside the local outer evaluator')
        self.task, self.timeout = problem_type, eval_timeout

    def get_reward(self, code, state):
        return asyncio.run(evaluate(self.task, code, self.timeout))


class ScienceTrainingEnv(Environment):
    reward_function = ScienceTrainingReward
    state_type = State

    def _should_keep_code_separators(self):
        return False

    def get_question(self):
        prompt = task_prompt(self.problem_type)
        state = self.initial_state
        if state.code:
            prompt += ('\nPrevious candidate:\n```python\n' + state.code + '\n```\n'
                       + 'Grader feedback:\n' + state.observation[-3000:] + '\nImprove this candidate.\n')
        return prompt

    async def _safe_grade(self, given_answer, step):
        ScienceTrainingReward(self.problem_type, self.log_path, self.eval_timeout,
                              self.num_cpus_per_task, self.eval_backend)
        result = await evaluate(self.problem_type, given_answer, self.eval_timeout)
        # Return scientific diagnostics alongside the bounded learning reward.
        metrics = result['metrics']
        feedback = {k: v for k, v in metrics.items() if k in (
            'mean_proxy_cost', 'weighted_candidate_cnots', 'weighted_baseline_cnots',
            'swaps', 'added_cnots', 'improvement', 'case_count', 'total_seconds')}
        result = dict(result, stdout=json.dumps(dict(reward=result['reward'], message=result['msg'],
                                                    metrics=feedback))[:3000])
        return VerifyResult(**{k: result[k] for k in (
            'reward', 'correctness', 'raw_score', 'msg', 'result_construction', 'stdout', 'metrics')})

    def _build_metrics(self, outs, correct_format, message, parsed_code):
        metrics = super()._build_metrics(outs, correct_format, message, parsed_code)
        metrics.update({'science/' + k: v for k, v in outs.metrics.items()
                        if isinstance(v, (int, float, str, bool))})
        return metrics


class RoutingTrainingEnv(ScienceTrainingEnv):
    pass


class PlacementTrainingEnv(ScienceTrainingEnv):
    pass
