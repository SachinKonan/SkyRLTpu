"""Science rewards in the unchanged native Discover training loop."""
import asyncio
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


def task_prompt(task, *, include_starter=True):
    name = 'prompts/rendered/routing.txt' if task == 'routing' else 'prompts/placement-jax-v6e.txt'
    cpu = task == 'placement' and os.environ.get('SCIENCE_PLACEMENT_BACKEND', 'tpu') == 'cpu'
    if cpu:
        name = 'prompts/placement-jax-cpu.txt'
    prompt = (Path(__file__).parent / name).read_text()
    if task == 'placement' and not cpu and placement_accelerator() == 'tpu-v4-64':
        prompt = prompt.replace('TPU v6e chip', 'TPU v4 chip')
    if not include_starter:
        marker = ('Valid initial policy block:' if task == 'routing' else
                  'Starting implementation (replace with your improved algorithm):')
        instructions, separator, _ = prompt.partition(marker)
        if not separator:
            raise ValueError(f'{task} prompt is missing its starter boundary')
        prompt = instructions.rstrip()
    if task == 'routing':
        from .routing_suite import validate_suite
        if validate_suite(os.environ.get('SCIENCE_ROUTING_SUITE', 'full')) == 'q20':
            prompt = ('Benchmark scope: Q20 ONLY, the 24 pinned circuits on the Q20 coupling graph. '
                      'Only their total added SWAPs S determines reward: max(1e-6,22714/(22714+S)). '
                      'Lower SWAPs and higher reward are better. All 24 cases must pass. '
                      'Willow and Heron are not evaluated in this experiment.\n\n') + prompt
    return prompt


def candidate_prompt(task, code='', feedback='', *, repair=False):
    prompt = task_prompt(task, include_starter=not bool(code))
    if code:
        instruction = ('Repair this invalid program using the grading error. First make it valid; '
                       'return a complete replacement program.' if repair else 'Improve this candidate.')
        prompt += ('\nPrevious candidate:\n```python\n' + code + '\n```\n'
                   + 'Grader feedback:\n' + feedback[-3000:] + '\n' + instruction + '\n')
    return prompt


async def evaluate(task, source, timeout):
    connect()
    root = os.environ['SCIENCE_WORKER_ROOT']
    refs = []
    try:
        if task == 'routing':
            from .ray_cpu import grade
            refs.append(grade.options(scheduling_strategy='SPREAD').remote(
                'routing', source, root, admission_timeout_s=timeout,
                slots_per_host=int(os.environ.get('SCIENCE_ROUTING_SLOTS_PER_HOST', '2')),
                routing_suite=os.environ.get('SCIENCE_ROUTING_SUITE','full')))
        elif task == 'placement' and os.environ.get('SCIENCE_PLACEMENT_BACKEND', 'tpu') == 'cpu':
            from .placement_ray import grade_cpu_case
            from .placement_task import CASES
            for case in CASES:
                refs.append(grade_cpu_case.options(scheduling_strategy='SPREAD').remote(
                    source, case, root, admission_timeout_s=timeout,
                    slots_per_host=int(os.environ.get('SCIENCE_PLACEMENT_SLOTS_PER_HOST', '16'))))
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
        state = self.initial_state
        return candidate_prompt(self.problem_type, state.code, state.observation)

    async def _safe_grade(self, given_answer, step):
        ScienceTrainingReward(self.problem_type, self.log_path, self.eval_timeout,
                              self.num_cpus_per_task, self.eval_backend)
        result = await evaluate(self.problem_type, given_answer, self.eval_timeout)
        from .feedback import diagnostic_message, observation
        result = dict(result, stdout=observation(self.problem_type, result),
                      msg=diagnostic_message(self.problem_type, result))
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
