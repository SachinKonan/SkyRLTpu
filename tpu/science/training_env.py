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


def task_prompt(task, *, include_starter=True, environment=None):
    environment = os.environ if environment is None else environment
    name = 'prompts/rendered/routing.txt' if task == 'routing' else 'prompts/placement-jax-v6e.txt'
    cpu = task == 'placement' and environment.get('SCIENCE_PLACEMENT_BACKEND', 'tpu') == 'cpu'
    if cpu:
        name = ('prompts/placement-fast-proxy-cpu-ibm17-v2.txt'
                if environment.get('SCIENCE_PLACEMENT_HELPER', 'none') == 'fast_proxy_v1'
                else 'prompts/placement-jax-cpu.txt')
    prompt = (Path(__file__).parent / name).read_text()
    if task == 'placement':
        from .challenge_contract import CASES, SUITE
        if environment.get('SCIENCE_PLACEMENT_SUITE', SUITE) != SUITE:
            raise ValueError('Circuit training requires the full IBM17 suite')
        prompt = prompt.replace('Trusted CPU grading has a separate 90-second cap; a case has a 300-second total cap.',
            'Trusted grading has a separate 180-second cap; each case has a 390-second worker envelope.')
        prompt = prompt.replace('The four-case suite has a 1,200-second execution cap, excluding queue wait.',
            'All 17 cases are required; queue waiting does not consume the candidate time budget.')
        prompt = (f'Benchmark scope: {SUITE}. Every submission is evaluated on ALL 17 IBM cases: '
                  + ', '.join(CASES) + '. The scientific objective is their equally weighted '
                  'arithmetic mean proxy cost, lower is better. No subset or baseline normalization. '
                  'Every case must be legal with zero hard-macro overlaps. Any invalid or missing '
                  'case gives reward zero; otherwise reward = max(1e-6, 1/(1+mean_proxy_cost)).\n\n') + prompt
    if cpu and environment.get('SCIENCE_PLACEMENT_RUNTIME') == 'cpu300-4g-v1':
        prompt=prompt.replace('8 GiB', '4 GiB')
        import re
        prompt=re.sub(r'Each case has a hard 180-second candidate deadline.*?Return before it expires\.',
            'Each case supplies 300 seconds of search time to place(). Return your best legal positions '
            'before this budget expires. The external candidate cap is 310 seconds, including startup and output.',
            prompt,flags=re.S)
        prompt=prompt.replace('390-second', '510-second')
    if task == 'placement' and not cpu:
        accelerator = environment['SCIENCE_ACCELERATOR']
        if accelerator not in ('tpu-v4-64', 'tpu-v6e-32'):
            raise ValueError('unsupported science placement accelerator')
        if accelerator == 'tpu-v4-64':
            prompt = prompt.replace('TPU v6e chip', 'TPU v4 chip')
    if not include_starter:
        marker = ('Valid initial policy block:' if task == 'routing' else
                  'Starting implementation (replace with your improved algorithm):')
        instructions, separator, _ = prompt.partition(marker)
        if not separator:
            raise ValueError(f'{task} prompt is missing its starter boundary')
        prompt = instructions.rstrip()
    if cpu and environment.get('SCIENCE_PLACEMENT_STARTS') == 'abuplace-xplace-three-starts-v1':
        prompt += ('\n\nAdditional starting layouts: problem["starting_layouts"] has shape [3,M,2], '
            'in the order problem["starting_names"]: off, rudy, rudy_hv. These are AbuPlace\'s '
            'three Xplace global-placement configurations, independently legalized and verified. '
            'They do not include AbuPlace\'s later refinement pipeline. '
            'problem["starting_scores"] has shape [3,4], with columns named by '
            'problem["starting_score_columns"]: proxy_cost, wirelength_cost, density_cost, congestion_cost. '
            'All starts use the same circuit and object ordering. initial_positions remains the '
            'original verified legal fallback. You may choose, copy, or refine any start. '
            'Do not assume the lowest initial cost has the best potential after optimization. '
            'Consider brief exploration of several starts, then allocate the remaining time to '
            'promising layouts. Preserve your best legal result across all attempts. '
            'Perturbations, swaps and coarse-to-fine moves may escape local minima; check legality '
            'and evaluate their actual benefit. Averaging layouts does not preserve legality. '
            'There is ONE shared 300-second search budget per case across all starts, not 300 '
            'seconds per start. Precomputation is excluded; evaluator construction and search '
            'consume your budget. Return only the single best legal layout you found.\n')
    if task == 'routing':
        prompt = ('Grader feedback includes a compact per-case SWAP table with columns named '
                  'in case_columns, plus per-topology totals. baseline_swaps is the fixed SABRE '
                  'reference; delta_swaps = candidate minus baseline, so negative is better. '
                  'Case IDs identify diagnostics only; do not hardcode cases. The single aggregate '
                  'reward remains the optimization objective. Failed execution may stop before '
                  'case counts are available; use the reported error in that situation.\n\n') + prompt
        from .routing_suite import validate_suite
        if validate_suite(environment.get('SCIENCE_ROUTING_SUITE', 'full')) == 'q20':
            prompt = ('Benchmark scope: Q20 ONLY, the 24 pinned circuits on the Q20 coupling graph. '
                      'Only their total added SWAPs S determines reward: max(1e-6,22714/(22714+S)). '
                      'Lower SWAPs and higher reward are better. All 24 cases must pass. '
                      'Willow and Heron are not evaluated in this experiment.\n\n') + prompt
    return prompt


def candidate_prompt(task, code='', feedback='', *, repair=False):
    from .feedback import feedback_limit
    prompt = task_prompt(task, include_starter=not bool(code))
    if code:
        instruction = ('Repair this invalid program using the grading error. First make it valid; '
                       'return a complete replacement program.' if repair else 'Improve this candidate.')
        prompt += ('\nPrevious candidate:\n```python\n' + code + '\n```\n'
                   + 'Grader feedback:\n' + feedback[-feedback_limit(task):] + '\n' + instruction + '\n')
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
            from .placement_resources import contract
            resources = contract(int(os.environ.get('SCIENCE_PLACEMENT_SLOTS_PER_HOST', '32'))) if os.environ.get('SCIENCE_PLACEMENT_RUNTIME') == 'cpu300-4g-v1' else None
            for case in CASES:
                refs.append(grade_cpu_case.options(scheduling_strategy='SPREAD', **(dict(memory=4*1024**3) if resources else {})).remote(
                    source, case, root, admission_timeout_s=timeout,
                    slots_per_host=int(os.environ.get('SCIENCE_PLACEMENT_SLOTS_PER_HOST', '16')),
                    helper=os.environ.get('SCIENCE_PLACEMENT_HELPER', 'none'),
                    **(dict(resource_contract=resources) if resources else {})))
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
        if self.problem_type == 'placement':
            from .placement_suite_guard import validate_state
            validate_state(state.to_dict())
        return candidate_prompt(self.problem_type, state.code, state.observation)

    async def _safe_grade(self, given_answer, step):
        ScienceTrainingReward(self.problem_type, self.log_path, self.eval_timeout,
                              self.num_cpus_per_task, self.eval_backend)
        if (self.problem_type == 'placement'
                and os.environ.get('SCIENCE_PLACEMENT_RUNTIME') == 'cpu300-4g-v1'):
            from .grading_dedup import grade_once
            scope = (step, self.eval_timeout, self.eval_backend,
                     tuple((key, os.environ.get(key)) for key in (
                         'TTD_RUN_DIR', 'RAY_NAMESPACE', 'SCIENCE_WORKER_ROOT',
                         'SCIENCE_PLACEMENT_RUNTIME', 'SCIENCE_PLACEMENT_BACKEND',
                         'SCIENCE_PLACEMENT_HELPER', 'SCIENCE_PLACEMENT_SUITE', 'SCIENCE_PLACEMENT_STARTS')))
            result, reused = await grade_once(scope, given_answer,
                lambda: evaluate(self.problem_type, given_answer, self.eval_timeout))
            result['metrics'] = dict(result.get('metrics', {}), grading_dedup_hit=reused)
        else:
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
