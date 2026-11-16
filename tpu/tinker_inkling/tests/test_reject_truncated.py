"""reject_truncated_no_reward: a zero-reward rollout cut off at the FIXED total budget (inside the context window)
must be dropped; a zero-reward rollout that stopped early must be kept; a valid long one kept."""
import ast, os, types
src = open('/scratch/gpfs/ZHUANGL/sk7524/discover-tinker-inkling/ttt_discover/rl/train.py').read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == 'reject_truncated_no_reward')


class TG:
    def __init__(self, trajectories_G, final_rewards_G, metrics_G):
        self.trajectories_G, self.final_rewards_G, self.metrics_G = trajectories_G, final_rewards_G, metrics_G

    def get_total_rewards(self):
        return self.final_rewards_G


ns = dict(os=os, TrajectoryGroup=TG, safezip=zip, list=list, dict=dict, float=float)
exec(ast.get_source_segment(src, fn).replace('list[TrajectoryGroup]', 'list').replace('tuple[list, dict[str, float]]', 'tuple'), ns)


def traj(prompt, gen):
    tr = types.SimpleNamespace(ac=types.SimpleNamespace(tokens=[0] * gen), ob=types.SimpleNamespace(to_ints=lambda: [0] * prompt))
    return types.SimpleNamespace(transitions=[tr])


group = TG([traj(4000, 32000), traj(4000, 9000), traj(4000, 32000), traj(4000, 15000)], [0.0, 0.0, 2.63, 2.6], [{}] * 4)
os.environ.pop('TTD_FIXED_THINK_TOKENS', None)
kept, m = ns['reject_truncated_no_reward']([group], 40960)
print('without fixed budget:', m, '(cut-off rollout NOT detected -> the bug)')
os.environ.update(TTD_FIXED_THINK_TOKENS='24000', TTD_FIXED_TOTAL_TOKENS='32000')
kept, m = ns['reject_truncated_no_reward']([group], 40960)
print('with fixed budget:   ', m, '| kept rewards', kept[0].final_rewards_G)
assert m['reject_truncated/dropped_trajs'] == 1 and kept[0].final_rewards_G == [0.0, 2.63, 2.6]
print('OK: the zero-reward rollout cut at 32k is dropped; early failure and valid rollouts kept')
