"""Fixed sampling budgets must survive early success and recovery."""
import asyncio
from types import SimpleNamespace

from tpu.swarm.ray_train import seed_bootstrap as bootstrap


def test_full_budget_ranks_late_candidates_and_resumes(tmp_path, monkeypatch):
    # Keep the journal test independent of the heavy training dependencies.
    import json
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    monkeypatch.setattr(bootstrap, 'save', save)
    config = SimpleNamespace(bootstrap_max_drafts=16, bootstrap_target_valid=4,
        bootstrap_group_size=4, bootstrap_max_groups=1, bootstrap_fixed_budget=True)
    calls = []
    async def generate(group):
        calls.append(group)
        return list(range(4))
    async def grade(group, sample, choice):
        name = f'bootstrap-{group:03d}-{sample:03d}'
        return dict(id=name, correctness=1,
                    state=dict(code=name, value=group * 4 + sample))
    rows, drafted = asyncio.run(bootstrap.collect(config, tmp_path, generate, grade))
    assert drafted == len(rows) == 16
    assert calls == [0, 1, 2, 3]
    selected = bootstrap.selected_states(rows, 4)
    assert [row['value'] for row in selected] == [15, 14, 13, 12]
    calls.clear()
    assert asyncio.run(bootstrap.collect(config, tmp_path, generate, grade)) == (rows, drafted)
    assert calls == []


def test_target_mode_still_stops_early():
    config = SimpleNamespace(bootstrap_target_valid=512, bootstrap_group_size=16,
                             bootstrap_max_groups=32)
    assert bootstrap.outstanding_limit(config, 512) == 0
    config.bootstrap_fixed_budget = True
    assert bootstrap.outstanding_limit(config, 512) == 32
