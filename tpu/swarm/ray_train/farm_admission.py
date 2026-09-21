"""Non-preemptive model-matched reservations; AC2 precedes qubit routing."""


def workload_priority(name):
    name = name.lower()
    return 0 if 'ac2' in name else 1 if 'qubit' in name else 2


def assignments(rows, farms, targets):
    """Return one URL per run, preserving live owners including unlisted runs.

    Discovery chooses candidates; the farm's atomic lease remains authoritative.
    No assignment revokes an existing lease or modifies a running experiment.
    """
    by_id = {r['job_id']: r for r in rows}
    ordered = sorted(targets, key=lambda job: (
        workload_priority(targets[job].get('workload', by_id[job].get('run_id', ''))),
        -by_id[job].get('priority', 0), job))
    result = {job: [] for job in targets}
    available = []
    for farm in sorted(farms, key=lambda f: f['job_id']):
        owner = (farm.get('owner_run') or '').split(':', 1)[0]
        occupied = farm.get('state') not in (None, 'unleased', 'expired') or farm.get('active', 0) > 0
        if occupied:
            for job in ordered:
                target = targets[job]
                if owner == target['run_id'] and target['model'] in farm['models'] and not result[job]:
                    result[job] = [farm['url']]
                    break
        else:
            available.append(farm)
    # An outstanding AC2 run, including one waiting for a TPU, has precedence
    # over a new qubit reservation. Existing qubit owners are never preempted.
    ac2_pending = any(workload_priority(r.get('run_id', '')) == 0 and
                      r['status'] not in ('SUCCEEDED', 'CANCELLED') and not r['status'].startswith('FAILED')
                      for r in rows)
    for job in ordered:
        if result[job]:
            continue
        target = targets[job]
        if ac2_pending and workload_priority(target.get('workload', target['run_id'])) > 0:
            continue
        for index, farm in enumerate(available):
            if target['model'] in farm['models']:
                result[job] = [farm['url']]
                available.pop(index)
                break
    return result
