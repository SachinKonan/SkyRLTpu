"""Non-preemptive model-matched reservations; AC2 precedes RG-LRU."""


def workload_priority(name):
    name = name.lower()
    return 0 if 'ac2' in name else 1 if any(s in name for s in ('rglru', 'rg-lru', 'rg_lru')) else 2


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
    pending_ac2 = [r for r in rows if workload_priority(r.get('run_id', '')) == 0 and
                   r['status'] not in ('SUCCEEDED', 'CANCELLED') and not r['status'].startswith('FAILED')]
    ac2_pending = bool(pending_ac2)
    for job in ordered:
        if result[job]:
            continue
        target = targets[job]
        priority = workload_priority(target.get('workload', target['run_id']))
        if ac2_pending and priority > 0:
            # The spare same-model farm can serve RG only after that model's
            # AC2 has a reservation. Pending/unreachable AC2 retains precedence.
            matching_ac2 = [job for job in ordered if targets[job]['model'] == target['model']
                            and workload_priority(targets[job].get('workload', targets[job]['run_id'])) == 0]
            family = next((name for name in ('qwen', 'gemma', 'muse')
                           if name in target['model'].lower()), None)
            waiting = any((family is None or family in row.get('run_id', '').lower())
                          and not result.get(row['job_id']) for row in pending_ac2)
            if priority != 1 or waiting or not all(result[job] for job in matching_ac2):
                continue
        for index, farm in enumerate(available):
            required = target.get('compatibility_sha256')
            offered = (farm.get('capabilities') or {}).get('compatibility_sha256')
            if target['model'] in farm['models'] and (not required or required == offered):
                result[job] = [farm['url']]
                available.pop(index)
                break
    return result
