"""Compatible farm discovery without workload-priority reservations."""


def hardware_priority(farm):
    """Prefer the measured faster farm hardware without disturbing leases."""
    identity = ' '.join(str(farm.get(key, '')) for key in
                        ('source_pool', 'source_cluster', 'source_name')).casefold()
    if 'v5p' in identity:
        return 0
    if 'v4-32' in identity or 'v4_32' in identity or 'v432' in identity:
        return 1
    return 2


def _ordered_candidates(job, farms):
    """Rotate for load spreading within each hardware tier, never across it."""
    ordered = []
    priorities = sorted({hardware_priority(farm) for farm in farms})
    for priority in priorities:
        tier = sorted((farm for farm in farms if hardware_priority(farm) == priority),
                      key=lambda farm: farm['job_id'])
        offset = job % len(tier)
        ordered.extend(tier[offset:] + tier[:offset])
    return ordered


def assignments(rows, farms, targets):
    """Offer free farms to all compatible runs, preserving existing leases.

    Candidate lists may overlap: the first successful atomic acquire wins.
    A borrower still holds only one lease. Two URLs are alternatives, bounded
    by the existing service-list validation, not two simultaneous reservations.
    No discovery update revokes an existing lease, including an unlisted owner's.
    """
    result = {job: [] for job in targets}
    available = []
    for farm in sorted(farms, key=lambda f: (hardware_priority(f), f['job_id'])):
        owner = (farm.get('owner_run') or '').split(':', 1)[0]
        occupied = farm.get('state') not in (None, 'unleased', 'expired') or farm.get('active', 0) > 0
        if occupied:
            for job, target in targets.items():
                if owner == target['run_id'] and target['model'] in farm['models'] and not result[job]:
                    result[job] = [farm['url']]
                    break
        else:
            available.append(farm)
    for job, target in targets.items():
        if result[job]:
            continue
        compatible = []
        required = target.get('compatibility_sha256')
        for farm in available:
            offered = (farm.get('capabilities') or {}).get('compatibility_sha256')
            if target['model'] in farm['models'] and (not required or required == offered):
                compatible.append(farm)
        if compatible:
            # Spread first choices within equal hardware. Hardware tiers retain
            # their measured ordering, so v5p always precedes v4-32.
            result[job] = [farm['url'] for farm in
                           _ordered_candidates(job, compatible)[:2]]
    return result
