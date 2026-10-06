"""Compatible farm discovery without workload-priority reservations."""


def hardware_priority(farm):
    """Prefer the measured faster farm hardware without disturbing leases."""
    identity = ' '.join(str(farm.get(key, '')) for key in
                        ('source_pool', 'source_cluster', 'source_name')).casefold()
    if 'v6e-32' in identity or 'v6e32' in identity:
        return 0
    if 'v6e-8' in identity or 'v6e8' in identity:
        return 1
    if 'v5p' in identity:
        return 2
    if 'v4-32' in identity or 'v4_32' in identity or 'v432' in identity:
        return 3
    return 4


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


def candidate_limit(target):
    """Legacy targets get two alternatives; multi-lease targets ask for more."""
    if target.get('lease_scope') == 'run' and (target.get('target_leases') or 1) > 1:
        return max(1, int(target.get('candidate_limit') or 2))
    return 2


def multi_lease(target):
    return candidate_limit(target) > 2 or (target.get('target_leases') or 1) > 1


def accepted_contracts(target):
    accepted = target.get('accepted_compatibility') or []
    required = target.get('compatibility_sha256')
    return set(accepted) | ({required} if required else set())


def assignments(rows, farms, targets):
    """Offer free farms to all compatible runs, preserving existing leases.

    Candidate lists may overlap: the first successful atomic acquire wins.
    A legacy borrower holds one lease; its two URLs are alternatives. A
    remote-only target (``target_leases > 1``) receives every farm it already
    owns plus free compatible farms up to its ``candidate_limit`` and holds
    several of them at once. No discovery update revokes an existing lease,
    including an unlisted owner's; no fairness rule ranks runs against each
    other, the farm's atomic acquire decides.
    """
    result = {job: [] for job in targets}
    available = []
    for farm in sorted(farms, key=lambda f: (hardware_priority(f), f['job_id'])):
        owner = (farm.get('owner_run') or '').split(':', 1)[0]
        occupied = farm.get('state') not in (None, 'unleased', 'expired') or farm.get('active', 0) > 0
        if occupied:
            for job, target in targets.items():
                if owner == target['run_id'] and target['model'] in farm['models'] and (
                        not result[job] or (multi_lease(target) and len(result[job]) < candidate_limit(target))):
                    result[job].append(farm['url'])
                    break
        else:
            available.append(farm)
    for job, target in targets.items():
        limit = candidate_limit(target)
        if result[job] and (not multi_lease(target) or len(result[job]) >= limit):
            continue
        compatible = []
        accepted = accepted_contracts(target)
        for farm in available:
            offered = (farm.get('capabilities') or {}).get('compatibility_sha256')
            if target['model'] in farm['models'] and (not accepted or offered in accepted):
                compatible.append(farm)
        if compatible:
            # Spread first choices within equal hardware. Hardware tiers retain
            # their measured ordering: v6e-32, v6e-8, v5p, then v4-32.
            room = limit - len(result[job])
            result[job] += [farm['url'] for farm in _ordered_candidates(job, compatible)[:room]]
    return result
