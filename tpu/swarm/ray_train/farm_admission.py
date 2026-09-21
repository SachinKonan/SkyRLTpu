"""Compatible farm discovery without workload-priority reservations."""


def assignments(rows, farms, targets):
    """Offer free farms to all compatible runs, preserving existing leases.

    Candidate lists may overlap: the first successful atomic acquire wins.
    A borrower still holds only one lease. Two URLs are alternatives, bounded
    by the existing service-list validation, not two simultaneous reservations.
    No discovery update revokes an existing lease, including an unlisted owner's.
    """
    result = {job: [] for job in targets}
    available = []
    for farm in sorted(farms, key=lambda f: f['job_id']):
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
                compatible.append(farm['url'])
        if compatible:
            # Spread first choices without preferring a workload or reserving
            # capacity for a job that has not acquired a lease.
            offset = job % len(compatible)
            result[job] = (compatible[offset:] + compatible[:offset])[:2]
    return result
