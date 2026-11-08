"""Refresh explicitly registered direct-SSH trainers from live farm discovery.

This only publishes candidates. The borrower owns attestation, leases, adapter
upload, renewal and in-flight draining. An inventory failure must skip the tick,
not be passed here as an empty successful inventory.
"""
from concurrent.futures import ThreadPoolExecutor


def stable_completion(events, stable_urls):
    """Evidence of a completed request after attesting an adapter on this farm."""
    ready = {}
    for event in events:
        key = (event.get('service'), event.get('phase'))
        if key[0] not in stable_urls:
            continue
        if event.get('event') == 'borrow_ready' and event.get('sha256'):
            ready[key] = event
        elif event.get('event') == 'borrow_generated' and key in ready:
            return dict(ready=ready[key], generated=event)
    return None


def candidates(target, registration, farms):
    """Keep the dedicated endpoint first; offer only compatible cloud farms."""
    urls = list(dict.fromkeys(registration['stable_urls']))
    if registration.get('allow_spot', False):
        accepted = set(target.get('accepted_compatibility') or [])
        if target.get('compatibility_sha256'):
            accepted.add(target['compatibility_sha256'])
        # Do not let an arbitrary first cloud farm define a fresh trainer's
        # serving contract. The dedicated engine establishes it first.
        if accepted:
            eligible = []
            for farm in farms:
                if target['model'] not in farm.get('models', []):
                    continue
                if (farm.get('capabilities') or {}).get('compatibility_sha256') not in accepted:
                    continue
                owner = farm.get('owner_run')
                # Borrower lease ownership and ingress update fencing use
                # different instance tokens; read the actual lease owner.
                ours = (target.get('borrowing') or {}).get('owner_run')
                # An expired lease keeps its last owner_run in the farm's status; the farm is free, so it must
                # not stay reserved for a borrower instance that no longer exists (a stopped trainer).
                if owner and owner != ours and farm.get('state') != 'expired':
                    continue
                eligible.append(farm['url'])
            urls.extend(sorted(set(eligible) - set(urls)))
    limit = target.get('candidate_limit', 16)
    if type(limit) is not int or limit < len(registration['stable_urls']):
        raise ValueError('invalid candidate limit for dedicated farms')
    return urls[:limit]


def tick(registrations, farms, call, trim=None):
    """call(registration, body=None) performs an instance-checked GET or POST; trim(target, urls) -> urls."""
    def update(registration):
        run_id = registration['run_id']
        try:
            descriptor = call(registration)
            if not descriptor.get('ok'):
                return dict(run_id=run_id, state='unreachable', detail=descriptor)
            target = descriptor['result']
            if target.get('run_id') != run_id or target.get('model') != registration['model']:
                return dict(run_id=run_id, state='identity_mismatch')
            if not target.get('instance') or target.get('enabled') is not True:
                return dict(run_id=run_id, state='not_opted_in')
            if target.get('lease_scope') != 'run':
                return dict(run_id=run_id, state='unsupported_lease_scope')
            urls = candidates(target, registration, farms)
            if trim:
                urls = trim(target, urls)
            if target.get('urls') == urls:
                return dict(run_id=run_id, state='unchanged', urls=urls,
                            instance=target['instance'], borrowing=target.get('borrowing'))
            body = {k: target[k] for k in ('run_id', 'instance', 'model')}
            body.update(group=registration['group'], urls=urls)
            result = call(registration, body)
            verified = (result.get('ok') and result['result'].get('instance') == target['instance']
                        and result['result'].get('run_id') == run_id
                        and result['result'].get('urls') == urls)
            return dict(run_id=run_id, state='updated' if verified else 'update_failed',
                        urls=urls, instance=target['instance'],
                        detail=None if verified else result)
        except Exception as exc:
            return dict(run_id=run_id, state='error', detail=type(exc).__name__ + ': ' + str(exc))
    with ThreadPoolExecutor(max_workers=8) as workers:
        return list(workers.map(update, registrations))
