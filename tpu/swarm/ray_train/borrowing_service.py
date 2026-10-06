"""Render a user-systemd unit for an explicitly scoped borrowing supervisor.

Rendering does not install, start, stop, or alter any service. Use a frozen
checkout for --checkout, and an explicit environment file for SkyPilot paths.
"""
import argparse
from pathlib import Path


def quote(value):
    value = str(value)
    if '\n' in value or '\r' in value or '\0' in value:
        raise ValueError('systemd arguments must be single-line strings')
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'


def unit(checkout, python, environment, ssh_dir, farm_pool, trainer_pools, lock_file,
         *, farm_name_contains='inference-farm', farm_pool_max_job_id=None):
    if farm_pool_max_job_id is not None and (not farm_pool or farm_pool_max_job_id < 1):
        raise ValueError('farm_pool_max_job_id requires farm_pool and a positive job ID')
    if not trainer_pools:
        raise ValueError('at least one explicit trainer pool is required')
    if not farm_pool and not farm_name_contains.strip():
        raise ValueError('a farm pool or nonempty farm name selector is required')
    command = [python, '-u', '-m', 'tpu.swarm.ray_train.borrowing_supervisor',
               '--farm-name-contains', farm_name_contains,
               '--ssh-config-dir', ssh_dir, '--lock-file', lock_file, '--run-scoped-only']
    if farm_pool:
        command += ['--farm-pool', farm_pool]
    if farm_pool_max_job_id is not None:
        command += ['--farm-pool-max-job-id', str(farm_pool_max_job_id)]
    for pool in trainer_pools:
        command += ['--trainer-pool', pool]
    working = str(checkout)
    environment = str(environment)
    output = str(Path(lock_file).with_suffix('.log'))
    for path in (working, environment, output):
        if not path.startswith('/') or any(c in path for c in '\n\r\0'):
            raise ValueError('unit paths must be absolute single-line paths')
    return '\n'.join([
        '[Unit]', 'Description=SkyRL exclusive inference farm discovery',
        'After=network-online.target', 'StartLimitIntervalSec=0', '', '[Service]',
        'Type=simple', 'WorkingDirectory=' + working.replace('%', '%%'),
        'EnvironmentFile=' + environment.replace('%', '%%'), 'ExecStart=' + ' '.join(map(quote, command)),
        'StandardOutput=append:' + output.replace('%', '%%'), 'StandardError=inherit',
        'Restart=always', 'RestartSec=10', 'TimeoutStopSec=60',
        'KillMode=control-group', 'UMask=0077', '', '[Install]', 'WantedBy=default.target', ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkout', 'python', 'environment', 'ssh-dir', 'lock-file', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--farm-pool')
    parser.add_argument('--farm-pool-max-job-id', type=int)
    parser.add_argument('--farm-name-contains', default='inference-farm')
    parser.add_argument('--trainer-pool', action='append', required=True)
    args = parser.parse_args()
    # Do not resolve the virtualenv Python symlink to the global interpreter.
    text = unit(args.checkout.resolve(), args.python.absolute(), args.environment.resolve(),
                args.ssh_dir.resolve(), args.farm_pool, args.trainer_pool, args.lock_file.resolve(),
                farm_name_contains=args.farm_name_contains,
                farm_pool_max_job_id=args.farm_pool_max_job_id)
    # Refuse to silently replace an existing service definition.
    with args.output.open('x') as stream:
        stream.write(text)


if __name__ == '__main__':
    main()
