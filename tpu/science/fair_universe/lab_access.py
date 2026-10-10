"""Access only the four hosts explicitly assigned in CODEX_HANDOVER.md."""
import argparse
import shlex
import subprocess
import sys

CLUSTER = 'tpuswarm-v5p32-east5a-erdos-3191'
SSH_CONFIG = '/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/' + CLUSTER
IPS = ['10.202.0.193', '10.202.0.178', '10.202.0.180', '10.202.0.197']
ROOT = '/dev/shm/fair-universe-codex'


def command(rank, script):
    if type(rank) is not int or rank not in range(4):
        raise ValueError('Only the four assigned lab hosts are allowed')
    if rank:
        script = ('ssh -F /dev/null -i ~/.ssh/sky-cluster-key -o StrictHostKeyChecking=no '
                  '-o ConnectTimeout=15 '+IPS[rank]+' '+shlex.quote(script))
    return ['ssh', '-F', SSH_CONFIG, '-o', 'ConnectTimeout=15', CLUSTER, script]


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--rank', type=int, required=True)
    a = p.parse_args()
    raise SystemExit(subprocess.call(command(a.rank, sys.stdin.read())))
