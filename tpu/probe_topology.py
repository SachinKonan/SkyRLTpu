"""Full-slice TPU topology probe: print each worker's physical chip coords.

Run one instance per TPU worker (all concurrently), e.g. via
tpu/probe_tpu_topology.sh. Each process prints its host's chip coordinates;
the z coordinate gives the host's physical position in the slice, which
determines which worker pairs are ICI-adjacent (required for multi-host
sub-slices such as a 2-host train mesh on a v5p-32).

Usage: probe_topology.py <process_id> <coordinator_host:port> <num_processes>
"""

import socket
import sys

import jax

process_id = int(sys.argv[1])
coordinator = sys.argv[2]
num_processes = int(sys.argv[3])

jax.distributed.initialize(
    coordinator_address=coordinator,
    num_processes=num_processes,
    process_id=process_id,
)

print(f"PROBE host={socket.gethostname()} process_id={process_id}", flush=True)
for d in jax.local_devices():
    print(f"PROBE dev process={d.process_index} id={d.id} coords={d.coords}", flush=True)
