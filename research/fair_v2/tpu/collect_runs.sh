#!/bin/bash
# collect_runs.sh: on host 0, wait for runs to finish then pull params/ref/log of each to /dev/shm/fair/pull/<run>.
declare -A HOST=( [d100]=10.202.0.134 [d101]=10.202.0.134 [d102]=10.202.0.134 [d103]=10.202.0.134
  [d104]=10.202.1.21 [d105]=10.202.1.21 [d106]=10.202.1.21 [d107]=10.202.1.21
  [d108]=10.202.1.22 [d109]=10.202.1.22 [d110]=10.202.1.22
  [w200]=10.202.1.23 [w201]=10.202.1.23 [w202]=10.202.1.23 [w203]=10.202.1.23 )
S="ssh -n -F /dev/null -o StrictHostKeyChecking=no -i $HOME/.ssh/sky-cluster-key"
runs="${@:-${!HOST[@]}}"
for r in $runs; do
  ip=${HOST[$r]}
  until $S $ip "grep -q finished /dev/shm/fair/runs/$r/train.log" 2>/dev/null; do sleep 20; done
  for f in params.npz ref.npz log.json; do bash /dev/shm/fair/code/fetch.sh $ip /dev/shm/fair/runs/$r/$f /dev/shm/fair/pull/$r/$f; done
done
echo all-collected
