#!/bin/bash
# clean /dev/shm/fair, then 4 concurrent drivers (one per chip), each grading in fresh child processes
set -u
cd "$(dirname "$0")"
[ -d /dev/shm/fair ] && mv /dev/shm/fair /dev/shm/fair_old_$(date +%s)
rm -f run4.done
python3 drive.py 0 v2 rl dirpower dog psband random_feats > d0.log 2>&1 &
python3 drive.py 1 v2 rl dog dirpower random_feats dirpower8 > d1.log 2>&1 &
python3 drive.py 2 v2 rl random_feats dirpower_wide dirpower batch_violator > d2.log 2>&1 &
python3 drive.py 3 v1 rl dirpower psband dirpower8 dirpower > d3.log 2>&1 &
wait
touch run4.done
