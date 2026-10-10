#!/bin/bash
# wave2.sh (host 0): when a first-wave run finishes, start the next job on the same host/chip.
S="ssh -n -F /dev/null -o StrictHostKeyChecking=no -i $HOME/.ssh/sky-cluster-key"
# old_run|ip|chip|runner|new_name|args
PLAN=(
"d100|10.202.0.134|0|run_chip_cnn.sh|c300|--seed 300 --epochs 20" "d101|10.202.0.134|1|run_chip_cnn.sh|c301|--seed 301 --epochs 20"
"d102|10.202.0.134|2|run_chip_cnn.sh|c302|--seed 302 --epochs 20" "d103|10.202.0.134|3|run_chip_cnn.sh|c303|--seed 303 --epochs 20"
"d104|10.202.1.21|0|run_chip_cnn.sh|c304|--seed 304 --epochs 20" "d105|10.202.1.21|1|run_chip_cnn.sh|c305|--seed 305 --epochs 20"
"d106|10.202.1.21|2|run_chip_cnn.sh|c306|--seed 306 --epochs 20" "d107|10.202.1.21|3|run_chip_cnn.sh|c307|--seed 307 --epochs 20"
"d108|10.202.1.22|0|run_chip_cnn.sh|c308|--seed 308 --epochs 20" "d109|10.202.1.22|1|run_chip_cnn.sh|c309|--seed 309 --epochs 20"
"d110|10.202.1.22|2|run_chip_cnn.sh|c310|--seed 310 --epochs 20"
"w200|10.202.1.23|0|run_chip.sh|d111|--seed 111 --epochs 30" "w201|10.202.1.23|1|run_chip.sh|d112|--seed 112 --epochs 30"
"w202|10.202.1.23|2|run_chip.sh|d113|--seed 113 --epochs 30" "w203|10.202.1.23|3|run_chip.sh|d114|--seed 114 --epochs 30"
)
declare -A started
while [ ${#started[@]} -lt ${#PLAN[@]} ]; do
  for p in "${PLAN[@]}"; do
    IFS='|' read -r old ip chip runner new args <<< "$p"
    [ -n "${started[$new]}" ] && continue
    if $S $ip "grep -q finished /dev/shm/fair/runs/$old/train.log && ! pgrep -f \"[r]uns/$old \" >/dev/null" 2>/dev/null; then
      $S $ip "RUNNER=$runner bash /dev/shm/fair/code/launch_host2.sh \"$chip|$new|$args\"" && started[$new]=1
    fi
  done
  sleep 30
done
echo wave2-all-started
