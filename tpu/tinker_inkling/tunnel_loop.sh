#!/usr/bin/env bash
# Keep local forwards 127.0.0.1:27861-6 -> gradefarm-harish-1..6 (:23850) open via the permanent OD v4 host 0.
# Restarts ssh whenever it exits. Run on della-vis1:  nohup bash tunnel_loop.sh > tunnel_loop.log 2>&1 &
K=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key
JUMP=gcpuser@35.186.57.88
L=""; i=0
for ip in 10.182.0.8 10.182.15.212 10.182.0.7 10.182.0.5 10.182.0.32 10.182.15.217; do
  i=$((i + 1)); L="$L -L 127.0.0.1:2786$i:$ip:23850"
done
while true; do
  echo "$(date -u +%FT%TZ) starting ssh"
  ssh -i "$K" -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      -o ExitOnForwardFailure=yes -N $L "$JUMP"
  echo "$(date -u +%FT%TZ) ssh exited rc=$?; restarting in 10 s"
  sleep 10
done
