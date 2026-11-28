#!/usr/bin/env bash
# Recreate the MOMO-era Inkling units on della-vis1 after a reboot (written 2026-11-27 when vis1 went down ~21:35Z).
# All of these were transient systemd-run units, so none survive a reboot. tinker-boot (resume-all) only restores the old
# tunnel_loop / tinker-watch / tinker-viz / unstopped grids. Run on della-vis1:  bash recover_vis1.sh
# Same settings as before the crash. The MOMO-train shim reuses its store (same --tag) and resumes from its ledger.
# The Inkling arm runs (mf9-s5, mcf-s5, mxe-*, mxc-*) had all finished, so none are restarted.
set -u
[[ $(hostname -s) == della-vis1* ]] || { echo "refusing: run this on della-vis1, not $(hostname -s)"; exit 2; }
O=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops
PY=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python
R=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs
MTRAIN_TAG="${MTRAIN_TAG:-circuit-b2-mtrain2}"

unit() {  # name, then the command (string); idempotent
  local u=$1; shift
  systemctl --user stop "$u" 2>/dev/null || true
  systemctl --user reset-failed "$u" 2>/dev/null || true
  systemd-run --user --unit="$u" --collect -p Restart=always -p RestartSec=30 -p StartLimitIntervalSec=0 /bin/bash -c "$*"
}

# 1. grading tunnels (sticky port maps portmap-27880/28000.json on GPFS keep every farm on its old port)
bash "$O/run_service.sh" tunnel-inkc /scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-circuit-farms-plus.json
bash "$O/run_service.sh" tunnel-v5p 8.234.57.45
# 2. watcher: registry sync (inkc + v6emath), stall watchdog incl. shim checks, prune, storage alert
unit tinker-fork-watch "exec /bin/bash $O/fork_watch.sh >> $O/fork_watch.log 2>&1"
# 3. frozen MOMO shims
unit tinker-shim-erdos "exec $PY -u $O/inkling_sample.py serve erdos 18790 $R/mf9-s1/env.txt SKYRL_GRADING_REGISTRY=$O/registry-v5p.json SKYRL_OPEN_GRADING_K=64 SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=8 >> $O/shim_erdos.log 2>&1"
unit tinker-shim-circuit "exec $PY -u $O/inkling_sample.py serve circuit 18791 >> $O/shim_circuit.log 2>&1"
# 4. MOMO-train shim (same store -> resumes its ledger; publish on per the user's decision)
unit tinker-shim-momo-train "exec $PY -u $O/inkling_momo_train_stream.py serve circuit 18793 --env-txt $R/mc-s5/env.txt --state tinker://b9352b42-346c-5f14-bafc-662d0219e950:train:0/weights/000012 --base-sampler tinker://b9352b42-346c-5f14-bafc-662d0219e950:train:0/sampler_weights/000012 --key HARISH --tag $MTRAIN_TAG --adapter-name circuit-inkling-s5-a12 --publish >> $O/shim_momo_train.log 2>&1"

# 5. Erdős MOMO-train shim (user 2026-11-28): mn9-s5 recipe, graded on the slim v6e math pool (registry-v5p = farms-v6emath)
MTRAIN_ERDOS_TAG="${MTRAIN_ERDOS_TAG:-erdos-b2-mtrain2}"
C=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code
unit tinker-shim-momo-train-erdos "exec $PY -u $O/inkling_momo_train_stream.py serve erdos 18795 --env-txt $R/mn9-s5/env.txt --state tinker://307a1783-ac19-5274-941a-308a43108638:train:0/weights/000012 --base-sampler tinker://307a1783-ac19-5274-941a-308a43108638:train:0/sampler_weights/000012 --key 1 --tag $MTRAIN_ERDOS_TAG --adapter-name erdos-inkling-s5-a12 --publish --set CODE_SKYRL=$C/skyrl-28b4c4aa866a-tfsci --set CODE_DISCOVER=$C/discover-0766e2413eb0 --set SKYRL_GRADING_REGISTRY=$O/registry-v5p.json --set SKYRL_GRADER_HASH_FILES=__init__.py,cgroup_limits.py,core_pool.py,cpu_slots.py,farm_resources.py,grading_dedup.py,grading_transport.py,math_grade.py,math_runner.py,placement_resources.py,placement_runtime.py,rewards.py,routing_resources.py,unit.py,worker.py --set SKYRL_OPEN_GRADING_K=64 --set SKYRL_OPEN_GRADING_HTTP_CONCURRENCY=8 >> $O/shim_momo_train_erdos.log 2>&1"

sleep 40
for p in 18790 18791; do
  printf '%s: ' "$p"; curl -s -m 20 -X POST -H 'Content-Type: application/json' -d '{}' "http://127.0.0.1:$p/sample"; echo
done
printf '18793: '; curl -s -m 20 http://127.0.0.1:18793/status; echo
printf '18795: '; curl -s -m 20 http://127.0.0.1:18795/status; echo
for u in tinker-tunnel-inkc tinker-tunnel-v5p tinker-fork-watch tinker-shim-erdos tinker-shim-circuit tinker-shim-momo-train tinker-shim-momo-train-erdos; do
  echo "$u $(systemctl --user is-active "$u")"
done
