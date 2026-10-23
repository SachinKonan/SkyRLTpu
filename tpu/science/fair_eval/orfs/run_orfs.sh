#!/bin/bash
# Run one macro placement through the challenge's Tier-2 flow: scripts/evaluate_with_orfs.py from the
# pinned challenge repo (996193c), unmodified, with --no-docker. OpenROAD, Yosys and KLayout come
# from the ORFS image built on the submission deadline (openroad/orfs:26Q2-385-gbd2a6b695, run with
# Apptainer); the flow scripts are the matching ORFS commit (bd2a6b695).
#
# evaluate_with_orfs.py edits the ORFS tree in place (flow/designs, flow/results, macro_place_util.tcl),
# so every run gets a private copy of flow/ (nangate45 only). Its tools/ is a symlink to the image's
# tools, which resolves inside the container; `make` and `python3` (genMetrics.py, which calls
# `openroad -version`) are run inside the container through wrappers first on PATH.
#
# Usage: run_orfs.sh NAME DESIGN POSITIONS.npy|none
#   NAME       label for the run (e.g. momofill-r1); DESIGN: ariane133 | ariane136 | mempool_tile | nvdla
#   POSITIONS  the [num_macros, 2] centers a placer returned (eval_one.py's positions.npy), or none to let
#              ORFS's own rtl_macro_placer place the macros (the script's baseline mode)
set -euo pipefail
O=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval/orfs
SIF=$O/orfs-26Q2-385.sif
PY=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval/env/cuda/bin/python
name=$1 design=$2 pos=$3
R=$O/runs/$name/$design

rm -rf "$R" && mkdir -p "$R/bin" "$R/ORFS"
rsync -a --include 'platforms/nangate45/***' --exclude 'platforms/*' \
      --include 'designs/nangate45/***' --exclude 'designs/*' \
      --exclude results --exclude logs --exclude objects --exclude reports \
      "$O/ORFS-src/flow/" "$R/ORFS/flow/"
ln -s /OpenROAD-flow-scripts/tools "$R/ORFS/tools"
for tool in make python3; do
  printf '#!/bin/bash\nexec apptainer exec --bind /scratch/gpfs/ZHUANGL %s %s "$@"\n' "$SIF" "$tool" > "$R/bin/$tool"
  chmod +x "$R/bin/$tool"
done

args=(--benchmark "${design}_ng45" --orfs-root "$R/ORFS" --output "$R/out" --no-docker)
if [ "$pos" != none ]; then
  "$PY" -c "import sys, numpy as np, torch; torch.save(torch.from_numpy(np.load(sys.argv[1])), sys.argv[2])" \
        "$pos" "$R/placement.pt"
  args+=(--placement "$R/placement.pt")
fi
echo "$(date -Is) start $name $design pos=$pos host=$(hostname) cpus=$(nproc)" | tee "$R/run.log"
cd "$O/challenge"
set +e
PATH="$R/bin:$PATH" "$PY" -u scripts/evaluate_with_orfs.py "${args[@]}" > "$R/eval.log" 2>&1
rc=$?
set -e
echo "$(date -Is) end rc=$rc" | tee -a "$R/run.log"
# Keep the evidence (logs, reports, summary, macro TCL and its spacing diff); drop bulky results.
cp -r "$R/ORFS/flow/logs" "$R/ORFS/flow/reports" "$R/" 2>/dev/null || true
cp "$R/ORFS/flow/designs/nangate45/"*/config.mk "$R/ORFS/flow/designs/nangate45/"*/macros.tcl "$R/" 2>/dev/null || true
ls "$R/ORFS/flow/results/nangate45/"*/base/ > "$R/results_listing.txt" 2>/dev/null || true
rm -rf "$R/ORFS"
exit $rc
