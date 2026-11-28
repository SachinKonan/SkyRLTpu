#!/usr/bin/env bash
# Inkling-2-Mini new-env generation probe (Kakeya-128 on key 2, AHC024 on key 3), 2 groups x 32 each, base model,
# Inkling production settings + Inkling's answer-length line (rq4-local/newproblems/build/inkling_answer_rules.md).
# Run ONLY after envs relays the go. Usage: bash run_new_env_probe.sh
set -u
PY=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python
P=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/inkling_probe.py
NP=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/newproblems
KAKEYA_PROMPT=$NP/build/kakeya_probe/probe_prompt.json
AHC_PROMPT=$NP/build/ahc024/probe/probe_prompt.json
KAKEYA_RULE="Answer length: your final program must be at most 12,000 characters (about 300 lines, roughly 4,500 tokens); a longer program is rejected and scores nothing, so keep it compact and do not let it grow from one version to the next."
AHC_RULE="Answer length: your final C++ program must be at most 12,000 characters (about 300 lines, roughly 5,500 tokens); a longer program is rejected and scores nothing, so keep it compact and do not let it grow from one version to the next."
for f in "$KAKEYA_PROMPT" "$AHC_PROMPT"; do [ -f "$f" ] || { echo "missing $f"; exit 2; }; done
mkdir -p $NP/probe/kakeya-128 $NP/probe/ahc024
$PY -W ignore $P --prompt "$KAKEYA_PROMPT" --problem kakeya-128 --out $NP/probe/kakeya-128/inkling.jsonl --key 2 \
    --groups 2 --n 32 --extra-rule "$KAKEYA_RULE" > $NP/probe/kakeya-128/inkling.log 2>&1 &
$PY -W ignore $P --prompt "$AHC_PROMPT" --problem ahc024 --out $NP/probe/ahc024/inkling.jsonl --key 3 \
    --groups 2 --n 32 --extra-rule "$AHC_RULE" > $NP/probe/ahc024/inkling.log 2>&1 &
wait
for p in kakeya-128 ahc024; do echo "$p: $(grep -c '"group"' $NP/probe/$p/inkling.jsonl 2>/dev/null) rows; $(tail -1 $NP/probe/$p/inkling.log)"; done
