#!/usr/bin/env bash
# Run the 9 neural runs R3-R5 on the Colab VM (resumable: finished runs are skipped,
# unfinished folds resume from checkpoints written every <= 15 min). Launched detached with nohup.
#   bash /content/pack/code/scripts/colab_run_nn.sh > /content/run.log 2>&1 &
set -u
cd /content/pack/code
IN=/content/pack/inputs
export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONIOENCODING=utf-8
run() {  # id config npz [extra overrides...]
  local id=$1 cfg=$2 npz=$3; shift 3
  if [ -f "outputs/nn/$id/nn_results.csv" ]; then echo "$(date -u +%FT%TZ) SKIP $id (done)"; return; fi
  echo "$(date -u +%FT%TZ) START $id"
  local t0=$(date +%s)
  python -m plasma_etch_vm.cli train-nn --config "$cfg" "model.inputs_npz=$IN/$npz" "targets_parquet=$IN/wafer_targets.parquet" \
    "run_name=$id" "checkpoint.dir=checkpoints/$id" checkpoint.every_minutes=15 "$@" > "logs_$id.txt" 2>&1
  echo "$(date -u +%FT%TZ) END $id exit=$? wall=$(( $(date +%s) - t0 ))s"
}
mkdir -p outputs/nn
for s in 0 1 2; do run "R3_cnn1d_s$s" models/cnn1d.yaml cnn1d.npz "seed=$s"; done
for s in 0 1 2; do run "R4_statfe_s$s" models/statfe_transformer.yaml statfe.npz "seed=$s"; done
for s in 0 1 2; do run "R5_statfe_tool_s$s" models/statfe_transformer.yaml statfe_tool.npz "seed=$s" "model.sources=[process]"; done
echo "$(date -u +%FT%TZ) ALL_DONE"
