#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

RUN_ROOT="outputs/train/wm3d_wam_k16_r5"
STAGE_A_OUT="$RUN_ROOT/stage_a_world_core_gpu1_4"
STAGE_A_CKPT="$STAGE_A_OUT/checkpoints/step_00030000"
WARMUP_OUT="$RUN_ROOT/stage_b_warmup_gpu1_4"
WARMUP_CKPT="$WARMUP_OUT/checkpoints/step_00002000"
WARMUP_LOG="$RUN_ROOT/stage_b_warmup_gpu1_4.formal.log"
MAIN_OUT="$RUN_ROOT/stage_b_main_gpu1_4"
MAIN_LOG="$RUN_ROOT/stage_b_main_gpu1_4.formal.log"
LOCK_PATH="$RUN_ROOT/.r5_stage_handoff.lock"
TORCHRUN_BIN="/data/Minko/.venvs/wm3d/bin/torchrun"
POLL_SECONDS="${WM3D_HANDOFF_POLL_SECONDS:-15}"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "WM3D_HANDOFF_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT"
exec 9>"$LOCK_PATH"
if ! flock -n 9; then
  echo "another Revision 5 handoff supervisor is already running" >&2
  exit 3
fi

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %z')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

no_tmp_files() {
  local checkpoint_dir="$1"
  [[ -z "$(find "$checkpoint_dir" -type f -name '*.tmp' -print -quit)" ]]
}

canonical_stage_a_complete() {
  local checkpoint_dir="$1"
  local rank

  [[ -s "$checkpoint_dir/metadata.json" ]] || return 1
  [[ -s "$checkpoint_dir/.metadata" ]] || return 1
  jq -e '
    .schema == "wm3d_wam_checkpoint_v1" and
    .storage == "canonical_dcp" and
    .phase == "world_core_pretrain" and
    .global_step == 30000 and
    .phase_total_steps == 30000 and
    .world_size == 4 and
    .physical_cuda_devices == [1, 2, 3, 4]
  ' "$checkpoint_dir/metadata.json" >/dev/null || return 1
  [[ "$(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.distcp' | wc -l)" -eq 4 ]] || return 1
  while IFS= read -r shard; do
    [[ -s "$shard" ]] || return 1
  done < <(find "$checkpoint_dir" -maxdepth 1 -type f -name '*.distcp' | sort)
  for rank in 000 001 002 003; do
    [[ -s "$checkpoint_dir/runtime_rank_${rank}.pt" ]] || return 1
  done
  no_tmp_files "$checkpoint_dir"
}

rank_local_warmup_complete() {
  local checkpoint_dir="$1"
  local rank

  [[ -s "$checkpoint_dir/metadata.json" ]] || return 1
  jq -e '
    .schema == "wm3d_wam_local_fsdp_checkpoint_v2" and
    .storage == "rank_local_fsdp" and
    .phase == "wan_action_warmup" and
    .global_step == 2000 and
    .phase_total_steps == 2000 and
    .world_size == 4 and
    .physical_cuda_devices == [1, 2, 3, 4]
  ' "$checkpoint_dir/metadata.json" >/dev/null || return 1
  for rank in 000 001 002 003; do
    [[ -s "$checkpoint_dir/model_rank_${rank}.pt" ]] || return 1
    [[ -s "$checkpoint_dir/optimizer_rank_${rank}.pt" ]] || return 1
    [[ -s "$checkpoint_dir/runtime_rank_${rank}.pt" ]] || return 1
  done
  no_tmp_files "$checkpoint_dir"
}

stage_running() {
  local output_dir="$1"
  ps -eo comm=,args= | awk -v needle="$output_dir" '
    ($1 == "pt_elastic" || $1 ~ /^python/) &&
    index($0, "scripts/train_wm3d_wam.py") &&
    index($0, needle) { found = 1 }
    END { exit(found ? 0 : 1) }
  '
}

wait_for_checkpoint() {
  local stage_name="$1"
  local checkpoint_dir="$2"
  local checker="$3"

  log "waiting for complete ${stage_name} checkpoint: ${checkpoint_dir}"
  until "$checker" "$checkpoint_dir"; do
    sleep "$POLL_SECONDS"
  done
  log "verified complete ${stage_name} checkpoint"
}

wait_for_stage_exit() {
  local stage_name="$1"
  local output_dir="$2"

  if stage_running "$output_dir"; then
    log "waiting for ${stage_name} training ranks to exit"
  fi
  while stage_running "$output_dir"; do
    sleep "$POLL_SECONDS"
  done
  log "${stage_name} training ranks have exited"
}

common_environment=(
  "CUDA_VISIBLE_DEVICES=1,2,3,4"
  "PYTHONPATH=src"
  "HF_HUB_OFFLINE=1"
  "TRANSFORMERS_OFFLINE=1"
  "TMPDIR=$PROJECT_ROOT/outputs/runtime_cache/tmp"
  "TORCHINDUCTOR_CACHE_DIR=$PROJECT_ROOT/outputs/runtime_cache/torchinductor"
  "TRITON_CACHE_DIR=$PROJECT_ROOT/outputs/runtime_cache/triton"
  "NCCL_NVLS_ENABLE=0"
  "TORCH_NCCL_TRACE_BUFFER_SIZE=1048576"
  "TORCH_NCCL_DUMP_ON_TIMEOUT=1"
  "TORCH_NCCL_DESYNC_DEBUG=1"
)

launch_warmup() {
  [[ ! -e "$WARMUP_OUT" ]] || fail "warmup output already exists: $WARMUP_OUT"
  [[ ! -e "$WARMUP_LOG" ]] || fail "warmup formal log already exists: $WARMUP_LOG"

  log "launching Stage B warmup from $STAGE_A_CKPT"
  nohup env "${common_environment[@]}" \
    "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    scripts/train_wm3d_wam.py \
    --phase wan_action_warmup \
    --max-steps 2000 \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --seed 20260819 \
    --warmup-steps 200 \
    --log-interval 10 \
    --validation-interval 500 \
    --validation-samples-per-rank 8 \
    --checkpoint-interval 500 \
    --keep-last-checkpoints 3 \
    --initialize-from "$STAGE_A_CKPT" \
    --output-dir "$WARMUP_OUT" \
    > "$WARMUP_LOG" 2>&1 < /dev/null &
  local launch_pid=$!
  sleep 2
  kill -0 "$launch_pid" 2>/dev/null || fail "Stage B warmup launcher exited immediately; inspect $WARMUP_LOG"
  log "Stage B warmup launch accepted with pid $launch_pid"
}

launch_main() {
  [[ ! -e "$MAIN_OUT" ]] || fail "main output already exists: $MAIN_OUT"
  [[ ! -e "$MAIN_LOG" ]] || fail "main formal log already exists: $MAIN_LOG"

  log "launching Stage B main from $WARMUP_CKPT"
  nohup env "${common_environment[@]}" \
    "$TORCHRUN_BIN" --standalone --nproc_per_node=4 \
    scripts/train_wm3d_wam.py \
    --phase wan_action_main \
    --max-steps 38000 \
    --micro-batch-size 1 \
    --gradient-accumulation-steps 1 \
    --num-workers 4 \
    --prefetch-factor 2 \
    --seed 20260819 \
    --warmup-steps 500 \
    --log-interval 10 \
    --validation-interval 500 \
    --validation-samples-per-rank 8 \
    --checkpoint-interval 500 \
    --keep-last-checkpoints 3 \
    --initialize-from "$WARMUP_CKPT" \
    --output-dir "$MAIN_OUT" \
    > "$MAIN_LOG" 2>&1 < /dev/null &
  local launch_pid=$!
  sleep 2
  kill -0 "$launch_pid" 2>/dev/null || fail "Stage B main launcher exited immediately; inspect $MAIN_LOG"
  log "Stage B main launch accepted with pid $launch_pid"
}

wait_for_checkpoint "Stage A" "$STAGE_A_CKPT" canonical_stage_a_complete
wait_for_stage_exit "Stage A" "$STAGE_A_OUT"

if rank_local_warmup_complete "$WARMUP_CKPT"; then
  log "Stage B warmup checkpoint is already complete"
elif stage_running "$WARMUP_OUT"; then
  log "Stage B warmup is already running; attaching to it"
else
  launch_warmup
fi

wait_for_checkpoint "Stage B warmup" "$WARMUP_CKPT" rank_local_warmup_complete
wait_for_stage_exit "Stage B warmup" "$WARMUP_OUT"

if stage_running "$MAIN_OUT"; then
  log "Stage B main is already running"
elif [[ -e "$MAIN_OUT" || -e "$MAIN_LOG" ]]; then
  fail "Stage B main has existing output/log but no running process; exact resume is required"
else
  launch_main
fi

log "Stage B main handoff complete; external monitor now owns ongoing health checks"
