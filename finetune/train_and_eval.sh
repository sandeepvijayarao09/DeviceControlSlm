#!/usr/bin/env bash
# Fine-tune (LoRA), fuse, and evaluate each candidate base on DeviceControlBench.
# RUN ON APPLE SILICON (needs MLX). CPU/Linux cannot run this.
#
#   pip install "mlx-lm>=0.19"
#   bash finetune/train_and_eval.sh
#
# For each base we report BEFORE (zero-shot) and AFTER (fine-tuned) benchmark
# metrics, so the effect of the fine-tune is explicit. A failure on one base
# (e.g. a missing 4-bit repo) is logged and the loop continues. Edit BASES.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/python"

# Apache-2.0, MLX-ready candidate bases (compare, then pick the winner).
BASES=(
  "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
  "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
  "mlx-community/SmolLM2-1.7B-Instruct-4bit"
)

ITERS="${ITERS:-800}"

# 0) generate data if missing (no GPU needed)
[ -f finetune/data/train.jsonl ] || python3 -m finetune.gen_data --n 2400

run_base() {
  local BASE="$1"
  local slug adapter fused
  slug="$(echo "$BASE" | sed 's#.*/##; s/-Instruct.*//' | tr 'A-Z' 'a-z')"
  adapter="finetune/adapters/${slug}"
  fused="finetune/fused/${slug}"
  echo "================ $BASE ================"

  # 1) BEFORE: zero-shot benchmark
  python3 -m bench.harness --backend mlx --model "$BASE" --tag base || return 1

  # 2) LoRA fine-tune  (module form works across mlx-lm versions)
  python3 -m mlx_lm.lora --model "$BASE" --train \
    --data finetune/data --adapter-path "$adapter" \
    --iters "$ITERS" --batch-size 4 --num-layers 16 \
    --learning-rate 1e-4 --max-seq-length 2048 --grad-checkpoint || return 1

  # 3) fuse adapter -> standalone model the harness can load
  python3 -m mlx_lm.fuse --model "$BASE" --adapter-path "$adapter" --save-path "$fused" || return 1

  # 4) AFTER: fine-tuned benchmark
  python3 -m bench.harness --backend mlx --model "$fused" --tag ft || return 1
}

for BASE in "${BASES[@]}"; do
  run_base "$BASE" || echo ">>> base '$BASE' failed; continuing to next base"
done

# 5) leaderboard (base vs ft, all models)
python3 -m bench.summarize
echo "Done. See results/leaderboard.md"
