#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Set this explicitly on shared machines; default targets a single visible GPU.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m pytest -q tests
python -m data.prepare_data --download
python -m eval.evaluate --config configs/eval_smoke.yaml --set 'variants={base: {adapter: null}}'
python -m train.sft --config configs/sft_smoke.yaml
python -m train.grpo --config configs/grpo_smoke.yaml
python -m eval.evaluate --config configs/eval_smoke.yaml
python -m eval.compare results/qwen3.5-2b/smoke-dev/{base,sft,grpo}/metrics.json --output results/qwen3.5-2b/smoke-comparison.json
python -m demo.app --config configs/eval_smoke.yaml --check
