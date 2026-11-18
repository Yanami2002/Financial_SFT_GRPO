#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# This entry starts real training only when explicitly invoked.
stage="${1:-}"
case "$stage" in
  sft|grpo) ;;
  *) echo "Usage: bash scripts/train.sh {sft|grpo} [--config PATH] [--set KEY=VALUE] [--dry-run]" >&2; exit 2 ;;
esac
shift
training_workers="${NPROC_PER_NODE:-3}"
training_config="configs/${stage}.yaml"
if [[ "${1:-}" == "--config" ]]; then
  training_config="${2:?Missing config path}"
  shift 2
fi
exec python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$training_workers" \
  -m "train.${stage}" --config "$training_config" "$@"
