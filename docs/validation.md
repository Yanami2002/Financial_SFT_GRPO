# Validation and experiment report

Last verified: **2026-11-20**, Asia/Tokyo. This report contains curated results and reproduction details. Raw logs, predictions, checkpoints, process identifiers, and machine-specific GPU identifiers remain in ignored local artifact directories.

## Reinforcement learning results

| Recipe | Selected step | Correct / 481 | Dev accuracy | Final step | Final dev accuracy |
|---|---:|---:|---:|---:|---:|
| [GRPO](grpo-baseline.md) | 400 | 245 | 50.94% | 1,681 | 48.65% |
| [DAPO](dapo.md) | 200 | 246 | 51.14% | 400 | 50.73% |

Both runs started from SFT step 500 and used the same training cohort and evaluation protocol. Compared with SFT, the selected checkpoints gained 2.91 and 3.12 percentage points, respectively. DAPO gained one additional correct answer over selected GRPO. The recipes have different sampling and compute budgets. This is a seed-42 comparison using checkpoints selected on dev. Continued-SFT controls, multi-seed comparisons, and held-out test evaluation remain pending. See the machine-readable [GRPO record](experiments/grpo-baseline.json) and [DAPO record](experiments/dapo-baseline.json).

## Completed Base and two-GPU SFT baseline

Base/dev evaluation, SFT training, and the protocol-checked comparison completed on **2026-11-17**.

| Model | Correct / scorable dev records | Numerical accuracy | Parseable outputs, all 883 dev records |
|---|---:|---:|---:|
| Base | 90 / 481 | 18.71% | 86.07% |
| SFT best, step 500 | 231 / 481 | 48.02% | 97.06% |
| SFT last, step 633 | 226 / 481 | 46.99% | 97.06% |

Selected SFT improves dev accuracy by **29.31 percentage points** on the ratio and percentage subset.

### Configuration

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-2B` |
| Model revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Loader | Text-only `Qwen3_5ForCausalLM`, BF16, SDPA |
| Thinking | Disabled throughout |
| FinQA revision | `0f16e2867befa6840783e58be38c9efb9229d742` |
| Train data | 3428 verified candidates; 3363 retained under 2048 tokens |
| Dev data | 883 records; 481 scorable, 402 unscorable |
| Hardware | 2 x NVIDIA RTX A6000 48 GB, DDP |
| Seed | 42 |
| LoRA | Rank 16, alpha 32, dropout 0.05, `all-linear` |
| Batch | 1 per device, accumulation 8, effective batch 16 |
| Schedule | 3 epochs, 633 updates, initial learning rate 1e-4, cosine decay |
| Selection | Greedy dev numerical accuracy every 100 steps and at completion |
| Dev generation | Prompt limit 2048, completion limit 128, no sampling |
| Tolerance | `atol=1e-6`, `rtol=1e-3`, plus reference rounding tolerance |

The frozen run configuration matches [configs/sft.yaml](../configs/sft.yaml). The [machine-readable report](experiments/sft-baseline.json) records package versions, original source/data fingerprints, metrics, and resources. The experiment predates the initial Git commit; original SHA256 fingerprints are retained instead of attributing it to a later commit.

### Reproduction

Use a fresh checkout or separate output paths. Replace the example device IDs with your allocated GPUs.

```bash
source .venv/bin/activate
python -m data.prepare_data --download
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=0 python -m eval.evaluate --config configs/eval_base_dev.yaml
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train.sh sft

python - <<'PYTHON'
import json
import subprocess
import sys
from pathlib import Path

selected = json.loads(Path('outputs/qwen3.5-2b/sft/best_checkpoint.json').read_text())
subprocess.run([
    sys.executable, '-m', 'eval.compare',
    'results/qwen3.5-2b/base-dev/base/metrics.json',
    selected['selected']['metrics'],
    '--output', 'results/qwen3.5-2b/sft-comparison.json',
], check=True)
PYTHON
```

The original unattended runner used equivalent frozen configurations and offline model-cache access. Process supervision is a local execution detail; these commands reproduce the data, evaluation, and training stages.

### Selection trajectory and verification

| Step | Correct / 481 | Numerical accuracy |
|---|---:|---:|
| 100 | 179 | 37.21% |
| 200 | 195 | 40.54% |
| 300 | 186 | 38.67% |
| 400 | 213 | 44.28% |
| 500 | 231 | **48.02%** |
| 600 | 224 | 46.57% |
| 633 | 226 | 46.99% |

All 372 tensors in the final adapter at `outputs/qwen3.5-2b/sft` were compared on CPU and exactly match `checkpoint-500`; all are finite. Training state reports step 633 and epoch 3, with finite recorded losses and gradient norms. Base and selected-SFT evaluation protocols match exactly.

### Runtime and memory

Base/dev took **15 minutes 18 seconds**. SFT took **4 hours 23 minutes 57 seconds**, including seven complete dev checks and checkpoint saving. The sequential job took approximately **4 hours 40 minutes**.

| DDP rank | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| 0 | 9.716 | 33.186 |
| 1 | 9.710 | 45.719 |

Reserved memory includes allocator cache and differs from live tensor allocation or `nvidia-smi` totals. These values are specific to this configuration. Optional convolution/linear-attention kernels were absent; PyTorch fallback implementations were used.

## Engineering verification

- Full dependency environment: **83 passed**, with CUDA hidden, including DAPO tests.
- Ruff F checks, configuration/API checks, and actual cached-tokenizer masking checks passed.
- Explicit tiny random CPU models and controlled outputs verify earlier-best retention, ties, final-step checks, exact best-weight export, and restoration of model mode, cache, padding, and RNG state.
- Three-process CPU/Gloo tests verify uneven shards, one-time aggregation, record order, and overlength examples staying in the scoring denominator.
- Dev evaluation leaves GRPO training reward statistics and frozen reference weights unchanged. Invalid validation splits and answer-bearing prompts are rejected.

Fixture scores demonstrate implementation behavior, not FinQA capability. See [checkpoint-selection.md](checkpoint-selection.md).

## GPU smoke history

Before full SFT, a real Qwen3.5 run completed one SFT update and four GRPO updates using 16 candidate train examples. Its fourth GRPO group received `[1.05, 0.05, 1.05, 0.05]`, producing nonzero gradients. The frozen reference contained 372 tensors and matched the initial SFT adapter. All 372 trained adapter tensors changed; final exports excluded reference weights.

A separate selection smoke run used two SFT steps and four GRPO steps, checking four dev records. Both restored earlier tied-best checkpoints, and final weights matched those selected checkpoints exactly. Three-model smoke dev scores were each 1/2 correct among two scorable records, with four parseable outputs. These small scores do not establish effectiveness.

The [DAPO smoke run](dapo.md#implementation-and-checks) completed two updates and exported the selected checkpoint.

## Local artifacts and publication scope

Training writes checkpoints, per-step dev predictions, and diagnostics under `outputs/`. Standalone evaluation writes metrics and predictions under `results/`. Git tracks the curated reports and JSON summaries under `docs/experiments/` and ignores the generated run artifacts.
