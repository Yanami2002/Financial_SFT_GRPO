# FinQA DAPO adaptation

This experiment adapts DAPO to short JSON answers on FinQA using Qwen3.5-2B and two RTX A6000 GPUs. It uses the same selected SFT initialization, training cohort, prompt, disabled thinking, and greedy dev scoring as the GRPO baseline.

See the [DAPO paper](https://arxiv.org/abs/2503.14476) and the [official verl recipe](https://github.com/verl-project/verl-recipe/tree/main/dapo) for the original long-reasoning setup. This implementation uses TRL 1.17.1.

## Results

The 400-update run selected step 200 with **246 / 481 correct (51.14%)**, up **3.12 percentage points** from SFT and **0.21 percentage points** from the selected GRPO checkpoint. Step 350 tied step 200, so checkpoint selection kept the earlier step. The final step scored **244 / 481 (50.73%)**. Relative to GRPO, DAPO corrected 11 answers and regressed on 10.

The comparison uses seed 42 and the same 481 scorable ratio and percentage questions from the FinQA dev split. The DAPO and GRPO recipes differ in sampling, rollout reuse, and compute. See the [machine-readable record](experiments/dapo-baseline.json) for the full dev trajectory and data fingerprints.

Dynamic sampling evaluated 5,024 candidate groups and selected 1,600 across 200 fresh rollout batches. All selected groups supplied nonzero reward variance. Each batch took two to six candidate generation batches to fill. Training, including eight dev evaluations, took about 9 hours 21 minutes. Peak allocated GPU memory was about 12.90 GiB per worker.

## Recipe

| Setting | Value |
|---|---|
| Initialization | Selected SFT, checkpoint 500; no continuation from GRPO |
| Train cohort | Same 3363 verified examples as completed GRPO |
| Outcome reward | +1 correct / -1 incorrect, no separate format bonus |
| Dynamic sampling | Keep complete groups with both correct and incorrect usable responses; refill from fresh train questions |
| Refill limit | At most 32 candidate generation batches to fill an accepted batch |
| Accepted batch on two GPUs | 8 questions × 8 responses = 64 responses |
| Per-device batch / accumulation | 1 / 32 |
| Optimization passes | 2 per accepted rollout batch; native behavior-policy log probabilities retained |
| Loss | Native TRL `loss_type: dapo`, normalized by global accepted completion tokens |
| Clip bounds | Lower 0.2, upper 0.28 |
| KL | Zero; no reference adapter |
| Sampling | Temperature 1.0, top-p 1.0, no top-k restriction |
| Learning rate | 1e-6, constant after 10 warmup updates |
| Completion budget | 128 tokens, 32-token soft overlength buffer with penalty up to -1 |
| Truncation | Native truncated-completion masking; a masked response cannot be the sole source of an outcome in an accepted group |
| Duration | 400 optimizer updates; 200 fresh accepted batches, each reused twice |
| Validation | All 883 dev records / 481 scorable; greedy, every 50 updates and at completion |
| Selection | Best dev numerical accuracy, earliest tie, final export restores best |

The completion budget, length penalty, and filtering parameters are set in [configs/dapo.yaml](../configs/dapo.yaml).

## Implementation and checks

`train.dapo_trainer` filters raw rollouts before TRL constructs loss tensors. All ranks gather rank-ordered candidate records, check complete group IDs, select identical global groups, and repartition examples, references, token IDs and decoded completions together. Native TRL then calculates selected rewards, group advantages, behavior-policy log probabilities and token counts. Candidate outputs with no reward differences never enter the loss. Replacement questions come only from the validated, length-filtered train cohort; all candidates in one refill cycle have distinct question IDs.

Accepted rollouts are reused for two native optimization passes, enabling asymmetric clipping to operate against fixed behavior-policy probabilities. CPU tests cover filtering, truncated groups, soft penalties, refill alignment, real gradient updates and exact best-weight export.

The two-GPU smoke completed two optimizer updates with finite nonzero gradients, selected eight mixed-outcome groups after two candidate batches, changed all 372 adapter tensors, and exported the selected checkpoint. Second-pass clipping was active.

## Reproduction

Follow the [user guide](guide.md) to prepare the data and reproduce SFT and GRPO. Build the shared cohort before starting DAPO if you have not already done so. Replace the example device IDs with your allocated GPUs.

```bash
python -m scripts.control_cohort
CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 bash scripts/train.sh dapo
```

Compare the selected dev checkpoints from the default reproduction paths:

```bash
python - <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path('outputs/qwen3.5-2b')
stages = ('sft', 'grpo', 'dapo')
metrics = [json.loads((root / stage / 'best_checkpoint.json').read_text())['selected']['metrics']
           for stage in stages]
subprocess.run([sys.executable, '-m', 'eval.compare', *metrics,
                '--output', 'results/qwen3.5-2b/dapo-dev-comparison.json'], check=True)
PY
```

Each run writes the following diagnostics under its output directory:

- `sampling_groups.jsonl`: every candidate group, correctness count, truncation count and eligibility.
- `dynamic_sampling.jsonl`: refill attempts, candidate/accepted counts, acceptance fraction and selected question IDs per fresh rollout batch.
- `reward_groups.jsonl`: selected groups actually used for optimization.
- `dev_history.jsonl` and `dev-evaluations/`: dev trajectory and per-example predictions.
- `best_checkpoint.json`: selected checkpoint and whether export completed.

Native completion-length and token-consumption metrics include all candidates, including rejected candidates. Native reward/zero-standard-deviation statistics describe selected training groups. Compare both sets when assessing training efficiency.
Native epoch values track dataloader progress; on-demand replacement questions are counted separately in the sampling logs.
