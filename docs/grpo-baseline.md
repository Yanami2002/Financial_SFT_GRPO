# Completed FinQA GRPO baseline

Recorded on **2026-11-20**, Asia/Tokyo. Training completed on **2026-11-19 at 03:14 JST**.
The [machine-readable record](experiments/grpo-baseline.json) retains the frozen configuration, source/data fingerprints, package versions, trajectory, reward diagnostics, and resources.

| Model | Correct / 481 scorable dev records | Accuracy |
|---|---:|---:|
| Base | 90 | 18.71% |
| Selected SFT, step 500 | 231 | 48.02% |
| GRPO best, step 400 | 245 | **50.94%** |
| GRPO last, step 1681 | 234 | 48.65% |

Best GRPO improves over selected SFT by **14 questions / 2.91 percentage points**.
This is a seed-42 result on the ratio and percentage subset of FinQA dev. All 883 dev records are retained, of which 481 have accepted numerical references. Held-out test evaluation and continued-SFT controls are pending.

## Configuration and selection

- Qwen3.5-2B at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, BF16, text-only loader, thinking disabled.
- Continue the selected SFT LoRA adapter, rank 16 / alpha 32 / all-linear.
- Same 3363 train records as the length-filtered SFT cohort; one epoch, 1681 optimizer updates, seed 42.
- Two GPUs, batch 1 per device, accumulation 8, eight responses per question. Each generation/update batch contains 16 responses for two distinct questions. The sampler processed 3362 questions, dropping the final incomplete batch.
- Reward: numerical correctness + 0.05 × valid JSON format; learning rate 5e-6, beta 0.04, temperature 0.8, top-p 0.95.
- Prompt limit 2048; completion limit 128; greedy full-dev validation every 100 steps and at completion. Final export contains checkpoint 400.

Use [configs/grpo_equal_data.yaml](../configs/grpo_equal_data.yaml) and the shared-cohort commands in the [user guide](guide.md#grpo-and-continued-sft-control) to reproduce this run.

## Diagnostics

| Group outcome | Groups | Share |
|---|---:|---:|
| Eight correct | 1816 | 54.0% |
| Eight incorrect | 682 | 20.3% |
| Mixed correctness | 864 | 25.7% |

Overall **74.3%** of groups had zero reward standard deviation. In **937 / 1681** updates, neither question supplied correctness differences; KL could still update the model. Training format accuracy was 100%, so the format component did not distinguish responses. Mean completion length was 19.08 tokens and no training responses were truncated.

At the selected checkpoint, 24 previously incorrect dev answers became correct, while 10 previously correct answers became incorrect. Of the 24 corrections, 23 changed the numerical answer by at most one percentage point. At the last checkpoint, the corresponding counts were 19 corrections and 16 regressions. These observations suggest numerical-output calibration and limited effective training signal; they are not causal ablations or proof about internal reasoning.

Full training took **10 h 41 min**, including 17 full-dev checks. Peak allocated memory was approximately **6.03 GiB** per rank.

## Follow-up

The [DAPO adaptation](dapo.md) starts from the same selected SFT adapter and adds dynamic filtering and refill, a loss normalized by accepted tokens, asymmetric clipping, and zero KL.
