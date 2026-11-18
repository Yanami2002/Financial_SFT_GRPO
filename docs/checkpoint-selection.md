# Automatic dev checkpoint selection

Training periodically evaluates generated answers on dev, retains the best checkpoint, and restores it before exporting the final adapter.

## Default behavior

1. SFT, GRPO, and both continued-training controls check the complete dev split every 100 optimizer updates.
2. The final update is checked even when it is not a multiple of 100.
3. Selection uses numerical accuracy of generated JSON answers. Ties retain the earliest evaluated step.
4. Native Transformers retention protects the best and latest checkpoints. All check metrics and predictions remain, even if older checkpoint weights are removed.
5. Training restores the best model before exporting the adapter to the stage output directory, which subsequent training or inference can use directly.

For example, if steps 100/200/300 scored 50%/60%/55%, the final `outputs/qwen3.5-2b/sft` adapter would contain step 200's weights. This is an illustration; actual results are in [validation.md](validation.md).

## Local outputs

Paths are relative to a stage output directory such as `outputs/qwen3.5-2b/sft/`:

| File | Purpose |
|---|---|
| `best_checkpoint.json` | Selected checkpoint, accuracy, and final-export status |
| `dev_history.jsonl` | Metrics at each dev check |
| `dev-evaluations/step-N/predictions.jsonl` | Per-example answers and errors |
| `dev-evaluations/step-N/metrics.json` | Aggregate scores and evaluation protocol |
| `dev-evaluations/step-N/run.json` | Configuration, source/data hashes, and adapter fingerprint |
| `checkpoint-N/` | Retained model and native training state |

`export_contains_best=true` means final export has completed. While training, `false` means a checkpoint is selected but the final adapter has not been exported. Storage and rotation use native Transformers behavior; validation does not load another base model.

## Data separation and scoring

Train is used for learning, dev for selection, and test only for final evaluation after settings are fixed. Entry points reject train/test validation records, duplicate dev IDs, overlap with training IDs, and assistant answers in dev prompts.

All 883 dev records are retained. Accuracy uses the 481 accepted numerical references; scorable overlength prompts count as incorrect. Parseability is separate and does not enter accuracy. Chat rendering, generation, and scoring are shared with standalone evaluation.

For DDP, each example belongs to one rank. Generation unwraps the DDP model, permits unequal shard lengths, and gathers results once. Validation restores model mode, cache, padding, and RNG state. It does not invoke GRPO training rewards, consume their statistics, or modify the frozen reference.

## Configuration

```yaml
training:
  eval_strategy: steps
  eval_steps: 100
  save_strategy: steps
  save_steps: 100
  save_total_limit: 3  # GRPO uses 2; keep at least the best and latest.
  load_best_model_at_end: true
  metric_for_best_model: numerical_accuracy
  greater_is_better: true
validation:
  data: data/processed/dev.jsonl
  limit: null
  generation:
    max_input_tokens: 2048
    max_new_tokens: 128
    do_sample: false
  tolerance:
    atol: 1.0e-6
    rtol: 0.001
```

`eval_steps` and `save_steps` must be equal positive integers. Smoke profiles use 16 candidate train records, four dev records, and shorter intervals. Dev checks contribute to training time, and each records its runtime.

## References and verification

- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer): native best-model tracking, retention, and restoration. Installed Transformers 5.22.1 source was checked for final-step evaluation behavior.
- [Transformers generation-based evaluation example](https://github.com/huggingface/transformers/blob/v4.56.2/examples/pytorch/summarization/run_summarization.py): generating answers before computing task metrics. The older API was not transplanted into this project's pinned Transformers 5 implementation.
- [TRL GRPO script](https://github.com/huggingface/trl/blob/main/trl/scripts/grpo.py): separate train/evaluation datasets. Here, fixed greedy numerical accuracy is used for selection instead of sampled training reward.

Explicit tiny random CPU models and controlled outputs test earlier-best retention, tie handling, final-step checks, state restoration, and exported-weight equality. Three-process tests cover uneven shards and overlength records. GPU smoke runs and the full two-GPU SFT experiment also verified selection; see the [report](validation.md).
