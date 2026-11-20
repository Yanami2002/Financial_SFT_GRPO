# User guide

Run all commands from the repository root. For project status and the current baseline, see the [experiment report](validation.md).

## Repository layout

```text
financial-sft-grpo/
├── configs/       # SFT, GRPO, DAPO, evaluation, smoke tests, and controls
├── data/          # Pinned data download and deterministic conversion
├── financeqa/     # Model loading, templates, provenance, and DDP helpers
├── train/         # Assistant-only SFT, GRPO, DAPO, and dev checkpoint selection
├── rewards/       # Numerical correctness and format rewards
├── eval/          # Scoring, protocol checks, and checkpoint comparison
├── inference/     # JSONL prediction CLI
├── demo/          # Local Gradio interface
├── scripts/       # Setup, configuration generation, and training entry points
├── tests/         # Data, scoring, model, checkpoint, and distributed tests
├── docs/          # Reproduction instructions and curated experiment summaries
└── requirements.txt
```

Downloaded data, weights, raw run outputs, caches, and virtual environments are excluded from Git. Their preparation commands, pinned versions, and curated results are included.

## Installation

Use Python 3.11. The tested environment uses Python 3.11.18, PyTorch 2.7.1/cu126, Transformers 5.22.1, PEFT 0.21.0, TRL 1.17.1, and Accelerate 1.15.0.

```bash
uv python install 3.11.18
uv venv --python 3.11.18 .venv
source .venv/bin/activate
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```

`requirements.txt` is the dependency installation entry point. The full tested package snapshot is in [docs/environment.txt](environment.txt). GRPO Triton compilation requires GCC, G++, and Python development headers; the tested uv-managed Python includes the headers.

The implementation uses native PyTorch/Transformers generation. Optional `causal-conv1d` and `flash-linear-attention` kernels are absent in the reported environment, which uses slower PyTorch fallback implementations. Optimized-kernel performance has not been measured.

## Prepare data and check the setup

```bash
python -m data.prepare_data --download
CUDA_VISIBLE_DEVICES='' python -m scripts.check_setup --world-size 2 --tokenizer
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests
```

The setup check reads model configuration and tokenizer metadata, not pretrained weights. Tests use explicitly identified tiny random models and controlled fixtures; they do not fine-tune the official Qwen checkpoint. Tests requiring optional ML packages are skipped when those packages are absent. A data/scoring-only environment requires `PyYAML==6.0.3` and `pytest==8.4.2`.

To check three-process CPU/Gloo communication and the reward-group protocol:

```bash
CUDA_VISIBLE_DEVICES='' python -m torch.distributed.run --standalone \
  --nnodes=1 --nproc_per_node=3 -m scripts.check_distributed \
  --output results/qwen3.5-2b/cpu-ddp-check
```

Offline preparation accepts official JSON files with `--raw-dir /path/to/official-json`. Sources, licenses, filtering, and limitations are documented in [data/README.md](../data/README.md).

## Model and scoring protocol

- Base: `Qwen/Qwen3.5-2B`, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.
- Text-only loading through `Qwen3_5ForCausalLM`; the vision encoder is not loaded. Missing or mismatched language weights cause an error.
- `enable_thinking: false` is consistent across token filtering, training, evaluation, inference, and control-data construction.
- LoRA uses rank 16, alpha 32, dropout 0.05, and `target_modules: all-linear`, covering full-attention, Gated DeltaNet, and MLP projections. Embeddings and `lm_head` remain frozen.
- SFT supervises annotated assistant answers and ending tokens. System/user text, the assistant header, empty thinking prefix, and padding are masked. No synthetic reasoning traces are added.
- Prompts contain complete `pre_text`, table, and `post_text`. Reference answers, annotated programs, and gold evidence never enter prompts.

Official train/dev/test sizes are **6251/883/1147**. The verified subset contains **3428/481/638** examples, currently all ratios or percentages. The reported SFT run retained **3363** training examples after length filtering.

Outputs use JSON such as `{"answer":20,"unit":"percent"}` or the equivalent `{"answer":0.2,"unit":"ratio"}`. Scoring checks units, sign, and numerical tolerance without guessing units or scales. Correctness requires error no greater than `max(config.atol, reference.atol, config.rtol * abs(reference.value))`. Defaults are `atol=1e-6`, `rtol=1e-3`; `reference.atol` captures annotation rounding. Final accuracy excludes the format reward.

Training skips whole overlength records and logs their IDs. Evaluation retains them as `context_too_long`; scorable overlength examples remain in the accuracy denominator.

## Reproduce the Base and SFT baseline

Choose allocated GPUs and use a fresh checkout or separate output directories for additional runs. The following reproduces the reported two-GPU configuration:

```bash
# Replace the example device IDs with your allocated GPUs.
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

# Base uses one GPU and the complete dev split.
CUDA_VISIBLE_DEVICES=0 python -m eval.evaluate --config configs/eval_base_dev.yaml

# This command starts real two-GPU LoRA training.
NPROC_PER_NODE=2 bash scripts/train.sh sft
```

SFT uses three epochs, batch size one per device, and gradient accumulation eight. Two workers give an effective batch of 16. DDP assigns devices by `LOCAL_RANK`; each worker holds a complete base model and adapter. GPU memory is not pooled, and FSDP/ZeRO/offloading are not configured. Single-GPU training uses one visible device and `NPROC_PER_NODE=1`.

Training checks the complete dev split every 100 optimizer steps and at the final step, selecting by generated-answer numerical accuracy. Ties keep the earliest evaluated step. Native Trainer retention protects the best and latest checkpoints. The best adapter is restored before final export. See [checkpoint selection](checkpoint-selection.md).

| Local artifact in each stage output directory | Contents |
|---|---|
| `best_checkpoint.json` | Selected step, score, and final-export status |
| `dev_history.jsonl` | Scores at each dev check |
| `dev-evaluations/step-N/` | Predictions, metrics, and evaluation provenance |
| `checkpoint-N/` | Retained model and training-state checkpoint |
| `run.json`, `train_metrics.json` | Configuration, hashes, versions, and resource use |

The completed run selected step 500 rather than the final step, 633. The adapter is available locally at `outputs/qwen3.5-2b/sft`; weights are not bundled with the repository. The [report](validation.md) includes a protocol-checked Base/SFT comparison command.

## GRPO and continued-SFT control

After SFT, build the shared training cohort and run GRPO with native TRL generation. These settings match the completed experiment:

```bash
python -m scripts.control_cohort
NPROC_PER_NODE=2 bash scripts/train.sh grpo --config configs/grpo_equal_data.yaml \
  --set training.output_dir=outputs/qwen3.5-2b/grpo
```

GRPO defaults to eight completions per question, one epoch, a 2048-token prompt budget, and a 128-token completion budget. Numerical correctness has weight 1.0 and format has weight 0.05. With `beta=0.04`, TRL copies the SFT adapter to a frozen KL reference; its precision, initial equality, and frozen state are checked. Only the trained default adapter is exported. Inference uses the original base plus that adapter.

Global generation batches must be divisible by `num_generations`. Configurations are checked for one, two, and three workers. Cross-rank diagnostics record complete reward groups and flag identical rewards. This profile does not use vLLM or an additional generation GPU.

The common cohort contains 3,363 training examples that fit both token budgets. To run the continued-SFT control from the same selected SFT model:

```bash
NPROC_PER_NODE=2 bash scripts/train.sh sft --config configs/continued_sft.yaml
```

Report sample exposure, generated tokens, and compute separately. Equal step counts can involve different amounts of compute. Continued-SFT results remain pending.

## DAPO

After SFT and `python -m scripts.control_cohort`, run:

```bash
NPROC_PER_NODE=2 bash scripts/train.sh dapo
```

DAPO starts from the same selected SFT adapter and uses dynamic sampling, a loss normalized by accepted tokens, asymmetric clipping, and zero KL. The default output is `outputs/qwen3.5-2b/dapo`. See the [recipe and results](dapo.md) for the exact configuration and a command to compare the selected dev checkpoints.

## Final evaluation and inference

After fixing settings and selecting checkpoints using dev:

```bash
python -m eval.evaluate --config configs/eval.yaml
python -m eval.compare results/qwen3.5-2b/test/{base,sft,grpo}/metrics.json \
  --output results/qwen3.5-2b/comparison.json
```

`configs/eval.yaml` targets **test**. To include DAPO in the test evaluation or demo, copy the config and add `dapo: {adapter: outputs/qwen3.5-2b/dapo}` under `variants`. The published metrics use dev only. Comparison rejects mismatched IDs, models, prompts, seeds, tolerances, or generation parameters. Evaluation, inference, and the demo use one process and load variants sequentially.

```bash
python -m inference.predict --input /path/to/input.jsonl --output results/predict.jsonl
python -m inference.predict --adapter outputs/qwen3.5-2b/sft \
  --input /path/to/input.jsonl --output results/predict-sft.jsonl
python -m demo.app --config configs/eval.yaml
```

CLI records contain `context` and `question`, with an optional `id`. The demo defaults to `localhost:7860` and lists available adapters. Outputs are experimental and must not be relied on for investment decisions.

## Validation and publication scope

The two-RTX-A6000 baseline took about 4 hours 24 minutes for SFT, including seven complete dev checks. Peak allocated memory was about 9.72 GiB per worker; reserved peaks were 33.19/45.72 GiB. These observations are configuration-specific, not universal memory requirements.

The full dependency environment passed 83 tests with CUDA hidden. Real GPU training and checkpoint selection were also checked in smoke runs. See [validation details](validation.md).

Commit source, configurations, tests, provenance documentation, and curated summaries. Raw logs/predictions, downloaded data, checkpoints, environments, credentials, and caches remain outside Git. The MIT [license](../LICENSE) covers project code; upstream data, models, and dependencies retain their own licenses.
