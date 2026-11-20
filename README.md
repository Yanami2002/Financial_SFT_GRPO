# FinQA SFT + GRPO

Given a financial report excerpt and a question, **Qwen3.5-2B** returns a numerical answer and its unit as JSON. This project compares LoRA SFT, GRPO, and a short-answer DAPO adaptation on the same FinQA evaluation subset.

[Quickstart](#quickstart) · [User guide](docs/guide.md) · [Experiment reports](docs/validation.md) · [Data and scoring](data/README.md)

## Results

| Model | Selected step | Correct / 481 | Dev accuracy | Change vs. SFT |
|---|---:|---:|---:|---:|
| Qwen3.5-2B Base | — | 90 | 18.71% | — |
| [LoRA SFT](docs/validation.md#completed-base-and-two-gpu-sft-baseline) | 500 | 231 | 48.02% | — |
| [SFT + GRPO](docs/grpo-baseline.md) | 400 | 245 | 50.94% | +2.91 pp |
| [SFT + DAPO](docs/dapo.md) | 200 | 246 | 51.14% | +3.12 pp |

All scores use the same 481 scorable ratio and percentage questions from 883 FinQA dev records. The fine-tuned checkpoints were selected on dev using seed 42. DAPO answered one more question correctly than GRPO at the selected checkpoints. The two recipes differ in sampling and compute. Multi-seed, continued-SFT, and held-out test comparisons remain open.

## Training

SFT learns to produce JSON from FinQA's annotated answers. GRPO and DAPO each start from the selected SFT adapter. GRPO rewards numerical correctness plus a 0.05 format bonus. DAPO filters groups that lack a mix of correct and incorrect answers and applies a soft penalty to long answers. Fine-tuning uses 3,363 training examples after length filtering. See the [configurations](configs/) and [scoring rules](data/README.md).

## Quickstart

The reported training runs used Python 3.11 and two CUDA GPUs. Replace the device IDs in the examples with your allocated GPUs.

```bash
git clone https://github.com/Yanami2002/Financial_SFT_GRPO.git
cd Financial_SFT_GRPO
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python -m data.prepare_data --download
python -m scripts.control_cohort
```

Train SFT first to create the starting adapter, then run either RL recipe or both. Each recipe writes its own adapter.

```bash
export CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2 bash scripts/train.sh sft
NPROC_PER_NODE=2 bash scripts/train.sh grpo --config configs/grpo_equal_data.yaml \
  --set training.output_dir=outputs/qwen3.5-2b/grpo
NPROC_PER_NODE=2 bash scripts/train.sh dapo

# Launch the demo with available models.
CUDA_VISIBLE_DEVICES=0 python -m demo.app --config configs/eval.yaml
```

Open http://127.0.0.1:7860 for the local demo. See the [user guide](docs/guide.md) for the repository layout, evaluation, and detailed reproduction instructions. Git tracks the code, configurations, and curated results. Downloaded data, adapters, and raw run outputs are generated locally.

## Example

Selected FinQA dev record: `CDNS/2012/page_30.pdf-1`. Context excerpt below; recorded predictions used the full report context.

| Cumulative investment value | Dec 29, 2007 | Dec 29, 2012 |
|---|---:|---:|
| Cadence Design Systems | 100.00 | 78.92 |

**Question:** What was the percentage cumulative 5-year total stockholder return for Cadence Design Systems Inc. for the five years ended 12/29/2012?

**Reference:** −21.08%, calculated as `(78.92 - 100) / 100`.

| Model | Recorded output | Correct |
|---|---|---|
| Base | `{"answer": 78.92, "unit": "percent"}` | No |
| SFT | `{"answer": -0.2108, "unit": "ratio"}` | Yes |

## License

[MIT](LICENSE). Upstream data and model licenses are listed in [data provenance](data/README.md).
