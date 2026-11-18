# FinQA SFT + GRPO

LoRA fine-tuning and verifiable-reward optimization for financial numerical QA with **Qwen3.5-2B**.

[Guide](docs/guide.md) · [Experiments](docs/validation.md) · [Data](data/README.md)

## Training Pipeline

FinQA filtering → LoRA SFT → GRPO → evaluation.

SFT learns annotated JSON answers. GRPO continues the SFT adapter using numerical correctness and output-format rewards.

## Results

| Model | Dev accuracy |
|---|---:|
| Base | 18.71% |
| LoRA SFT | **48.02%** |

Single-seed results on 481 scorable ratio/percentage questions from 883 FinQA dev records. SFT is dev-selected; these are not full-FinQA or held-out test scores.

SFT baseline complete. Full GRPO and continued-SFT experiments are pending.

## SFT Setup

| Setting | Value |
|---|---|
| Model | Qwen3.5-2B, BF16, thinking disabled |
| Data | FinQA ratio/percentage subset; 3,363 training examples after length filtering |
| Supervision | Assistant answers only; no synthetic reasoning traces |
| LoRA | Rank 16, alpha 32, dropout 0.05, `all-linear` |
| Training | 3 epochs, learning rate 1e-4, cosine decay, maximum sequence length 2048 |

See the [full configuration](configs/sft.yaml) for details.

## GRPO Setup

Current recipe; full GRPO results are pending.

| Setting | Value |
|---|---|
| Initialization | Trained SFT adapter |
| Sampling | 8 responses per question, up to 128 new tokens each |
| Reward | `1.0 × numerical correctness + 0.05 × valid output format` |
| Training | 1 epoch, learning rate 5e-6, KL coefficient 0.04 |

Correctness checks units, scales, and numerical tolerance. Format requires parseable JSON with a valid answer and supported unit. See the [configuration](configs/grpo.yaml).

## Quickstart

Python 3.11 and CUDA GPUs.

```bash
git clone https://github.com/Yanami2002/Financial_SFT_GRPO.git
cd Financial_SFT_GRPO
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python -m data.prepare_data --download
```

Train SFT before GRPO; adapters are not bundled. Replace the example GPU IDs with your allocated devices.

```bash
export CUDA_VISIBLE_DEVICES=0,1
NPROC_PER_NODE=2 bash scripts/train.sh sft
NPROC_PER_NODE=2 bash scripts/train.sh grpo

# Launch the demo with available models.
CUDA_VISIBLE_DEVICES=0 python -m demo.app --config configs/eval.yaml
```

Open http://127.0.0.1:7860. See the [guide](docs/guide.md#reproduce-the-base-and-sft-baseline) for full reproduction and evaluation instructions.

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
