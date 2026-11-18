# FinQA SFT + GRPO

LoRA fine-tuning and verifiable-reward optimization for financial numerical QA with **Qwen3.5-2B**.

[Guide](docs/guide.md) · [Experiments](docs/validation.md) · [Data](data/README.md)

## Results

| Model | Dev accuracy |
|---|---:|
| Base | 18.71% |
| LoRA SFT | **48.02%** |

Single-seed results on 481 scorable ratio/percentage questions from 883 FinQA dev records. SFT is dev-selected; these are not full-FinQA or held-out test scores.

SFT baseline complete. Full GRPO and continued-SFT experiments are pending.

## Quickstart

Python 3.11 and CUDA GPUs.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python -m data.prepare_data --download
```

See the [guide](docs/guide.md#reproduce-the-base-and-sft-baseline) for training, evaluation, and the demo.

## License

[MIT](LICENSE). Upstream data and model licenses are listed in [data provenance](data/README.md).
