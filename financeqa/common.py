import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from financeqa.distributed import is_main_process, rank, world_size, local_rank

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = '''Answer the financial numerical question using only the supplied report. Return exactly one JSON object with keys "answer" (a number) and "unit". Allowed units: ratio (dimensionless fraction), percent (percentage points: 20 means 20%), count, years, usd, usd_million, usd_billion. For example a percentage can be expressed as percent or its equivalent fraction in ratio. Preserve the sign. Do not guess an unknown unit. If the context is insufficient, return {"answer": null, "unit": "unknown"}. No markdown or prose.'''


def messages(context, question):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Report context:\n{context}\n\nQuestion: {question}"}]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def config_cli(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True)
    p.add_argument("--set", action="append", default=[], metavar="KEY=YAML_VALUE")
    p.add_argument("--dry-run", action="store_true", help="Validate config/data, without loading weights")
    args = p.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    for override in args.set:
        key, value = override.split("=", 1)
        target = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return cfg, args


def record_run(out, cfg, files=()):
    packages = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "gradio"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"started_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {str(p.relative_to(ROOT)): digest(p) for directory in ("financeqa", "data", "rewards", "train", "eval", "inference", "demo", "scripts") for p in sorted((ROOT / directory).glob("*.py"))},
            "config": cfg, "python": platform.python_version(), "packages": packages,
            "files_sha256": {str(p): digest(p) for p in files},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "distributed": {"rank": rank(), "local_rank": local_rank(), "world_size": world_size()},
            "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()}
    try:
        import torch
        info["gpu"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    if is_main_process():
        write_json(Path(out) / "run.json", info)
    return info


def training_rows(path, limit=None):
    rows = read_jsonl(path)
    if not rows or any(r.get("split") != "train" for r in rows):
        raise ValueError("Training data must be nonempty and exclusively official train split")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    return rows[:limit] if limit else rows


def validation_data(cfg, train_rows):
    """Validate checkpoint selection without importing ML packages or loading weights."""
    from rewards.numerical import validate_reference
    spec, training = cfg['validation'], cfg['training']
    if (training.get('eval_strategy') != 'steps' or training.get('save_strategy') != 'steps'
            or training.get('eval_steps') != training.get('save_steps')
            or not isinstance(training.get('eval_steps'), int) or training['eval_steps'] < 1):
        raise ValueError('Checkpoint selection requires equal positive integer eval_steps/save_steps and steps strategies')
    if (training.get('load_best_model_at_end') is not True
            or training.get('metric_for_best_model') != 'numerical_accuracy'
            or training.get('greater_is_better') is not True):
        raise ValueError('Select checkpoints by numerical_accuracy, greater_is_better=true, load_best_model_at_end=true')
    if training.get('save_total_limit') is not None and training['save_total_limit'] < 2:
        raise ValueError('save_total_limit must retain at least the best and latest checkpoints')
    if spec['generation'].get('do_sample') is not False:
        raise ValueError('Dev checkpoint selection requires greedy generation: do_sample=false')
    for key in ('max_input_tokens', 'max_new_tokens'):
        if not isinstance(spec['generation'].get(key), int) or spec['generation'][key] < 1:
            raise ValueError(f'Validation {key} must be a positive integer')
    limit = spec.get('limit')
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError('Validation limit must be null or a positive integer')
    rows = read_jsonl(spec['data'])
    if not rows or any(row.get('split') != 'dev' for row in rows):
        raise ValueError('Checkpoint selection requires exclusively official dev records, never train/test')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate dev IDs')
    if {r['id'] for r in rows} & {r['id'] for r in train_rows}:
        raise ValueError('Dev IDs overlap training IDs')
    if any(not r.get('prompt') or any(m['role'] not in ('system', 'user') for m in r['prompt']) for r in rows):
        raise ValueError('Dev prompts must contain only system/user messages, without reference answers')
    selected = rows[:limit] if limit else rows
    if not any(validate_reference(r.get('reference')) for r in selected):
        raise ValueError('No scorable dev records for checkpoint selection')
    return selected
