"""Continue SFT adapter using TRL 1.17.1; references remain reward-only columns."""
import json
from collections import defaultdict
from pathlib import Path
import statistics
from financeqa.common import config_cli, record_run, training_rows, validation_data, write_json
from financeqa.distributed import is_main_process, gather_records, validate_batch, save_training
from financeqa.modeling import load_model, adapter_fingerprint, resource_start, check_training_environment, render_chat, chat_kwargs
from rewards.numerical import score, completion_text, validate_reference
from rewards.format_reward import format_reward


class DiagnosticReward:
    __name__ = "numerical_correctness"

    def __init__(self, path, tolerance, format_weight=.05, num_generations=None):
        self.path = Path(path)
        self.tolerance = tolerance
        self.format_weight = format_weight
        self.call = 0
        self.num_generations = num_generations
        if is_main_process():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def __call__(self, completions, reference, id, **kwargs):
        if len(completions) != len(reference) or len(id) != len(reference):
            raise ValueError("Reward columns do not align")
        scores = [score(completion_text(c), r, **self.tolerance) for c, r in zip(completions, reference)]
        # Callbacks receive LOCAL completions; TRL gathers their returned rewards later.
        # Gather diagnostics explicitly before grouping so a group can span ranks.
        records = gather_records([{"id": key, "score": result} for key, result in zip(id, scores)])
        if is_main_process():
            self.call += 1
            groups = diagnostic_groups(records, self.num_generations)
            with self.path.open("a", encoding="utf-8") as f:
                for group_index, (key, values) in enumerate(groups):
                    rewards = [float(v["correct"]) + self.format_weight * float(v["parseable"]) for v in values]
                    row = {"call": self.call, "group": group_index, "id": key, "n": len(values), "rewards": rewards,
                           "mean": statistics.mean(rewards), "std": statistics.pstdev(rewards),
                           "parseable_rate": statistics.mean(float(v["parseable"]) for v in values),
                           "accuracy": statistics.mean(float(v["correct"]) for v in values)}
                    f.write(json.dumps(row) + "\n")
                    if row["std"] == 0:
                        print(f"GRPO diagnostic: id={key} all {len(values)} rewards equal ({row['mean']}); no relative learning signal", flush=True)
        return [float(v["correct"]) for v in scores]


def diagnostic_groups(records, num_generations=None):
    if num_generations is None:
        groups = defaultdict(list)
        for row in records:
            groups[row["id"]].append(row["score"])
        return list(groups.items())
    if len(records) % num_generations:
        raise ValueError("Global reward records do not form complete generation groups")
    groups = []
    for start in range(0, len(records), num_generations):
        group = records[start:start + num_generations]
        if len({r["id"] for r in group}) != 1:
            raise ValueError("GRPO diagnostic group order disagrees with TRL's rank-ordered reward groups")
        groups.append((group[0]["id"], [r["score"] for r in group]))
    return groups


def align_reference_adapter(model):
    """TRL/PEFT may create ref in BF16 while the trained SFT adapter is FP32.

    Verify the upstream copy (at its dtype), then preserve exact SFT precision.
    This runs before optimizer construction/training and never updates the policy.
    """
    import torch
    refs = [(n, p) for n, p in model.named_parameters() if ".ref." in n]
    if not refs or any(p.requires_grad for _, p in refs):
        raise RuntimeError("Expected a frozen SFT reference adapter")
    aligned = 0
    for name, param in refs:
        source = model.get_parameter(name.replace(".ref.", ".default."))
        if not torch.equal(param, source.to(param.dtype)):
            raise RuntimeError("Unexpected reference values before precision alignment")
        if param.dtype != source.dtype:
            param.data = source.detach().clone()
            aligned += 1
        if not torch.equal(param, source):
            raise RuntimeError("Reference adapter must equal the initial SFT adapter")
    return {"reference_tensor_count": len(refs), "precision_aligned_tensor_count": aligned,
            "initial_reference_equals_sft": True}


def main(dapo=False):
    cfg, args = config_cli(__doc__)
    if dapo:
        sampling = cfg['dynamic_sampling']
        if not isinstance(sampling['max_generation_batches'], int) or sampling['max_generation_batches'] < 1:
            raise ValueError('DAPO max_generation_batches must be a positive integer')
        if cfg['training']['loss_type'] != 'dapo' or cfg['training']['beta'] != 0:
            raise ValueError('DAPO requires loss_type=dapo and beta=0')
        if not 0 < cfg['overlong_buffer']['length'] < cfg['training']['max_completion_length']:
            raise ValueError('DAPO overlength buffer must fit the completion budget')
    rows = training_rows(cfg["data"], cfg.get("limit"))
    dev_rows = validation_data(cfg, rows)
    if any(not validate_reference(r.get("reference")) for r in rows):
        raise ValueError("GRPO requires a verified numerical reference for every row")
    if not cfg.get("sft_adapter"):
        raise ValueError("GRPO requires an existing SFT adapter")
    batch = validate_batch(cfg["training"], grpo=True)
    template_kwargs = chat_kwargs(cfg["model"])
    if cfg["training"].get("use_vllm", False):
        raise ValueError("This profile uses native TRL generation on DDP workers; vLLM is not configured")
    if args.dry_run:
        if is_main_process():
            print({"algorithm": "dapo" if dapo else "grpo", "samples": len(rows), "dev_samples": len(dev_rows), "sft_adapter": cfg["sft_adapter"], "num_generations": cfg["training"]["num_generations"], **batch})
        return
    check_training_environment(grpo=True)
    if Path(cfg["sft_adapter"]).resolve() == Path(cfg["training"]["output_dir"]).resolve():
        raise ValueError("GRPO output directory must differ from SFT input adapter")
    import importlib.metadata
    if importlib.metadata.version("trl") != "1.17.1":
        raise RuntimeError("This implementation was checked against trl==1.17.1; install requirements.txt")
    from transformers import set_seed
    from trl import GRPOConfig
    from train.validation import NumericalGRPOTrainer
    from datasets import Dataset
    weights = [1.0] if dapo else [1.0, cfg["format_weight"]]
    if not dapo and not 0 <= weights[1] <= .1:
        raise ValueError("format_weight must be between 0 and 0.1")
    training_args = GRPOConfig(seed=cfg["seed"], data_seed=cfg["seed"], report_to="none",
                               reward_weights=weights, remove_unused_columns=False,
                               chat_template_kwargs=template_kwargs, **cfg["training"])
    set_seed(cfg["seed"])
    start = resource_start()
    model, tokenizer = load_model(cfg["model"], trainable=True, adapter=cfg["sft_adapter"])
    tokenizer.padding_side = "left"
    model.enable_input_require_grads()
    kept, dropped = [], []
    for row in rows:
        ids = render_chat(tokenizer, row["prompt"], generation=True, template_kwargs=template_kwargs)
        if len(ids) > cfg["max_prompt_tokens"]:
            dropped.append(row["id"])
        else:
            kept.append({k: row[k] for k in ("prompt", "reference", "id")})
    if not kept:
        raise ValueError("No GRPO prompts fit max_prompt_tokens")
    if cfg["max_prompt_tokens"] + cfg["training"]["max_completion_length"] > model.config.get_text_config().max_position_embeddings:
        raise ValueError("Prompt+completion budget exceeds model context window")
    out = Path(cfg["training"]["output_dir"])
    if is_main_process():
        provenance = record_run(out, cfg, [cfg["data"], cfg["validation"]["data"], Path(cfg["data"]).parent / "manifest.json"])
        provenance["sft_adapter_sha256"] = adapter_fingerprint(cfg["sft_adapter"])
        provenance["batch"] = batch
        write_json(out / "run.json", provenance)
        write_json(out / "length_filter.json", {"kept": len(kept), "dropped_ids": dropped})
    extra = {}
    trainer_class = NumericalGRPOTrainer
    if dapo:
        from rewards.dapo import DAPOReward
        from train.dapo_trainer import NumericalDAPOTrainer
        trainer_class = NumericalDAPOTrainer
        reward = DAPOReward(out / 'reward_groups.jsonl', cfg['tolerance'], training_args.num_generations,
                            training_args.max_completion_length, cfg['overlong_buffer']['length'],
                            cfg['overlong_buffer']['penalty_factor'])
        extra = {'dynamic_sampling': cfg['dynamic_sampling'], 'dapo_reward': reward}
        reward_funcs = [reward]
    else:
        reward = DiagnosticReward(out / "reward_groups.jsonl", cfg["tolerance"], cfg["format_weight"], training_args.num_generations)
        reward_funcs = [reward, format_reward]
    # No peft_config: train the loaded default adapter. With beta>0 TRL copies it to frozen ref.
    trainer = trainer_class(model=model, processing_class=tokenizer, args=training_args,
                          reward_funcs=reward_funcs, train_dataset=Dataset.from_list(kept),
                          eval_dataset=dev_rows, validation_config=cfg['validation'], model_spec=cfg['model'], **extra)
    if training_args.beta > 0:
        setup = align_reference_adapter(model)
        if is_main_process():
            write_json(out / "policy_setup.json", {"mode": "continue_default_adapter", "reference": "frozen_sft_copy",
                       **setup, "disable_dropout": training_args.disable_dropout})
    elif dapo and is_main_process():
        write_json(out / 'policy_setup.json', {'mode': 'continue_default_adapter', 'algorithm': 'dapo',
                   'reference': None, 'beta': 0, 'disable_dropout': training_args.disable_dropout,
                   'dynamic_sampling': cfg['dynamic_sampling'], 'overlong_buffer': cfg['overlong_buffer'],
                   'epsilon_low': training_args.epsilon, 'epsilon_high': training_args.epsilon_high,
                   'loss_type': training_args.loss_type})
    result = trainer.train()
    save_training(trainer, tokenizer, out, result, start)
    trainer.write_selection()

if __name__ == "__main__":
    main()
