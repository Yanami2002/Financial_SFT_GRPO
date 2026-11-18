"""LoRA SFT with explicit assistant-only loss and no silent truncation."""
from pathlib import Path
from financeqa.distributed import is_main_process, validate_batch, save_training
from financeqa.common import config_cli, record_run, training_rows, validation_data, write_json
from financeqa.modeling import load_model, resource_start, check_training_environment, adapter_fingerprint, render_chat, chat_kwargs


def tokenize_example(row, tokenizer, max_length, template_kwargs=None):
    msgs = row["messages"]
    if len(msgs) != 3 or [x["role"] for x in msgs] != ["system", "user", "assistant"]:
        raise ValueError("SFT requires system/user/assistant messages")
    prompt = render_chat(tokenizer, msgs[:-1], generation=True, template_kwargs=template_kwargs)
    full = render_chat(tokenizer, msgs, generation=False, template_kwargs=template_kwargs)
    if full[:len(prompt)] != prompt:
        raise ValueError("Chat-template prompt is not a prefix: cannot safely mask assistant loss")
    if len(full) > max_length:
        return None
    if len(full) <= len(prompt):
        raise ValueError("Empty assistant supervision")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}


class AssistantCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        import torch
        length = max(len(r["input_ids"]) for r in rows)
        batch = {}
        for key, pad in (("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)):
            batch[key] = torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in rows])
        return batch


def main():
    cfg, args = config_cli(__doc__)
    rows = training_rows(cfg["data"], cfg.get("limit"))
    dev_rows = validation_data(cfg, rows)
    batch = validate_batch(cfg["training"])
    template_kwargs = chat_kwargs(cfg["model"])
    if args.dry_run:
        if is_main_process():
            print({"samples": len(rows), "dev_samples": len(dev_rows), "assistant_only": True, **batch})
        return
    check_training_environment()
    if cfg.get("init_adapter") and Path(cfg["init_adapter"]).resolve() == Path(cfg["training"]["output_dir"]).resolve():
        raise ValueError("Output directory must differ from the initial adapter directory")
    from transformers import TrainingArguments, set_seed
    from train.validation import NumericalSFTTrainer
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    training_args = TrainingArguments(seed=cfg["seed"], data_seed=cfg["seed"], report_to="none",
                                     remove_unused_columns=False, **cfg["training"])
    set_seed(cfg["seed"])
    start = resource_start()
    model, tokenizer = load_model(cfg["model"], trainable=True, adapter=cfg.get("init_adapter"))
    if not cfg.get("init_adapter"):
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **cfg["lora"]))
    model.enable_input_require_grads()
    tokenized, dropped = [], []
    for row in rows:
        item = tokenize_example(row, tokenizer, cfg["max_length"], template_kwargs)
        if item is None:
            dropped.append(row["id"])
        else:
            tokenized.append(item)
    if not tokenized:
        raise ValueError("No examples fit max_length; increase it. Contexts are never silently truncated.")
    out = cfg["training"]["output_dir"]
    if is_main_process():
        provenance = record_run(out, cfg, [cfg["data"], cfg["validation"]["data"], Path(cfg["data"]).parent / "manifest.json"])
        provenance["init_adapter_sha256"] = adapter_fingerprint(cfg.get("init_adapter"))
        provenance["batch"] = batch
        write_json(Path(out) / "run.json", provenance)
        write_json(Path(out) / "length_filter.json", {"kept": len(tokenized), "dropped_ids": dropped})
    trainer = NumericalSFTTrainer(model=model, args=training_args, train_dataset=Dataset.from_list(tokenized),
                      data_collator=AssistantCollator(tokenizer.pad_token_id), processing_class=tokenizer,
                      eval_dataset=dev_rows, validation_config=cfg['validation'], model_spec=cfg['model'])
    result = trainer.train()
    save_training(trainer, tokenizer, out, result, start)
    trainer.write_selection()

if __name__ == "__main__":
    main()
