"""Single-node torchrun/DDP helpers; collectives are called by every rank."""
import os


def rank():
    return int(os.environ.get("RANK", "0"))


def world_size():
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process():
    return rank() == 0


def gather_records(records):
    """Flatten rank-ordered records, matching TRL's reward gather order."""
    if world_size() == 1:
        return records
    import torch.distributed as dist
    if not dist.is_initialized():
        raise RuntimeError("Distributed collectives require an initialized process group")
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, records)
    return [row for rows in gathered for row in rows]


def validate_batch(training, *, grpo=False, workers=None):
    workers = world_size() if workers is None else workers
    micro = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    if min(workers, micro, accumulation) < 1:
        raise ValueError("world size, micro batch, and accumulation must be positive")
    effective = workers * micro * accumulation
    result = {"world_size": workers, "effective_batch_size": effective}
    if grpo:
        generations = training["num_generations"]
        generation_batch = training.get("generation_batch_size")
        steps = training.get("steps_per_generation")
        if generation_batch is not None and steps is not None:
            raise ValueError("Specify generation_batch_size OR steps_per_generation")
        generation_batch = generation_batch or workers * micro * (steps or accumulation)
        if generations < 2 or generation_batch % generations:
            raise ValueError(f"Generation batch {generation_batch} must be divisible by num_generations={generations} >= 2")
        if generation_batch % (workers * micro):
            raise ValueError("Generation batch must be divisible by world_size * per-device batch")
        result.update(generation_batch_size=generation_batch,
                      prompts_per_generation_batch=generation_batch // generations)
    return result


def save_training(trainer, tokenizer, out, result, start):
    """DDP final adapter/tokenizer/state: one writer, all-rank resource records."""
    from pathlib import Path
    from financeqa.common import write_json
    from financeqa.modeling import resource_finish
    trainer.accelerator.wait_for_everyone()
    resources = gather_records([{"rank": rank(), **resource_finish(start)}])
    if trainer.is_world_process_zero():
        model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
        model.save_pretrained(out, selected_adapters=["default"])
        tokenizer.save_pretrained(out)
        trainer.save_state()
        write_json(Path(out) / "train_metrics.json", {**result.metrics, "resources_by_rank": resources})
    trainer.accelerator.wait_for_everyone()
