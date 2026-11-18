"""Text-only Qwen3.5 loading and shared chat rendering; no eager ML imports."""
import json
import os
from collections.abc import Mapping
from pathlib import Path
import time
from financeqa.common import digest
from financeqa.distributed import local_rank, world_size

DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DEFAULT_CHAT_KWARGS = {"enable_thinking": False}


def chat_kwargs(spec):
    kwargs = dict(spec.get("chat_template_kwargs", DEFAULT_CHAT_KWARGS))
    if kwargs.get("enable_thinking") is not False:
        raise ValueError("This answer-only experiment requires enable_thinking=false in every stage")
    return kwargs


def render_chat(tokenizer, messages, *, generation, template_kwargs=None):
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation,
                                            **(template_kwargs or {}))
    # Transformers 5 can return BatchEncoding by default; older versions return IDs.
    ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("render_chat expects exactly one conversation")
        ids = ids[0]
    return list(ids)


def load_tokenizer(spec, trainable=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"],
                                              cache_dir=spec.get("cache_dir"), trust_remote_code=False)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Model must provide chat template and EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" if trainable else "left"
    return tokenizer


def resolve_device(spec, *, trainable=False):
    import torch
    requested = spec.get("device", "auto")
    if trainable and world_size() > 1:
        if "LOCAL_RANK" not in os.environ:
            raise RuntimeError("Use torchrun to supply LOCAL_RANK for DDP")
        if requested != "auto" and not requested.startswith("cuda"):
            raise ValueError("Distributed model training is configured for CUDA DDP")
        device = f"cuda:{local_rank()}"
    elif requested == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = requested
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use device=cpu and dtype=float32 for CPU inference")
        torch.cuda.set_device(device)
    return device


def load_backbone(spec, trainable=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    kwargs = {"revision": spec["revision"], "cache_dir": spec.get("cache_dir"), "trust_remote_code": False}
    config = AutoConfig.from_pretrained(spec["name"], **kwargs)
    loader = spec.get("loader", "qwen3_5_text")
    if loader == "qwen3_5_text":
        from transformers import Qwen3_5ForCausalLM
        if config.model_type not in ("qwen3_5", "qwen3_5_text"):
            raise ValueError("qwen3_5_text loader requires a Qwen3.5 config")
        config = config.get_text_config()
        model_class = Qwen3_5ForCausalLM
    elif loader == "auto_causal_lm":
        model_class = AutoModelForCausalLM
    else:
        raise ValueError(f"Unsupported loader: {loader}")
    device = resolve_device(spec, trainable=trainable)
    dtype = getattr(torch, spec.get("dtype", "bfloat16"))
    if device == "cpu" and dtype != torch.float32:
        raise RuntimeError("CPU execution requires an explicit dtype=float32")
    model, loading = model_class.from_pretrained(
        spec["name"], config=config, dtype=dtype,
        attn_implementation=spec.get("attn_implementation", "sdpa"),
        output_loading_info=True, **kwargs)
    # Reject accidental random initialization when extracting the text backbone.
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Incomplete text weights: missing={loading.get('missing_keys')}, mismatched={loading.get('mismatched_keys')}")
    model.to(device)
    model.config.use_cache = not trainable
    return model


def load_model(spec, trainable=False, adapter=None):
    chat_kwargs(spec)
    tokenizer = load_tokenizer(spec, trainable=trainable)
    if adapter:
        parent_run = Path(adapter) / "run.json"
        if not parent_run.exists():
            parent_run = Path(adapter).parent / "run.json"
        if parent_run.exists():
            trained_on = json.loads(parent_run.read_text())["config"]["model"]
            if trained_on["name"] != spec["name"] or trained_on["revision"] != spec["revision"]:
                raise ValueError("Adapter provenance model/revision differs from requested base")
        from peft import PeftConfig
        ac = PeftConfig.from_pretrained(adapter)
        if ac.base_model_name_or_path != spec["name"]:
            raise ValueError(f"Adapter base {ac.base_model_name_or_path} differs from requested {spec['name']}")
    model = load_backbone(spec, trainable=trainable)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    if not trainable:
        model.eval()
    return model, tokenizer


def adapter_fingerprint(adapter):
    if not adapter:
        return None
    p = Path(adapter)
    files = sorted(p.glob("adapter_model.*")) + [p / "adapter_config.json"]
    if not files or not all(f.exists() for f in files):
        raise FileNotFoundError(f"Incomplete adapter: {adapter}")
    return {str(f): digest(f) for f in files}


class Predictor:
    def __init__(self, spec, adapter=None):
        if world_size() > 1:
            raise RuntimeError("Inference/evaluation run in one process; do not launch them with torchrun")
        self.spec = spec
        self.model, self.tokenizer = load_model(spec, adapter=adapter)

    def predict(self, prompt, generation, seed=42):
        from transformers import set_seed
        set_seed(seed)
        return generate_answer(self.model, self.tokenizer, self.spec, prompt, generation)


def generate_answer(model, tokenizer, spec, prompt, generation):
    """Shared greedy/scored inference, including identical context-length handling."""
    import torch
    start = time.perf_counter()
    ids = render_chat(tokenizer, prompt, generation=True, template_kwargs=chat_kwargs(spec))
    if len(ids) > generation['max_input_tokens']:
        return {'output':'', 'error':'context_too_long', 'input_tokens':len(ids), 'seconds':0}
    if len(ids) + generation['max_new_tokens'] > model.config.get_text_config().max_position_embeddings:
        raise ValueError('Input+completion exceeds model context window')
    inputs = torch.tensor([ids], device=model.device)
    options = {k:v for k,v in generation.items() if k != 'max_input_tokens'}
    if not options.get('do_sample', False):
        options.update(temperature=None, top_p=None, top_k=None)
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                pad_token_id=tokenizer.pad_token_id, synced_gpus=False, **options)
    if model.device.type == 'cuda':
        torch.cuda.synchronize(model.device)
    return {'output':tokenizer.decode(output[0, len(ids):], skip_special_tokens=True),
            'error':None, 'input_tokens':len(ids), 'generated_tokens':output.shape[1]-len(ids),
            'eos_terminated':int(output[0,-1]) == tokenizer.eos_token_id,
            'seconds':time.perf_counter()-start}


def resource_start():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def resource_finish(start):
    import torch
    return {"seconds": time.perf_counter() - start,
            "max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None,
            "max_memory_reserved_gib": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None}


def check_training_environment(grpo=False):
    import os
    import shutil
    import sysconfig
    import torch
    count = torch.cuda.device_count()
    if count > 1 and world_size() == 1:
        raise RuntimeError("Multiple GPUs visible: launch with torchrun --nproc_per_node=N, or expose only one GPU")
    if world_size() > 1:
        if "LOCAL_RANK" not in os.environ or not 0 <= local_rank() < count:
            raise RuntimeError("Invalid LOCAL_RANK / visible CUDA devices; launch one process per GPU with torchrun")
        torch.cuda.set_device(local_rank())
    if grpo:
        if not torch.cuda.is_available():
            raise RuntimeError("TRL fused GRPO kernel requires CUDA in this configuration")
        if not (Path(sysconfig.get_paths()["include"]) / "Python.h").is_file():
            raise RuntimeError("GRPO Triton needs Python.h. Install python3.11-dev or use uv Python 3.11.18")
        if not shutil.which("gcc") or not shutil.which("g++"):
            raise RuntimeError("GRPO Triton requires GCC and G++ on PATH")
