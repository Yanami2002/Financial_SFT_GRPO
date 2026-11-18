import argparse
import json
from financeqa.common import messages, read_jsonl, write_jsonl
from financeqa.modeling import Predictor, DEFAULT_MODEL, DEFAULT_REVISION
from rewards.numerical import parse_output


def main():
    p = argparse.ArgumentParser(description="Predict JSONL {context, question}, sequentially on one model")
    p.add_argument("--base", default=DEFAULT_MODEL)
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--adapter")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--cache-dir", default=".cache/huggingface")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--device", default="auto")
    p.add_argument("--loader", choices=["qwen3_5_text", "auto_causal_lm"], default="qwen3_5_text")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-input-tokens", type=int, default=2048)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    rows = read_jsonl(a.input)
    if not rows or any(not isinstance(r.get("context"), str) or not isinstance(r.get("question"), str) for r in rows):
        p.error("Input must be nonempty JSONL with string context and question")
    if a.dry_run:
        print(json.dumps({"n": len(rows), "base": a.base, "adapter": a.adapter})); return
    predictor = Predictor({"name": a.base, "revision": a.revision, "cache_dir": a.cache_dir,
                           "dtype": a.dtype, "device": a.device, "loader": a.loader,
                           "chat_template_kwargs": {"enable_thinking": False}}, a.adapter)
    outputs = []
    for i, row in enumerate(rows):
        result = predictor.predict(messages(row["context"], row["question"]),
                                   {"max_input_tokens": a.max_input_tokens, "max_new_tokens": a.max_new_tokens,
                                    "do_sample": False}, a.seed)
        outputs.append({"id": row.get("id", i), **result, "parsed": parse_output(result["output"]),
                        "base": a.base, "revision": a.revision, "adapter": a.adapter, "seed": a.seed})
    write_jsonl(a.output, outputs)

if __name__ == "__main__":
    main()
