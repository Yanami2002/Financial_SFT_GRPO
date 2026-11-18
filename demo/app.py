"""Local Gradio comparison: one resident model, sequential switching."""
import argparse
import gc
import json
import threading
from pathlib import Path
from financeqa.common import messages
from financeqa.modeling import Predictor
from rewards.numerical import parse_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--check", action="store_true", help="Construct UI without starting server/loading weights")
    a = p.parse_args()
    import yaml
    import gradio as gr
    cfg = yaml.safe_load(Path(a.config).read_text())
    variants = {k: v for k, v in cfg["variants"].items() if not v.get("adapter") or (Path(v["adapter"]) / "adapter_config.json").is_file()}
    state = {"name": None, "predictor": None}
    lock = threading.Lock()

    def compare(context, question, selected):
        if not selected or not context or not question or not context.strip() or not question.strip():
            return "Please supply context, question, and at least one model."
        results = []
        with lock:
            for name in selected:
                if name not in variants:
                    raise gr.Error("Unavailable model")
                import time
                start = time.perf_counter()
                if state["name"] != name:
                    state["predictor"] = None
                    state["name"] = None
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    state["predictor"] = Predictor(cfg["model"], variants[name].get("adapter"))
                    state["name"] = name
                result = state["predictor"].predict(messages(context, question), cfg["generation"], cfg["seed"])
                results.append({"model": name, **result, "parsed": parse_output(result["output"]),
                                "total_seconds_including_load": time.perf_counter() - start})
        return json.dumps(results, indent=2, ensure_ascii=False)

    with gr.Blocks(title="FinQA research comparison") as app:
        gr.Markdown("# Financial numerical QA research prototype\nFirst load/model switch may be slow. Models run sequentially; one model stays resident. Outputs are experimental and cannot be relied upon for investment decisions.")
        context = gr.Textbox(label="Report context (include units and table)", lines=10)
        question = gr.Textbox(label="Question")
        selected = gr.CheckboxGroup(list(variants), value=[next(iter(variants))], label="Available models")
        button = gr.Button("Run comparison")
        output = gr.Code(language="json", label="Answers, parsed values, and latency")
        button.click(compare, [context, question, selected], output, concurrency_limit=1)
    if a.check:
        print({"ui_constructed": True, "models": list(variants)}); return
    app.queue().launch(server_name=a.host, server_port=a.port, share=False)

if __name__ == "__main__":
    main()
