"""Same prompt/data/generation for all variants; references only reach scoring."""
from pathlib import Path
from financeqa.common import config_cli, read_jsonl, record_run, write_json, digest
from eval.metrics import evaluate_prediction, aggregate
from financeqa.modeling import Predictor, adapter_fingerprint, resource_start, resource_finish


def run(cfg, dry_run=False, only=None):
    rows = read_jsonl(cfg["data"])
    if cfg.get("limit"):
        rows = rows[:cfg["limit"]]
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Empty evaluation set or duplicate IDs")
    if any(r["split"] != cfg["split"] for r in rows) or cfg["split"] not in ("dev", "test"):
        raise ValueError("Evaluation requires the configured official dev/test split")
    if dry_run:
        print({"samples": len(rows), "split": cfg["split"], "variants": list(cfg["variants"])})
        return
    summaries = {}
    for variant, spec in cfg["variants"].items():
        if only and variant not in only:
            continue
        output_dir = Path(cfg["output_dir"]) / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        provenance = record_run(output_dir, cfg, [cfg["data"], Path(cfg["data"]).parent / "manifest.json"])
        provenance["variant"] = variant
        provenance["adapter"] = adapter_fingerprint(spec.get("adapter"))
        start = resource_start()
        predictor = Predictor(cfg["model"], spec.get("adapter"))
        outputs = []
        # Incremental output survives interruptions; repeated runs overwrite explicitly.
        pred_path = output_dir / "predictions.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            import json
            for i, row in enumerate(rows):
                result = predictor.predict(row["prompt"], cfg["generation"], cfg["seed"])
                record = evaluate_prediction(row, result["output"], **cfg["tolerance"])
                record.update({k: v for k, v in result.items() if k != "output"})
                outputs.append(record)
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"); f.flush()
                print(f"{variant}: {i + 1}/{len(rows)} correct={record['score']['correct']}", flush=True)
        summary = aggregate(outputs)
        summary["resources"] = resource_finish(start)
        summary["protocol"] = {"data_sha256": digest(cfg["data"]), "ids": [r["id"] for r in rows],
                               "prompt_sha256": provenance["prompt_sha256"], "split": cfg["split"],
                               "generation": cfg["generation"], "tolerance": cfg["tolerance"],
                               "model": cfg["model"], "seed": cfg["seed"]}
        write_json(output_dir / "run.json", provenance)
        write_json(output_dir / "metrics.json", summary)
        summaries[variant] = summary
        del predictor
        import gc, torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(Path(cfg["output_dir"]) / "summary.json", summaries)


def main():
    cfg, args = config_cli(__doc__)
    run(cfg, args.dry_run)

if __name__ == "__main__":
    main()
