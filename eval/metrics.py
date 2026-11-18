from collections import defaultdict
from rewards.numerical import score


def evaluate_prediction(row, output, atol=1e-6, rtol=1e-3):
    result = score(output, row.get("reference"), atol, rtol)
    return {"id": row["id"], "split": row["split"], "question": row["question"], "output": output,
            "reference": row.get("reference"), "original_answer": row.get("answer"),
            "skip_reason": row.get("skip_reason"), "score": result,
            "answer_type": row["reference"]["unit"] if row.get("reference") else "unscorable"}


def aggregate(rows):
    def metrics(items):
        n = len(items)
        valid = [r for r in items if r["score"]["scorable"]]
        return {"n": n, "n_scorable": len(valid), "n_unscorable": n - len(valid),
                "correct": sum(r["score"]["correct"] for r in valid),
                "numerical_accuracy": sum(r["score"]["correct"] for r in valid) / len(valid) if valid else None,
                "parseable_rate": sum(r["score"]["parseable"] for r in items) / n if n else None,
                "parseable_rate_scorable": sum(r["score"]["parseable"] for r in valid) / len(valid) if valid else None}
    groups = defaultdict(list)
    for row in rows:
        groups[row["answer_type"]].append(row)
    return {**metrics(rows), "by_answer_type": {k: metrics(v) for k, v in groups.items()}}
