"""FinQA official full-context conversion; no retrieved/gold evidence enters prompts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import urllib.request
from financeqa.common import messages, write_json, write_jsonl, digest
from rewards.numerical import number

REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
SOURCE = "https://github.com/czyssrs/FinQA"
SPLITS = ("train", "dev", "test")


def reference_for(qa):
    """Accept only explicit units or an explicitly dimensionless ratio question.

    Require written-answer and executed-answer agreement within written rounding.
    Never infer a percent scale from numerical similarity alone.
    """
    raw = str(qa.get("answer", "")).strip().lower().replace("−", "-")
    try:
        executed = number(qa.get("exe_ans"))
    except ValueError:
        return None, "nonnumeric_execution"
    if not raw:
        return None, "missing_written_answer"
    unit, scale = None, 1
    value_text = raw
    if raw.endswith("%"):
        unit, scale, value_text = "ratio", .01, raw[:-1].strip()
    else:
        patterns = [(r"\s+years?", "years", 1), (r"\s+shares?", "count", 1),
                    (r"\s+times", "ratio", 1), (r"\s+usd", "usd", 1),
                    (r"\s+million\s+usd", "usd", 1e6), (r"\s+billion\s+usd", "usd", 1e9)]
        for pat, u, factor in patterns:
            m = re.fullmatch(r"(.+?)" + pat, raw)
            if m:
                value_text, unit, scale = m[1], u, factor
                break
        if unit is None and re.search(r"\bratio\b", qa.get("question", "").lower()):
            unit = "ratio"
    if unit is None:
        return None, "ambiguous_unit"
    try:
        written = number(value_text)
    except ValueError:
        return None, "unparsed_written_answer"
    # Half of last written digit, in the written answer's units (incl exponent).
    mantissa, *exponent = value_text.replace(",", "").strip("()").lower().split("e")
    digits = len(mantissa.split(".")[1]) if "." in mantissa else 0
    resolution = 10.0 ** ((int(exponent[0]) if exponent else 0) - digits)
    tol = .5 * resolution + 1e-8
    # % annotation explicitly licenses percent-vs-fraction convention reconciliation.
    candidates = [executed]
    if raw.endswith("%"):
        candidates = [executed / scale, executed]
    matching = [v for v in candidates if abs(v - written) <= tol]
    if not matching:
        return None, "annotation_execution_disagreement"
    # If both scales match but represent different values, skip rather than choose.
    if len(matching) > 1 and not math.isclose(matching[0], matching[1], abs_tol=1e-10):
        return None, "ambiguous_percent_scale"
    return {"value": matching[0] * scale, "unit": unit,
            "atol": tol * scale, "written_value": written * scale,
            "convention": "explicit_annotation_validated_against_exe_ans"}, None


def convert(row, split, revision):
    for key in ("id", "pre_text", "post_text", "table", "qa"):
        if key not in row:
            raise ValueError(f"FinQA schema missing {key}")
    qa = row["qa"]
    if not isinstance(row["id"], str) or not qa.get("question"):
        raise ValueError("Missing ID/question")
    if not all(isinstance(row[k], list) for k in ("pre_text", "post_text", "table")):
        raise ValueError("Unexpected FinQA context schema")
    context = "\n".join(row["pre_text"]) + "\n\nTable (each row is a JSON array):\n"
    context += "\n".join(json.dumps(r, ensure_ascii=False) for r in row["table"])
    context += "\n\n" + "\n".join(row["post_text"])
    ref, reason = reference_for(qa)
    return {"id": row["id"], "source": SOURCE, "data_revision": revision, "split": split,
            "pre_text": row["pre_text"], "post_text": row["post_text"], "table": row["table"],
            "context": context, "question": qa["question"], "answer": qa.get("answer"),
            "exe_ans": qa.get("exe_ans"), "program": qa.get("program"),
            "reference": ref, "skip_reason": reason,
            "prompt": messages(context, qa["question"])}


def duplicate_report(splits):
    ids, exact, questions, reports = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(set)
    norm = lambda s: " ".join(s.lower().split())
    for split, rows in splits.items():
        for row in rows:
            loc = {"split": split, "id": row["id"]}
            ids[row["id"]].append(loc)
            key = hashlib.sha256(norm(row["question"] + "\n" + row["context"]).encode()).hexdigest()
            exact[key].append(loc)
            questions[norm(row["question"])].append(loc)
            reports[row["id"].rsplit("-", 1)[0]].add(split)
    return {"duplicate_ids": [v for v in ids.values() if len(v) > 1],
            "exact_context_question_duplicates": [v for v in exact.values() if len(v) > 1],
            "cross_split_question_matches": [v for v in questions.values() if len({x['split'] for x in v}) > 1],
            "cross_split_report_count": sum(len(v) > 1 for v in reports.values()),
            "scope": "Whitespace/case normalized full context+question, ID, question-only, report ID; no semantic equivalence detection."}


def prepare(raw_dir, out_dir, revision=REVISION, download=False):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest, splits = {}, {}
    for split in SPLITS:
        path = raw_dir / f"{split}.json"
        url = f"https://raw.githubusercontent.com/czyssrs/FinQA/{revision}/dataset/{split}.json"
        if download:
            try:
                with urllib.request.urlopen(url, timeout=90) as response:
                    payload = response.read()
                json.loads(payload)  # fail before replacing a valid cached file
                path.write_bytes(payload)
            except Exception as e:
                raise RuntimeError(f"FinQA download failed: {url}: {e}. Provide official local JSON files with --raw-dir.") from e
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; use --download or --raw-dir")
        manifest[split] = {"url": url if download else None, "sha256": digest(path)}
        splits[split] = [convert(r, split, revision if download else "local-unverified:" + digest(path))
                         for r in json.loads(path.read_text())]
    duplicates = duplicate_report(splits)
    write_json(out_dir / "duplicates.json", duplicates)
    if duplicates["duplicate_ids"] or any(len({x['split'] for x in g}) > 1 for g in duplicates["exact_context_question_duplicates"]):
        raise ValueError("Duplicate IDs or exact cross-split duplicates; see duplicates.json. No splits have been shuffled or silently repaired.")
    stats = {}
    for split, rows in splits.items():
        verified = [r for r in rows if r["reference"] is not None]
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        write_jsonl(out_dir / f"{split}_sft.jsonl", [
            {"id": r["id"], "split": split, "messages": r["prompt"] + [{"role": "assistant", "content": json.dumps(
                {"answer": r["reference"]["written_value"], "unit": r["reference"]["unit"]})}]} for r in verified])
        # Non-prompt columns reach only reward callbacks, never chat rendering.
        if split == "train":
            write_jsonl(out_dir / "train_grpo.jsonl", [
                {"id": r["id"], "split": split, "prompt": r["prompt"], "reference": r["reference"]} for r in verified])
        stats[split] = {"total": len(rows), "verified_numeric": len(verified),
                        "skip_reasons": dict(Counter(r["skip_reason"] for r in rows if r["skip_reason"])),
                        "types": dict(Counter(r["reference"]["unit"] for r in verified))}
    write_json(out_dir / "manifest.json", {"source": SOURCE, "requested_revision": revision,
               "provenance": "downloaded_pinned" if download else "local_hash_only",
               "files": manifest, "stats": stats})
    print(json.dumps(stats, indent=2))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument("--out-dir", default="data/processed")
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--download", action="store_true")
    a = p.parse_args(); prepare(a.raw_dir, a.out_dir, a.revision, a.download)

if __name__ == "__main__":
    main()
