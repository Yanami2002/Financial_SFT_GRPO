"""Strict, finite numeric parsing; percent is a fraction, currencies use base units."""
import json
import math
import re
from decimal import Decimal, InvalidOperation

UNITS = {"ratio": ("ratio", 1), "percent": ("ratio", .01), "%": ("ratio", .01),
         "count": ("count", 1), "years": ("years", 1),
         "usd": ("usd", 1), "usd_million": ("usd", 1e6),
         "usd_billion": ("usd", 1e9)}
NUM = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("invalid_number_type")
    s = str(value).strip().replace("−", "-")
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1].strip()
    if "," in s:
        if not re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", s):
            raise ValueError("invalid_grouping")
        s = s.replace(",", "")
    if not NUM.fullmatch(s):
        raise ValueError("invalid_number")
    try:
        result = float(Decimal(s))
    except (InvalidOperation, OverflowError):
        raise ValueError("invalid_number") from None
    if not math.isfinite(result):
        raise ValueError("nonfinite_number")
    return result


def _pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate_json_key")
        d[k] = v
    return d


def parse_output(text):
    """No regex extraction from prose, no guessing missing units, no NaN/Infinity."""
    try:
        if not isinstance(text, str) or len(text) > 32768:
            raise ValueError("invalid_output")
        obj = json.loads(text.strip(), object_pairs_hook=_pairs,
                         parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_number")))
        if not isinstance(obj, dict) or "answer" not in obj:
            raise ValueError("missing_answer")
        unit = obj.get("unit")
        if not isinstance(unit, str) or unit not in UNITS:
            raise ValueError("missing_or_unknown_unit")
        ans = obj["answer"]
        if isinstance(ans, str) and ans.strip().endswith("%"):
            if unit not in ("percent", "%"):
                raise ValueError("conflicting_percent_unit")
            ans = ans.strip()[:-1].strip()
        value = number(ans)
        dimension, scale = UNITS[unit]
        value *= scale
        if not math.isfinite(value):
            raise ValueError("nonfinite_number")
        return {"parseable": True, "value": value, "unit": dimension, "error": None}
    except (ValueError, TypeError, OverflowError, RecursionError) as e:
        return {"parseable": False, "value": None, "unit": None, "error": str(e)}


def validate_reference(ref):
    try:
        return (isinstance(ref, dict) and ref.get("unit") in {x[0] for x in UNITS.values()}
                and math.isfinite(number(ref["value"]))
                and math.isfinite(float(ref.get("atol", 0))) and float(ref.get("atol", 0)) >= 0)
    except (ValueError, KeyError, TypeError):
        return False


def score(text, reference, atol=1e-6, rtol=1e-3):
    if not math.isfinite(atol) or not math.isfinite(rtol) or min(atol, rtol) < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    parsed = parse_output(text)
    valid = validate_reference(reference)
    correct = False
    if valid and parsed["parseable"] and parsed["unit"] == reference["unit"]:
        target = number(reference["value"])
        tol = max(atol, float(reference.get("atol", 0)), rtol * abs(target))
        correct = abs(parsed["value"] - target) <= tol
    return {**parsed, "scorable": valid, "correct": correct}


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1 and isinstance(completion[0], dict):
        return completion[0].get("content", "")
    return ""


def numerical_reward(completions, reference, atol=1e-6, rtol=1e-3, **kwargs):
    if len(completions) != len(reference):
        raise ValueError("Completion/reference length mismatch")
    return [float(score(completion_text(c), r, atol, rtol)["correct"]) for c, r in zip(completions, reference)]
