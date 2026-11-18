import argparse
import json
from financeqa.common import write_json


def main():
    p = argparse.ArgumentParser(description="Compare evaluation metrics, rejecting protocol mismatches")
    p.add_argument("metrics", nargs="+")
    p.add_argument("--output", default="results/comparison.json")
    a = p.parse_args()
    rows = {path: json.load(open(path)) for path in a.metrics}
    protocols = [r["protocol"] for r in rows.values()]
    if any(p != protocols[0] for p in protocols[1:]):
        raise ValueError("Comparison rejected: samples, model, seed, prompt, tolerance or generation differ")
    write_json(a.output, rows)
    print("model\tn\tscorable\tunscorable\taccuracy\tparseable")
    for name, row in rows.items():
        print(f"{name}\t{row['n']}\t{row['n_scorable']}\t{row['n_unscorable']}\t{row['numerical_accuracy']}\t{row['parseable_rate']}")

if __name__ == "__main__":
    main()
