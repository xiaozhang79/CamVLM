from __future__ import annotations

import argparse
import json
from pathlib import Path

from .task import load_samples, record_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CCTV-Anomaly inference chunks.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path, required=True)
    args = parser.parse_args()
    order = {record_key(sample): sample["sample_idx"] for sample in load_samples(args.data)}
    merged = {}
    for path in args.inputs:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    merged[record_key(record)] = record
    records = sorted(merged.values(), key=lambda record: order.get(record_key(record), len(order)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.jsonl_output.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8")
    print(f"Merged {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
