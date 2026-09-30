from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .infer import load_samples, record_key


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge UDVideoQA chunk prediction files.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jsonl-output", default=None)
    parser.add_argument("--question_files", nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.question_files)
    order = {record_key(sample): idx for idx, sample in enumerate(samples)}
    merged: Dict[Any, Dict[str, Any]] = {}
    for input_path in args.inputs:
        for record in load_jsonl(Path(input_path)):
            merged[record_key(record)] = record
    records = sorted(merged.values(), key=lambda record: order.get(record_key(record), len(order)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if args.jsonl_output:
        jsonl_output = Path(args.jsonl_output)
        jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_output.open("w", encoding="utf-8") as handle:
            for record in records:
                json.dump(record, handle, ensure_ascii=False)
                handle.write("\n")
    print(f"Merged {len(records)} records to {output}")


if __name__ == "__main__":
    main()
