from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from .task import CATEGORIES, parse_prediction


FIELDS = ("location", "time_of_day", "subjects", "activities", "objects")
DEFAULT_STRUCTURED_DATA = Path(os.environ.get("CCTV_DATA_ROOT", Path(__file__).resolve().parents[2] / "datasets" / "cctv_anomaly")) / "test.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CCTV-Anomaly captions and categories.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--structured-data", type=Path, default=DEFAULT_STRUCTURED_DATA)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=512)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--requests-per-minute", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def canonical_category(value: Any) -> str:
    text = normalized(value).lower().strip(" `\"'.:;")
    return text if text in CATEGORIES else ""


def is_empty_gt(value: Any) -> bool:
    return value is None or value == "" or value == []


def field_items(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalized(item) for item in value if normalized(item)]
    return [] if is_empty_gt(value) else [normalized(value)]


def load_structured_annotations(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Structured annotation file is missing: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotations = []
    for index, row in enumerate(rows):
        structured = row.get("structured")
        if not isinstance(structured, dict):
            raise ValueError(f"{path}:{index + 1} has no structured annotation object")
        annotations.append(
            {
                "id": str(row.get("video_id") or f"{index:05d}"),
                "caption": normalized(row.get("caption")),
                "category": canonical_category(
                    row.get("Category") or (row.get("categories") or [{}])[0].get("category")
                ),
                "structured": {field: structured.get(field) for field in FIELDS},
            }
        )
    return annotations


def attach_structured_annotations(
    records: List[Dict[str, Any]], annotations: List[Dict[str, Any]], source: Path
) -> List[Dict[str, Any]]:
    by_id = {annotation["id"]: annotation for annotation in annotations}
    attached = []
    for record in records:
        sample_id = str(record.get("id", ""))
        annotation = by_id.get(sample_id)
        if annotation is None:
            raise ValueError(f"No structured annotation for sample id={sample_id!r} in {source}")
        if normalized(record.get("gt_caption")) != annotation["caption"]:
            raise ValueError(f"Caption mismatch for sample id={sample_id!r} in {source}")
        if canonical_category(record.get("gt_category")) != annotation["category"]:
            raise ValueError(f"Category mismatch for sample id={sample_id!r} in {source}")
        item = dict(record)
        item["structured"] = annotation["structured"]
        attached.append(item)
    return attached


def restore_prediction_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(record)
    if not normalized(item.get("raw_model_output")):
        return item
    if normalized(item.get("pred_caption")) and normalized(item.get("pred_category")):
        return item
    caption, category = parse_prediction(item["raw_model_output"])
    if not normalized(item.get("pred_caption")):
        item["pred_caption"] = caption
    if not normalized(item.get("pred_category")):
        item["pred_category"] = category
    return item


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def key(record: Dict[str, Any], model: str) -> str:
    payload = {"model": model, "prompt": prompt(record)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prompt(record: Dict[str, Any]) -> str:
    description = normalized(record.get("pred_caption"))
    structured = record.get("structured") or {}
    return f"""
You are a lenient semantic evaluator for security-video descriptions.
Judge whether the model description covers every GT item in the structured fields.

General rules:
- Synonyms, paraphrases, broader terms, and common variants count when they express the same visible meaning.
- Do not require exact wording, direction, position, order, or object subtype unless a subject attribute is required.
- Do not reward information absent from the description. Contradictions are not hits.

Field rules:
- location: a compatible scene type counts; driveway and street may represent the same relevant outdoor access area.
- time_of_day: compatible cues such as night/dark/evening or day/daylight count.
- subjects: identify the same entity. If a GT subject contains attributes such as color, clothing, count, role, age,
  species, or vehicle type, the description must contain the entity plus at least one compatible GT attribute.
  Saying only "a man" does not hit "a man wearing a dark blue top and black pants"; "a person in blue" does.
  If the GT item is generic and has no attribute, such as "a person", "a man", "people", or "vehicles",
  matching the entity alone is sufficient.
- activities: the core action is enough. walk/run/move/approach/head/go and similar movement expressions may match.
- objects: the object meaning is enough and attributes are ignored. car/pickup/sedan/vehicle and similar variants may match.

Model description:
{description}

GT structured fields:
{json.dumps(structured, ensure_ascii=False, indent=2)}

Return strict JSON only. Preserve each GT list item exactly and return one result per item:
{{
  "location": {{"hit": true, "reason": "..."}},
  "time_of_day": {{"hit": true, "reason": "..."}},
  "subjects": [{{"item": "...", "hit": true, "reason": "..."}}],
  "activities": [{{"item": "...", "hit": true, "reason": "..."}}],
  "objects": [{{"item": "...", "hit": true, "reason": "..."}}]
}}
""".strip()


def json_object(text: str) -> Dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("Judge output must be a JSON object")
    return value


def as_hit(value: Any) -> bool:
    return value is True or value == 1 or str(value).strip().lower() == "true"


def zero_caption_score() -> Dict[str, Any]:
    return {
        "caption_score": 0.0,
        "field_scores": {field: 0.0 for field in FIELDS},
        "field_details": {field: {"score": 0.0, "items": []} for field in FIELDS},
    }


def score_from_judgment(structured: Dict[str, Any], judgment: Dict[str, Any]) -> Dict[str, Any]:
    scores: Dict[str, float] = {}
    details: Dict[str, Any] = {}
    for field in FIELDS:
        value = structured.get(field)
        if is_empty_gt(value):
            scores[field] = 20.0
            details[field] = {"score": 20.0, "auto_full_credit": True, "items": []}
            continue

        items = field_items(value)
        if field in ("location", "time_of_day"):
            judged = judgment.get(field) or {}
            hit = as_hit(judged.get("hit")) if isinstance(judged, dict) else False
            scores[field] = 20.0 if hit else 0.0
            details[field] = {
                "score": scores[field],
                "hit": hit,
                "reason": judged.get("reason", "") if isinstance(judged, dict) else "",
            }
            continue

        judged_items = judgment.get(field) or []
        if not isinstance(judged_items, list):
            judged_items = []
        by_item = {
            normalized(item.get("item")): item
            for item in judged_items
            if isinstance(item, dict) and normalized(item.get("item"))
        }
        per_item = 20.0 / len(items)
        item_details = []
        score = 0.0
        for index, item in enumerate(items):
            judged = by_item.get(item)
            if judged is None and len(judged_items) == len(items):
                judged = judged_items[index]
            hit = as_hit(judged.get("hit")) if isinstance(judged, dict) else False
            item_score = per_item if hit else 0.0
            score += item_score
            item_details.append(
                {
                    "item": item,
                    "hit": hit,
                    "score": item_score,
                    "reason": judged.get("reason", "") if isinstance(judged, dict) else "",
                }
            )
        scores[field] = score
        details[field] = {"score": score, "items": item_details}
    return {"caption_score": sum(scores.values()), "field_scores": scores, "field_details": details}

class Gate:
    def __init__(self, rpm: int) -> None:
        self.rpm, self.next_time, self.lock = max(0, rpm), 0.0, threading.Lock()

    def acquire(self) -> None:
        with self.lock:
            if not self.rpm:
                return
            now = time.monotonic()
            wait = max(0.0, self.next_time - now)
            self.next_time = max(now, self.next_time) + 60.0 / self.rpm
        if wait:
            time.sleep(wait)


class MissingResponseTextError(RuntimeError):
    pass


def extract_text(response: Any) -> str:
    if getattr(response, "output_text", None):
        return str(response.output_text)
    parts = []
    for output in getattr(response, "output", []) or []:
        for content in getattr(output, "content", []) or []:
            if getattr(content, "text", None):
                parts.append(str(content.text))
    if not parts:
        raise MissingResponseTextError("GPT judge response has no text output")
    return "\n".join(parts)


def extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if choices:
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if content:
            return str(content)
    raise MissingResponseTextError("GPT judge chat fallback has no text output")


def status_code(exc: BaseException) -> Optional[int]:
    value = getattr(exc, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int):
        return value
    match = re.search(r"Error code:\s*(\d{3})", str(exc))
    return int(match.group(1)) if match else None


def judge(record: Dict[str, Any], client: Any, args: argparse.Namespace, gate: Gate) -> Dict[str, Any]:
    error = None
    for attempt in range(args.max_retries + 1):
        try:
            gate.acquire()
            response = client.responses.create(model=args.model, input=[{"role": "user", "content": [{"type": "input_text", "text": prompt(record)}]}], max_output_tokens=args.max_output_tokens)
            try:
                raw = extract_text(response)
            except MissingResponseTextError:
                gate.acquire()
                chat_response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt(record)}],
                    max_tokens=args.max_output_tokens,
                    response_format={"type": "json_object"},
                )
                raw = extract_chat_text(chat_response)
            parsed = json_object(raw)
            parsed["raw_response"] = raw
            parsed["model"] = args.model
            return parsed
        except Exception as exc:
            error = exc
            if status_code(exc) in {400, 401, 403, 404}:
                break
            if attempt < args.max_retries:
                time.sleep(min(args.retry_sleep * 2**attempt, 60.0) + random.uniform(0, 1))
    raise RuntimeError(f"GPT caption judge failed: {error}")


def main() -> None:
    args = parse_args()
    args.output = args.output or args.input.with_name(args.input.stem + ".gpt_scored.json")
    args.summary_output = args.summary_output or args.input.with_name(args.input.stem + ".gpt_summary.json")
    args.cache = args.cache or args.input.with_name(args.input.stem + ".gpt_caption_cache.json")
    records = [restore_prediction_fields(record) for record in json.loads(args.input.read_text(encoding="utf-8"))]
    records = attach_structured_annotations(
        records, load_structured_annotations(args.structured_data), args.structured_data
    )
    cache = json.loads(args.cache.read_text(encoding="utf-8")) if args.cache.exists() else {}

    def caption_parse_success(record: Dict[str, Any]) -> bool:
        return bool(not record.get("error") and normalized(record.get("pred_caption")))

    def category_parse_success(record: Dict[str, Any]) -> bool:
        return bool(
            not record.get("error")
            and canonical_category(record.get("pred_category"))
        )

    def parse_success(record: Dict[str, Any]) -> bool:
        return caption_parse_success(record) and category_parse_success(record)

    def requires_judge(record: Dict[str, Any]) -> bool:
        return caption_parse_success(record) and any(
            not is_empty_gt((record.get("structured") or {}).get(field)) for field in FIELDS
        )

    candidates = [
        (idx, record)
        for idx, record in enumerate(records)
        if requires_judge(record) and (args.force or key(record, args.model) not in cache)
    ]
    if candidates:
        api_key = os.getenv(args.api_key_env)
        if not api_key:
            raise RuntimeError(f"{args.api_key_env} is not set")
        from openai import OpenAI
        client = OpenAI(base_url=args.base_url, api_key=api_key, timeout=args.timeout, max_retries=0)
        gate = Gate(args.requests_per_minute)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            iterator = iter(candidates)
            futures = {}
            for _ in range(min(args.max_inflight, len(candidates))):
                idx, record = next(iterator)
                futures[pool.submit(judge, record, client, args, gate)] = (idx, record)
            completed = 0
            while futures:
                future = next(as_completed(futures))
                idx, record = futures.pop(future)
                try:
                    cache[key(record, args.model)] = future.result()
                except BaseException:
                    save(args.cache, cache)
                    for pending in futures:
                        pending.cancel()
                    raise
                completed += 1
                if completed % 10 == 0:
                    save(args.cache, cache)
                try:
                    next_idx, next_record = next(iterator)
                except StopIteration:
                    continue
                futures[pool.submit(judge, next_record, client, args, gate)] = (next_idx, next_record)
        save(args.cache, cache)

    caption_score_sum = category_correct = errors = 0
    field_score_sums = {field: 0.0 for field in FIELDS}
    scored = []
    for record in records:
        item = dict(record)
        valid = parse_success(record)
        item["parse_success"] = valid
        item["caption_parse_success"] = caption_parse_success(record)
        item["category_parse_success"] = category_parse_success(record)
        item["category_correct"] = int(
            item["category_parse_success"]
            and canonical_category(record.get("pred_category"))
            == canonical_category(record.get("gt_category"))
        )
        category_correct += item["category_correct"]

        if not item["caption_parse_success"]:
            item["caption_judgment"] = {
                "unscored": "inference_error_or_invalid_output"
            }
            caption_result = zero_caption_score()
        elif requires_judge(record):
            item["caption_judgment"] = cache.get(key(record, args.model), {})
            caption_result = score_from_judgment(
                record.get("structured") or {}, item["caption_judgment"]
            )
        else:
            item["caption_judgment"] = {
                "auto_full_credit": "all_structured_fields_are_empty"
            }
            caption_result = score_from_judgment(record.get("structured") or {}, {})

        item.update(caption_result)
        caption_score_sum += float(item["caption_score"])
        for field in FIELDS:
            field_score_sums[field] += float(item["field_scores"][field])
        errors += int(bool(record.get("error")))
        scored.append(item)

    total = len(records)
    average_caption_score = caption_score_sum / total if total else 0.0
    summary = {
        "total": total,
        "error_count": errors,
        "parse_success_count": sum(int(item["parse_success"]) for item in scored),
        "caption_parse_success_count": sum(
            int(item["caption_parse_success"]) for item in scored
        ),
        "category_parse_success_count": sum(
            int(item["category_parse_success"]) for item in scored
        ),
        "caption": {
            "total": total,
            "score_sum": caption_score_sum,
            "average_score": average_caption_score,
            "accuracy": average_caption_score / 100.0,
            "field_average_scores": {
                field: field_score_sums[field] / total if total else 0.0 for field in FIELDS
            },
        },
        "category": {
            "total": total,
            "correct": category_correct,
            "accuracy": category_correct / total if total else 0.0,
        },
    }
    save(args.output, scored)
    save(args.summary_output, summary)
    print(
        f"CCTV-Anomaly accuracy (%): Caption: {summary['caption']['average_score']:.1f} "
        f"| Category: {summary['category']['accuracy'] * 100:.1f}"
    )


if __name__ == "__main__":
    main()
