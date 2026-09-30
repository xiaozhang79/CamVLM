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
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = "gpt-5.5"
CATEGORY_WEIGHTS = {
    "Basic Understanding": 1.0,
    "Attribution": 1.2,
    "Event Reasoning": 1.3,
    "Reverse Reasoning": 1.3,
    "Counterfactual Inference": 1.5,
}
CATEGORY_LOG_ORDER = [
    ("BU", "Basic Understanding"),
    ("Atr", "Attribution"),
    ("ER", "Event Reasoning"),
    ("RR", "Reverse Reasoning"),
    ("CI", "Counterfactual Inference"),
]
SMART_QUOTES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score UDVideoQA free-text predictions with a GPT LLM judge.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--model", default=OPENAI_MODEL)
    parser.add_argument("--base-url", default=OPENAI_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--workers", type=int, default=1024)
    parser.add_argument("--max-inflight", type=int, default=128)
    parser.add_argument("--requests-per-minute", type=int, default=0, help="0 disables RPM throttling and relies on max-inflight.")
    parser.add_argument("--min-requests-per-minute", type=int, default=10)
    parser.add_argument("--rate-limit-cooldown", type=float, default=60.0)
    parser.add_argument("--rate-limit-backoff", type=float, default=0.7)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"], default="none")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def default_output_path(path: Path) -> Path:
    return path.with_name(path.stem + ".gpt_scored.json")


def default_summary_path(path: Path) -> Path:
    return path.with_name(path.stem + ".gpt_summary.json")


def default_cache_path(path: Path) -> Path:
    return path.with_name(path.stem + ".gpt_score_cache.json")


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        os.chmod(path, 0o644)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).translate(SMART_QUOTES)
    return re.sub(r"\s+", " ", text).strip()


def create_openai_client(api_key: Optional[str], api_key_env: str, base_url: str, timeout: float):
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not set. Export it before running this script.")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required: pip install openai") from exc
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)


class RequestGate:
    """Spread requests evenly and slow down globally when the upstream is unstable."""

    def __init__(self, requests_per_minute: int, min_requests_per_minute: int) -> None:
        self.requests_per_minute = max(0, requests_per_minute)
        self.min_requests_per_minute = max(1, min_requests_per_minute)
        self.lock = threading.Condition()
        self.next_request_at = 0.0
        self.cooldown_until = 0.0

    def acquire(self) -> None:
        with self.lock:
            while True:
                now = time.monotonic()
                wait_for = max(0.0, self.cooldown_until - now)
                if wait_for <= 0.0 and (self.requests_per_minute == 0 or now >= self.next_request_at):
                    if self.requests_per_minute > 0:
                        self.next_request_at = now + 60.0 / self.requests_per_minute
                    return
                if wait_for <= 0.0:
                    wait_for = self.next_request_at - now
                self.lock.wait(timeout=max(0.01, wait_for))

    def backoff(self, status_code: Optional[int], cooldown: float, backoff: float) -> None:
        if status_code != 429:
            return
        with self.lock:
            previous = self.requests_per_minute
            if previous <= 0:
                self.cooldown_until = max(self.cooldown_until, time.monotonic() + max(0.0, cooldown))
                self.lock.notify_all()
                print(
                    f"GPT judge upstream returned 429; applying a {cooldown:.1f}s global cooldown.",
                    flush=True,
                )
                return
            reduced = max(self.min_requests_per_minute, int(previous * backoff))
            if reduced >= previous and previous > self.min_requests_per_minute:
                reduced = previous - 1
            self.requests_per_minute = max(self.min_requests_per_minute, reduced)
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + max(0.0, cooldown))
            self.lock.notify_all()
        if self.requests_per_minute < previous:
            print(
                f"GPT judge upstream returned 429; reducing global request rate "
                f"from {previous} to {self.requests_per_minute} requests/minute.",
                flush=True,
            )


def exception_status_code(exc: BaseException) -> Optional[int]:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    text = str(exc)
    match = re.search(r"Error code:\s*(\d{3})", text)
    if match:
        return int(match.group(1))
    match = re.search(r"statusCode['\"]?\s*:\s*(\d{3})", text)
    if match:
        return int(match.group(1))
    return None


class MissingResponseTextError(RuntimeError):
    pass


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return normalize_text(output_text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    if chunks:
        return normalize_text("\n".join(chunks))
    raise MissingResponseTextError(f"GPT judge response has no text output: {response}")


def extract_chat_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if choices:
        content = getattr(getattr(choices[0], "message", None), "content", None)
        if content:
            return normalize_text(content)
    raise MissingResponseTextError(f"GPT judge chat fallback has no text output: {response}")


def call_openai(client: Any, args: argparse.Namespace, prompt: str, gate: RequestGate) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(args.max_retries + 1):
        try:
            request_args: Dict[str, Any] = {
                "model": args.model,
                "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                "max_output_tokens": args.max_output_tokens,
            }
            if args.reasoning_effort != "none":
                request_args["reasoning"] = {"effort": args.reasoning_effort}
            gate.acquire()
            response = client.responses.create(**request_args)
            try:
                return extract_response_text(response)
            except MissingResponseTextError:
                gate.acquire()
                chat_response = client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_output_tokens,
                    response_format={"type": "json_object"},
                )
                return extract_chat_text(chat_response)
        except Exception as exc:
            last_error = exc
            status_code = exception_status_code(exc)
            gate.backoff(status_code, args.rate_limit_cooldown, args.rate_limit_backoff)
            if attempt < args.max_retries:
                delay = min(args.retry_sleep * (2**attempt), 60.0) + random.uniform(0.0, 1.0)
                print(
                    f"GPT judge request failed (attempt {attempt + 1}/{args.max_retries + 1}): "
                    f"{exc}. Retrying in {delay:.1f}s.",
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(f"GPT judge request failed: {last_error}")


def load_records(path: Path) -> List[Dict[str, Any]]:
    value = load_json(path)
    if isinstance(value, list):
        return value
    raise ValueError(f"Expected JSON array: {path}")


def canonical_category(value: Any) -> str:
    text = normalize_text(value).lower().replace("counterfactual", "counter factual")
    if text in {"basic understanding", "basic"}:
        return "Basic Understanding"
    if text == "attribution":
        return "Attribution"
    if text in {"event reasoning", "event"}:
        return "Event Reasoning"
    if text in {"reverse reasoning", "reverse"}:
        return "Reverse Reasoning"
    if text in {"counter factual", "counter factual inference"}:
        return "Counterfactual Inference"
    return normalize_text(value) or "Unknown"


def infer_subtype(question: Any) -> str:
    text = normalize_text(question).lower()
    pedestrian_terms = {"pedestrian", "pedestrians", "person", "people", "man", "woman", "child", "cyclist", "bicyclist", "walker"}
    vehicular_terms = {"car", "cars", "vehicle", "vehicles", "truck", "bus", "van", "sedan", "suv", "motorcycle", "lane", "traffic", "driving", "overtake", "approach", "foreground"}

    def has_term(term: str) -> bool:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None

    if any(has_term(term) for term in pedestrian_terms):
        return "pedestrian"
    if any(has_term(term) for term in vehicular_terms):
        return "vehicular"
    return "background"


def final_answer_text(record: Dict[str, Any]) -> str:
    answer = normalize_text(record.get("pred_answer"))
    if answer:
        return answer
    output = normalize_text(record.get("final_model_output"))
    match = re.search(r"<answer>(.*?)</answer>", output, flags=re.IGNORECASE | re.DOTALL)
    return normalize_text(match.group(1) if match else output)


def cache_key(record: Dict[str, Any], judge_model: str) -> str:
    payload = {
        "judge_model": normalize_text(judge_model),
        "question": normalize_text(record.get("question")),
        "gt_answer": normalize_text(record.get("gt_answer")),
        "model_answer": final_answer_text(record),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_prompt(record: Dict[str, Any]) -> str:
    question = normalize_text(record.get("question"))
    ground_truth = normalize_text(record.get("gt_answer"))
    model_answer = final_answer_text(record)
    return (
        "You are an impartial, strict grader for VideoQA semantic correctness.\n\n"
        f"Question: {question}\n"
        f"Ground truth answer: {ground_truth}\n"
        f"Model answer: {model_answer}\n\n"
        "Assign correct as 1 or 0 only; no partial credit.\n"
        "Accept exact or unambiguous semantic equivalence for binary or categorical answers.\n"
        "Accept a short textual paraphrase only when it has the same meaning. Missing key information or contradicting the ground truth is 0.\n"
        "A blank answer, NA, I don't know, unsure, or a contradiction is 0.\n"
        "Do not infer beyond the ground truth.\n\n"
        'Return JSON only: {"correct": 0|1}'
    )


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_judgment(raw_text: str) -> Dict[str, Any]:
    value = extract_json_object(raw_text) or {}
    correct = value.get("correct", 0)
    if isinstance(correct, str):
        correct = 1 if correct.strip().lower() in {"1", "true", "yes", "correct"} else 0
    return {
        "correct": 1 if correct == 1 else 0,
        "raw_response": raw_text,
        "model": None,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def judge_one(
    index: int,
    record: Dict[str, Any],
    client: Any,
    args: argparse.Namespace,
    gate: RequestGate,
) -> Tuple[int, Dict[str, Any]]:
    judgment = parse_judgment(call_openai(client, args, build_prompt(record), gate))
    judgment["model"] = args.model
    return index, judgment


def empty_stats() -> Dict[str, Any]:
    return {"total": 0, "correct": 0, "weight_sum": 0.0, "weighted_correct": 0.0}


def add_stat(stats: Dict[str, Any], correct: int, weight: float) -> None:
    stats["total"] += 1
    stats["correct"] += int(correct)
    stats["weight_sum"] += weight
    stats["weighted_correct"] += int(correct) * weight


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    total = stats["total"]
    weight_sum = stats["weight_sum"]
    return {
        **stats,
        "unweighted_accuracy": stats["correct"] / total if total else 0.0,
        "weighted_score": stats["weighted_correct"] / total if total else 0.0,
        "weighted_accuracy": stats["weighted_correct"] / weight_sum if weight_sum else 0.0,
    }


def compute_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = empty_stats()
    by_set: Dict[str, Dict[str, Any]] = {}
    by_category: Dict[str, Dict[str, Any]] = {}
    by_subtype: Dict[str, Dict[str, Any]] = {}
    error_count = 0
    for record in records:
        if record.get("error"):
            error_count += 1
        judgment = record.get("gpt_judgment", {})
        category = canonical_category(record.get("category"))
        subtype = infer_subtype(record.get("question"))
        correct = int(judgment.get("correct", 0))
        weight = CATEGORY_WEIGHTS.get(category, 1.0)
        add_stat(overall, correct, weight)
        add_stat(by_set.setdefault(str(record.get("set_name", "unknown")), empty_stats()), correct, weight)
        add_stat(by_category.setdefault(category, empty_stats()), correct, weight)
        add_stat(by_subtype.setdefault(subtype, empty_stats()), correct, weight)
    return {
        **finalize_stats(overall),
        "error_count": error_count,
        "category_weights": CATEGORY_WEIGHTS,
        "by_set": {key: finalize_stats(value) for key, value in sorted(by_set.items())},
        "by_category": {key: finalize_stats(value) for key, value in sorted(by_category.items())},
        "by_subtype": {key: finalize_stats(value) for key, value in sorted(by_subtype.items())},
    }


def format_log_summary(summary: Dict[str, Any]) -> str:
    by_category = summary.get("by_category", {})
    parts = []
    for short_name, category in CATEGORY_LOG_ORDER:
        category_stats = by_category.get(category, {})
        accuracy = float(category_stats.get("unweighted_accuracy", 0.0)) * 100.0
        parts.append(f"{short_name}: {accuracy:.1f}")
    weighted_average = float(summary.get("weighted_accuracy", 0.0)) * 100.0
    parts.append(f"Average: {weighted_average:.1f}")
    return "UDVideoQA accuracy (%): " + " | ".join(parts)


def apply_judgments(records: List[Dict[str, Any]], cache: Dict[str, Any], judge_model: str) -> List[Dict[str, Any]]:
    output = []
    for record in records:
        new_record = dict(record)
        judgment = dict(cache.get(cache_key(record, judge_model), {}))
        if not judgment:
            judgment = {
                "correct": 0,
                "raw_response": "",
            }
        new_record["model_answer"] = final_answer_text(record)
        new_record["canonical_category"] = canonical_category(record.get("category"))
        new_record["subtype"] = infer_subtype(record.get("question"))
        new_record["gpt_judgment"] = judgment
        output.append(new_record)
    return output


def main() -> None:
    args = parse_args()
    args.output = args.output or default_output_path(args.input)
    args.summary_output = args.summary_output or default_summary_path(args.input)
    args.cache = args.cache or default_cache_path(args.input)
    records = load_records(args.input)
    cache = load_json(args.cache, default={})
    if not isinstance(cache, dict):
        raise ValueError(f"Cache must be a JSON object: {args.cache}")
    missing = [index for index, record in enumerate(records) if args.force or cache_key(record, args.model) not in cache]
    print(f"Loaded {len(records)} records; GPT judge model={args.model}; calls needed={len(missing)}.")
    if args.dry_run:
        return
    if missing:
        client = create_openai_client(os.getenv(args.api_key_env), args.api_key_env, args.base_url, args.timeout)
        if (
            args.workers <= 0
            or args.max_inflight <= 0
            or args.requests_per_minute < 0
            or args.min_requests_per_minute <= 0
            or args.max_output_tokens <= 0
        ):
            raise ValueError(
                "--workers, --max-inflight, non-negative --requests-per-minute, --min-requests-per-minute, "
                "and --max-output-tokens must be positive"
            )
        if args.requests_per_minute > 0 and args.min_requests_per_minute > args.requests_per_minute:
            raise ValueError("--min-requests-per-minute cannot exceed --requests-per-minute")
        if not 0.0 < args.rate_limit_backoff < 1.0:
            raise ValueError("--rate-limit-backoff must be strictly between 0 and 1")
        max_inflight = min(args.workers, args.max_inflight)
        print(
            f"GPT judge concurrency: workers={args.workers}, max_inflight={max_inflight}, "
            f"requests_per_minute={args.requests_per_minute}, max_output_tokens={args.max_output_tokens}, "
            f"timeout={args.timeout:.0f}s, attempts={args.max_retries + 1}.",
            flush=True,
        )
        gate = RequestGate(args.requests_per_minute, args.min_requests_per_minute)
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            remaining = iter(missing)
            futures = {}
            for _ in range(min(max_inflight, len(missing))):
                index = next(remaining)
                futures[executor.submit(judge_one, index, records[index], client, args, gate)] = index
            completed = 0
            failed_indices = []
            while futures:
                future = next(as_completed(futures))
                submitted_index = futures.pop(future)
                try:
                    index, judgment = future.result()
                except Exception as exc:
                    failed_indices.append(submitted_index)
                    print(f"GPT judge failed for sample_idx={submitted_index}: {exc}", flush=True)
                else:
                    cache[cache_key(records[index], args.model)] = judgment
                completed += 1
                if completed % 10 == 0 or completed == len(missing):
                    save_json_atomic(args.cache, cache)
                    print(f"Saved cache progress: {completed}/{len(missing)}")
                try:
                    next_index = next(remaining)
                except StopIteration:
                    continue
                futures[executor.submit(judge_one, next_index, records[next_index], client, args, gate)] = next_index
        save_json_atomic(args.cache, cache)
        if failed_indices:
            preview = ", ".join(str(index) for index in failed_indices[:20])
            raise RuntimeError(
                f"GPT judging finished with {len(failed_indices)} failed samples ({preview}). "
                f"Successful judgments were saved to {args.cache}; rerun the same command to resume only failed samples."
            )
    scored_records = apply_judgments(records, cache, args.model)
    summary = compute_summary(scored_records)
    save_json_atomic(args.output, scored_records)
    save_json_atomic(args.summary_output, summary)
    print(format_log_summary(summary))
    print(f"Saved scored predictions to {args.output}")
    print(f"Saved summary to {args.summary_output}")


if __name__ == "__main__":
    main()
