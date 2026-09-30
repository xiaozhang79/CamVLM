from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


TASK_PROMPT = """Task:
Analyze the video and:
1. Provide a concise but complete description of the main events in the video.
2. Identify ONE single best category that most accurately describes the primary event.

Classification Rules:
- Choose exactly ONE category only.
- Select the category based on the dominant or most important event in the video.
- Prioritize safety/security incidents over normal activity.
- If multiple events occur, choose the most severe or highest-risk event.
- Only use the category names exactly as listed below.
- If none of the risk categories apply, use "normal activity".

Categories:
- break-in: Illegal entry or attempted intrusion into a property, home, room, or enclosed area.
- theft: Taking or attempting to take any property without permission, including packages, car, or belongings.
- suspicious behavior: Loitering, face occlude, peeping, casing a location, or other abnormal behavior suggesting intent to commit a crime.
- violence: Physical assault, fighting, armed threat, visible weapon, shooting, or any aggressive act against a person.
- vandalism: Intentional damage, graffiti, arson, scratching or defacement of property.
- fire hazard: Visible fire, smoke, explosion, or any fire-related danger.
- personal emergency: A person slips, falls, drowning, or shows urgent danger.
- wild animal: Presence of a wild or potentially dangerous animal (e.g., wolf, bear, raccoon, deer) in the area.
- vehicle incident: Vehicle collision or a vehicle striking a person or object.
- normal activity: Ordinary daily activity with no safety or security risk.

Output Format (strictly follow):
Video category: <exact category name>
Video description: <event summary>"""


CATEGORIES = (
    "break-in",
    "fire hazard",
    "normal activity",
    "personal emergency",
    "suspicious behavior",
    "theft",
    "vandalism",
    "vehicle incident",
    "violence",
    "wild animal",
)


def load_samples(path: Path) -> List[Dict[str, Any]]:
    import json

    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            expected = ("video_id", "duration", "video_path", "caption", "Category", "structured")
            if tuple(row) != expected:
                raise ValueError(f"{path}:{line_number} must use fields in order {expected}, got {tuple(row)}")
            video = Path(str(row["video_path"]))
            if not video.is_file():
                raise FileNotFoundError(f"{path}:{line_number} video is missing: {video}")
            category = str(row["Category"]).strip().lower()
            if category not in CATEGORIES:
                raise ValueError(f"{path}:{line_number} has unknown Category={row['Category']!r}")
            structured = row["structured"]
            if not isinstance(structured, dict):
                raise ValueError(f"{path}:{line_number} must contain a structured object")
            samples.append(
                {
                    "sample_idx": len(samples),
                    "id": str(row["video_id"]),
                    "duration": str(row["duration"]),
                    "video": str(video),
                    "gt_caption": str(row["caption"]).strip(),
                    "gt_category": category,
                    "structured": structured,
                }
            )
    return samples


def parse_prediction(text: Any) -> Tuple[str, str]:
    value = str(text or "").replace("\r", "\n").strip()
    description_match = re.search(
        r"(?:^|[\n>])\s*video\s*description\s*:\s*(.*?)(?=\s*video\s*category\s*:|\Z)",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    category_match = re.search(
        r"(?:^|[\n>])\s*video\s*category\s*:\s*([^\n<]+)",
        value,
        flags=re.IGNORECASE,
    )
    description = re.sub(r"\s+", " ", description_match.group(1)).strip() if description_match else ""
    category = category_match.group(1).strip().lower().rstrip(". ") if category_match else ""
    return description, category


def record_key(record: Dict[str, Any]) -> str:
    return str(record.get("id", ""))
