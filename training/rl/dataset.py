import copy
import json
from typing import Any, Dict, List

from torch.utils.data import Dataset


def _is_video_item(item: Dict[str, Any]) -> bool:
    return (
        isinstance(item, dict)
        and item.get("type") == "video"
        and isinstance(item.get("video"), str)
        and bool(item["video"])
    )


def _is_complete_second_video_item(item: Dict[str, Any]) -> bool:
    try:
        return float(item.get("video_end", 0.0)) - float(item.get("video_start", 0.0)) >= 1.0 - 1e-9
    except (TypeError, ValueError):
        return False


def extract_video_items(sample: Dict[str, Any], fps: float) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for message in sample.get("messages", []):
        if message.get("role") != "user":
            continue
        for raw_item in message.get("content", []):
            if not _is_video_item(raw_item):
                continue
            if not _is_complete_second_video_item(raw_item):
                continue
            item = copy.deepcopy(raw_item)
            window = item.get("window")
            if not isinstance(window, list) or len(window) != 4:
                raise ValueError(f"Sample {sample.get('id')} has a video turn without a four-value window.")
            object_ids = item.get("target_object_ids")
            bboxes = item.get("target_bboxes")
            fallbacks = item.get("target_bbox_fallbacks")
            if not isinstance(object_ids, list) or not object_ids:
                raise ValueError(f"Sample {sample.get('id')} has a video turn without target object ids.")
            if not isinstance(bboxes, list) or len(bboxes) != len(object_ids):
                raise ValueError(f"Sample {sample.get('id')} has mismatched target bboxes.")
            if not isinstance(fallbacks, list) or len(fallbacks) != len(object_ids):
                raise ValueError(f"Sample {sample.get('id')} has mismatched target bbox fallbacks.")
            for object_id, bbox, fallback in zip(object_ids, bboxes, fallbacks):
                if not isinstance(object_id, int):
                    raise ValueError(f"Sample {sample.get('id')} has a non-integer target object id.")
                if not isinstance(fallback, bool):
                    raise ValueError(f"Sample {sample.get('id')} has a non-boolean bbox fallback flag.")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(f"Sample {sample.get('id')} has an invalid target bbox.")
                try:
                    x1, y1, x2, y2 = (float(value) for value in bbox)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Sample {sample.get('id')} has a non-numeric target bbox.") from exc
                if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
                    raise ValueError(f"Sample {sample.get('id')} has an out-of-range target bbox: {bbox}")
            items.append(item)
    if not items:
        raise ValueError(f"Sample {sample.get('id')} has no video turns.")
    return items


class CamVLMRLDataset(Dataset):
    def __init__(self, path: str, *, fps: float = 2.0):
        with open(path, "r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Expected a non-empty JSON list in {path}")

        original_count = len(rows)
        rows = [row for row in rows if self._is_trainable_sample(row)]
        dropped_count = original_count - len(rows)
        if not rows:
            raise ValueError(f"No trainable RL samples with complete video turns in {path}")
        if dropped_count:
            print(
                "Dropped RL samples without complete 1-second video turns: "
                f"{dropped_count}/{original_count}",
                flush=True,
            )

        self.rows = rows
        self.fps = fps

    @staticmethod
    def _is_trainable_sample(row: Dict[str, Any]) -> bool:
        try:
            extract_video_items(row, fps=2.0)
        except ValueError:
            return False
        return True

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        options = row.get("options")
        if not isinstance(options, list) or len(options) < 2:
            raise ValueError(f"Sample {row.get('id')} has invalid options.")
        normalized_options = [str(option).strip() for option in options]
        answer = str(row["Answer"]).strip()
        if answer not in normalized_options:
            raise ValueError(
                f"Sample {row.get('id')} Answer does not exactly match any option."
            )
        return {
            "id": str(row.get("id", index)),
            "Question": str(row["Question"]).strip(),
            "options": normalized_options,
            "Answer": answer,
            "video_items": extract_video_items(row, self.fps),
        }
