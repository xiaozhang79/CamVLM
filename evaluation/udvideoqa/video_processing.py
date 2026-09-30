from __future__ import annotations

import importlib.util
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


TARGET_FPS = float(os.environ.get("VIDEO_FPS", "2"))
MIN_TOKENS = int(os.environ.get("VIDEO_MIN_TOKENS", "64"))
TOTAL_TOKENS = int(os.environ.get("VIDEO_TOTAL_TOKENS", "14336"))
_VIEWPORT_METADATA_KEYS = {
    "source_height",
    "source_width",
    "viewport_height",
    "viewport_width",
}
# Serialize Decord access: concurrent reads can abort on malformed H.264 streams.
_DECORD_READ_LOCK = threading.Lock()
def _load_chat_template() -> str:
    prompts_path = Path(__file__).resolve().parents[2] / "training" / "rl" / "prompts.py"
    spec = importlib.util.spec_from_file_location("camvlm_rl_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load CamVLM prompts from {prompts_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.QWEN3_CHAT_TEMPLATE


QWEN3_CHAT_TEMPLATE = _load_chat_template()


def _crop_video_to_window(video: torch.Tensor, window: Optional[Sequence[float]]) -> torch.Tensor:
    _, _, height, width = video.shape
    cropped = video
    if window and len(window) == 4:
        x1, y1, x2, y2 = window
        if not (x1 <= 0.0 and y1 <= 0.0 and x2 >= 1.0 and y2 >= 1.0):
            px1 = int(round(min(max(x1, 0.0), 1.0) * width))
            py1 = int(round(min(max(y1, 0.0), 1.0) * height))
            px2 = int(round(min(max(x2, 0.0), 1.0) * width))
            py2 = int(round(min(max(y2, 0.0), 1.0) * height))
            px1 = max(0, min(width - 1, px1))
            py1 = max(0, min(height - 1, py1))
            px2 = max(px1 + 1, min(width, px2))
            py2 = max(py1 + 1, min(height, py2))
            cropped = video[:, :, py1:py2, px1:px2]
    return cropped


def _video_tensor_to_pil_frames(video: torch.Tensor) -> List[Image.Image]:
    frames = video.detach().cpu().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _sample_absolute_indices(
    video_start: float,
    video_end: float,
    video_fps: float,
    total_frames: int,
    max_frames: Optional[int] = None,
) -> List[int]:
    duration = max(video_end - video_start, 0.0)
    nframes = max(2, int(math.floor(duration * TARGET_FPS + 1e-6)))
    if nframes % 2:
        nframes += 1
    if max_frames is not None:
        max_frames = max(2, int(max_frames))
        if max_frames % 2:
            max_frames -= 1
        nframes = min(nframes, max_frames)
    if nframes < int(math.floor(duration * TARGET_FPS + 1e-6)):
        last_timestamp = max(video_start, video_end - 1.0 / max(video_fps, 1.0))
        timestamps = np.linspace(video_start, last_timestamp, nframes).tolist()
    else:
        timestamps = [video_start + frame_idx / TARGET_FPS for frame_idx in range(nframes)]
    indices = []
    for timestamp in timestamps:
        if timestamp >= video_end:
            timestamp = max(video_start, video_end - 1.0 / max(video_fps, 1.0))
        indices.append(min(max(int(round(timestamp * video_fps)), 0), total_frames - 1))
    return indices


@lru_cache(maxsize=256)
def _mp4_metadata(video_path: str) -> Tuple[int, float]:
    import decord

    with _DECORD_READ_LOCK:
        reader = decord.VideoReader(video_path)
        return len(reader), float(reader.get_avg_fps())


def _mp4_time_bounds(ele: Dict[str, Any], video_fps: float, total_frames: int) -> Tuple[float, float]:
    duration = total_frames / video_fps
    start = max(0.0, min(float(ele.get("video_start", 0.0)), duration))
    end = max(0.0, min(float(ele.get("video_end", duration)), duration))
    if end <= start:
        end = min(duration, start + 1.0 / max(video_fps, 1.0))
    return start, end


def _fetch_video_decord(ele: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, Any]]:
    import decord

    with _DECORD_READ_LOCK:
        reader = decord.VideoReader(str(ele["video"]))
        total_frames = len(reader)
        video_fps = float(reader.get_avg_fps())
        if total_frames <= 0 or video_fps <= 0:
            raise ValueError(f"Invalid decord metadata for {ele['video']}: frames={total_frames}, fps={video_fps}")
        start, end = _mp4_time_bounds(ele, video_fps, total_frames)
        indices = _sample_absolute_indices(start, end, video_fps, total_frames, ele.get("max_frames"))
        video = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2).contiguous()
    return video, {
        "fps": video_fps,
        "frames_indices": indices,
        "total_num_frames": total_frames,
        "video_backend": "decord",
    }


def _fetch_video_qwen_backend(ele: Dict[str, Any]) -> Tuple[List[Image.Image], Dict[str, Any]]:
    video, metadata = _fetch_video_decord(ele)
    metadata = dict(metadata)
    metadata["source_height"] = int(video.shape[2])
    metadata["source_width"] = int(video.shape[3])
    video = _crop_video_to_window(video, ele.get("window"))
    metadata["viewport_height"] = int(video.shape[2])
    metadata["viewport_width"] = int(video.shape[3])
    return _video_tensor_to_pil_frames(video), metadata


def resize_viewport_frames_to_source(
    frames: Sequence[Image.Image], metadata: Dict[str, Any]
) -> List[Image.Image]:
    source_width = int(metadata.get("source_width", 0))
    source_height = int(metadata.get("source_height", 0))
    if source_width <= 0 or source_height <= 0:
        raise ValueError("Viewport metadata is missing the source frame dimensions.")
    source_size = (source_width, source_height)
    return [
        frame if frame.size == source_size else frame.resize(source_size, Image.Resampling.BICUBIC)
        for frame in frames
    ]


def processor_video_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in _VIEWPORT_METADATA_KEYS}


def _build_qwen_video_item(
    item: Dict[str, Any],
    frames: List[Image.Image],
    metadata: Dict[str, Any],
    pixel_scale: int,
    clip_total_pixels: Optional[int],
) -> Dict[str, Any]:
    qwen_item = {
        "type": "video",
        "video": frames,
        "sample_fps": TARGET_FPS,
        "raw_fps": float(metadata["fps"]),
        "min_pixels": MIN_TOKENS * pixel_scale,
        "_camvlm_video_metadata": metadata,
    }
    if clip_total_pixels is not None:
        qwen_item["total_pixels"] = clip_total_pixels
    return qwen_item


def _process_qwen_video_items(
    qwen_video_items: List[Dict[str, Any]],
    image_patch_size: int,
) -> Tuple[None, Optional[List[torch.Tensor]], Dict[str, Any]]:
    if not qwen_video_items:
        return None, None, {"do_sample_frames": False}

    from qwen_vl_utils import vision_process as qwen_vision_process

    video_inputs = []
    video_metadata = []
    for item in qwen_video_items:
        qwen_item = {key: value for key, value in item.items() if not key.startswith("_camvlm_")}
        (video_input, _), _ = qwen_vision_process.fetch_video(
            qwen_item,
            image_patch_size=image_patch_size,
            return_video_sample_fps=True,
            return_video_metadata=True,
        )
        video_inputs.append(video_input)
        video_metadata.append(processor_video_metadata(item["_camvlm_video_metadata"]))
    return None, video_inputs, {
        "do_sample_frames": False,
        "video_metadata": video_metadata,
    }
