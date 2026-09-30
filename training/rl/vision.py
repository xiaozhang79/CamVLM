import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from qwen_vl_utils import smart_resize
from torchvision import io

try:
    from transformers.video_utils import VideoMetadata
except ImportError:
    VideoMetadata = None


QWEN3_VIDEO_PIXEL_SCALE = 32 * 32
VIDEO_FRAME_FACTOR = 2


class VisionLoader:
    def __init__(
        self,
        *,
        fps: float,
        min_tokens: int,
        total_tokens: int,
        max_frames: int,
        mevis_fps: float,
        youtube_vos_fps: float,
        jpeg_read_workers: int = 4,
    ):
        self.fps = fps
        self.min_tokens = min_tokens
        self.total_tokens = total_tokens
        self.max_frames = max_frames
        self.mevis_fps = mevis_fps
        self.youtube_vos_fps = youtube_vos_fps
        self.jpeg_read_workers = max(1, jpeg_read_workers)

    @staticmethod
    def is_video_item(item: Dict[str, Any]) -> bool:
        return (
            isinstance(item, dict)
            and item.get("type") == "video"
            and isinstance(item.get("video"), str)
            and bool(item["video"])
        )

    @staticmethod
    def _make_metadata(**kwargs):
        return VideoMetadata(**kwargs) if VideoMetadata is not None else kwargs

    @staticmethod
    def _metadata_dict(metadata: Any) -> Dict[str, Any]:
        if isinstance(metadata, dict):
            value = dict(metadata)
        else:
            value = {
                key: getattr(metadata, key)
                for key in (
                    "total_num_frames",
                    "fps",
                    "duration",
                    "frames_indices",
                    "height",
                    "width",
                    "video_backend",
                )
                if hasattr(metadata, key)
            }
        value["do_sample_frames"] = False
        return value

    def _infer_source_fps(self, video_path: str) -> float:
        normalized = video_path.replace("\\", "/")
        if "/MeViS/" in normalized or normalized.startswith("MeViS/"):
            return self.mevis_fps
        if "/Refer-YouTube-VOS/" in normalized or normalized.startswith("Refer-YouTube-VOS/"):
            return self.youtube_vos_fps
        raise ValueError(f"Could not infer source FPS for frame directory: {video_path}")

    @staticmethod
    def _frame_sort_key(path: str) -> Tuple[int, Any]:
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            return 0, int(stem)
        except ValueError:
            return 1, stem

    @staticmethod
    @lru_cache(maxsize=256)
    def _list_frame_paths(video_dir: str) -> Tuple[str, ...]:
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"JPEG sequence directory not found: {video_dir}")
        paths = [
            os.path.join(video_dir, name)
            for name in os.listdir(video_dir)
            if os.path.splitext(name)[1].lower() in {".jpg", ".jpeg", ".png"}
        ]
        paths.sort(key=VisionLoader._frame_sort_key)
        if not paths:
            raise ValueError(f"No image frames found in: {video_dir}")
        return tuple(paths)

    def _sample_indices(
        self,
        video_start: float,
        video_end: float,
        video_fps: float,
        total_frames: int,
    ) -> List[int]:
        duration = max(video_end - video_start, 0.0)
        requested_frames = max(2, int(math.floor(duration * self.fps + 1e-6)))
        if requested_frames % 2:
            requested_frames += 1
        nframes = min(requested_frames, self.max_frames)
        end_time = max(video_start, video_end - 1.0 / max(video_fps, 1.0))
        if requested_frames > self.max_frames:
            timestamps = np.linspace(video_start, end_time, num=nframes).tolist()
        else:
            timestamps = [min(video_start + frame_idx / self.fps, end_time) for frame_idx in range(nframes)]
        return [min(max(int(round(timestamp * video_fps)), 0), total_frames - 1) for timestamp in timestamps]

    def _crop_and_resize(
        self,
        video: torch.Tensor,
        window: Optional[Sequence[float]],
        clip_total_pixels: Optional[int],
    ) -> torch.Tensor:
        nframes, _, height, width = video.shape
        cropped = video
        if window and len(window) == 4:
            x1, y1, x2, y2 = map(float, window)
            px1 = max(0, min(width - 1, int(round(min(max(x1, 0.0), 1.0) * width))))
            py1 = max(0, min(height - 1, int(round(min(max(y1, 0.0), 1.0) * height))))
            px2 = max(px1 + 1, min(width, int(round(min(max(x2, 0.0), 1.0) * width))))
            py2 = max(py1 + 1, min(height, int(round(min(max(y2, 0.0), 1.0) * height))))
            cropped = video[:, :, py1:py2, px1:px2]

        if not clip_total_pixels:
            return cropped
        min_pixels = self.min_tokens * QWEN3_VIDEO_PIXEL_SCALE
        max_pixels = max(
            int(clip_total_pixels / max(nframes, 1) * VIDEO_FRAME_FACTOR),
            int(min_pixels * 1.05),
        )
        target_height, target_width = smart_resize(
            int(cropped.shape[2]),
            int(cropped.shape[3]),
            factor=32,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        if cropped.shape[2:] == (target_height, target_width):
            return cropped
        resized = torch.nn.functional.interpolate(
            cropped.float(),
            size=(target_height, target_width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.clamp(0, 255).to(video.dtype)

    def _load_jpeg_item(
        self,
        item: Dict[str, Any],
        clip_total_pixels: Optional[int],
    ) -> Tuple[torch.Tensor, Any]:
        paths = self._list_frame_paths(item["video"])
        source_fps = float(item.get("source_fps") or self._infer_source_fps(item["video"]))
        start = float(item.get("video_start", 0.0))
        end = float(item.get("video_end", len(paths) / source_fps))
        indices = self._sample_indices(start, end, source_fps, len(paths))
        selected_paths = [paths[index] for index in indices]
        with ThreadPoolExecutor(max_workers=min(self.jpeg_read_workers, len(selected_paths))) as executor:
            frames = list(executor.map(lambda path: io.read_image(path, mode=io.ImageReadMode.RGB), selected_paths))
        first_shape = frames[0].shape
        if any(frame.shape != first_shape for frame in frames[1:]):
            raise ValueError(f"Inconsistent JPEG frame sizes in {item['video']}")
        video = self._crop_and_resize(torch.stack(frames), item.get("window"), clip_total_pixels)
        metadata = self._make_metadata(
            total_num_frames=len(paths),
            fps=source_fps,
            duration=len(paths) / source_fps,
            frames_indices=indices,
            height=int(video.shape[2]),
            width=int(video.shape[3]),
            video_backend="jpeg_sequence",
        )
        return video, metadata

    def _load_mp4_item(
        self,
        item: Dict[str, Any],
        clip_total_pixels: Optional[int],
    ) -> Tuple[torch.Tensor, Any]:
        try:
            import decord
        except ImportError as exc:
            raise RuntimeError("decord is required for MP4 CamVLM RL samples.") from exc
        reader = decord.VideoReader(item["video"])
        total_frames = len(reader)
        source_fps = float(reader.get_avg_fps())
        start = float(item.get("video_start", 0.0))
        end = float(item.get("video_end", total_frames / source_fps))
        indices = self._sample_indices(start, end, source_fps, total_frames)
        frames = torch.tensor(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
        video = self._crop_and_resize(frames, item.get("window"), clip_total_pixels)
        metadata = self._make_metadata(
            total_num_frames=total_frames,
            fps=source_fps,
            duration=total_frames / source_fps,
            frames_indices=indices,
            height=int(video.shape[2]),
            width=int(video.shape[3]),
            video_backend="decord",
        )
        return video, metadata

    def load_video_items(
        self,
        items: Sequence[Dict[str, Any]],
        *,
        budget_video_turns: Optional[int] = None,
    ) -> Tuple[List[torch.Tensor], List[Any]]:
        clip_total_pixels = None
        if items and self.total_tokens > 0:
            divisor = max(1, budget_video_turns or len(items))
            clip_total_pixels = int(
                self.total_tokens * QWEN3_VIDEO_PIXEL_SCALE / divisor
            )
        videos: List[torch.Tensor] = []
        metadata: List[Any] = []
        for item in items:
            if os.path.isdir(item["video"]):
                video, video_metadata = self._load_jpeg_item(item, clip_total_pixels)
            else:
                video, video_metadata = self._load_mp4_item(item, clip_total_pixels)
            videos.append(video)
            metadata.append(video_metadata)
        return videos, metadata

    def collect_from_messages(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        budget_video_turns: Optional[int] = None,
    ) -> Tuple[None, Optional[List[torch.Tensor]], Optional[List[Any]]]:
        items = [
            item
            for message in messages
            for item in message.get("content", [])
            if self.is_video_item(item)
        ]
        if not items:
            return None, None, None
        videos, metadata = self.load_video_items(
            items,
            budget_video_turns=budget_video_turns,
        )
        return None, videos, metadata

    def vllm_multimodal_data(
        self,
        messages: Sequence[Dict[str, Any]],
        *,
        budget_video_turns: Optional[int] = None,
    ) -> Dict[str, List[Tuple[np.ndarray, Dict[str, Any]]]]:
        items = [
            item
            for message in messages
            for item in message.get("content", [])
            if self.is_video_item(item)
        ]
        videos, metadata = self.load_video_items(
            items,
            budget_video_turns=budget_video_turns,
        )
        arrays = [
            video.permute(0, 2, 3, 1).contiguous().cpu().numpy().astype(np.uint8)
            for video in videos
        ]
        return {
            "video": [
                (array, self._metadata_dict(video_metadata))
                for array, video_metadata in zip(arrays, metadata)
            ]
        }
