import os
import re
import textwrap
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from transformers import AutoModelForImageTextToText, AutoModelForVision2Seq, Qwen3VLForConditionalGeneration
except ImportError:
    AutoModelForImageTextToText = None
    AutoModelForVision2Seq = None
    Qwen3VLForConditionalGeneration = None


MIN_VIEWPORT_RATIO = 0.05
HARD_MAX_VIEWPORT_RATIO = 0.80
MIN_VISUALIZATION_BOX_WIDTH = 8
VISUALIZATION_BOX_WIDTH_DIVISOR = 250


def load_model(model_path: str, torch_dtype: str):
    dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    kwargs = {"torch_dtype": dtype, "device_map": "auto", "trust_remote_code": True}
    if Qwen3VLForConditionalGeneration is not None:
        return _sanitize_generation_config(Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs))
    if AutoModelForImageTextToText is not None:
        return _sanitize_generation_config(AutoModelForImageTextToText.from_pretrained(model_path, **kwargs))
    if AutoModelForVision2Seq is not None:
        return _sanitize_generation_config(AutoModelForVision2Seq.from_pretrained(model_path, **kwargs))
    raise ImportError("No supported Qwen3-VL model class is available in this transformers installation.")


def _sanitize_generation_config(model):
    generation_config = getattr(model, "generation_config", None)
    if generation_config is None:
        return model

    generation_config.do_sample = False
    for field in ("temperature", "top_p", "top_k"):
        if hasattr(generation_config, field):
            setattr(generation_config, field, None)
    return model


def parse_action_string(text: str) -> List[Dict[str, Any]]:
    action_match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, flags=re.DOTALL)
    if not action_match:
        return []

    action_text = action_match.group(0)
    chunks = [chunk.strip().strip("{}").strip() for chunk in action_text.split(";")]
    actions = []
    for chunk in chunks:
        action = re.search(r'"action"\s*:\s*"?([A-Za-z_]+|None)"?', chunk)
        if not action:
            continue
        name = action.group(1)
        if name == "None":
            actions.append({"action": "None"})
            continue

        offset = re.search(r'"offset"\s*:\s*([0-9.]+)', chunk)
        scale = re.search(r'"scale"\s*:\s*([0-9.]+)', chunk)
        parsed = {"action": name}
        if offset:
            parsed["offset"] = float(offset.group(1))
        if scale:
            parsed["scale"] = float(scale.group(1))
        actions.append(parsed)
    return actions


def split_think_action(text: str) -> Tuple[str, str, List[Dict[str, Any]]]:
    think = ""
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    action_match = re.search(r"\{[^{}]*\"action\"[^{}]*\}", text, flags=re.DOTALL)
    answer_match = re.search(r"<answer>.*?</answer>", text, flags=re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    elif text.strip().startswith("<think>"):
        think_start = text.find("<think>") + len("<think>")
        end_candidates = [
            match.start()
            for match in (action_match, answer_match)
            if match is not None and match.start() >= think_start
        ]
        think_end = min(end_candidates) if end_candidates else len(text)
        think = text[think_start:think_end].strip()
    actions = parse_action_string(text)
    action_text = ""
    if action_match:
        action_text = action_match.group(0).strip()
    return think, action_text, actions


def split_final_answer(text: str) -> Tuple[str, str]:
    think = ""
    answer = ""
    think_match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    if answer_match:
        answer = answer_match.group(1).strip()
    return think, answer


def load_font(size: int = 16):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def parse_initial_window(value: str) -> Tuple[float, float, float, float]:
    parts = [float(x) for x in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--initial_window must be x1,y1,x2,y2 in normalized coordinates.")
    x1, y1, x2, y2 = parts
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError("--initial_window values must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1.")
    return clamp_window_size_and_position((x1, y1, x2, y2))


def clamp_window_size_and_position(window: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = window
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    width = min(max(x2 - x1, MIN_VIEWPORT_RATIO), HARD_MAX_VIEWPORT_RATIO)
    height = min(max(y2 - y1, MIN_VIEWPORT_RATIO), HARD_MAX_VIEWPORT_RATIO)

    x1 = min(max(cx - width / 2, 0.0), 1.0 - width)
    y1 = min(max(cy - height / 2, 0.0), 1.0 - height)
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def apply_actions(
    window: Tuple[float, float, float, float],
    actions: List[Dict[str, Any]],
    *,
    scale_is_factor: bool = False,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = clamp_window_size_and_position(window)
    for action in actions:
        name = str(action.get("action")).lower()
        if name in ("none", "null"):
            continue
        if name in {"left", "right", "up", "down"}:
            offset = float(action.get("offset", 0.0))
            dx = (-offset if name == "left" else offset if name == "right" else 0.0)
            dy = (-offset if name == "up" else offset if name == "down" else 0.0)
            x1, x2 = x1 + dx, x2 + dx
            y1, y2 = y1 + dy, y2 + dy
        elif name in {"zoom_in", "zoom_out"}:
            scale = float(action.get("scale", 0.0))
            if scale_is_factor and (
                (name == "zoom_in" and not 0.0 < scale < 1.0)
                or (name == "zoom_out" and scale <= 1.0)
            ):
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            width, height = x2 - x1, y2 - y1
            if scale_is_factor:
                factor = scale
            else:
                factor = 1.0 / (1.0 + scale) if name == "zoom_in" else 1.0 + scale
            width, height = width * factor, height * factor
            x1, x2 = cx - width / 2, cx + width / 2
            y1, y2 = cy - height / 2, cy + height / 2

        x1, y1, x2, y2 = clamp_window_size_and_position((x1, y1, x2, y2))
    return x1, y1, x2, y2


def to_pixels(window: Tuple[float, float, float, float], size: Tuple[int, int]) -> List[int]:
    width, height = size
    x1, y1, x2, y2 = window
    return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]


def visualization_box_width(size: Tuple[int, int]) -> int:
    return max(MIN_VISUALIZATION_BOX_WIDTH, int(round(min(size) / VISUALIZATION_BOX_WIDTH_DIVISOR)))


def interpolate_window(
    before: Tuple[float, float, float, float],
    after: Tuple[float, float, float, float],
    progress: float,
) -> Tuple[float, float, float, float]:
    progress = min(max(progress, 0.0), 1.0)
    return tuple(b + (a - b) * progress for b, a in zip(before, after))  # type: ignore[return-value]


def draw_visualization(
    frame: Image.Image,
    before: Tuple[float, float, float, float],
    after: Tuple[float, float, float, float],
    title: str,
    actions: List[Dict[str, Any]],
) -> Image.Image:
    image = frame.copy()
    draw = ImageDraw.Draw(image)
    box_width = visualization_box_width(image.size)
    draw.rectangle(to_pixels(before, image.size), outline=(255, 0, 0), width=box_width)
    draw.rectangle(to_pixels(after, image.size), outline=(0, 220, 80), width=box_width)
    text = f"{title} | red=before green=after | actions={actions}"
    draw.rectangle([0, 0, image.size[0], 24], fill=(0, 0, 0))
    draw.text((6, 5), text, fill=(255, 255, 255))
    return image


def draw_gif_frame(
    frame: Image.Image,
    before: Tuple[float, float, float, float],
    current: Tuple[float, float, float, float],
    step: Dict[str, Any],
    frame_time: float,
    font,
) -> Image.Image:
    panel_height = 170
    image = Image.new("RGB", (frame.size[0], frame.size[1] + panel_height), (20, 20, 20))
    image.paste(frame.convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(image)
    box_width = visualization_box_width(frame.size)
    draw.rectangle(to_pixels(before, frame.size), outline=(255, 0, 0), width=box_width)
    draw.rectangle(to_pixels(current, frame.size), outline=(0, 220, 80), width=box_width)

    header = (
        f"step={step['step']}  t={frame_time:.2f}s  "
        f"red=before  green=current  action={step.get('action_text') or step.get('actions')}"
    )
    text = "CoT: " + (step.get("think") or "(empty)")
    lines = [header]
    lines.extend(textwrap.wrap(text, width=110)[:6])

    y = frame.size[1] + 8
    for line in lines:
        draw.text((8, y), line, fill=(255, 255, 255), font=font)
        y += 22
    return image


def make_summary(images: List[Image.Image], cols: int = 2, gap: int = 12) -> Image.Image:
    if not images:
        raise ValueError("No images to summarize")
    width, height = images[0].size
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * width + (cols - 1) * gap, rows * height + (rows - 1) * gap), (30, 30, 30))
    for idx, image in enumerate(images):
        row, col = divmod(idx, cols)
        canvas.paste(image, (col * (width + gap), row * (height + gap)))
    return canvas
