import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training.rl.window_utils import apply_actions


Window = Tuple[float, float, float, float]
REWARD_WEIGHTS = {
    "accuracy": 1.0,
    "viewpoint": 0.6,
    "action_reward": 0.0,
}
_ACTION_OBJECT_RE = re.compile(r"^\s*\{(?P<body>.*?)\}\s*$", re.DOTALL)
_ACTION_CHUNK_RE = re.compile(
    r'^\s*"action"\s*:\s*"?(?P<name>None|left|right|up|down|zoom_in|zoom_out)"?'
    r'(?:\s*,\s*"(?P<value_name>offset|scale)"\s*:\s*(?P<value>[0-9]+(?:\.[0-9]+)?))?\s*$',
    re.IGNORECASE,
)
_ANSWER_RE = re.compile(
    r"^\s*<answer>\s*(?P<body>.*?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
_BARE_ANSWER_LETTER_RE = re.compile(r"^(?P<letter>[A-Z])$", re.IGNORECASE)
_LABELED_ANSWER_RE = re.compile(
    r"^\(\s*(?P<letter>[A-Z])\s*\)(?:\s+(?P<option_text>\S.*))?$",
    re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedAction:
    valid: bool
    actions: List[Dict[str, Any]]
    action_bearing: bool


@dataclass(frozen=True)
class RewardBreakdown:
    answer_accuracy: float
    mean_viewpoint: float
    action_reward: float
    total_reward: float
    action_seconds: int
    decision_seconds: int
    invalid_action_seconds: int
    bbox_transition_count: int
    bbox_fallback_transition_count: int
    final_answer_valid: bool


def parse_action_output(text: str) -> ParsedAction:
    match = _ACTION_OBJECT_RE.fullmatch(text or "")
    if not match:
        return ParsedAction(False, [], True)
    chunks = [chunk.strip() for chunk in match.group("body").split(";")]
    if not chunks or any(not chunk for chunk in chunks):
        return ParsedAction(False, [], True)

    actions: List[Dict[str, Any]] = []
    saw_none = False
    for chunk in chunks:
        chunk_match = _ACTION_CHUNK_RE.fullmatch(chunk)
        if not chunk_match:
            return ParsedAction(False, [], True)
        name = chunk_match.group("name")
        normalized = name.lower()
        value_name = chunk_match.group("value_name")
        value = chunk_match.group("value")
        if normalized == "none":
            if len(chunks) != 1 or value_name is not None:
                return ParsedAction(False, [], True)
            saw_none = True
            actions.append({"action": "None"})
            continue
        if normalized in {"left", "right", "up", "down"}:
            if value_name != "offset" or value is None or not 0.0 <= float(value) <= 1.0:
                return ParsedAction(False, [], True)
            actions.append({"action": normalized, "offset": float(value)})
        else:
            if value_name != "scale" or value is None:
                return ParsedAction(False, [], True)
            scale = float(value)
            if normalized == "zoom_in" and not 0.0 < scale < 1.0:
                return ParsedAction(False, [], True)
            if normalized == "zoom_out" and not scale > 1.0:
                return ParsedAction(False, [], True)
            actions.append({"action": normalized, "scale": scale})
    return ParsedAction(True, actions, not saw_none)


def parse_answer_letter(text: str, num_options: int) -> Tuple[Optional[str], bool]:
    match = _ANSWER_RE.fullmatch(text or "")
    if not match:
        return None, False
    body = match.group("body").strip()
    letter_match = _BARE_ANSWER_LETTER_RE.fullmatch(body)
    if letter_match is None:
        letter_match = _LABELED_ANSWER_RE.fullmatch(body)
    if letter_match is None:
        return None, False
    letter = letter_match.group("letter").upper()
    valid_letters = {chr(ord("A") + index) for index in range(num_options)}
    return (letter, True) if letter in valid_letters else (None, False)


def answer_letter_from_option(answer: str, num_options: int) -> str:
    match = re.match(r"^\s*\(\s*([A-Z])\s*\)", answer or "")
    if not match:
        raise ValueError(f"Could not extract option letter from Answer: {answer!r}")
    letter = match.group(1).upper()
    if letter not in {chr(ord("A") + index) for index in range(num_options)}:
        raise ValueError(f"Answer letter {letter!r} is outside the available options.")
    return letter


def viewpoint(viewport: Sequence[float], bbox: Sequence[float]) -> float:
    """Return the fraction of a target bbox visible inside a viewport (IoB)."""
    ax1, ay1, ax2, ay2 = map(float, viewport)
    bx1, by1, bx2, by2 = map(float, bbox)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return intersection / area_b if area_b > 0 else 0.0


def replay_action(window: Sequence[float], parsed: ParsedAction) -> Window:
    current = tuple(map(float, window))
    if not parsed.valid:
        return current
    return apply_actions(current, parsed.actions, scale_is_factor=True)


def compute_trajectory_reward(
    predicted_windows_after: Sequence[Sequence[float]],
    target_bboxes_by_turn: Sequence[Sequence[Sequence[float]]],
    target_bbox_fallbacks_by_turn: Sequence[Sequence[bool]],
    parsed_actions: Sequence[ParsedAction],
    final_output: str,
    gt_answer: str,
    num_options: int,
    *,
    reward_funcs: Sequence[str] = (
        "accuracy",
        "viewpoint",
    ),
) -> RewardBreakdown:
    pred_letter, final_valid = parse_answer_letter(final_output, num_options)
    gt_letter = answer_letter_from_option(gt_answer, num_options)
    accuracy = float(final_valid and pred_letter == gt_letter)

    transition_count = min(
        len(predicted_windows_after),
        max(0, len(target_bboxes_by_turn) - 1),
    )
    transition_coverages: List[float] = []
    fallback_transition_count = 0
    for index in range(transition_count):
        bboxes = target_bboxes_by_turn[index + 1]
        fallbacks = target_bbox_fallbacks_by_turn[index + 1]
        if not bboxes or len(bboxes) != len(fallbacks):
            raise ValueError(
                "Every reward transition must provide equally-sized non-empty "
                "target_bboxes and target_bbox_fallbacks."
            )
        transition_coverages.append(
            sum(viewpoint(predicted_windows_after[index], bbox) for bbox in bboxes) / len(bboxes)
        )
        fallback_transition_count += int(any(fallbacks))
    mean_coverage = (
        sum(transition_coverages) / len(transition_coverages)
        if transition_coverages
        else 0.0
    )

    scored_actions = list(parsed_actions)
    predicted_move_mask = [parsed.action_bearing for parsed in scored_actions]
    action_seconds = sum(predicted_move_mask)
    decision_seconds = len(scored_actions)
    invalid_action_seconds = sum(not parsed.valid for parsed in scored_actions)
    action_reward = 0.0
    reward_values = {
        "accuracy": accuracy,
        "viewpoint": mean_coverage,
        "action_reward": action_reward,
    }
    unknown = set(reward_funcs) - set(REWARD_WEIGHTS)
    if unknown:
        raise ValueError(f"Unknown reward functions: {sorted(unknown)}")
    total = sum(
        REWARD_WEIGHTS[name] * reward_values[name]
        for name in reward_funcs
    )
    return RewardBreakdown(
        answer_accuracy=accuracy,
        mean_viewpoint=mean_coverage,
        action_reward=action_reward,
        total_reward=total,
        action_seconds=action_seconds,
        decision_seconds=decision_seconds,
        invalid_action_seconds=invalid_action_seconds,
        bbox_transition_count=len(transition_coverages),
        bbox_fallback_transition_count=fallback_transition_count,
        final_answer_valid=final_valid,
    )
