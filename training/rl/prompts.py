from typing import Any, Dict, List


CAMVLM_SYSTEM_TEMPLATE = (
    "You are an expert in video tracking and video analysis. In a continuous video, each second you can "
    "only see a cropped local window from the current video frame, not the full frame. Based on the visible "
    "content inside the window, you need to decide each second whether to move or zoom the window to track "
    "all objects referred to by the multiple-choice question, ensuring they remain clearly visible inside "
    "the window with an appropriate size. After the video ends, answer the question using the window views "
    "observed throughout the process.\n\n"
    "Question:\n{question}\n\n"
    "Options:\n{options}\n\n"
    "During each second, output only the action JSON. "
    "Use this format:\n\n"
    "{{\"action\": ...}}\n\n"
    "Available actions: {{\"action\": None}} (no movement); {{\"action\": left/right/up/down, \"offset\": 0-1}} "
    "(move the window left/right/up/down, range 0-1); {{\"action\": zoom_in, \"scale\": s}}, where 0<s<1 "
    "(shrink the window by the direct scale factor); {{\"action\": zoom_out, \"scale\": s}}, where s>1 "
    "(enlarge the window by the direct scale factor). "
    "To apply multiple actions, separate them with semicolons, for example: "
    "{{\"action\": zoom_out, \"scale\": 1.5; \"action\": down, \"offset\": 0.12}}.\n\n"
    "When the user later indicates that the video has ended, stop window tracking and answer directly with "
    "<answer>X</answer>, where X is the selected option letter."
)
CAMVLM_QA_SYSTEM_TEMPLATE = (
    "You are an expert in video tracking and video analysis. In a continuous video, each second you can "
    "only see a cropped local window from the current video frame, not the full frame. Based on the visible "
    "content inside the window, you need to decide each second whether to move or zoom the window to track "
    "all objects referred to by the question, ensuring they remain clearly visible inside the window with "
    "an appropriate size. After the video ends, answer the question using the window views observed "
    "throughout the process.\n\n"
    "Question:\n{question}\n\n"
    "During each second, output only the action JSON. "
    "Use this format:\n\n"
    "{{\"action\": ...}}\n\n"
    "Available actions: {{\"action\": None}} (no movement); {{\"action\": left/right/up/down, \"offset\": 0-1}} "
    "(move the window left/right/up/down, range 0-1); {{\"action\": zoom_in, \"scale\": s}}, where 0<s<1 "
    "(shrink the window by the direct scale factor); {{\"action\": zoom_out, \"scale\": s}}, where s>1 "
    "(enlarge the window by the direct scale factor). "
    "To apply multiple actions, separate them with semicolons, for example: "
    "{{\"action\": zoom_out, \"scale\": 1.5; \"action\": down, \"offset\": 0.12}}.\n\n"
    "When the user later indicates that the video has ended, stop window tracking and answer directly with "
    "<answer>your answer</answer>."
)
FINAL_ANSWER_PROMPT_TEMPLATE = (
    "The video has ended. Stop window tracking and answer the multiple-choice question directly. "
    "Select exactly one option from {choices}. Output only its letter using this format: "
    "<answer>X</answer>."
)
QA_FINAL_ANSWER_PROMPT = (
    "The video has ended. Stop window tracking and answer the question directly. "
    "Output only the answer using this format: <answer>your answer</answer>."
)

QWEN3_CHAT_TEMPLATE = r"""
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for item in content -%}
            {%- if item.type == 'image' or item.image is string or item.image_url is string -%}
                <|vision_start|><|image_pad|><|vision_end|>
            {%- elif item.type == 'video' and item.video is string -%}
                <|vision_start|><|video_pad|><|vision_end|>
            {%- elif 'text' in item -%}
                {{- item.text -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- for message in messages -%}
    {{- '<|im_start|>' + message.role + '\n' -}}
    {{- render_content(message.content) -}}
    {{- '<|im_end|>\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
{%- endif -%}
"""

QWEN3_CAMTRACK_CHAT_TEMPLATE = QWEN3_CHAT_TEMPLATE


def format_options(options: List[Any]) -> str:
    return "\n".join(str(option).strip() for option in options)


def build_initial_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = str(sample["Question"]).strip()
    options = format_options(sample["options"])
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": CAMVLM_SYSTEM_TEMPLATE.format(question=question, options=options),
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Question:\n{question}\n\nOptions:\n{options}",
            }],
        },
    ]


def build_qa_initial_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    question = str(sample["Question"]).strip()
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": CAMVLM_QA_SYSTEM_TEMPLATE.format(question=question),
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Question:\n{question}",
            }],
        },
    ]


def build_final_prompt(num_options: int) -> str:
    choices = "/".join(chr(ord("A") + index) for index in range(num_options))
    return FINAL_ANSWER_PROMPT_TEMPLATE.format(choices=choices)


def build_qa_final_prompt() -> str:
    return QA_FINAL_ANSWER_PROMPT
