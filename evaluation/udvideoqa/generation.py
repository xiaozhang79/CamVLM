from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GenerationSettings:
    max_new_tokens: int
    do_sample: bool
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


def resolve_generation_settings(
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        top_p=top_p,
    )


def transformers_generation_kwargs(settings: GenerationSettings):
    kwargs = {
        "max_new_tokens": settings.max_new_tokens,
        "do_sample": settings.do_sample,
    }
    if settings.temperature is not None:
        kwargs["temperature"] = settings.temperature
    if settings.top_p is not None:
        kwargs["top_p"] = settings.top_p
    if settings.top_k is not None:
        kwargs["top_k"] = settings.top_k
    if settings.repetition_penalty is not None:
        kwargs["repetition_penalty"] = settings.repetition_penalty
    return kwargs


def vllm_generation_kwargs(settings: GenerationSettings):
    return {
        "max_tokens": settings.max_new_tokens,
        "temperature": (
            settings.temperature
            if settings.do_sample and settings.temperature is not None
            else 1.0 if settings.do_sample else 0.0
        ),
        "top_p": settings.top_p if settings.top_p is not None else 1.0,
        "top_k": settings.top_k if settings.top_k is not None else -1,
        "repetition_penalty": (
            settings.repetition_penalty
            if settings.repetition_penalty is not None
            else 1.0
        ),
    }
