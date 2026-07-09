"""Thin Fireworks client wrapper: retries, cost logging, mock mode."""

import time

from openai import OpenAI

from . import config
from .costs import tracker

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.FIREWORKS_API_KEY and not config.MOCK:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. Put it in .env or the environment, "
                "or run with STYLEFORGE_MOCK=1 for an offline smoke test."
            )
        _client = OpenAI(base_url=config.FIREWORKS_BASE_URL, api_key=config.FIREWORKS_API_KEY)
    return _client


def chat(
    messages: list[dict],
    model: str,
    *,
    mock_response: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
    reasoning_effort: str | None = None,
    retries: int = 3,
) -> str:
    if config.MOCK:
        return mock_response if mock_response is not None else "MOCK"

    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort is not None:
        # Kimi K2.x: "none" fully disables thinking -> fast, cheap, clean content.
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            usage = resp.usage
            tracker.record(
                model,
                usage.prompt_tokens if usage else 0,
                usage.completion_tokens if usage else 0,
            )
            choice = resp.choices[0]
            content = choice.message.content or ""
            # Reasoning models (Kimi K2.x): thinking normally lands in the separate
            # reasoning_content field, but leaks into content if truncated mid-thought.
            if "</think>" in content:
                content = content.split("</think>")[-1]
            content = content.strip()
            if not content or choice.finish_reason == "length":
                raise RuntimeError(
                    f"empty/truncated completion (finish_reason={choice.finish_reason}) "
                    f"— raise max_tokens for reasoning models"
                )
            return content
        except Exception as e:  # noqa: BLE001 — retry any transport/API error
            last_err = e
            if attempt < retries - 1:  # no dead sleep after the final attempt
                time.sleep(2**attempt)
    raise RuntimeError(f"Fireworks call to {model} failed after {retries} attempts: {last_err}")
