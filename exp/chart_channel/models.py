"""Model clients. HF router models use HF_TOKEN; "openai:<model>" uses OPENAI_API_KEY. Keys come only from the environment and are never printed."""

from __future__ import annotations

import base64
import os
import random
import re
import threading
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from chart_channel import spend
from chart_channel.sandbox import measure
from chart_channel.svg import json_values

ROUTER = "https://router.huggingface.co/v1"
_STOP = {400, 401, 403, 404, 422}


def token() -> str:
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            return value.strip()
    raise SystemExit("No Hugging Face token. Set HF_TOKEN.")


_clients: dict = {}


def client() -> OpenAI:
    if "hf" not in _clients:
        _clients["hf"] = OpenAI(base_url=ROUTER, api_key=token(), timeout=240, max_retries=0)
    return _clients["hf"]


def _backend(model: str):
    """'openai:<model>' goes to api.openai.com with OPENAI_API_KEY. Anything else goes to the HF router.
    Returns (client, model name sent, provider label)."""
    if model.startswith("openai:"):
        if "openai" not in _clients:
            _clients["openai"] = OpenAI(base_url="https://api.openai.com/v1",
                                        api_key=os.environ["OPENAI_API_KEY"], timeout=300, max_retries=0)
        return _clients["openai"], model.split(":", 1)[1], "openai"
    return client(), model, None


def parse_json(text: str) -> dict | None:
    found = [value for value in json_values(text) if isinstance(value, dict)]
    for preferred in ("signal", "detected", "score", "svg"):
        for value in reversed(found):
            if preferred in value:
                return value
    return found[-1] if found else None


_TAG = threading.local()


def set_tag(tag: str) -> None:
    _TAG.value = tag


def _chat(model: str, messages: list, max_tokens: int, temperature: float = 0.0, seed: int | None = None) -> str:
    delay = 3.0
    last: Exception | None = None
    api, name, fixed_provider = _backend(model)
    for _ in range(7):
        spend.check(model)
        try:
            if fixed_provider == "openai":
                kwargs = dict(model=name, messages=messages, max_completion_tokens=max(max_tokens, 32000),
                              reasoning_effort=os.environ.get("OPENAI_EFFORT", "low"))
            else:
                kwargs = dict(model=name, messages=messages, temperature=temperature, max_tokens=max_tokens)
                if seed is not None:
                    kwargs["seed"] = seed
            raw = api.chat.completions.with_raw_response.create(**kwargs)
            reply = raw.parse()
            provider = fixed_provider or raw.headers.get("x-inference-provider")
            usage = reply.usage
            if usage is not None:
                spend.record(model, provider, usage.prompt_tokens or 0, usage.completion_tokens or 0,
                             getattr(_TAG, "value", ""))
            return reply.choices[0].message.content or ""
        except APIStatusError as exc:
            if exc.status_code in _STOP:
                raise
            last = exc
        except (APITimeoutError, APIConnectionError) as exc:
            last = exc
        time.sleep(delay + random.random() * delay)
        delay = min(delay * 2, 60)
    assert last is not None
    raise last


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode()


def complete(model: str, prompt: str, image: Path | None = None, temperature: float = 0.0,
             seed: int | None = None, max_tokens: int = 12000) -> str:
    content: list | str
    if image is None:
        content = prompt
    else:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image)}"}},
        ]
    return _chat(model, [{"role": "user", "content": content}], max_tokens=max_tokens,
                 temperature=temperature, seed=seed)


def _code_block(text: str) -> str | None:
    for match in re.finditer(r"```([a-zA-Z]*)\s*(.*?)```", text, flags=re.S):
        lang, body = match.group(1).lower(), match.group(2).strip()
        if lang == "python" or (lang == "" and re.search(r"\b(im|np|print)\b", body)):
            return body
    return None


def decode(model: str, prompt: str, image: Path, turns: int = 5, extra: list[Path] | None = None) -> tuple[dict | None, list[str]]:
    """Text tool loop. The router model that rejected function calls can still emit Python."""
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(image)}"}},
    ]
    messages: list = [{"role": "user", "content": content}]
    logs: list[str] = []
    for _ in range(turns):
        text = _chat(model, messages, max_tokens=2000)
        messages.append({"role": "assistant", "content": text})
        parsed = parse_json(text)
        code = _code_block(text)
        if parsed and "signal" in parsed and code is None:
            return parsed, logs
        if code:
            output = measure(image, code, extra=extra)
            logs.append(output)
            messages.append({
                "role": "user",
                "content": (
                    f"Measurement output:\n{output}\n"
                    "Read im.size before indexing. No imports. "
                    "When you are done, reply with the JSON only and no code block."
                ),
            })
            continue
        messages.append({"role": "user", "content": "Reply with the JSON only and no code block."})
    text = _chat(model, messages, max_tokens=600)
    return parse_json(text), logs
