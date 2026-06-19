"""Parse model JSON responses with retry support."""

from __future__ import annotations

import json
import re
from typing import Any


class JSONParseError(ValueError):
    """Raised when model output cannot be parsed as the expected JSON object."""


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json_text(raw: str) -> str:
    """Extract a JSON object string from free-form model output."""
    text = raw.strip()
    if not text:
        raise JSONParseError("Empty model response")

    block_match = _JSON_BLOCK_RE.search(text)
    if block_match:
        return block_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text

    object_match = _JSON_OBJECT_RE.search(text)
    if object_match:
        return object_match.group(0).strip()

    raise JSONParseError("No JSON object found in model response")


def parse_prediction_json(raw: str, required_keys: list[str]) -> dict[str, Any]:
    """Parse and validate a prediction JSON object from model text."""
    json_text = extract_json_text(raw)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise JSONParseError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise JSONParseError("Parsed JSON is not an object")

    missing = [key for key in required_keys if key not in data]
    if missing:
        raise JSONParseError(f"Missing required keys: {missing}")

    return {key: data[key] for key in required_keys}
