"""Chinese phrases and matching terms shared by navigation components."""

from functools import lru_cache
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Union

Message = Union[str, tuple[str, ...]]


@lru_cache(maxsize=1)
def _messages() -> Mapping[str, Message]:
    path = Path(__file__).parent / "locales" / "zh-CN.json"
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("Language resources must be a JSON object")
    messages = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("Language resource keys must be strings")
        if isinstance(value, str):
            messages[key] = value
        elif isinstance(value, list) and all(isinstance(item, str) for item in value):
            messages[key] = tuple(value)
        else:
            raise ValueError(f"Language resource {key!r} must be text or a list of terms")
    return MappingProxyType(messages)


def text(key: str) -> str:
    """Return a phrase, label, or template."""
    value = _messages()[key]
    if not isinstance(value, str):
        raise TypeError(f"Language resource {key!r} is a term group")
    return value


def terms(key: str) -> tuple[str, ...]:
    """Return an ordered, immutable group of commands or matching terms."""
    value = _messages()[key]
    if not isinstance(value, tuple):
        raise TypeError(f"Language resource {key!r} is text")
    return value


FULL_STOP = text("punctuation.full_stop")
SENTENCE_ENDINGS = text("punctuation.sentence_endings")
PHRASE_SEPARATOR = text("punctuation.phrase_separator")
