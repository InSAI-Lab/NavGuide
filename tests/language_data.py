"""Chinese input and output samples for language-dependent regression tests."""

from functools import lru_cache
import json
from pathlib import Path


@lru_cache(maxsize=1)
def _samples():
    path = Path(__file__).parent / "fixtures" / "locales" / "zh-CN.json"
    return json.loads(path.read_text(encoding="utf-8"))


def text(key):
    return _samples()[key]
