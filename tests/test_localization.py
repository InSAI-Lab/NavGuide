import importlib.util
import json
from pathlib import Path
import re
import zipfile

from fastapi.testclient import TestClient

from navguide.runtime.config import Settings
from navguide.runtime.server import create_app


ROOT = Path(__file__).resolve().parents[1]
TOKEN = "localization-test-token-at-least-24-characters"
LOADER_PATH = "/static/localization.js"
LOCALE_PATH = "/static/locales/zh-CN.json"


def test_public_language_assets_preserve_device_authentication():
    expected_messages = json.loads((ROOT / "web" / LOCALE_PATH.lstrip("/")).read_text())
    with TestClient(create_app(Settings(device_token=TOKEN))) as client:
        assert client.get("/").status_code == 200
        loader = client.get(LOADER_PATH)
        assert loader.status_code == 200
        assert "javascript" in loader.headers["content-type"]
        assert LOCALE_PATH in loader.text
        messages = client.get(LOCALE_PATH)
        assert messages.status_code == 200
        assert messages.json() == expected_messages
        assert client.get("/api/status").status_code == 401
        assert client.get("/api/status", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
        for path in ("/static/locales/other.json", "/static/localization.js/private", "/static/navigation.js"):
            assert client.get(path).status_code == 401


def test_both_interfaces_load_complete_language_resources():
    console = (ROOT / "web/index.html").read_text()
    navigation = (ROOT / "web/templates/navigation.html").read_text()
    script = (ROOT / "web/static/navigation.js").read_text()
    loader = (ROOT / "web/static/localization.js").read_text()
    messages = json.loads((ROOT / "web/static/locales/zh-CN.json").read_text())

    assert LOADER_PATH in console
    assert 'src="/static/navigation.js"' in navigation
    assert 'from "./localization.js"' in script
    assert LOCALE_PATH in loader

    sources = (console, navigation, script)
    attribute_keys = {
        key for source in sources
        for key in re.findall(r'data-i18n(?:-placeholder)?="([^"]+)"', source)
    }
    runtime_keys = {
        key for source in sources
        for key in re.findall(r"['\"]((?:console|navigation|imu)\.[a-z_]+)['\"]", source)
    }
    assert attribute_keys
    assert runtime_keys
    for key in attribute_keys | runtime_keys:
        assert key in messages, key
        assert isinstance(messages[key], str) and messages[key], key


def test_source_archive_contains_interface_language_resources(tmp_path):
    spec = importlib.util.spec_from_file_location("localization_release", ROOT / "scripts/package_source.py")
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    result = release.build(tmp_path / "source.zip", ROOT)

    required = (
        "web/index.html",
        "web/templates/navigation.html",
        "web/static/navigation.js",
        "web/static/localization.js",
        "web/static/locales/zh-CN.json",
    )
    with zipfile.ZipFile(result["archive"]) as archive:
        for path in required:
            assert archive.read("NavGuide/" + path) == (ROOT / path).read_bytes()
