import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest

spec = importlib.util.spec_from_file_location("release", Path(__file__).resolve().parents[1] / "scripts/package_source.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def test_export_excludes_private_config_history_weights_and_recordings(tmp_path):
    for name in ["module.py", ".env", ".env.example", "firmware/config.h", "firmware/config.example.h",
                 ".git/config", "model/private.pt", "recordings/person.wav", "web/static/models/private.glb"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("example")
    output = tmp_path / "dist" / "source.zip"
    release.build(output, tmp_path)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert "NavGuide/module.py" in names
    assert "NavGuide/.env.example" in names
    assert "NavGuide/firmware/config.example.h" in names
    assert not any(name.endswith((".pt", ".wav", ".glb", "/config.h", "/.env")) for name in names)
    assert not any("/.git/" in name for name in names)


def test_export_refuses_credential_without_printing_it(tmp_path):
    (tmp_path / "module.py").write_text('KEY="sk-' + 'a' * 32 + '"')
    with pytest.raises(ValueError, match="Potential credential in module.py"):
        release.build(tmp_path / "source.zip", tmp_path)


def test_export_contains_every_cloud_build_input(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = release.build(tmp_path / "source.zip", root)
    with zipfile.ZipFile(result["archive"]) as archive:
        names = set(archive.namelist())
    for path in ("deploy/Dockerfile.cloud", "deploy/.env.cloud.example", "deploy/Caddyfile",
                 "deploy/compose.cloud.yaml", "navguide/cloud/gateway.py", "navguide/cloud/asr.py", "requirements-cloud.txt", "constraints.txt"):
        assert "NavGuide/" + path in names


def test_nested_private_data_and_symlinks_are_excluded(tmp_path):
    private_files = ("deploy/.env.cloud", "deploy/credentials.json", "docs/private/notes.md",
                     "web/static/recordings/transcript.json", "scripts/config.local.json", "firmware/firmware.bin",
                     "web/static/models/model.json", "web/static/voice/transcript.txt")
    for name in private_files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private")
    (tmp_path / "module.py").write_text("public = True")
    (tmp_path / "linked.py").symlink_to(tmp_path / "deploy/.env.cloud")
    output = tmp_path / "output.zip"
    release.build(output, tmp_path)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {"NavGuide/module.py", "NavGuide/RELEASE_MANIFEST.json"}


@pytest.mark.parametrize("value", [
    "sk-proj-" + "a_b-" * 12,
    '#define NAVGUIDE_WIFI_PASS "private-campus-password"',
    'NAVGUIDE_DEVICE_TOKEN=' + "a" * 48,
])
def test_export_rejects_provider_device_and_wifi_credentials(tmp_path, value):
    (tmp_path / "config.example.h").write_text(value)
    output = tmp_path / "output.zip"
    with pytest.raises(ValueError) as error:
        release.build(output, tmp_path)
    assert value not in str(error.value)
    assert not output.exists()


def test_manifest_matches_scanned_bytes_and_archive_is_reproducible(tmp_path, monkeypatch):
    source = tmp_path / "module.py"
    source.write_text("public = True\n")
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    real_read = Path.read_bytes
    reads = []

    def read_once(path):
        if path == source:
            reads.append(path)
            # A second read simulates a secret injected after the validation pass.
            if len(reads) > 1:
                return b"unscanned secret"
        return real_read(path)

    with monkeypatch.context() as context:
        context.setattr(Path, "read_bytes", read_once)
        release.build(first, tmp_path)
    assert len(reads) == 1
    release.build(second, tmp_path)
    assert first.read_bytes() == second.read_bytes()
    with zipfile.ZipFile(first) as archive:
        manifest = json.loads(archive.read("NavGuide/RELEASE_MANIFEST.json"))
        for record in manifest:
            contents = archive.read("NavGuide/" + record["path"])
            assert record["bytes"] == len(contents)
            assert record["sha256"] == hashlib.sha256(contents).hexdigest()


def test_gitignore_keeps_deployment_template_and_rejects_live_config(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git not installed")
    root = Path(__file__).resolve().parents[1]
    shutil.copy(root / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for path, expected in (("deploy/.env.cloud", 0), ("firmware/config.h", 0),
                           ("deploy/.env.cloud.example", 1), ("firmware/config.example.h", 1)):
        result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "-q", path])
        assert result.returncode == expected, path


def test_minimal_dependencies_accept_socks_proxy_configuration():
    # Client construction must work on hosts with SOCKS settings, without
    # contacting a proxy or triggering any paid upstream request.
    import asyncio
    import httpx

    async def scenario():
        async with httpx.AsyncClient(proxy="socks5://127.0.0.1:9"):
            pass
    asyncio.run(scenario())
