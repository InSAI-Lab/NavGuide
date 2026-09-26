#!/usr/bin/env python3
"""Create a source-only snapshot without Git history, private config or media."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DIRS = {"navguide", "firmware", "deploy", "docs", "examples", "scripts", "tests", "web", ".github"}
EXTENSIONS = {".py", ".md", ".txt", ".json", ".jsonl", ".yml", ".yaml", ".toml", ".ini", ".cpp", ".h", ".ino", ".html", ".js", ".css", ".sh", ".bat", ".cff", ".example"}
NAMES = {"LICENSE", "Dockerfile", "Dockerfile.cloud", "Caddyfile", ".gitignore", ".dockerignore"}
PRIVATE_DIRS = {"__pycache__", ".pio", ".pytest_cache", ".git", "node_modules", "venv", "dist", "build",
                "recordings", "recording", "captures", "uploads", "output", "outputs", "logs", "private", "secrets", "credentials",
                "voice", "music", "media", "models"}
PRIVATE_NAMES = {"config.h", "credentials.json", "secrets.json", "service-account.json", "service_account.json", "RELEASE_MANIFEST.json"}
SECRET = re.compile(r"(?:sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
# Device and Wi-Fi secrets need not look like a provider's API key.
CONFIG_SECRET = re.compile(
    r'''(?im)^[ \t]*(?:\#define[ \t]+)?(?:NAVGUIDE_(?:WIFI_PASS|DEVICE_TOKEN|CLOUD_TOKEN|GATEWAY_TOKEN)|DASHSCOPE_API_KEY)[ \t]*(?:=|:)?[ \t]*(?:["']([^"'\r\n]+)["']|([A-Za-z0-9_-]{16,})[ \t]*$)'''
)
STAMP = (2026, 9, 26, 0, 0, 0)


def candidates(root=ROOT):
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if len(relative.parts) > 1 and relative.parts[0] not in DIRS:
            if str(relative) not in {"model/README.md", "model/manifest.example.json"}:
                continue
        if any(p.lower() in PRIVATE_DIRS or p.startswith(".venv") for p in relative.parts):
            continue
        if (path.name in PRIVATE_NAMES or ".local." in path.name or ".private." in path.name
                or (path.name.startswith(".env") and not path.name.endswith(".example"))):
            continue
        if path.suffix not in EXTENSIONS and path.name not in NAMES:
            continue
        yield path, relative


def build(output, root=ROOT):
    entries = list(candidates(root))
    files = []
    snapshots = []
    for path, relative in entries:
        data = path.read_bytes()
        text = data.decode("utf-8", errors="ignore")
        if SECRET.search(text) or CONFIG_SECRET.search(text):
            raise ValueError(f"Potential credential in {relative}; source archive not written")
        files.append({"path": str(relative), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        snapshots.append((relative, data))
    output.parent.mkdir(parents=True, exist_ok=True)

    def write(archive, relative, data):
        info = zipfile.ZipInfo("NavGuide/" + str(relative), date_time=STAMP)
        info.create_system = 3
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = (0o100755 if Path(relative).suffix == ".sh" else 0o100644) << 16
        archive.writestr(info, data)

    # Write the scanned bytes once, so a concurrent source edit cannot inject an
    # unscanned value or make the archive disagree with its manifest.
    descriptor, temporary = tempfile.mkstemp(prefix=".navguide-release-", suffix=".zip", dir=output.parent)
    os.close(descriptor)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for relative, data in snapshots:
                write(archive, relative, data)
            write(archive, "RELEASE_MANIFEST.json", json.dumps(files, indent=2).encode())
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {"archive": str(output), "files": len(files), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / "NavGuide-source.zip")
    args = parser.parse_args()
    print(json.dumps(build(args.output), indent=2))
