#!/usr/bin/env python3
"""Check separately supplied model hashes without executing model code."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    errors = []
    for item in manifest["models"]:
        path = args.manifest.parent / item["file"]
        if not item.get("sha256") or len(item["sha256"]) != 64:
            errors.append(f"{item['file']}: supply an actual SHA256 from the model owner")
            continue
        if not path.is_file():
            errors.append(f"{item['file']}: missing")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != item["sha256"]:
            errors.append(f"{item['file']}: checksum mismatch")
    for error in errors:
        print(error)
    if errors:
        return 1
    print("All model checksums match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
