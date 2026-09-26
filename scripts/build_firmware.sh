#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
firmware_env="${FIRMWARE_ENV:-xiao_esp32s3}"
if [[ "$firmware_env" != "xiao_esp32s3" && "$firmware_env" != "firmware_ci" && "$firmware_env" != "firmware_ci_tls" ]]; then
  echo 'FIRMWARE_ENV must be xiao_esp32s3, firmware_ci or firmware_ci_tls' >&2
  exit 2
fi
if [[ -x .venv-firmware/bin/pio ]]; then
  pio_cmd=(.venv-firmware/bin/pio)
else
  pio_cmd=(pio)
fi
if [[ "$firmware_env" == "xiao_esp32s3" && ! -f firmware/config.h ]]; then
  echo 'Copy firmware/config.example.h to firmware/config.h and configure it first.' >&2
  echo 'For a credential-free build use FIRMWARE_ENV=firmware_ci.' >&2
  exit 2
fi
"${pio_cmd[@]}" run -e "$firmware_env" "$@"
output="build/release/$firmware_env"
mkdir -p "$output"
for file in firmware.bin bootloader.bin partitions.bin firmware.elf; do
  cp ".pio/build/$firmware_env/$file" "$output/$file"
done
python3 - "$output" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.suffix in ('.bin', '.elf')}
(root / 'sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
PY
printf 'Firmware artifacts: %s\n' "$output"
printf 'Private builds embed WiFi credentials and the device token. Keep their artifacts private.\n'
