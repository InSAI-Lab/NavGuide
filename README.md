# NavGuide

Wearable spoken guidance with task-aware visual content selection and inertial gating. NavGuide includes a local device service, tactile paving and street crossing workflows, item search, an optional cloud description gateway, and XIAO ESP32S3 Sense firmware.

## Quick Start

Use Python 3.11:

```bash
git clone https://github.com/InSAI-Lab/NavGuide.git
cd NavGuide
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt -c constraints.txt
cp .env.example .env
python -m navguide
```

Open <http://127.0.0.1:8081>. The default `observations` mode accepts detection results without loading a vision model. Camera navigation requires the models and device setup described below.

Run tests with `python -m pytest -q`.

## Documentation

- [Local deployment, device interfaces, and field validation](docs/deployment.md)
- [Cloud gateway deployment](docs/cloud.md)
- [Hardware wiring, firmware configuration, and flashing](docs/hardware.md)
- [Model files and dependencies](model/README.md)
- [Algorithm, parameter provenance, and evaluation](docs/architecture.md)
- [Project structure](PROJECT_STRUCTURE.md)

For extended navigation, install `requirements-navigation.txt`, prepare the task models and audio assets, then run `python -m navguide.navigation.app`.

## License

See [LICENSE](LICENSE) for license terms.
