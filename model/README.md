# Model Files

Model binaries are supplied separately. Record each model's source, version, SHA256 checksum, license, and usage restrictions in the model manifest.

| File | Purpose |
| :--- | :--- |
| `yoloe-11l-seg.pt` | Open vocabulary detection and segmentation |
| `yolo-seg.pt` | Tactile paving and crossing segmentation |
| `trafficlight.pt` | Traffic light recognition |
| `shoppingbest5.pt` | Product search |
| `hand_landmarker.task` | MediaPipe hand detection |

Algorithm replay and the `observations` service mode do not require these files. Initialize and cache the YOLOE text encoder before deployment, following the provider's requirements. Redistribution is subject to each model owner's license.

Copy `model/manifest.example.json` to `model/manifest.json` and fill in the model details, including a valid SHA256 checksum. Run the check from the project root:

```bash
python scripts/check_models.py model/manifest.json
```
