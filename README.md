# GeoAI Pavement Survey

A vehicle-mounted Android phone records pavement video with exact per-frame time and position. An offline PC pipeline then cuts out every pavement distress as a geotagged, time-stamped crop and sends it to a separate VLM, which classifies it to IRC terms.

| Part | Status | Where |
|---|---|---|
| Capture app (Samsung Galaxy S21 5G pilot) | **Built**; tested on the emulator; S21 field checks pending | [android/](android/README.md) |
| Phone ↔ PC data contract | Spec v1.1, matches the app | [docs/session-package-spec.md](docs/session-package-spec.md) |
| USB transfer and session validation | **Built and tested** | [tools/pull_sessions.py](tools/pull_sessions.py), [tools/validate_session.py](tools/validate_session.py) |
| Extraction pipeline (PC, offline) | **Built and tested** (label-free; runs on the pipeline PC) | [pipeline/](pipeline/README.md), design in [docs/pipeline/](docs/pipeline/README.md) |
| VLM hand-off | **Built and tested** (folder batch over USB; runner for the VLM machine) | [pipeline/geoai/vlm.py](pipeline/geoai/vlm.py), [VLM team's answers](vlm_handoff/vlm-information-response.md) |
| Detector training | Not needed for the pilot (optional later) | [docs/detection-model-plan.md](docs/detection-model-plan.md) |
| Overall research and plan | v2 | [docs/PLAN.md](docs/PLAN.md) |
| Mount calculator | Built | [tools/mount_geometry.py](tools/mount_geometry.py) |

Everything is offline: the phone, the pipeline PC and the VLM machine exchange data only over USB, or optionally an isolated LAN between the PC and the VLM machine.
