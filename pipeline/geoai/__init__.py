"""GeoAI extraction pipeline.

Reads verified GeoAI Capture sessions, finds road-surface regions that do not look like healthy
pavement (no labelled training data needed), and writes geotagged crops sized exactly for the
Qwen2.5-VL classifier that runs on a separate machine.
"""

__version__ = "0.1.0"
