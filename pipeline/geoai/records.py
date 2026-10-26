"""Writing crops (PNG with EXIF + embedded metadata), sidecar JSON, CSV and GeoJSON.

Crops are PNG, not JPEG: the frames are already video-compressed, and the VLM team measured
that re-encoding as JPEG q95 changed the label set on 9 of 49 photos (response notes 1.7).
PNG adds no further loss and is deterministic.

Metadata is written three ways so no single reader matters: EXIF in the PNG (GPS = camera,
GPSDest = anomaly centroid), a `geoai` PNG text chunk with the full record, and a sidecar JSON.
The VLM ignores all of it (it reads pixels only).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import ExifTags, Image
from PIL.PngImagePlugin import PngInfo
from PIL.TiffImagePlugin import IFDRational

from .util import to_json, write_json_atomic


def unique_id(name: str) -> str:
    """32 hex characters (EXIF ImageUniqueID) derived deterministically from a record name."""
    return hashlib.sha256(name.encode("utf-8")).hexdigest()[:32]


def _dms(deg: float) -> tuple[IFDRational, IFDRational, IFDRational]:
    d = abs(float(deg))
    D = int(d)
    m = (d - D) * 60
    M = int(m)
    S = round((m - M) * 60, 6)
    if S >= 60:
        S -= 60
        M += 1
    if M >= 60:
        M -= 60
        D += 1
    return IFDRational(D, 1), IFDRational(M, 1), IFDRational(int(round(S * 1_000_000)), 1_000_000)


def _rat(v: float, den: int = 1000) -> IFDRational:
    return IFDRational(int(round(float(v) * den)), den)


def utc_strings(utc_ms: float, offset_minutes: int) -> dict:
    t = datetime.fromtimestamp(utc_ms / 1000.0, tz=timezone.utc)
    loc = t + timedelta(minutes=offset_minutes)
    sign = "+" if offset_minutes >= 0 else "-"
    off = f"{sign}{abs(offset_minutes) // 60:02d}:{abs(offset_minutes) % 60:02d}"
    return {
        "utc_iso": t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z",
        "local_iso": loc.strftime("%Y-%m-%dT%H:%M:%S.") + f"{loc.microsecond // 1000:03d}{off}",
        "exif_local": loc.strftime("%Y:%m:%d %H:%M:%S"),
        "exif_subsec": f"{loc.microsecond // 1000:03d}",
        "exif_offset": off,
        "gps_date": t.strftime("%Y:%m:%d"),
        "gps_time": (t.hour, t.minute, t.second + t.microsecond / 1e6),
    }


def build_exif(rec: dict, make: str, model: str, software: str, offset_minutes: int) -> Image.Exif:
    """EXIF for a crop record (crop mode) or a `geoai.segment.v1` record (segment mode).

    GPS = camera position at the capture time; GPSDest = what the image is about: the anomaly centroid
    (`rec["anomaly"]`, crop mode) or the segment centre (`rec["target"]`, segment mode). The capture time
    is the record's top-level `utc_ms` (crop mode) or, when that key is absent, `rec["frame"]["utc_ms"]`
    (segment mode keeps every time inside its frame block). Crop records always carry `utc_ms` and no
    `target`, so their EXIF bytes are exactly what they were before segment mode existed.
    """
    ex = Image.Exif()
    ex[ExifTags.Base.Make] = make or "unknown"
    ex[ExifTags.Base.Model] = model or "unknown"
    ex[ExifTags.Base.Software] = software
    ex[ExifTags.Base.Orientation] = 1  # pixels are stored upright
    exif_ifd = {ExifTags.Base.ImageUniqueID: unique_id(rec["id"])}
    gps: dict = {ExifTags.GPS.GPSVersionID: b"\x02\x03\x00\x00", ExifTags.GPS.GPSMapDatum: "WGS-84"}
    utc_ms = rec["utc_ms"] if "utc_ms" in rec else (rec.get("frame") or {}).get("utc_ms")
    if utc_ms is not None:
        ts = utc_strings(utc_ms, offset_minutes)
        exif_ifd.update({ExifTags.Base.DateTimeOriginal: ts["exif_local"],
                         ExifTags.Base.SubsecTimeOriginal: ts["exif_subsec"],
                         ExifTags.Base.OffsetTimeOriginal: ts["exif_offset"]})
        h, m, s = ts["gps_time"]
        gps[ExifTags.GPS.GPSDateStamp] = ts["gps_date"]
        gps[ExifTags.GPS.GPSTimeStamp] = (IFDRational(h, 1), IFDRational(m, 1), _rat(s, 1000))
    cam = rec.get("camera") or {}
    if cam.get("lat") is not None:
        gps[ExifTags.GPS.GPSLatitudeRef] = "N" if cam["lat"] >= 0 else "S"
        gps[ExifTags.GPS.GPSLatitude] = _dms(cam["lat"])
        gps[ExifTags.GPS.GPSLongitudeRef] = "E" if cam["lon"] >= 0 else "W"
        gps[ExifTags.GPS.GPSLongitude] = _dms(cam["lon"])
        if cam.get("alt_msl_m") is not None:  # never the ellipsoidal height (docs/pipeline/08 §2)
            gps[ExifTags.GPS.GPSAltitudeRef] = b"\x00" if cam["alt_msl_m"] >= 0 else b"\x01"
            gps[ExifTags.GPS.GPSAltitude] = _rat(abs(cam["alt_msl_m"]), 100)
        if cam.get("h_sigma_m") is not None:
            gps[31] = _rat(cam["h_sigma_m"], 100)  # GPSHPositioningError (EXIF 2.31)
        if cam.get("speed_mps") is not None:
            gps[ExifTags.GPS.GPSSpeedRef] = "K"
            gps[ExifTags.GPS.GPSSpeed] = _rat(cam["speed_mps"] * 3.6, 100)
        if cam.get("heading_deg") is not None:
            gps[ExifTags.GPS.GPSTrackRef] = "T"
            gps[ExifTags.GPS.GPSTrack] = _rat(cam["heading_deg"] % 360, 100)
            gps[ExifTags.GPS.GPSImgDirectionRef] = "T"
            gps[ExifTags.GPS.GPSImgDirection] = _rat(cam.get("view_direction_deg", cam["heading_deg"]) % 360, 100)
    tgt = rec.get("target") or rec.get("anomaly") or {}
    if tgt.get("lat") is not None:
        gps[ExifTags.GPS.GPSDestLatitudeRef] = "N" if tgt["lat"] >= 0 else "S"
        gps[ExifTags.GPS.GPSDestLatitude] = _dms(tgt["lat"])
        gps[ExifTags.GPS.GPSDestLongitudeRef] = "E" if tgt["lon"] >= 0 else "W"
        gps[ExifTags.GPS.GPSDestLongitude] = _dms(tgt["lon"])
        if tgt.get("bearing_deg") is not None and tgt.get("distance_m") is not None:
            gps[ExifTags.GPS.GPSDestBearingRef] = "T"
            gps[ExifTags.GPS.GPSDestBearing] = _rat(tgt["bearing_deg"] % 360, 100)
            gps[ExifTags.GPS.GPSDestDistanceRef] = "K"
            gps[ExifTags.GPS.GPSDestDistance] = _rat(tgt["distance_m"] / 1000.0, 100000)
    ex[ExifTags.IFD.Exif] = exif_ifd
    ex[ExifTags.IFD.GPSInfo] = gps
    return ex


def write_png(path: Path, rgb: np.ndarray, rec: dict | None, *, make: str = "", model: str = "",
              software: str = "GeoAI pipeline", offset_minutes: int = 330, compress_level: int = 6) -> str:
    """Write an RGB uint8 image as PNG with metadata; returns the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    kwargs = {"format": "PNG", "compress_level": compress_level}
    if rec is not None:
        info = PngInfo()
        info.add_text("geoai", to_json(rec, indent=None))
        kwargs["pnginfo"] = info
        kwargs["exif"] = build_exif(rec, make, model, software, offset_minutes).tobytes()
    tmp = path.with_suffix(".tmp.png")
    img.save(tmp, **kwargs)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PngPool:
    """Writes PNGs on worker threads; zlib releases the GIL, so encoding runs in parallel with detection.

    The image and the record are copied at submit time, so later changes to either don't reach the file
    and the output is the same as writing in order. `drain()` waits for every file and runs the callbacks
    (in submit order) with each file's SHA-256.
    """

    def __init__(self, workers: int | None = None, max_pending: int = 64):
        self._ex = ThreadPoolExecutor(workers or min(8, os.cpu_count() or 4), thread_name_prefix="png")
        self._jobs: list[tuple[Future, object]] = []
        self._max_pending = max_pending

    def submit(self, path: Path, rgb: np.ndarray, rec: dict | None, on_done=None, **kwargs) -> None:
        img = np.array(rgb, dtype=np.uint8, copy=True, order="C")
        rec_copy = json.loads(to_json(rec, indent=None)) if rec is not None else None
        self._jobs.append((self._ex.submit(write_png, path, img, rec_copy, **kwargs), on_done))
        pending = [f for f, _ in self._jobs if not f.done()]
        if len(pending) > self._max_pending:
            pending[0].result()  # bound the memory held by queued images

    def drain(self) -> None:
        jobs, self._jobs = self._jobs, []
        for fut, cb in jobs:
            sha = fut.result()
            if cb is not None:
                cb(sha)

    def close(self) -> None:
        try:
            self.drain()
        finally:
            self._ex.shutdown(wait=True)


def read_png_record(path: Path) -> tuple[dict | None, dict]:
    """(embedded geoai record, EXIF GPS IFD) of a crop, for checks."""
    with Image.open(path) as im:
        txt = im.text.get("geoai")
        gps = dict(im.getexif().get_ifd(ExifTags.IFD.GPSInfo))
    return (json.loads(txt) if txt else None), gps


def dms_to_deg(dms, ref: str) -> float:
    d = float(dms[0]) + float(dms[1]) / 60 + float(dms[2]) / 3600
    return -d if ref in ("S", "W") else d


CSV_FIELDS = ["id", "status", "session_id", "segment", "frame_in_segment", "frame_global", "utc_iso", "local_iso",
              "anomaly_lat", "anomaly_lon", "anomaly_sigma_m", "camera_lat", "camera_lon", "heading_deg",
              "speed_mps", "distance_m", "lateral_m", "ahead_m", "area_m2", "evidence", "kinds", "crop_file",
              "frame_request", "out_w", "out_h", "tokens", "scale", "truncated", "flags"]


def write_csv(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel reads UTF-8 correctly
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            a, c, cr = r.get("anomaly") or {}, r.get("camera") or {}, r.get("crop") or {}
            w.writerow({
                "id": r["id"], "status": r.get("status"), "session_id": r.get("session_id"),
                "segment": r.get("segment"), "frame_in_segment": r.get("frame_in_segment"),
                "frame_global": r.get("frame_global"), "utc_iso": r.get("utc_iso"), "local_iso": r.get("local_iso"),
                "anomaly_lat": _fmt(a.get("lat"), 9), "anomaly_lon": _fmt(a.get("lon"), 9),
                "anomaly_sigma_m": _fmt(a.get("sigma_m"), 2), "camera_lat": _fmt(c.get("lat"), 9),
                "camera_lon": _fmt(c.get("lon"), 9), "heading_deg": _fmt(c.get("heading_deg"), 2),
                "speed_mps": _fmt(c.get("speed_mps"), 2), "distance_m": _fmt(a.get("distance_m"), 3),
                "lateral_m": _fmt(a.get("lateral_m"), 3), "ahead_m": _fmt(a.get("ahead_m"), 3),
                "area_m2": _fmt(a.get("area_m2"), 4), "evidence": _fmt(r.get("evidence"), 2),
                "kinds": "+".join(r.get("kinds", [])), "crop_file": cr.get("file"),
                "frame_request": r.get("frame_request"), "out_w": cr.get("out_w"), "out_h": cr.get("out_h"),
                "tokens": cr.get("tokens"), "scale": _fmt(cr.get("scale"), 4), "truncated": int(bool(cr.get("truncated"))),
                "flags": "+".join(r.get("flags", [])),
            })


def _fmt(v, nd):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    return f"{float(v):.{nd}f}"


def write_geojson(path: Path, recs: list[dict]) -> None:
    """Anomaly points, RFC 7946 (WGS-84, [longitude, latitude])."""
    feats = []
    for r in recs:
        a = r.get("anomaly") or {}
        if a.get("lat") is None:
            continue
        props = {k: r.get(k) for k in ("id", "status", "session_id", "frame_global", "utc_iso", "local_iso",
                                         "evidence", "kinds", "flags", "frame_request")}
        props.update({"crop_file": (r.get("crop") or {}).get("file"), "area_m2": a.get("area_m2"),
                      "sigma_m": a.get("sigma_m")})
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [a["lon"], a["lat"]]},
                      "properties": props})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def write_sidecar(path: Path, rec: dict) -> None:
    write_json_atomic(path, rec)
