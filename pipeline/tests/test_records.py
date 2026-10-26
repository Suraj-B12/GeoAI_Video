import numpy as np
from PIL import ExifTags, Image

from geoai.records import dms_to_deg, read_png_record, unique_id, utc_strings, write_png

REC = {
    "id": "SESSION_0000123_00",
    "utc_ms": 1_792_650_123_456.0,
    "camera": {"lat": 12.9715987, "lon": 77.5945627, "alt_msl_m": 902.4, "h_sigma_m": 3.2, "speed_mps": 11.1,
               "heading_deg": 271.5, "view_direction_deg": 271.5},
    "anomaly": {"lat": 12.9716102, "lon": 77.5945401, "bearing_deg": 300.0, "distance_m": 2.7},
}


def test_png_exif_round_trip(tmp_path):
    p = tmp_path / "c.png"
    rgb = (np.random.default_rng(0).random((56, 84, 3)) * 255).astype(np.uint8)
    write_png(p, rgb, REC, make="samsung", model="SM-G991B")
    rec, gps = read_png_record(p)
    assert rec["id"] == REC["id"]
    G = ExifTags.GPS
    assert abs(dms_to_deg(gps[G.GPSLatitude], gps[G.GPSLatitudeRef]) - REC["camera"]["lat"]) < 1e-7
    assert abs(dms_to_deg(gps[G.GPSLongitude], gps[G.GPSLongitudeRef]) - REC["camera"]["lon"]) < 1e-7
    assert abs(dms_to_deg(gps[G.GPSDestLatitude], gps[G.GPSDestLatitudeRef]) - REC["anomaly"]["lat"]) < 1e-7
    assert abs(dms_to_deg(gps[G.GPSDestLongitude], gps[G.GPSDestLongitudeRef]) - REC["anomaly"]["lon"]) < 1e-7
    assert gps[G.GPSMapDatum] == "WGS-84"
    assert abs(float(gps[G.GPSAltitude]) - 902.4) < 0.01 and gps[G.GPSAltitudeRef] == b"\x00"
    with Image.open(p) as im:
        ex = im.getexif()
        exif = ex.get_ifd(ExifTags.IFD.Exif)
        assert ex[ExifTags.Base.Orientation] == 1
    assert exif[ExifTags.Base.ImageUniqueID] == unique_id(REC["id"]) and len(unique_id("x")) == 32
    assert exif[ExifTags.Base.OffsetTimeOriginal] == "+05:30"
    # Pixels unchanged (lossless).
    with Image.open(p) as im:
        assert np.array_equal(np.asarray(im.convert("RGB")), rgb)


def test_ist_is_utc_plus_5_30():
    s = utc_strings(1_792_650_123_456.0, 330)
    assert s["utc_iso"].endswith("Z") and s["local_iso"].endswith("+05:30")
    hh_utc, mm_utc = int(s["utc_iso"][11:13]), int(s["utc_iso"][14:16])
    hh_loc, mm_loc = int(s["local_iso"][11:13]), int(s["local_iso"][14:16])
    assert ((hh_loc * 60 + mm_loc) - (hh_utc * 60 + mm_utc)) % (24 * 60) == 330
    assert s["utc_iso"].endswith(".456Z")


def test_southern_western_hemisphere_refs(tmp_path):
    rec = {**REC, "camera": {**REC["camera"], "lat": -33.8688, "lon": -70.6693},
           "anomaly": {**REC["anomaly"], "lat": -33.86885, "lon": -70.66935}}
    p = tmp_path / "s.png"
    write_png(p, np.zeros((28, 28, 3), np.uint8), rec)
    _, gps = read_png_record(p)
    G = ExifTags.GPS
    assert gps[G.GPSLatitudeRef] == "S" and gps[G.GPSLongitudeRef] == "W"
    assert abs(dms_to_deg(gps[G.GPSLatitude], "S") + 33.8688) < 1e-7


def test_deterministic_bytes(tmp_path):
    rgb = np.full((56, 56, 3), 100, np.uint8)
    h1 = write_png(tmp_path / "a.png", rgb, REC)
    h2 = write_png(tmp_path / "b.png", rgb, REC)
    assert h1 == h2
