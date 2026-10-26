import math

import numpy as np

from geoai.config import GeometryConfig
from geoai.geometry import CameraModel, cam_from_device, intrinsics_from_camera_meta, nominal_model, vehicle_from_camera


def _model(pitch=45.0, h=1.6, w=3840, hgt=2160, hfov=67.0, facing="forward", dist=None):
    f = (w / 2) / math.tan(math.radians(hfov) / 2)
    K = np.array([[f, 0, (w - 1) / 2], [0, f, (hgt - 1) / 2], [0, 0, 1.0]])
    th = math.radians(pitch)
    up_cam = np.array([0.0, -math.cos(th), -math.sin(th)])
    R = vehicle_from_camera(up_cam, facing)
    return CameraModel(w, hgt, K, np.zeros(5) if dist is None else np.asarray(dist, float), R,
                       np.array([0.0, 0.0, h]))


def test_ground_round_trip_sub_millimetre():
    m = _model(dist=[0.05, -0.02, 0.001, -0.0005, 0.003])
    xs, ys = np.meshgrid(np.linspace(-1.5, 1.5, 13), np.linspace(0.9, 3.2, 12))
    u, v, ok = m.ground_to_pixel(xs, ys)
    assert ok.all()
    x2, y2, ok2 = m.pixel_to_ground(u, v)
    assert ok2.all()
    assert np.hypot(x2 - xs, y2 - ys).max() < 1e-3


def test_image_centre_ray_hits_ground_at_h_over_tan_pitch():
    m = _model(pitch=45.0, h=1.6)
    x, y, ok = m.pixel_to_ground(np.array([(m.width - 1) / 2]), np.array([(m.height - 1) / 2]))
    assert ok[0] and abs(x[0]) < 1e-9 and abs(y[0] - 1.6) < 1e-6
    assert abs(m.pitch_deg - 45.0) < 1e-9 and abs(m.roll_deg) < 1e-9


def test_rows_above_the_horizon_are_invalid():
    m = _model(pitch=10.0)
    _, _, ok = m.pixel_to_ground(np.array([1920.0]), np.array([10.0]))
    assert not ok[0]


def test_rearward_camera_sees_behind():
    m = _model(facing="rearward")
    x, y, ok = m.pixel_to_ground(np.array([(m.width - 1) / 2]), np.array([(m.height - 1) / 2]))
    assert ok[0] and abs(y[0] + 1.6) < 1e-6  # 1.6 m behind the camera, not ahead


def test_rotated_model_matches_rotated_pixels():
    m = _model(dist=[0.05, -0.02, 0.001, -0.0005, 0.003])
    r = m.rotated_180()
    u = np.array([500.0, 2000.0, 3500.0])
    v = np.array([1300.0, 1700.0, 2100.0])
    x1, y1, _ = m.pixel_to_ground(u, v)
    x2, y2, _ = r.pixel_to_ground(m.width - 1 - u, m.height - 1 - v)
    assert np.allclose(x1, x2, atol=1e-6) and np.allclose(y1, y2, atol=1e-6)


def test_scaled_model_consistent():
    m = _model()
    s = m.scaled(0.5)
    u = np.array([1000.0, 2500.0])
    v = np.array([1500.0, 2000.0])
    x1, y1, _ = m.pixel_to_ground(u, v)
    x2, y2, _ = s.pixel_to_ground((u + 0.5) * 0.5 - 0.5, (v + 0.5) * 0.5 - 0.5)
    assert np.allclose(x1, x2, atol=1e-6) and np.allclose(y1, y2, atol=1e-6)


def test_gravity_of_spec_example_gives_45_degree_pitch():
    # spec §4 example: sensor orientation 90, gravity (6.9, 0.1, 6.9)
    meta = {"camera": {"sensor_orientation_deg": 90}, "mount": {"nominal_height_m": 1.6, "camera_facing": "forward"}}
    m = nominal_model(meta, 3840, 2160, np.array([6.9, 0.1, 6.9]), GeometryConfig())
    assert abs(m.pitch_deg - 45.0) < 0.1
    assert abs(m.roll_deg) < 1.0
    assert m.height_m == 1.6 and m.source == "nominal"


def test_cam_from_device_is_a_rotation():
    for so in (0, 90, 180, 270):
        R = cam_from_device(so)
        assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-12


def test_intrinsics_from_characteristics_crop_and_scale():
    # 4000x3000 active array, factory fx = 3000 px, principal point at the centre.
    cam = {"active_array_size": [0, 0, 4000, 3000], "pixel_array_size": [4000, 3000],
           "lens_intrinsic_calibration": [3000.0, 3000.0, 2000.0, 1500.0, 0.0]}
    K, dist, src = intrinsics_from_camera_meta(cam, 3840, 2160, 67.0)
    assert src == "camera_characteristics"
    # 16:9 stream = 4000x2250 centred crop scaled by 0.96.
    assert abs(K[0, 0] - 2880.0) < 1e-9
    assert abs(K[0, 2] - 1920.0) < 1e-9 and abs(K[1, 2] - (1500 - 375) * 0.96) < 1e-9


def test_gsd_increases_with_range():
    m = _model()
    along, across = m.gsd_mm(np.array([1920.0, 1920.0]), np.array([2100.0, 900.0]))
    assert along[0] < along[1] and across[0] < across[1]
    assert 0.3 < along[0] < 1.5
