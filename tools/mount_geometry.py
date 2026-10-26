"""Camera-mount geometry calculator for vehicle pavement video.

Model: pinhole camera, flat road, square pixels, no lens distortion, EIS off.
World axes: X right, Y forward (direction of travel), Z up; camera at (0, 0, h).

Prints, for one mount configuration:
  - where each image row lands on the road, swath width, and ground sampling distance (GSD)
  - ground travel per frame, how many frames a road point stays in the useful lower part of the frame
  - motion blur at the image centre for a set of exposure times

Examples:
  python tools/mount_geometry.py --height 1.6 --pitch 45
  python tools/mount_geometry.py --width 1920 --rows 1080 --hfov 67 --height 1.3 --pitch 25 --speeds 20 30 40
"""
import argparse
import math

import numpy as np


def make_camera(width, rows, hfov_deg, height_m, pitch_deg):
    f = (width / 2) / math.tan(math.radians(hfov_deg) / 2)
    p = math.radians(pitch_deg)
    return {
        "W": width, "H": rows, "f": f, "cx": width / 2, "cy": rows / 2, "h": height_m,
        "x_c": np.array([1.0, 0.0, 0.0]),                      # image right
        "y_c": np.array([0.0, -math.sin(p), -math.cos(p)]),    # image down
        "z_c": np.array([0.0, math.cos(p), -math.sin(p)]),     # optical axis
        "vfov": 2 * math.degrees(math.atan((rows / 2) / f)),
    }


def pixel_to_ground(cam, u, v):
    d = cam["x_c"] * (u - cam["cx"]) / cam["f"] + cam["y_c"] * (v - cam["cy"]) / cam["f"] + cam["z_c"]
    if d[2] >= 0:
        return None  # ray at or above the horizon
    t = -cam["h"] / d[2]
    return np.array([0.0, 0.0, cam["h"]]) + t * d


def ground_to_pixel(cam, point):
    rel = point - np.array([0.0, 0.0, cam["h"]])
    xc, yc, zc = rel @ cam["x_c"], rel @ cam["y_c"], rel @ cam["z_c"]
    return np.array([cam["cx"] + cam["f"] * xc / zc, cam["cy"] + cam["f"] * yc / zc])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=3840, help="video width in px (long side, across the road)")
    ap.add_argument("--rows", type=int, default=2160, help="video height in px")
    ap.add_argument("--hfov", type=float, default=67.0,
                    help="horizontal field of view in degrees for the video mode (measure it; ~67 for a 26 mm-equiv main camera)")
    ap.add_argument("--height", type=float, required=True, help="camera height above the road in metres")
    ap.add_argument("--pitch", type=float, required=True, help="camera pitch below horizontal in degrees")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--speeds", type=float, nargs="+", default=[20, 40, 60], help="vehicle speeds in km/h")
    ap.add_argument("--exposures", type=float, nargs="+", default=[1000, 2000, 4000, 8000],
                    help="exposure times as 1/x seconds")
    ap.add_argument("--band", type=float, default=0.6,
                    help="fraction of the frame, from the bottom, treated as the useful high-resolution band")
    a = ap.parse_args()

    cam = make_camera(a.width, a.rows, a.hfov, a.height, a.pitch)
    W, H = a.width, a.rows
    print(f"{W}x{H}, HFOV {a.hfov:.1f} deg (VFOV {cam['vfov']:.1f}), camera {a.height} m high, pitched {a.pitch} deg down")
    if a.pitch - cam["vfov"] / 2 <= 0:
        print("note: the horizon is inside the frame (top rows see sky / far road)")

    for label, v in [("bottom", H - 1), ("3/4 down", 0.75 * H), ("centre", H / 2), ("1/4 down", 0.25 * H), ("top", 0)]:
        pc = pixel_to_ground(cam, W / 2, v)
        if pc is None:
            print(f"  row {label:9s}: above horizon")
            continue
        pl, pr = pixel_to_ground(cam, 0, v), pixel_to_ground(cam, W - 1, v)
        pu = pixel_to_ground(cam, W / 2 + 1, v)
        pv = pixel_to_ground(cam, W / 2, v + 1) if v + 1 < H else pixel_to_ground(cam, W / 2, v - 1)
        print(f"  row {label:9s}: {pc[1]:6.2f} m ahead | swath {np.linalg.norm(pr - pl):5.2f} m | "
              f"GSD across {np.linalg.norm(pu - pc) * 1000:5.2f} mm/px, along {np.linalg.norm(pv - pc) * 1000:5.2f} mm/px")

    y_bottom = pixel_to_ground(cam, W / 2, H - 1)[1]
    band_top = pixel_to_ground(cam, W / 2, (1 - a.band) * H)
    centre = pixel_to_ground(cam, W / 2, H / 2)
    for kmh in a.speeds:
        mps = kmh / 3.6
        step = mps / a.fps
        frames = (band_top[1] - y_bottom) / step if band_top is not None else float("inf")
        shift_px = np.linalg.norm(ground_to_pixel(cam, centre - np.array([0, step, 0])) - ground_to_pixel(cam, centre))
        blur = []
        for x in a.exposures:
            e = 1.0 / x
            moved = centre - np.array([0.0, mps * e, 0.0])
            px = np.linalg.norm(ground_to_pixel(cam, moved) - ground_to_pixel(cam, centre))
            blur.append(f"1/{x:g}s {px:4.1f}px ({mps * e * 1000:4.1f}mm)")
        print(f"  {kmh:4.0f} km/h: {step * 100:5.1f} cm/frame (~{shift_px:4.0f} px at centre) | "
              f"~{frames:4.1f} frames in lower {a.band:.0%} | centre blur: " + ", ".join(blur))


if __name__ == "__main__":
    main()
