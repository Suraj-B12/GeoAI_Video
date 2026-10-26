#!/usr/bin/env python3
"""Train the DINOv2 distress probe on the Attain dataset (CC BY 4.0).

    python -I tools/train_probe.py data/attain models/probe

Steps:
1. Patch features for the road part of every image (lower 55 %), with exactly the pipeline's
   feature settings (geoai.config.AnomalyConfig: blocks, tiles, illumination normalisation).
2. Patch labels from the annotations:
   - positive: at least 25 % of the patch inside a distress polygon, or its centre inside a
     distress box (boxes are looser, so their patches get half weight);
   - negative: no distress pixel within one patch of it; patches on faded markings and manholes
     are kept as hard negatives (they look different but are not distress).
3. Split by blocks of 20 consecutive frames (neighbouring frames are near-duplicates). The 27
   frames the VLM team sent (vlm_handoff/sample_images) are forced into the test split.
4. L2-regularised logistic regression, C chosen on the validation split.
5. Test metrics: patch AUROC; at a threshold with 1 % false-positive rate on test road patches,
   the share of annotated distresses with at least one patch above it (per class), and the same
   for markings/manholes.
Writes <out>/attain_dinov2_probe_v1.npz and <out>/MODEL_CARD.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

PIPELINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE))  # -I removes the script directory from sys.path

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from geoai.anomaly import normalise_illumination  # noqa: E402
from geoai.config import AnomalyConfig  # noqa: E402
from geoai.dinov2 import PATCH, PatchFeatures  # noqa: E402
from geoai.probe import DistressProbe  # noqa: E402
from geoai.util import sha256_file  # noqa: E402

ROAD_FROM = 0.45      # images are windshield views: road in the lower part
LOOKALIKE = ("faded marking", "manhole")


def family(name: str) -> str:
    n = name.lower().split(" - ")[0].strip()
    n = re.sub(r"\s+", " ", n.replace("-", " ").strip())
    for key, fam in (("alligator", "alligator"), ("block", "block"), ("linear", "linear"), ("pothole", "pothole"),
                     ("ravel", "raveling"), ("weather", "weathering"), ("patch", "patch"), ("drop", "dropoff"),
                     ("marking", "marking"), ("manhole", "manhole")):
        if key in n:
            return fam
    return "unknown:" + n


def load_classes(root: Path) -> dict[str, list[str]]:
    out = {}
    for y in root.glob("*_data.yaml"):
        txt = y.read_text(encoding="utf-8")
        names = re.findall(r"""['"]([^'"]+)['"]""", txt[txt.index("names"):txt.index("]", txt.index("names"))])
        sub = re.search(r"dataset_name:\s*(\S+)", txt).group(1)          # Attain_SMP_WS_v1.0
        out[sub.rsplit(".", 1)[0]] = names                                # Attain_SMP_WS_v1
    return out


def annotations(img_path: Path, classes: dict[str, list[str]], w: int, h: int):
    """[(family, kind, points (N,2) in pixels)] with kind 'poly' or 'box'."""
    sub = img_path.parent.name
    out = []
    xml = img_path.with_suffix(".xml")
    txt = img_path.with_suffix(".txt")
    if xml.exists():
        root = ET.fromstring(xml.read_bytes())
        for obj in root.iter("object"):
            bb = obj.find("bndbox")
            x0, x1 = float(bb.findtext("xmin")), float(bb.findtext("xmax"))
            y0, y1 = float(bb.findtext("ymin")), float(bb.findtext("ymax"))
            out.append((family(obj.findtext("name")), "box", np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
    elif txt.exists():
        names = classes[sub]
        for line in txt.read_text().splitlines():
            v = line.split()
            if len(v) < 5:
                continue
            fam = family(names[int(v[0])])
            vals = np.array(v[1:], float)
            if len(vals) == 4:
                cx, cy, bw, bh = vals * np.array([w, h, w, h])
                out.append((fam, "box", np.array([[cx - bw / 2, cy - bh / 2], [cx + bw / 2, cy - bh / 2],
                                                  [cx + bw / 2, cy + bh / 2], [cx - bw / 2, cy + bh / 2]])))
            elif len(vals) >= 6 and len(vals) % 2 == 0:
                out.append((fam, "poly", vals.reshape(-1, 2) * np.array([w, h])))
    return out


def frame_index(name: str) -> int:
    return int(re.search(r"_(\d{6})", name).group(1))


def split_of(sub: str, idx: int, forced_test: set) -> str:
    if (sub, idx // 20) in forced_test:
        return "test"
    r = ((idx // 20) * 2654435761 + hash_sub(sub)) % 100
    return "test" if r < 15 else ("val" if r < 30 else "train")


def hash_sub(sub: str) -> int:
    return sum(ord(c) * (i + 1) for i, c in enumerate(sub)) * 97


def patch_labels(anns, gh, gw, y_off, H, W, rng):
    """Per patch: label (1 pos, 0 neg, -1 ignore), weight, and object ids for evaluation."""
    hh, ww = gh * PATCH, gw * PATCH
    dpoly = np.zeros((H, W), np.uint8)
    dbox = np.zeros((H, W), np.uint8)
    look = np.zeros((H, W), np.uint8)
    objs = []
    for k, (fam, kind, pts) in enumerate(anns):
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.round(pts).astype(np.int32)], 1)
        if fam in ("marking", "manhole"):
            look |= m
        elif kind == "poly":
            dpoly |= m
        else:
            dbox |= m
        objs.append((fam, kind, m[y_off:y_off + hh, :ww]))
    def frac(mask):
        return mask[y_off:y_off + hh, :ww].reshape(gh, PATCH, gw, PATCH).mean(axis=(1, 3))
    fp, fb, fl = frac(dpoly), frac(dbox), frac(look)
    any_d = ((fp > 0) | (fb > 0)).astype(np.uint8)
    near = cv2.dilate(any_d, np.ones((3, 3), np.uint8)) > 0
    label = np.full((gh, gw), -1, np.int8)
    weight = np.zeros((gh, gw), np.float32)
    pos_poly = fp >= 0.25
    pos_box = (fb >= 0.5) & ~pos_poly
    label[pos_poly] = 1
    weight[pos_poly] = 1.0
    label[pos_box] = 1
    weight[pos_box] = 0.5
    neg = ~near
    label[neg] = 0
    weight[neg] = np.where(fl[neg] > 0.2, 3.0, 1.0)  # markings / manholes: hard negatives
    obj_patches = []
    for fam, kind, m in objs:
        if m.size == 0:
            continue
        f = m.reshape(gh, PATCH, gw, PATCH).mean(axis=(1, 3))
        cells = np.argwhere(f >= (0.25 if kind == "poly" else 0.5))
        if len(cells) == 0:
            cells = np.argwhere(f > 0)
        if len(cells):
            obj_patches.append((fam, kind, cells))
    return label, weight, fl > 0.2, obj_patches


def build(root: Path, cfg: AnomalyConfig, cache: Path, forced_test: set, pf, max_pos: int = 200, max_neg: int = 200):
    """Sampled training patches for every image (features + labels + split)."""
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    classes = load_classes(root)
    rng = np.random.default_rng(0)
    X, Y, Wt, LOOK, SPLIT, IMG = [], [], [], [], [], []
    imgs = sorted(p for p in root.rglob("*.jpg"))
    t0 = time.time()
    for n, p in enumerate(imgs):
        feats, label, weight, look, _ = image_patches(p, root, classes, cfg, pf)
        sp = split_of(p.parent.name, frame_index(p.name), forced_test)
        pos = np.argwhere(label == 1)
        neg = np.argwhere(label == 0)
        if len(pos) > max_pos:
            pos = pos[rng.choice(len(pos), max_pos, replace=False)]
        if len(neg) > max_neg:
            # Keep hard negatives (markings, manholes) first, then sample the rest.
            hard = neg[look[neg[:, 0], neg[:, 1]]]
            easy = neg[~look[neg[:, 0], neg[:, 1]]]
            if len(hard) > max_neg // 2:
                hard = hard[rng.choice(len(hard), max_neg // 2, replace=False)]
            k = max_neg - len(hard)
            if len(easy) > k:
                easy = easy[rng.choice(len(easy), k, replace=False)]
            neg = np.concatenate([hard, easy])
        cells = np.concatenate([pos, neg]) if len(pos) else neg
        X.append(feats[cells[:, 0], cells[:, 1]].astype(np.float16))
        Y.append(label[cells[:, 0], cells[:, 1]])
        Wt.append(weight[cells[:, 0], cells[:, 1]])
        LOOK.append(look[cells[:, 0], cells[:, 1]])
        SPLIT.append(np.full(len(cells), {"train": 0, "val": 1, "test": 2}[sp], np.int8))
        IMG.append(np.full(len(cells), n, np.int32))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(imgs)} images, {time.time() - t0:.0f} s", flush=True)
    data = {"X": np.concatenate(X), "Y": np.concatenate(Y), "W": np.concatenate(Wt), "LOOK": np.concatenate(LOOK),
            "SPLIT": np.concatenate(SPLIT), "IMG": np.concatenate(IMG),
            "img_names": np.array([p.relative_to(root).as_posix() for p in imgs])}
    np.savez(cache, **data)
    return data


def image_patches(p: Path, root: Path, classes, cfg: AnomalyConfig, pf):
    im = np.asarray(Image.open(p).convert("RGB"))
    H, W = im.shape[:2]
    y_off = int(H * ROAD_FROM)
    feats = pf(normalise_illumination(im[y_off:], cfg.illum_sigma_px))
    gh, gw = feats.shape[:2]
    label, weight, look, objs = patch_labels(annotations(p, classes, W, H), gh, gw, y_off, H, W, None)
    return feats, label, weight, look, objs


def evaluate_objects(root: Path, cfg: AnomalyConfig, pf, w, b, seed, forced_test: set):
    """Test split, object level: does each annotated object get at least one patch >= seed?"""
    classes = load_classes(root)
    per = {}
    img_max = {"with_distress": [], "without_distress": []}
    for p in sorted(root.rglob("*.jpg")):
        if split_of(p.parent.name, frame_index(p.name), forced_test) != "test":
            continue
        feats, label, _, look, objs = image_patches(p, root, classes, cfg, pf)
        logit = feats @ w + b
        for fam, kind, cells in objs:
            o = per.setdefault(fam, [0, 0])
            o[0] += 1
            o[1] += bool((logit[cells[:, 0], cells[:, 1]] >= seed).any())
        distress = any(f not in ("marking", "manhole") for f, _, _ in objs)
        img_max["with_distress" if distress else "without_distress"].append(float(logit.max()))
    per_class = {k: {"objects": v[0], "flagged": v[1], "share": v[1] / v[0]} for k, v in sorted(per.items())}
    return per_class, img_max


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--samples", type=Path, default=PIPELINE.parent / "vlm_handoff" / "sample_images" / "manifest.csv")
    a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    cfg = AnomalyConfig()
    forced = set()
    if a.samples.exists():
        for line in a.samples.read_text(encoding="utf-8").splitlines()[1:]:
            m = re.search(r"Attain_SMP_WS_v2_(\d{6})", line)
            if m:
                forced.add(("Attain_SMP_WS_v2", int(m.group(1)) // 20))
    cache = a.cache or (a.out / "attain_features_cache.npz")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pf = PatchFeatures(Path(cfg.weights), cfg.device, cfg.tile_patches, cfg.tile_overlap_patches,
                       cfg.batch_tiles, cfg.weights_sha256, cfg.layers)
    d = build(a.data, cfg, cache, forced, pf)
    print(f"features: {len(d['Y'])} patches ({time.time() - t0:.0f} s)")
    lab = d["Y"] >= 0
    X = d["X"][lab].astype(np.float32)
    Y = d["Y"][lab].astype(int)
    Wt = d["W"][lab]
    S = d["SPLIT"][lab]
    LOOK = d["LOOK"][lab]
    tr, va, te = S == 0, S == 1, S == 2
    rng = np.random.default_rng(1)
    if tr.sum() > 500_000:  # bound memory/time of the fit
        idx = np.flatnonzero(tr)
        keep = np.zeros_like(tr)
        keep[rng.choice(idx, 500_000, replace=False)] = True
        tr = keep
    print(f"train {tr.sum()} (pos {Y[tr].sum()}), val {va.sum()} (pos {Y[va].sum()}), test {te.sum()} (pos {Y[te].sum()})")

    best = None
    for C in (0.03, 0.1, 0.3, 1.0, 3.0):
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        clf.fit(X[tr], Y[tr], sample_weight=Wt[tr])
        auc = roc_auc_score(Y[va], clf.decision_function(X[va]))
        print(f"  C={C}: val AUROC {auc:.4f}")
        if best is None or auc > best[0]:
            best = (auc, C, clf)
    val_auc, C, clf = best
    w = clf.coef_[0].astype(np.float32)
    b = float(clf.intercept_[0])

    # Threshold: 1 % false positives on held-out (validation) road patches, markings included.
    sv = X[va] @ w + b
    seed = float(np.quantile(sv[Y[va] == 0], 0.99))
    st = X[te] @ w + b
    test_auc = float(roc_auc_score(Y[te], st))
    test_fpr = float((st[Y[te] == 0] >= seed).mean())
    look_fpr = float((st[(Y[te] == 0) & LOOK[te]] >= seed).mean()) if ((Y[te] == 0) & LOOK[te]).any() else None
    # Object level, test split (features re-extracted for test images only).
    per_class, img_max = evaluate_objects(a.data, cfg, pf, w, b, seed, forced)
    meta = {
        "name": "attain_dinov2_probe_v1", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_data": "Attain v1 (doi:10.17632/nykrzdm74f.1), CC BY 4.0, all three subsets",
        "features": {"model": "dinov2_vits14_reg4", "weights_sha256": cfg.weights_sha256, "layers": list(cfg.layers),
                     "illum_sigma_px": cfg.illum_sigma_px, "tile_patches": cfg.tile_patches,
                     "tile_overlap_patches": cfg.tile_overlap_patches, "road_from": ROAD_FROM},
        "split": "blocks of 20 consecutive frames per subset (~70/15/15); VLM sample frames' blocks forced to test",
        "C": C, "val_auroc": val_auc, "test_auroc": test_auc, "seed_rule": "1 % FPR on validation road patches",
        "test_fpr_at_seed": test_fpr, "test_lookalike_patch_fpr_at_seed": look_fpr,
        "test_objects_flagged": per_class,
        "test_image_max_logit": {k: {"n": len(v), "median": float(np.median(v)) if v else None,
                                     "above_seed": float(np.mean(np.array(v) >= seed)) if v else None}
                                 for k, v in img_max.items()},
        "n_patches": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "illum_sigma_px": cfg.illum_sigma_px,
    }
    probe = DistressProbe(w, b, tuple(cfg.layers), cfg.illum_sigma_px, seed, meta)
    path = a.out / "attain_dinov2_probe_v1.npz"
    probe.save(path)
    meta["sha256"] = sha256_file(path)
    (a.out / "probe_metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("C", "val_auroc", "test_auroc", "test_fpr_at_seed",
                                            "test_lookalike_patch_fpr_at_seed")}, indent=1))
    for k, v in per_class.items():
        print(f"  {k:12s} {v['flagged']:5d}/{v['objects']:5d} flagged ({100 * v['share']:.0f} %)")
    print(f"wrote {path} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
