#!/usr/bin/env python3
"""Write models/probe/MODEL_CARD.md from probe_metrics.json (so the card can't drift from the numbers)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "models" / "probe"
    m = json.loads((d / "probe_metrics.json").read_text(encoding="utf-8"))
    m["seed_logit"] = float(np.load(d / "attain_dinov2_probe_v1.npz")["seed_logit"])
    f = m["features"]
    rows = "\n".join(f"| {k} | {v['objects']} | {v['flagged']} | {100 * v['share']:.0f} % | "
                     + (f"{100 * v['chance_share']:.0f} %" if "chance_share" in v else "n/a") + " |"
                     for k, v in m["test_objects_flagged"].items())
    im = m.get("test_image_max_logit", {})
    card = f"""# {m['name']}

A linear probe (logistic regression) on DINOv2 patch features. Its output is an absolute "this patch looks like
pavement distress" score used by the extraction pipeline (`geoai/anomaly.py`) to:
- keep likely distress out of the session's "normal pavement" bank;
- add candidates of its own.

| Item | Value |
|---|---|
| File | `attain_dinov2_probe_v1.npz` (SHA-256 `{m.get('sha256', '?')}`) |
| Created | {m['created_utc']} |
| Training data | {m['training_data']} |
| Features | `{f['model']}` blocks {f['layers']}, illumination normalisation sigma {f['illum_sigma_px']} px, tiles of {f['tile_patches']} patches; road part of each image (rows below {f['road_from']:.0%}) |
| Labels | Positive: at least 25 % of the patch inside a distress polygon, or its centre inside a distress box (half weight). Negative: no distress within one patch; faded markings and manholes are hard negatives (weight 3) |
| Split | {m['split']} |
| Patches | train {m['n_patches']['train']:,}, val {m['n_patches']['val']:,}, test {m['n_patches']['test']:,} |
| Regularisation | C = {m['C']} (chosen on validation AUROC) |
| Seed threshold | logit {float(m.get('seed_logit', 0) or 0):.3f}: {m['seed_rule']} |

## Held-out results (test split)

| Metric | Value |
|---|---|
| Patch AUROC (validation) | {m['val_auroc']:.3f} |
| Patch AUROC (test) | {m['test_auroc']:.3f} |
| False positives at the seed threshold, test road patches | {100 * m['test_fpr_at_seed']:.2f} % |
| ... of which marking/manhole patches above the threshold | {('%.1f %%' % (100 * m['test_lookalike_patch_fpr_at_seed'])) if m['test_lookalike_patch_fpr_at_seed'] is not None else 'n/a'} |
| Test images with distress: share with any patch above the threshold | {('%.0f %%' % (100 * im['with_distress']['above_seed'])) if im.get('with_distress', {}).get('above_seed') is not None else 'n/a'} ({im.get('with_distress', {}).get('n', 0)} images) |
| Test images without annotated distress: share with any patch above the threshold | {('%.0f %%' % (100 * im['without_distress']['above_seed'])) if im.get('without_distress', {}).get('above_seed') is not None else 'n/a'} ({im.get('without_distress', {}).get('n', 0)} images) |

Annotated objects in test images with at least one patch above the threshold. The last column is the share expected **by chance**: random patch scores at the same false-positive rate still hit large objects. A family is only really detected where its share clearly exceeds chance.

| Family | Objects | Flagged | Share | By chance |
|---|---|---|---|---|
{rows}

So the probe really detects linear cracks, potholes and alligator cracking, and patches moderately. Block cracks and weathering are large, so chance alone nearly matches their shares. Ravelling and lane-edge drop-off score **below** chance: the probe doesn't see them, and they are left to the session-bank detector. Faded markings also score below chance, which is the wanted behaviour for a look-alike.

## Limits

- **Viewpoint.** Attain images are windshield views (forward and rear, 640×640 to 1920×1080). The S21 rig looks down at the road at about 1 mm/px, so the probe is in a different domain there. Confirm on real sessions, using the VLM feedback in `vlm-import`.
- **Labels.** Box annotations include background around thin cracks (half weight). "Unannotated" doesn't guarantee healthy, so some negatives are unlabelled distress.
- **Classes.** Distress families covered: linear, alligator and block cracks, potholes, patches and utility cuts, weathering, ravelling, lane/shoulder drop-off. Not covered: bleeding, rutting, shoving and other deformations.

## Licence and attribution

Training data: Attain, Amirkabir University of Technology, Mendeley Data doi:10.17632/nykrzdm74f.1, CC BY 4.0.
Credit it when sharing this probe. Feature model: DINOv2 (Meta), Apache-2.0.
"""
    (d / "MODEL_CARD.md").write_text(card, encoding="utf-8", newline="\n")
    print(d / "MODEL_CARD.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
