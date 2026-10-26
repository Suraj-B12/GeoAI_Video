# attain_dinov2_probe_v1

A linear probe (logistic regression) on DINOv2 patch features. Its output is an absolute "this patch looks like
pavement distress" score used by the extraction pipeline (`geoai/anomaly.py`) to:
- keep likely distress out of the session's "normal pavement" bank;
- add candidates of its own.

| Item | Value |
|---|---|
| File | `attain_dinov2_probe_v1.npz` (SHA-256 `6b3786b8cf85abca0d1a2adca1abc482d7b30be3bc6219b5d896b173afa263f1`) |
| Created | 2026-10-22T14:08:38Z |
| Training data | Attain v1 (doi:10.17632/nykrzdm74f.1), CC BY 4.0, all three subsets |
| Features | `dinov2_vits14_reg4` blocks [4, 8], illumination normalisation sigma 24.0 px, tiles of 32 patches; road part of each image (rows below 45%) |
| Labels | Positive: at least 25 % of the patch inside a distress polygon, or its centre inside a distress box (half weight). Negative: no distress within one patch; faded markings and manholes are hard negatives (weight 3) |
| Split | blocks of 20 consecutive frames per subset (~70/15/15); VLM sample frames' blocks forced to test |
| Patches | train 391,461, val 97,416, test 214,148 |
| Regularisation | C = 3.0 (chosen on validation AUROC) |
| Seed threshold | logit 2.091: 1 % FPR on validation road patches |

## Held-out results (test split)

| Metric | Value |
|---|---|
| Patch AUROC (validation) | 0.878 |
| Patch AUROC (test) | 0.863 |
| False positives at the seed threshold, test road patches | 1.84 % |
| ... of which marking/manhole patches above the threshold | 3.1 % |
| Test images with distress: share with any patch above the threshold | 85 % (588 images) |
| Test images without annotated distress: share with any patch above the threshold | 55 % (89 images) |

Annotated objects in test images with at least one patch above the threshold. The last column is the share expected **by chance**: random patch scores at the same false-positive rate still hit large objects. A family is only really detected where its share clearly exceeds chance.

| Family | Objects | Flagged | Share | By chance |
|---|---|---|---|---|
| alligator | 937 | 884 | 94 % | 52 % |
| block | 31 | 31 | 100 % | 90 % |
| dropoff | 129 | 3 | 2 % | 22 % |
| linear | 2013 | 1718 | 85 % | 30 % |
| manhole | 50 | 35 | 70 % | 72 % |
| marking | 515 | 143 | 28 % | 47 % |
| patch | 179 | 106 | 59 % | 39 % |
| pothole | 99 | 80 | 81 % | 24 % |
| raveling | 165 | 35 | 21 % | 44 % |
| weathering | 1136 | 826 | 73 % | 64 % |

So the probe really detects linear cracks, potholes and alligator cracking, and patches moderately. Block cracks and weathering are large, so chance alone nearly matches their shares. Ravelling and lane-edge drop-off score **below** chance: the probe doesn't see them, and they are left to the session-bank detector. Faded markings also score below chance, which is the wanted behaviour for a look-alike.

## Limits

- **Viewpoint.** Attain images are windshield views (forward and rear, 640×640 to 1920×1080). The S21 rig looks down at the road at about 1 mm/px, so the probe is in a different domain there. Confirm on real sessions, using the VLM feedback in `vlm-import`.
- **Labels.** Box annotations include background around thin cracks (half weight). "Unannotated" doesn't guarantee healthy, so some negatives are unlabelled distress.
- **Classes.** Distress families covered: linear, alligator and block cracks, potholes, patches and utility cuts, weathering, ravelling, lane/shoulder drop-off. Not covered: bleeding, rutting, shoving and other deformations.

## Licence and attribution

Training data: Attain, Amirkabir University of Technology, Mendeley Data doi:10.17632/nykrzdm74f.1, CC BY 4.0.
Credit it when sharing this probe. Feature model: DINOv2 (Meta), Apache-2.0.
