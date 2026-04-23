# Calibration Dataset

Small labeled validation set used by [`../calibrate_threshold.py`](../calibrate_threshold.py)
to pick the detector's decision threshold via Youden's J. Aim for ~15 images per
class minimum; more is better. Images are **not** tracked by git (see
[`.gitignore`](.gitignore)) — each contributor populates their own folders.

## Layout

```
calibration_data/
├── real/   — genuine face photos (class label 1)
└── fake/   — any generated face (class label 0)
```

## How to populate

Drop images directly into each subfolder. Any of `.png`, `.jpg`, `.jpeg`, `.webp`
works. The calibration script loads everything in these folders, runs inference,
and computes the threshold.

**Real sources** (drop in `real/`):
- Celebrity reference photos from `inputs/` folders
- Any real face downloaded from the web
- Test images from the Kaggle `140k-real-and-fake-faces` validation split

**Fake sources** (drop in `fake/`):
- `../../stylegan/outputs/*.png` — StyleGAN+CLIP generations
- SDXL outputs downloaded from Colab (`/content/outputs/*.png`)
- IP-Adapter edits from the SDXL notebook

Useful commands from project root:

```bash
# Copy some StyleGAN outputs as fake samples
cp stylegan/outputs/*.png deepfake_detector/calibration_data/fake/

# Copy SDXL outputs (after downloading outputs.zip from Colab)
unzip outputs.zip -d deepfake_detector/calibration_data/fake/
```

## Run calibration

From project root:

```bash
python -m deepfake_detector.calibrate_threshold
```

This produces:
- `../threshold.json` — the calibrated decision boundary
- `../outputs/roc_curve.png` + `confusion_matrix.png`
- An MLflow run under the `detector_calibration` experiment
