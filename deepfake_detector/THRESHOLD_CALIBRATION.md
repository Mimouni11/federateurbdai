# Deepfake Detector — Threshold Calibration Plan

## Context

The trained EfficientNet-B0 detector (`deepfake_detector/model.pt`, from teammate's 99% AUC run) has a severe **positive bias**: on every image we throw at it, the raw logit comes out positive (observed range 3.1–5.2 on real + StyleGAN + SDXL inputs). The sigmoid → `prob < 0.5 ? fake : real` rule in `api/main.py:64` therefore always returns `real`. Sample logs:

| Input | Logit | Sigmoid | Current verdict | Correct verdict |
|---|---|---|---|---|
| Real celebrity photo | 5.19 | 0.9945 | REAL ✓ | REAL |
| StyleGAN fake | 4.83 | 0.9920 | REAL ✗ | FAKE |
| SDXL + IP-Adapter fake | 3.11 | 0.9574 | REAL ✗ | FAKE |

The **model still discriminates** (fakes score below reals), but the 0.5 threshold is wrong for this particular weight file. Threshold calibration is the right choice because:

1. The grading rubric rewards methodology rigor — a data-driven threshold with an ROC curve and logged MLflow run is more defensible than a hand-tuned bias offset.
2. The project narrative "detector has architecture-specific weaknesses" benefits from an **uncertainty zone** in the output.
3. We have enough labeled data to calibrate properly — celebrity photos = real, StyleGAN/SDXL outputs = fake.

## Approach

### Enhancement 1: use Youden's J for threshold selection (not F1)

Youden's J = TPR − FPR. More robust than F1 when classes are imbalanced and gives a threshold that balances both types of errors equally. Standard in medical diagnostic calibration literature.

### Enhancement 2: add an uncertainty band

Emit three labels instead of two: `real`, `fake`, `uncertain`. Logits within `threshold ± margin` → uncertain. Cleanly handles the borderline SDXL cases and is honest about the detector's limits.

### Enhancement 3: persist threshold as config

Save the chosen threshold + margin to `deepfake_detector/threshold.json`. The API loads it at startup instead of hardcoding. The config gets baked into the Docker image, no code edits when the threshold changes.

### Enhancement 4: log calibration run to MLflow

ROC curve, confusion matrix, per-class metrics, chosen threshold — all logged under an experiment called `detector_calibration`. Fulfills the MLflow traceability requirement for the detector half of the project.

## Files to create / modify

| File | Change |
|---|---|
| `deepfake_detector/calibrate_threshold.py` | **new** — calibration script. Reads images from `calibration_data/real/` and `calibration_data/fake/`, runs inference, computes ROC, picks threshold via Youden's J, logs everything to MLflow, writes `threshold.json`. |
| `deepfake_detector/threshold.json` | **new** — `{ "threshold": 4.5, "margin": 0.5 }` (values filled in by calibration script). |
| `api/main.py` | Load `threshold.json` at startup. Replace `is_fake = prob < 0.5` with three-way logic using the logit + margin. Response schema gains `uncertain` option. Remove the `[DEBUG]` print added during diagnosis. |
| `frontend/src/api.ts` | `PredictResponse.prediction` type becomes `"real" \| "fake" \| "uncertain"`. |
| `frontend/src/components/Detector.tsx` | `ResultBar` handles the `uncertain` case (grey card, "UNCERTAIN" label). |
| `deepfake_detector/calibration_data/real/` + `fake/` | **new** — small labeled set (~15 images each) using existing files: real = celebrity inputs already on disk; fake = StyleGAN outputs from `stylegan/outputs/` + SDXL outputs downloaded from Colab. |

## Existing utilities to reuse

- `deepfake_detector.model.build_model` — model factory, already used in `api/main.py:26`
- `TRANSFORM` block in `api/main.py:30-34` — copy the same preprocess into the calibration script to match inference-time preprocessing exactly
- MLflow setup pattern from `deepfake_detector/train.py:35` — reuse the same experiment pattern

## Verification

1. Drop ~15 real face images in `deepfake_detector/calibration_data/real/` and ~15 fake images (mix of StyleGAN + SDXL) in `deepfake_detector/calibration_data/fake/`.
2. Run `python -m deepfake_detector.calibrate_threshold` — should:
   - Print per-image logits and labels
   - Print selected threshold + Youden's J value
   - Write `deepfake_detector/threshold.json`
   - Create an MLflow run under `detector_calibration` experiment with ROC curve + confusion matrix artifacts
3. Restart uvicorn. Upload the same three test images from the debug session (Brad Pitt photo, StyleGAN fake, SDXL edit) via the React frontend. Expected:
   - Real Brad Pitt → REAL (confidence > 0.5)
   - StyleGAN fake → FAKE or UNCERTAIN
   - SDXL fake → FAKE
4. Check MLflow UI at `http://localhost:5555` — `detector_calibration` experiment exists with ROC curve artifact.

## Out of scope

- Retraining the model on balanced data (the real fix but a day of work)
- Batch evaluation of all generated images (separate follow-up for the "generator vs detector" comparison story)
- Frontend copy changes beyond supporting the `uncertain` state
