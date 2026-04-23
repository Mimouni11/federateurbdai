import io
import json
import base64
from pathlib import Path

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from deepfake_detector.model import build_model
from deepfake_detector.gradcam import get_gradcam_image

DETECTOR_DIR   = Path(__file__).parent.parent / "deepfake_detector"
MODEL_PATH     = DETECTOR_DIR / "model.pt"
THRESHOLD_PATH = DETECTOR_DIR / "threshold.json"

# Load calibrated decision boundary. If threshold.json is missing (first-time
# setup before running calibrate_threshold.py), fall back to logit=0 which is
# equivalent to the original 0.5-sigmoid rule.
if THRESHOLD_PATH.exists():
    _cfg = json.loads(THRESHOLD_PATH.read_text())
    THRESHOLD = float(_cfg["threshold"])
    MARGIN    = float(_cfg["margin"])
    print(f"[calibration] threshold={THRESHOLD:.4f}  margin={MARGIN:.4f}")
else:
    THRESHOLD = 0.0
    MARGIN    = 0.0
    print("[calibration] threshold.json not found — using uncalibrated logit>0 rule")

app = FastAPI(title="Deepfake Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.get("/health")
def health():
    return {
        "status":    "ok",
        "device":    DEVICE,
        "threshold": THRESHOLD,
        "margin":    MARGIN,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a JPEG or PNG image.",
        )

    try:
        contents = await file.read()
        img      = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor).item()

    # AI probability = calibrated P(fake). Sigmoid on the *centered* logit
    # (logit - threshold) so 50% aligns with the decision boundary instead of
    # the model's biased raw sigmoid. 0% = very real, 100% = very fake.
    centered = logit - THRESHOLD
    ai_probability = float(torch.sigmoid(torch.tensor(centered)).item())

    # Three-way verdict using the uncertainty band on the logit axis
    if centered > MARGIN:
        prediction = "fake"
    elif centered < -MARGIN:
        prediction = "real"
    else:
        prediction = "uncertain"

    # Qualitative confidence: how far beyond the uncertainty band are we?
    # HIGH   = |centered| > 2 × margin  (clearly decided)
    # MEDIUM = |centered| > 1 × margin  (past the band but not deep)
    # LOW    = within the band (uncertain zone)
    if MARGIN > 0:
        distance_ratio = abs(centered) / MARGIN
    else:
        distance_ratio = abs(centered)
    if distance_ratio > 2.0:
        confidence = "high"
    elif distance_ratio > 1.0:
        confidence = "medium"
    else:
        confidence = "low"

    gradcam_img = get_gradcam_image(model, img, DEVICE)
    buf = io.BytesIO()
    gradcam_img.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "prediction":     prediction,
        "ai_probability": round(ai_probability, 4),
        "confidence":     confidence,
        "logit":          round(logit, 4),
        "gradcam_image":  gradcam_b64,
    }
