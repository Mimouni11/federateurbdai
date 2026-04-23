import io
import json
import time
import base64
from pathlib import Path

import numpy as np
import torch
import mlflow
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from deepfake_detector.model import build_model
from deepfake_detector.gradcam import get_gradcam_image

ROOT           = Path(__file__).parent.parent
DETECTOR_DIR   = ROOT / "deepfake_detector"
MODEL_PATH     = DETECTOR_DIR / "model.pt"
THRESHOLD_PATH = DETECTOR_DIR / "threshold.json"
MLFLOW_URI     = f"sqlite:///{ROOT / 'mlflow.db'}"

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

app = FastAPI(title="Deepfake Detector + StyleGAN Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Detector (loaded at startup) ────────────────────────────────────────
model = build_model(pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── StyleGAN generator (lazy-loaded on first /generate call) ────────────
# Loading pulls in ~2GB of GPU memory (StyleGAN2 + CLIP ViT-B/32). Done on
# first request so the API boots fast even when the generator isn't used.
_generator = None


def get_generator():
    global _generator
    if _generator is None:
        from stylegan.clip_guidance import CLIPGuidedGenerator
        print("[generator] loading StyleGAN2 + CLIP ...")
        _generator = CLIPGuidedGenerator(device=DEVICE)
        print("[generator] ready")
    return _generator


@app.get("/health")
def health():
    return {
        "status":             "ok",
        "device":             DEVICE,
        "threshold":          THRESHOLD,
        "margin":             MARGIN,
        "generator_loaded":   _generator is not None,
    }


# ── /predict ────────────────────────────────────────────────────────────

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

    if centered > MARGIN:
        prediction = "fake"
    elif centered < -MARGIN:
        prediction = "real"
    else:
        prediction = "uncertain"

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


# ── /generate ───────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=300)
    steps:  int = Field(default=300, ge=20, le=1000)
    lr:     float = Field(default=0.02, gt=0.0, le=0.5)
    seed:   int = Field(default=42)


@app.post("/generate")
async def generate(req: GenerateRequest):
    """Generate a face image via StyleGAN2 + CLIP optimization.

    Returns the final image as base64 PNG plus per-generation metrics.
    Runs the same pipeline as `python -m stylegan.clip_guidance`.
    Logs the run to MLflow under the `user_generations` experiment.
    """
    generator = get_generator()

    start = time.perf_counter()
    final_np, history, _snapshots, _w = generator.optimize(
        req.prompt, num_steps=req.steps, lr=req.lr, seed=req.seed,
    )
    elapsed = time.perf_counter() - start

    final_clip_sim = float(history[-1]) if history else 0.0

    # Encode image as base64 PNG
    pil_img = Image.fromarray((np.clip(final_np, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Log to MLflow
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("user_generations")
        with mlflow.start_run(run_name=req.prompt[:40]):
            mlflow.log_params({
                "prompt":     req.prompt,
                "num_steps":  req.steps,
                "lr":         req.lr,
                "seed":       req.seed,
                "model":      "StyleGAN2-FFHQ + CLIP ViT-B/32",
                "source":     "api_user",
            })
            mlflow.log_metrics({
                "final_clip_similarity": final_clip_sim,
                "generation_time_s":     elapsed,
            })
    except Exception as e:
        print(f"[generate] MLflow logging failed (non-fatal): {e}")

    return {
        "prompt":                  req.prompt,
        "image":                   image_b64,
        "final_clip_similarity":   round(final_clip_sim, 4),
        "generation_time_s":       round(elapsed, 2),
        "num_steps":               req.steps,
        "lr":                      req.lr,
        "seed":                    req.seed,
    }
