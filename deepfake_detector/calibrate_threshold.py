"""Calibrate the detector's decision threshold from a small labeled validation set.

Runs inference on images in `calibration_data/real/` and `calibration_data/fake/`,
computes the ROC curve, and picks the threshold that maximizes Youden's J statistic
(TPR - FPR). Writes `threshold.json` for the API to load at startup.

Logs the calibration run to MLflow (experiment `detector_calibration`) with:
- ROC curve and confusion matrix PNGs
- Per-class precision / recall / F1 metrics
- The saved threshold.json as an artifact

Usage (from project root):
    python -m deepfake_detector.calibrate_threshold
    python -m deepfake_detector.calibrate_threshold --margin 0.8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from torchvision import transforms

from deepfake_detector.model import build_model


ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
MODEL_PATH = ROOT / "model.pt"
THRESHOLD_PATH = ROOT / "threshold.json"
DEFAULT_DATA_DIR = ROOT / "calibration_data"
MLFLOW_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"

# Same preprocess as api/main.py → matches inference conditions exactly
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def collect_logits(model, data_dir: Path, device: str):
    """Run inference on every image in real/ and fake/. Returns (logits, labels).

    Label convention: real=0, fake=1. This matches the model's *actual* training
    polarity (verified empirically: feeding the opposite convention to roc_curve
    produces AUC 0.40, i.e. below random). So high logit → model says fake.
    """
    logits, labels = [], []
    for cls_name, cls_label in [("real", 0), ("fake", 1)]:
        cls_dir = data_dir / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(
                f"Missing calibration folder: {cls_dir}\n"
                f"Create {data_dir}/real/ and {data_dir}/fake/ and drop ~15 images in each."
            )
        images = sorted(
            p for p in cls_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if not images:
            raise ValueError(f"No images in {cls_dir}")
        for img_path in images:
            img = Image.open(img_path).convert("RGB")
            tensor = TRANSFORM(img).unsqueeze(0).to(device)
            with torch.no_grad():
                logit = model(tensor).item()
            logits.append(logit)
            labels.append(cls_label)
            print(f"  {cls_name}/{img_path.name}: logit={logit:.4f}")
    return np.array(logits), np.array(labels)


def find_best_threshold(logits, labels):
    """Pick the threshold on the logit axis that maximizes Youden's J (TPR − FPR).

    Youden's J balances sensitivity and specificity equally — robust to class imbalance.
    """
    fpr, tpr, thresholds = roc_curve(labels, logits)
    j_scores = tpr - fpr
    best_idx = int(j_scores.argmax())
    return float(thresholds[best_idx]), float(j_scores[best_idx]), (fpr, tpr)


def plot_roc(fpr, tpr, auc, threshold, out_path: Path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, linewidth=2, color="#F7820F", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Detector ROC curve — threshold = {threshold:.3f}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_cm(cm, out_path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Oranges",
        xticklabels=["real", "fake"],
        yticklabels=["real", "fake"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (calibrated threshold)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory with real/ and fake/ subfolders of labeled images",
    )
    parser.add_argument(
        "--margin", type=float, default=0.5,
        help="Uncertainty band around the threshold (in logit units). "
             "Logits within threshold ± margin → 'uncertain' at inference time.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval().to(device)
    print(f"Model loaded: {MODEL_PATH}")

    print(f"\nCollecting logits from {args.data_dir}/")
    logits, labels = collect_logits(model, args.data_dir, device)
    # Label convention here: real=0, fake=1 (matches training, see collect_logits)
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    print(f"\nTotal: {len(logits)} images ({n_real} real, {n_fake} fake)")

    threshold, youden_j, (fpr, tpr) = find_best_threshold(logits, labels)
    auc = float(roc_auc_score(labels, logits))
    print(f"\nBest threshold (logit): {threshold:.4f}")
    print(f"Youden's J:             {youden_j:.4f}")
    print(f"AUC:                    {auc:.4f}")

    preds = (logits > threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    acc = float((preds == labels).mean())
    # Labels: 0 = real, 1 = fake (matches training polarity)
    prec_real = float(precision_score(labels, preds, pos_label=0, zero_division=0))
    rec_real = float(recall_score(labels, preds, pos_label=0, zero_division=0))
    f1_real = float(f1_score(labels, preds, pos_label=0, zero_division=0))
    prec_fake = float(precision_score(labels, preds, pos_label=1, zero_division=0))
    rec_fake = float(recall_score(labels, preds, pos_label=1, zero_division=0))
    f1_fake = float(f1_score(labels, preds, pos_label=1, zero_division=0))
    print(f"\nAccuracy at chosen threshold: {acc:.4f}")
    print(f"Real  — P {prec_real:.3f}  R {rec_real:.3f}  F1 {f1_real:.3f}")
    print(f"Fake  — P {prec_fake:.3f}  R {rec_fake:.3f}  F1 {f1_fake:.3f}")

    config = {
        "threshold": threshold,
        "margin": float(args.margin),
        "auc": auc,
        "youden_j": youden_j,
        "n_real": n_real,
        "n_fake": n_fake,
    }
    THRESHOLD_PATH.write_text(json.dumps(config, indent=2))
    print(f"\nSaved {THRESHOLD_PATH}")

    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    roc_path = outputs_dir / "roc_curve.png"
    cm_path = outputs_dir / "confusion_matrix.png"
    plot_roc(fpr, tpr, auc, threshold, roc_path)
    plot_cm(cm, cm_path)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("detector_calibration")
    with mlflow.start_run(run_name=f"calibration_t{threshold:.2f}"):
        mlflow.log_params({
            "data_dir": str(args.data_dir),
            "n_real": n_real,
            "n_fake": n_fake,
            "model": "efficientnet_b0",
            "model_path": str(MODEL_PATH),
            "margin_logit": float(args.margin),
        })
        mlflow.log_metrics({
            "threshold_logit": threshold,
            "youden_j": youden_j,
            "auc": auc,
            "accuracy": acc,
            "real_precision": prec_real,
            "real_recall": rec_real,
            "real_f1": f1_real,
            "fake_precision": prec_fake,
            "fake_recall": rec_fake,
            "fake_f1": f1_fake,
        })
        mlflow.log_artifact(str(roc_path))
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(THRESHOLD_PATH))
        print("\nLogged to MLflow experiment 'detector_calibration'")


if __name__ == "__main__":
    main()
