"""
CLIP-guided face generation using StyleGAN2.

Text prompt -> optimize W latent -> StyleGAN2 renders face matching the description.

Usage:
    python -m models.clip_guidance --prompt "a young woman with red hair"
    python -m models.clip_guidance --prompt "an old man with a beard" --steps 400 --seed 123
"""

import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image

from utils.config import (
    CLIP_MODEL, CLIP_NORMALIZE_MEAN, CLIP_NORMALIZE_STD,
    NUM_STEPS, LEARNING_RATE, W_MEAN_SAMPLES,
    OUTPUT_DIR, MLFLOW_DB_URI, USE_FP16,
)
from models.architecture import load_generator, generate_from_w, compute_mean_w
from utils.metrics import compute_all_metrics


clip_normalize = transforms.Normalize(mean=CLIP_NORMALIZE_MEAN, std=CLIP_NORMALIZE_STD)


class CLIPGuidedGenerator:

    def __init__(self, checkpoint=None, device="cuda"):
        self.device = device

        # Load generator (fp16 locally, fp32 on Kaggle)
        self.G = load_generator(checkpoint=checkpoint, device=device)

        # Load CLIP
        import clip
        self.clip_model, _ = clip.load(CLIP_MODEL, device=device)
        self.clip_model.eval()
        print(f"CLIP {CLIP_MODEL} loaded")

        # Precompute mean W
        self.w_mean = compute_mean_w(self.G, num_samples=W_MEAN_SAMPLES, device=device)

    def encode_text(self, prompt):
        """Encode text prompt to normalized CLIP features."""
        import clip
        tokens = clip.tokenize([prompt]).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features

    def clip_loss(self, image, text_features):
        """Negative cosine similarity between image and text in CLIP space."""
        # image: [1, 3, H, W] in [0, 1], possibly fp16
        image_224 = F.interpolate(image.float(), size=(224, 224), mode="bilinear", align_corners=False)
        image_224 = clip_normalize(image_224).half()
        image_features = self.clip_model.encode_image(image_224)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (image_features * text_features).sum(dim=-1)
        return -similarity.mean()

    def optimize(self, prompt, num_steps=NUM_STEPS, lr=LEARNING_RATE, seed=42):
        """Optimize W to match a text prompt. Returns (image_np, history, snapshots, w)."""
        print(f'Generating: "{prompt}" ({num_steps} steps, lr={lr})')

        text_features = self.encode_text(prompt)

        # Init W from mean + small noise
        torch.manual_seed(seed)
        w = (self.w_mean.clone() + torch.randn_like(self.w_mean) * 0.05).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)

        history = []
        snapshots = []

        for step in range(num_steps):
            optimizer.zero_grad()

            img = generate_from_w(self.G, w)
            loss = self.clip_loss(img, text_features)
            loss.backward()
            optimizer.step()

            sim = -loss.item()
            history.append(sim)

            if step % 50 == 0 or step == num_steps - 1:
                print(f"  step {step:3d}/{num_steps}  similarity: {sim:.4f}")
                with torch.no_grad():
                    snap = generate_from_w(self.G, w)
                    snapshots.append(snap[0].float().permute(1, 2, 0).cpu().numpy())

        with torch.no_grad():
            final = generate_from_w(self.G, w)
            final_np = final[0].float().permute(1, 2, 0).cpu().numpy()

        return final_np, history, snapshots, w.detach()

    def interpolate(self, w1, w2, n_steps=8):
        """Interpolate between two W vectors. Returns list of image arrays."""
        alphas = np.linspace(0, 1, n_steps)
        images = []
        for alpha in alphas:
            w_interp = (1 - alpha) * w1 + alpha * w2
            with torch.no_grad():
                img = generate_from_w(self.G, w_interp)
                images.append(img[0].float().permute(1, 2, 0).cpu().numpy())
        return images, alphas


def save_image(image_np, prompt, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = prompt.replace(" ", "_")[:50] + ".png"
    path = os.path.join(output_dir, filename)
    img = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
    img.save(path)
    print(f"Saved: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="CLIP-guided face generation")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--steps", type=int, default=NUM_STEPS)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected, this will be very slow")

    gen = CLIPGuidedGenerator(device=device)

    # MLflow setup
    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        mlflow.set_experiment("clip_guidance")

    final_img, history, snapshots, w = gen.optimize(
        args.prompt, num_steps=args.steps, lr=args.lr, seed=args.seed,
    )

    img_path = save_image(final_img, args.prompt, args.output_dir)

    if not args.no_mlflow:
        import mlflow
        with mlflow.start_run(run_name=args.prompt[:30]):
            mlflow.log_params({
                "prompt": args.prompt,
                "num_steps": args.steps,
                "lr": args.lr,
                "seed": args.seed,
                "fp16": USE_FP16,
            })
            # Log per-step CLIP similarity
            for i, sim in enumerate(history):
                if i % 10 == 0:
                    mlflow.log_metric("clip_similarity_step", sim, step=i)

            # Compute and log all metrics
            with torch.no_grad():
                final_tensor = generate_from_w(gen.G, w)
                initial_tensor = generate_from_w(gen.G, gen.w_mean)
            text_features = gen.encode_text(args.prompt)

            all_metrics = compute_all_metrics(
                final_image_tensor=final_tensor,
                initial_image_tensor=initial_tensor,
                text_features=text_features,
                clip_model=gen.clip_model,
                w_optimized=w,
                w_mean=gen.w_mean,
                final_image_np=final_img,
            )
            mlflow.log_metrics(all_metrics)

            mlflow.log_artifact(img_path)


if __name__ == "__main__":
    main()
