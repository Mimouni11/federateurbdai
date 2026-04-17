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

from stylegan.config import (
    CLIP_MODEL, CLIP_NORMALIZE_MEAN, CLIP_NORMALIZE_STD,
    NUM_STEPS, LEARNING_RATE, W_MEAN_SAMPLES,
    OUTPUT_DIR, MLFLOW_DB_URI, USE_FP16,
)
from stylegan.architecture import load_generator, generate_from_w, compute_mean_w
from stylegan.encoder import load_celebrity, FaceEncoder
from stylegan.identity import extract_identity_from_numpy, identity_loss
from stylegan.metrics import compute_all_metrics


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
        # image: [1, 3, H, W] in [0, 1]
        clip_input_size = self.clip_model.visual.input_resolution
        image_resized = F.interpolate(image.float(), size=(clip_input_size, clip_input_size), mode="bilinear", align_corners=False)
        image_resized = clip_normalize(image_resized)
        # Match CLIP model dtype (fp16 for ViT-B/32, fp32 for ViT-L/14)
        dtype = next(self.clip_model.parameters()).dtype
        image_features = self.clip_model.encode_image(image_resized.to(dtype))
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

    def optimize_with_identity(self, prompt, w_start, reference_image_np,
                                  num_steps=NUM_STEPS, lr=LEARNING_RATE,
                                  lambda_id=0.5, seed=42, use_gemini=False,
                                  judge_every=100):
        """Optimize W to match prompt while preserving identity.

        Args:
            prompt: text description of desired edit
            w_start: starting W vector (from encoder or celebrity DB)
            reference_image_np: original face image as numpy array [H, W, 3] in [0, 1]
            num_steps: optimization steps
            lr: learning rate
            lambda_id: weight for identity loss (higher = more identity preservation)
            seed: random seed
            use_gemini: enable Gemini judge + dynamic loss adjustment
            judge_every: run Gemini judge every N steps
        Returns:
            (final_image_np, history, snapshots, w, judge_history)
        """
        print(f'Editing: "{prompt}" ({num_steps} steps, lr={lr}, lambda_id={lambda_id})')

        text_features = self.encode_text(prompt)

        # Get reference identity embedding
        ref_embedding = extract_identity_from_numpy(reference_image_np, self.device)

        # Start from the provided W (identity-locked starting point)
        torch.manual_seed(seed)
        w = w_start.clone().to(self.device).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)

        history = {"clip_similarity": [], "identity_loss": [], "total_loss": []}
        snapshots = []
        judge_history = []

        for step in range(num_steps):
            optimizer.zero_grad()

            img = generate_from_w(self.G, w)

            # CLIP loss (text alignment)
            l_clip = self.clip_loss(img, text_features)

            # Identity loss (face preservation) — differentiable
            id_loss = identity_loss(img, ref_embedding, self.device)

            # Combined loss
            loss = l_clip + lambda_id * id_loss
            loss.backward()
            optimizer.step()

            clip_sim = -l_clip.item()
            history["clip_similarity"].append(clip_sim)
            history["identity_loss"].append(id_loss.item())
            history["total_loss"].append(loss.item())

            if step % 50 == 0 or step == num_steps - 1:
                print(f"  step {step:3d}/{num_steps}  clip_sim: {clip_sim:.4f}  "
                      f"id_loss: {id_loss:.4f}  total: {loss.item():.4f}")
                with torch.no_grad():
                    snap = generate_from_w(self.G, w)
                    snapshots.append(snap[0].float().permute(1, 2, 0).cpu().numpy())

            # Gemini judge + dynamic adjustment
            if use_gemini and step > 0 and step % judge_every == 0:
                from stylegan.gemini import judge_identity, adjust_loss_weights
                with torch.no_grad():
                    current_np = generate_from_w(self.G, w)[0].float().permute(1, 2, 0).cpu().numpy()
                scores = judge_identity(current_np, reference_image_np, step=step)
                judge_history.append({"step": step, **scores})

                adjustments = adjust_loss_weights(scores, lambda_id, lr)
                lambda_id = adjustments["lambda_id"]
                new_lr = adjustments["lr"]
                if new_lr != lr:
                    lr = new_lr
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

        with torch.no_grad():
            final = generate_from_w(self.G, w)
            final_np = final[0].float().permute(1, 2, 0).cpu().numpy()

        return final_np, history, snapshots, w.detach(), judge_history

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
    # Identity-preserving edit mode
    parser.add_argument("--celebrity", type=str, help="Celebrity name from DB (use --list to see)")
    parser.add_argument("--image", type=str, help="Path to face photo to edit")
    parser.add_argument("--lambda-id", type=float, default=0.5, help="Identity loss weight")
    parser.add_argument("--list", action="store_true", help="List saved celebrities")
    parser.add_argument("--encode-steps", type=int, default=500, help="Steps for photo inversion")
    # Gemini features
    parser.add_argument("--gemini", action="store_true", help="Enable Gemini prompt enrichment + judge + report")
    parser.add_argument("--judge-every", type=int, default=100, help="Run Gemini judge every N steps")
    args = parser.parse_args()

    if args.list:
        from stylegan.encoder import list_celebrities
        celebs = list_celebrities()
        print(f"Saved celebrities: {celebs if celebs else '(none)'}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: No GPU detected, this will be very slow")

    gen = CLIPGuidedGenerator(device=device)

    # MLflow setup
    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_DB_URI)
        mlflow.set_experiment("clip_guidance")

    # Determine mode: identity edit or free generation
    use_identity = args.celebrity or args.image
    w_start = None
    reference_np = None
    original_image_path = args.image

    if args.celebrity:
        # Load W + original image path from celebrity DB
        w_start, original_image_path = load_celebrity(args.celebrity)
        w_start = w_start.to(device)
        with torch.no_grad():
            ref_img = generate_from_w(gen.G, w_start)
            reference_np = ref_img[0].float().permute(1, 2, 0).cpu().numpy()
        print(f"Loaded celebrity: {args.celebrity}")

    elif args.image:
        # Encode photo on the fly
        encoder = FaceEncoder(G=gen.G, device=device)
        w_start, reference_np = encoder.encode(args.image, num_steps=args.encode_steps)

    # Gemini prompt enrichment (Feature 1) — always uses the ORIGINAL photo
    actual_prompt = args.prompt
    if args.gemini and use_identity:
        from stylegan.gemini import enrich_prompt
        actual_prompt = enrich_prompt(None, args.prompt,
                                      image_path=original_image_path)

    judge_history = []
    if use_identity:
        final_img, history, snapshots, w, judge_history = gen.optimize_with_identity(
            actual_prompt, w_start=w_start, reference_image_np=reference_np,
            num_steps=args.steps, lr=args.lr, lambda_id=args.lambda_id,
            seed=args.seed, use_gemini=args.gemini, judge_every=args.judge_every,
        )
    else:
        final_img, history, snapshots, w = gen.optimize(
            actual_prompt, num_steps=args.steps, lr=args.lr, seed=args.seed,
        )

    img_path = save_image(final_img, args.prompt, args.output_dir)

    if not args.no_mlflow:
        import mlflow
        run_name = args.prompt[:30]
        if args.celebrity:
            run_name = f"{args.celebrity}: {run_name}"

        with mlflow.start_run(run_name=run_name):
            params = {
                "prompt": args.prompt,
                "num_steps": args.steps,
                "lr": args.lr,
                "seed": args.seed,
                "fp16": USE_FP16,
                "mode": "identity_edit" if use_identity else "free_generation",
                "gemini_enabled": args.gemini,
            }
            if args.gemini and actual_prompt != args.prompt:
                params["enriched_prompt"] = actual_prompt
            if args.celebrity:
                params["celebrity"] = args.celebrity
            if args.image:
                params["source_image"] = os.path.basename(args.image)
            if use_identity:
                params["lambda_id"] = args.lambda_id
            mlflow.log_params(params)

            # Log per-step metrics
            if use_identity:
                for i, sim in enumerate(history["clip_similarity"]):
                    if i % 10 == 0:
                        mlflow.log_metric("clip_similarity_step", sim, step=i)
                        mlflow.log_metric("identity_loss_step", history["identity_loss"][i], step=i)
            else:
                for i, sim in enumerate(history):
                    if i % 10 == 0:
                        mlflow.log_metric("clip_similarity_step", sim, step=i)

            # Compute and log all metrics
            with torch.no_grad():
                final_tensor = generate_from_w(gen.G, w)
                initial_tensor = generate_from_w(gen.G, w_start if w_start is not None else gen.w_mean)
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

            # Gemini judge history + final evaluation (Features 2, 4)
            if args.gemini and use_identity:
                from stylegan.gemini import final_evaluation
                import json

                # Log judge scores per step
                for jh in judge_history:
                    step = jh["step"]
                    if "overall" in jh:
                        mlflow.log_metric("judge_overall", jh["overall"], step=step)
                    if "eyes" in jh:
                        mlflow.log_metric("judge_eyes", jh["eyes"], step=step)

                # Final evaluation report
                report = final_evaluation(final_img, reference_np, args.prompt, all_metrics)
                if "gemini_final_score" in report:
                    mlflow.log_metric("gemini_final_score", report["gemini_final_score"])
                mlflow.log_text(json.dumps(report, indent=2), "gemini_report.json")

            mlflow.log_artifact(img_path)
            if args.image:
                mlflow.log_artifact(args.image)


if __name__ == "__main__":
    main()
