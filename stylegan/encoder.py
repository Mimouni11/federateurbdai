"""Face encoder: real photo → W-space latent + celebrity database.

Uses optimization-based inversion (no external encoder model needed).
Projects a real face photo into StyleGAN2's W-space so it can be edited with CLIP.

Usage:
    # Encode a celebrity photo and save to DB
    python -m models.encoder --image path/to/brad_pitt.jpg --name "brad_pitt" --save

    # Encode and immediately edit with CLIP
    python -m models.encoder --image path/to/brad_pitt.jpg --prompt "with a red beard" --steps 200
"""

import argparse
import os

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

from stylegan.config import (
    CLIP_MODEL, CLIP_NORMALIZE_MEAN, CLIP_NORMALIZE_STD,
    NUM_STEPS, LEARNING_RATE, W_MEAN_SAMPLES,
    OUTPUT_DIR, MLFLOW_DB_URI, CELEBRITY_DB_DIR,
)
from stylegan.architecture import load_generator, generate_from_w, compute_mean_w

# VGG16 for perceptual loss during inversion
_vgg = None


def _get_vgg(device="cuda"):
    global _vgg
    if _vgg is None:
        from torchvision.models import vgg16, VGG16_Weights
        _vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:23].to(device).eval()
        for p in _vgg.parameters():
            p.requires_grad = False
        print("VGG16 perceptual model loaded")
    return _vgg


def perceptual_loss(img1, img2, device="cuda"):
    """LPIPS-style perceptual loss using VGG16 features."""
    vgg = _get_vgg(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    x1 = normalize(F.interpolate(img1.float(), size=(256, 256), mode="bilinear", align_corners=False))
    x2 = normalize(F.interpolate(img2.float(), size=(256, 256), mode="bilinear", align_corners=False))
    feat1 = vgg(x1)
    feat2 = vgg(x2)
    return F.mse_loss(feat1, feat2)


def load_target_image(image_path, resolution=1024, device="cuda", align=True):
    """Load, align (FFHQ template), and preprocess a target face image."""
    if align:
        from stylegan.alignment import align_face_ffhq
        img = align_face_ffhq(image_path, output_size=resolution, device=device)
    else:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.LANCZOS)
    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return tensor


class FaceEncoder:
    """Projects real face photos into StyleGAN2 W+ space via optimization.

    W+ gives each StyleGAN layer its own latent vector, allowing much more
    faithful reconstructions than single-W inversion.
    """

    def __init__(self, G=None, device="cuda"):
        self.device = device
        if G is not None:
            self.G = G
        else:
            self.G = load_generator(device=device)
        self.w_mean = compute_mean_w(self.G, device=device)
        self.num_ws = self.G.mapping.num_ws

    def encode(self, image_path, num_steps=1000, lr=0.01, lambda_id=1.0):
        """Invert a face image into W+ space with identity preservation.

        Pipeline: align face to FFHQ template → optimize W+ using pixel + VGG perceptual
        + FaceNet identity losses. Identity loss is the key addition — forces the W+ to
        preserve the actual person, not just pixel similarity.

        Args:
            image_path: path to the face photo
            num_steps: optimization steps (1000 recommended for W+)
            lr: learning rate for inversion
            lambda_id: weight for identity loss during inversion
        Returns:
            (w_plus_latent [1, num_ws, 512], final_reconstruction_np)
        """
        from stylegan.identity import identity_loss, extract_identity_embedding

        target = load_target_image(image_path, self.G.img_resolution, self.device, align=True)
        print(f'Encoding face from: {image_path} ({num_steps} steps, W+ space, aligned)')

        # Extract identity embedding from aligned target
        ref_embedding = extract_identity_embedding(target, self.device)

        # Init W+ from mean W broadcast to all layers [1, num_ws, 512]
        w_plus = self.w_mean.unsqueeze(1).repeat(1, self.num_ws, 1).clone().detach().requires_grad_(True)
        w_mean_plus = w_plus.clone().detach()

        optimizer = torch.optim.Adam([w_plus], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)

        for step in range(num_steps):
            optimizer.zero_grad()

            img = generate_from_w(self.G, w_plus)

            l_pixel = F.mse_loss(img, target)
            l_percep = perceptual_loss(img, target, self.device)
            l_id = identity_loss(img, ref_embedding, self.device)
            l_reg = 5e-4 * (w_plus - w_mean_plus).pow(2).mean()

            loss = l_pixel + 0.8 * l_percep + lambda_id * l_id + l_reg
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 200 == 0 or step == num_steps - 1:
                print(f"  step {step:4d}/{num_steps}  pixel: {l_pixel.item():.4f}  "
                      f"percep: {l_percep.item():.4f}  id: {l_id.item():.4f}  reg: {l_reg.item():.6f}")

        with torch.no_grad():
            final = generate_from_w(self.G, w_plus)
            final_np = final[0].float().permute(1, 2, 0).cpu().numpy()

        return w_plus.detach(), final_np


# ── Celebrity Database ──────────────────────────────────────────────────

def save_celebrity(name, w_latent, reference_image_path=None):
    """Save a celebrity's W vector to the database."""
    os.makedirs(CELEBRITY_DB_DIR, exist_ok=True)
    save_path = CELEBRITY_DB_DIR / f"{name}.pt"
    data = {"w": w_latent.cpu(), "name": name}
    if reference_image_path:
        data["reference_path"] = str(reference_image_path)
    torch.save(data, save_path)
    print(f"Saved celebrity W vector: {save_path}")


def load_celebrity(name):
    """Load a celebrity's W vector and reference image path from the database."""
    load_path = CELEBRITY_DB_DIR / f"{name}.pt"
    if not load_path.exists():
        available = list_celebrities()
        raise FileNotFoundError(
            f"Celebrity '{name}' not found. Available: {available}"
        )
    data = torch.load(load_path, weights_only=False)
    print(f"Loaded celebrity: {name}")
    return data["w"], data.get("reference_path")


def list_celebrities():
    """List all celebrities in the database."""
    if not CELEBRITY_DB_DIR.exists():
        return []
    return [f.stem for f in CELEBRITY_DB_DIR.glob("*.pt")]


def main():
    parser = argparse.ArgumentParser(description="Face encoder + celebrity DB")
    parser.add_argument("--image", type=str, help="Path to face photo to encode")
    parser.add_argument("--name", type=str, help="Celebrity name for DB storage")
    parser.add_argument("--save", action="store_true", help="Save W vector to celebrity DB")
    parser.add_argument("--list", action="store_true", help="List saved celebrities")
    parser.add_argument("--steps", type=int, default=500, help="Inversion steps")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if args.list:
        celebs = list_celebrities()
        print(f"Saved celebrities: {celebs if celebs else '(none)'}")
        return

    if not args.image:
        parser.error("--image is required for encoding")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = FaceEncoder(device=device)
    w, reconstruction = encoder.encode(args.image, num_steps=args.steps, lr=args.lr)

    # Save reconstruction
    os.makedirs(args.output_dir, exist_ok=True)
    name = args.name or os.path.splitext(os.path.basename(args.image))[0]
    recon_path = os.path.join(args.output_dir, f"{name}_reconstruction.png")
    img = Image.fromarray((np.clip(reconstruction, 0, 1) * 255).astype(np.uint8))
    img.save(recon_path)
    print(f"Reconstruction saved: {recon_path}")

    # Save to celebrity DB
    if args.save:
        if not args.name:
            parser.error("--name is required with --save")
        save_celebrity(args.name, w, args.image)


if __name__ == "__main__":
    main()
