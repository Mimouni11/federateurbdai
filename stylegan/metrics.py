"""Metrics for StyleGAN2 + CLIP-guided image generation."""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models

from stylegan.config import CLIP_NORMALIZE_MEAN, CLIP_NORMALIZE_STD


# ── CLIP Similarity (text-image alignment) ──────────────────────────────

def clip_similarity(image_tensor, text_features, clip_model):
    """Cosine similarity between generated image and text prompt in CLIP space.

    Args:
        image_tensor: [1, 3, H, W] in [0, 1]
        text_features: precomputed CLIP text features
        clip_model: loaded CLIP model
    Returns:
        float similarity score
    """
    normalize = transforms.Normalize(mean=CLIP_NORMALIZE_MEAN, std=CLIP_NORMALIZE_STD)
    clip_input_size = clip_model.visual.input_resolution
    img_resized = F.interpolate(image_tensor.float(), size=(clip_input_size, clip_input_size), mode="bilinear", align_corners=False)
    img_resized = normalize(img_resized)
    dtype = next(clip_model.parameters()).dtype

    with torch.no_grad():
        image_features = clip_model.encode_image(img_resized.to(dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return (image_features * text_features).sum(dim=-1).item()


# ── LPIPS (perceptual distance from mean face) ──────────────────────────

_lpips_net = None


def _get_lpips():
    global _lpips_net
    if _lpips_net is None:
        import lpips
        _lpips_net = lpips.LPIPS(net="vgg", verbose=False)
        _lpips_net.eval()
    return _lpips_net


def lpips_distance(image_tensor, reference_tensor):
    """LPIPS perceptual distance between two images.

    Lower = more perceptually similar to reference.
    Higher = more diverse/different from starting point.

    Args:
        image_tensor: [1, 3, H, W] in [0, 1] (generated image)
        reference_tensor: [1, 3, H, W] in [0, 1] (mean face or initial image)
    Returns:
        float LPIPS distance
    """
    net = _get_lpips()
    device = image_tensor.device
    net = net.to(device)

    # LPIPS expects [-1, 1]
    img = image_tensor.float() * 2 - 1
    ref = reference_tensor.float() * 2 - 1

    with torch.no_grad():
        dist = net(img, ref)
    return dist.item()


# ── Inception Score components (image quality/sharpness) ────────────────

_inception_model = None


def _get_inception():
    global _inception_model
    if _inception_model is None:
        _inception_model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        _inception_model.eval()
    return _inception_model


def inception_confidence(image_tensor):
    """Softmax confidence from InceptionV3 as a proxy for image quality.

    Higher confidence = sharper, more recognizable image.
    Returns max class probability and entropy of the distribution.

    Args:
        image_tensor: [1, 3, H, W] in [0, 1]
    Returns:
        dict with 'inception_confidence' and 'inception_entropy'
    """
    model = _get_inception()
    device = image_tensor.device
    model = model.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img = F.interpolate(image_tensor.float(), size=(299, 299), mode="bilinear", align_corners=False)
    img = normalize(img)

    with torch.no_grad():
        logits = model(img)
        probs = F.softmax(logits, dim=-1)
        max_conf = probs.max().item()
        entropy = -(probs * probs.log()).sum().item()

    return {"inception_confidence": max_conf, "inception_entropy": entropy}


# ── Face Detection Confidence (is it actually a face?) ──────────────────

def face_sharpness(image_np):
    """Laplacian variance as a sharpness/quality proxy (no extra deps).

    Higher = sharper image. Blurry/collapsed outputs score low.

    Args:
        image_np: H x W x 3 numpy array in [0, 1]
    Returns:
        float sharpness score
    """
    gray = np.dot(image_np[..., :3], [0.2989, 0.5870, 0.1140])
    gray_uint8 = (np.clip(gray, 0, 1) * 255).astype(np.uint8).astype(np.float64)

    # Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    from scipy.ndimage import convolve
    laplacian = convolve(gray_uint8, kernel)
    return float(laplacian.var())


# ── W-space metrics (latent health) ─────────────────────────────────────

def w_space_distance(w_optimized, w_mean):
    """L2 distance from optimized W to mean W.

    Too high = out-of-distribution (artifacts likely).
    Too low = barely changed from mean face.

    Args:
        w_optimized: optimized W tensor
        w_mean: mean W tensor
    Returns:
        float L2 distance
    """
    return (w_optimized.float() - w_mean.float()).norm().item()


# ── Pixel-level diversity (mode collapse check) ────────────────────────

def pixel_std(image_tensor):
    """Standard deviation across pixels. Near-zero = mode collapse / solid color.

    Args:
        image_tensor: [1, 3, H, W] in [0, 1]
    Returns:
        float mean pixel std across channels
    """
    return image_tensor.float().std().item()


# ── Aggregate all metrics ──────────────────────────────────────────────

def compute_all_metrics(
    final_image_tensor,
    initial_image_tensor,
    text_features,
    clip_model,
    w_optimized,
    w_mean,
    final_image_np,
):
    """Compute all metrics for a generation run.

    Args:
        final_image_tensor: [1, 3, H, W] in [0, 1] — final generated image
        initial_image_tensor: [1, 3, H, W] in [0, 1] — image before optimization
        text_features: precomputed CLIP text features
        clip_model: loaded CLIP model
        w_optimized: final W latent
        w_mean: mean W latent
        final_image_np: H x W x 3 numpy in [0, 1]
    Returns:
        dict of all metric name -> value
    """
    metrics = {}

    # CLIP alignment
    metrics["clip_similarity"] = clip_similarity(final_image_tensor, text_features, clip_model)

    # Perceptual distance from starting point
    metrics["lpips_from_mean"] = lpips_distance(final_image_tensor, initial_image_tensor)

    # Image quality
    inception = inception_confidence(final_image_tensor)
    metrics.update(inception)

    # Sharpness
    metrics["sharpness"] = face_sharpness(final_image_np)

    # Latent health
    metrics["w_distance_from_mean"] = w_space_distance(w_optimized, w_mean)

    # Mode collapse check
    metrics["pixel_diversity"] = pixel_std(final_image_tensor)

    return metrics
