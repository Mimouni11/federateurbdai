"""Identity loss for face preservation during CLIP-guided editing.

Uses FaceNet (InceptionResnetV1) from facenet-pytorch — pure Python, no C++ build needed.
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


_facenet_model = None


def _get_facenet(device="cuda"):
    """Load pretrained FaceNet model."""
    global _facenet_model
    if _facenet_model is None:
        from facenet_pytorch import InceptionResnetV1
        _facenet_model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
        for p in _facenet_model.parameters():
            p.requires_grad = False
        print("FaceNet identity model loaded")
    return _facenet_model


_facenet_normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


def extract_identity_embedding(image_tensor, device="cuda"):
    """Extract FaceNet identity embedding from a face image tensor.

    Args:
        image_tensor: [1, 3, H, W] in [0, 1] (generated image)
        device: torch device
    Returns:
        normalized embedding tensor [1, 512]
    """
    model = _get_facenet(device)
    # FaceNet expects 160x160 images normalized to [-1, 1]
    img = F.interpolate(image_tensor.float(), size=(160, 160), mode="bilinear", align_corners=False)
    img = _facenet_normalize(img).to(device)
    with torch.no_grad():
        embedding = model(img)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    return embedding


def extract_identity_from_numpy(image_np, device="cuda"):
    """Extract identity embedding from a numpy image.

    Args:
        image_np: H x W x 3 numpy array in [0, 1]
        device: torch device
    Returns:
        normalized embedding tensor [1, 512]
    """
    tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)
    return extract_identity_embedding(tensor, device)


def identity_loss(current_image_tensor, reference_embedding, device="cuda"):
    """Compute identity loss between current generated image and reference.

    Returns 1 - cosine_similarity (0 = same person, 2 = opposite).
    This is differentiable — gradients flow through the generated image.

    Args:
        current_image_tensor: [1, 3, H, W] in [0, 1] (must have grad)
        reference_embedding: [1, 512] normalized FaceNet embedding (detached)
        device: torch device
    Returns:
        scalar loss tensor
    """
    model = _get_facenet(device)
    # FaceNet forward (with gradients for backprop through the generator)
    img = F.interpolate(current_image_tensor.float(), size=(160, 160), mode="bilinear", align_corners=False)
    img = _facenet_normalize(img).to(device)
    current_embedding = model(img)
    current_embedding = current_embedding / current_embedding.norm(dim=-1, keepdim=True)

    cosine_sim = (current_embedding * reference_embedding.detach()).sum(dim=-1)
    return (1.0 - cosine_sim).mean()
