"""FFHQ face alignment using MTCNN landmarks.

StyleGAN2 FFHQ expects faces cropped/aligned to a specific template.
Unaligned input → garbage inversion. This module fixes that.
"""

import numpy as np
from PIL import Image

_mtcnn = None


def _get_mtcnn(device="cuda"):
    global _mtcnn
    if _mtcnn is None:
        from facenet_pytorch import MTCNN
        # keep_all=False: return only the most prominent face
        _mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
        print("MTCNN face detector loaded")
    return _mtcnn


def align_face_ffhq(image_path_or_pil, output_size=1024, device="cuda"):
    """Align a face image to the FFHQ template.

    Uses MTCNN to detect 5 landmarks (eyes, nose, mouth corners), then computes
    an affine crop that matches how StyleGAN2-FFHQ's training data was prepared.

    Args:
        image_path_or_pil: path to image OR PIL Image
        output_size: final image size (1024 for StyleGAN2-FFHQ)
        device: torch device for MTCNN
    Returns:
        PIL.Image of the aligned face, or None if no face detected
    """
    mtcnn = _get_mtcnn(device)

    if isinstance(image_path_or_pil, str):
        img = Image.open(image_path_or_pil).convert("RGB")
    else:
        img = image_path_or_pil.convert("RGB")

    # Detect face + landmarks
    boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
    if landmarks is None or len(landmarks) == 0:
        print("WARNING: No face detected, skipping alignment")
        return img.resize((output_size, output_size), Image.LANCZOS)

    lm = landmarks[0]  # [5, 2]: left eye, right eye, nose, left mouth, right mouth

    # FFHQ alignment (from official FFHQ prep script, NVIDIA's recipe)
    eye_left = lm[0]
    eye_right = lm[1]
    mouth_left = lm[3]
    mouth_right = lm[4]

    eye_avg = (eye_left + eye_right) / 2
    eye_to_eye = eye_right - eye_left
    mouth_avg = (mouth_left + mouth_right) / 2
    eye_to_mouth = mouth_avg - eye_avg

    # Compute the crop rectangle
    x = eye_to_eye.copy()
    x[1] = -x[1]  # flip y since FFHQ uses inverted y
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.array([-x[1], x[0]])  # perpendicular
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

    # Apply quad transform
    aligned = img.transform(
        (output_size, output_size),
        Image.QUAD,
        quad.flatten().tolist(),
        Image.BILINEAR,
    )
    return aligned
