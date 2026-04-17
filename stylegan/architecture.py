"""StyleGAN2 wrapper: clone, patch, load, generate."""

import sys
import subprocess
import pickle

import torch

from stylegan.config import STYLEGAN_DIR, CHECKPOINT_PATH, USE_FP16


def ensure_stylegan2():
    """Clone stylegan2-ada-pytorch to vendor/ and patch for modern PyTorch."""
    if not STYLEGAN_DIR.exists():
        print(f"Cloning stylegan2-ada-pytorch to {STYLEGAN_DIR}...")
        subprocess.run(
            ["git", "clone", "https://github.com/NVlabs/stylegan2-ada-pytorch.git", str(STYLEGAN_DIR)],
            check=True,
        )

    # Patch misc.py — IterableDataset.__init__ signature changed
    misc_path = STYLEGAN_DIR / "torch_utils" / "misc.py"
    text = misc_path.read_text()
    if "super().__init__(dataset)" in text:
        misc_path.write_text(text.replace("super().__init__(dataset)", "super().__init__()"))
        print("Patched misc.py")

    # Patch grid_sample_gradfix — double-backward removed in newer PyTorch
    grid_path = STYLEGAN_DIR / "torch_utils" / "ops" / "grid_sample_gradfix.py"
    grid_path.write_text(
        "import torch\n\nenabled = False\n\n"
        "def grid_sample(input, grid, **kwargs):\n"
        "    return torch.nn.functional.grid_sample(input, grid, **kwargs)\n"
    )

    # Patch upfirdn2d + bias_act — force pure-Python fallback (CUDA ops won't compile on Windows)
    for op_name in ("upfirdn2d", "bias_act"):
        op_path = STYLEGAN_DIR / "torch_utils" / "ops" / f"{op_name}.py"
        text = op_path.read_text()
        if "return False  # patched" not in text:
            text = text.replace(
                "def _init():",
                "def _init():\n    return False  # patched",
            )
            op_path.write_text(text)
            print(f"Patched {op_name}.py")

    # Add to sys.path so pickle can find training.networks
    stylegan_str = str(STYLEGAN_DIR)
    if stylegan_str not in sys.path:
        sys.path.insert(0, stylegan_str)


def load_generator(checkpoint=None, device="cuda", fp16=None):
    """Load StyleGAN2 G_ema from a .pkl checkpoint."""
    ensure_stylegan2()

    checkpoint = checkpoint or CHECKPOINT_PATH
    fp16 = fp16 if fp16 is not None else USE_FP16

    with open(checkpoint, "rb") as f:
        G = pickle.load(f)["G_ema"].to(device).eval()

    if fp16:
        G = G.half()

    print(f"Generator loaded: {G.img_resolution}x{G.img_resolution} ({'fp16' if fp16 else 'fp32'})")
    return G


def generate_from_w(G, w):
    """Generate image from a W or W+ vector. Returns [1, 3, H, W] in [0, 1]."""
    if w.ndim == 2:
        # W-space: [1, 512] -> broadcast to all layers
        ws = w.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
    else:
        # W+-space: [1, num_ws, 512] -> use per-layer latents directly
        ws = w
    img = G.synthesis(ws.to(dtype=next(G.parameters()).dtype))
    return (img.clamp(-1, 1) + 1) / 2


def compute_mean_w(G, num_samples=1000, device="cuda"):
    """Compute mean W latent from random Z samples."""
    dtype = next(G.parameters()).dtype
    with torch.no_grad():
        z = torch.randn(num_samples, G.z_dim, device=device, dtype=dtype)
        label = torch.zeros([num_samples, G.c_dim], device=device) if G.c_dim > 0 else None
        w_samples = G.mapping(z, label)
    return w_samples[:, 0, :].mean(dim=0, keepdim=True).float()