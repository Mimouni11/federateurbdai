# SDXL on Google Colab — Phase 1

## Context

StyleGAN2 pipeline hit a quality ceiling for face generation/editing. Next step of the comparison study is to prove SDXL (diffusion-based) produces dramatically better outputs. Phase 1 is **entirely on Google Colab**, no local code, no FastAPI, no ngrok, no IP-Adapter, no celebrity manipulation. Just: SDXL loads, generates good images from text prompts, metrics are logged. FastAPI + ngrok + local client come in Phase 2. Celebrity manipulation + IP-Adapter come in Phase 3. Gemini integration comes last.

## Goal

A single self-contained Colab notebook (`SDXL/sdxl_colab.ipynb`) that:
1. Loads SDXL Base (+ Refiner if VRAM permits on T4)
2. Generates high-quality images from text prompts
3. Computes per-image metrics (CLIP similarity, sharpness, diversity)
4. Logs everything to MLflow (running inside the Colab VM)
5. Saves outputs with descriptive filenames to `/content/outputs/`

## File to create

`SDXL/sdxl_colab.ipynb` — new Colab notebook, fully standalone (no imports from the `stylegan/` package, since Colab won't have access to the local codebase)

## Notebook structure (cell by cell)

### Cell 1 — Environment setup
```
!pip install -q diffusers==0.30.0 transformers accelerate safetensors \
                invisible_watermark mlflow ftfy
!pip install -q git+https://github.com/openai/CLIP.git
```
Mount Drive only if user wants persistent storage (optional).

### Cell 2 — Load SDXL Base
```python
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
).to("cuda")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()  # fallback to SDPA if unavailable
```
**Decision point:** Start with Base only. On T4 (16GB) the Refiner doubles VRAM + time; add it in a second pass only if Base output needs more polish.

### Cell 3 — Load CLIP for metrics
```python
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")
```
Same model as StyleGAN pipeline → metrics comparable.

### Cell 4 — Metrics module (self-contained)
Define inline:
- `clip_similarity(pil_image, prompt)` — cosine similarity between image and text in CLIP space
- `sharpness(pil_image)` — Laplacian variance on greyscale
- `pixel_diversity(pil_image)` — channel-wise std dev
- `generation_time_s` — wall clock from time.perf_counter
- `vram_peak_gb` — from `torch.cuda.max_memory_allocated`

Signatures mirror the stylegan `metrics.py` so numbers are directly comparable later.

### Cell 5 — MLflow setup
```python
import mlflow
mlflow.set_tracking_uri("sqlite:////content/mlflow.db")
mlflow.set_experiment("sdxl_phase1")
```
Same SQLite backend pattern used locally → can merge runs later by copying `/content/mlflow.db` out of Colab.

### Cell 6 — Generation function
```python
def generate(prompt, negative_prompt="", steps=30, guidance=7.5, seed=42,
             width=1024, height=1024):
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator("cuda").manual_seed(seed)
    start = time.perf_counter()
    image = pipe(prompt, negative_prompt=negative_prompt,
                 num_inference_steps=steps, guidance_scale=guidance,
                 generator=gen, width=width, height=height).images[0]
    elapsed = time.perf_counter() - start
    return image, elapsed
```

### Cell 7 — Run prompts + log
Start with **3 portrait prompts** matching what StyleGAN was tested on (for direct comparison):
- `"portrait of a young woman with red hair, photorealistic"`
- `"portrait of an elderly man with white beard and long hair, photorealistic"`
- `"portrait of a middle-aged man with blonde hair and blue eyes, photorealistic"`

For each: start MLflow run, log params (prompt, steps, guidance, seed, model=SDXL-Base), generate image, compute metrics, log metrics, save artifact, log artifact.

### Cell 8 — Quick parameter sweep (optional)
Same prompt with different `guidance_scale` ∈ {5, 7.5, 10, 12} and `steps` ∈ {20, 30, 50}. Produces 12 images → visual grid + metrics table → pick best combo for Phase 2.

### Cell 9 — Display results
Inline grid of all generated images with metrics captions. Use matplotlib or IPython.display.

## Metrics logged per image

| Metric | Source | Comparable to StyleGAN? |
|---|---|---|
| `clip_similarity` | CLIP ViT-B/32 | ✅ same model, same formula |
| `sharpness` | Laplacian variance | ✅ |
| `pixel_diversity` | pixel std | ✅ |
| `generation_time_s` | wall clock | ➖ not measured in StyleGAN runs |
| `vram_peak_gb` | torch.cuda | ➖ |

## Verification

1. Open the notebook on Colab with T4 runtime (free tier)
2. Run Cell 1 → Cell 7 top to bottom, no errors
3. Cell 7 produces 3 images saved to `/content/outputs/`
4. Visually: quality must be clearly higher than StyleGAN outputs (sharper, no plastic smoothing, realistic skin)
5. MLflow UI inside Colab: `!mlflow ui --backend-store-uri sqlite:////content/mlflow.db --port 5000 &` + ngrok/public URL → dashboard shows 3 runs with clip_similarity, sharpness, etc.
6. CLIP similarity should land in the 0.28–0.32 range for portrait prompts (SDXL's typical range; StyleGAN+CLIP hit 0.40–0.49 but that's because it optimizes directly against CLIP — not a fair direct comparison)

## Explicitly out of scope for Phase 1

- FastAPI endpoint / ngrok tunnel
- Local `sdxl_client.py` calling Colab
- IP-Adapter identity conditioning
- Celebrity photo input
- Gemini enrichment / judge
- Dynamic loss scheduling

These are deliberate phase boundaries — get Phase 1 (model works, images are good, metrics logged) fully solid first.
