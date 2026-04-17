# StyleGAN2 + CLIP-Guided Face Generation

StyleGAN2-ADA with CLIP guidance for text-driven face generation and identity-preserving editing.

## Commands

### Generate a face from text prompt
```bash
python -m stylegan.clip_guidance --prompt "a young woman with red hair" --steps 300 --lr 0.02
```

### Encode a celebrity photo into the DB
```bash
python -m stylegan.encoder --image inputs/brad-pitt.webp --name brad_pitt --save --steps 1000
```

### Edit a celebrity with identity preservation
```bash
python -m stylegan.clip_guidance --celebrity brad_pitt --prompt "with a red beard" --steps 300 --lr 0.02
```

### Edit directly from a photo (no DB)
```bash
python -m stylegan.clip_guidance --image inputs/face.jpg --prompt "smiling" --steps 300 --lr 0.02 --lambda-id 0.5
```

### Edit with Gemini intelligence (prompt enrichment + judge + dynamic adjustment + report)
```bash
python -m stylegan.clip_guidance --celebrity brad_pitt --prompt "with a red beard" --steps 300 --lr 0.02 --gemini
```

### Gemini judge runs every N steps (default 100)
```bash
python -m stylegan.clip_guidance --celebrity brad_pitt --prompt "with a red beard" --steps 300 --gemini --judge-every 50
```

### List saved celebrities
```bash
python -m stylegan.encoder --list
```

### MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Files

| File | Description |
|------|-------------|
| `config.py` | All hyperparameters and paths |
| `architecture.py` | StyleGAN2 loading, patching, W/W+ generation |
| `clip_guidance.py` | CLIP-guided optimization (free gen + identity edit) |
| `encoder.py` | Face photo inversion to W+ space + celebrity DB |
| `identity.py` | FaceNet identity loss for face preservation |
| `metrics.py` | CLIP similarity, LPIPS, inception, sharpness, etc. |
| `gemini.py` | Gemini LLM: prompt enrichment, judge, dynamic loss, final report |
