# FederateurBDAI — Project Plan
## StyleGAN2 vs SDXL: A Celebrity Face Generation & Manipulation Study

---

## Project Goal

Prove through empirical comparison that StyleGAN2 is architecturally limited for identity-specific face generation, and that a diffusion-based pipeline (SDXL + IP-Adapter) solves the core problems. Gemini acts as an objective judge for both pipelines.(*fro claude :gemini is the last step ignore it untill you're told about it  )

---

## Architecture Overview

```
INPUT: Celebrity photo + edit request ("Brad Pitt with red beard" (*for claude: celebrity photo manupulation is the final goal and it's not that critical , good photos generation is the core ))
              │
              ▼
┌─────────────────────────────────────────────────────┐
│              GEMINI ENRICHMENT LAYER                │
│  Photo → detailed facial description → CLIP prompt  │
└─────────────────────────────────────────────────────┘
              │
      ┌───────┴───────┐
      ▼               ▼
┌──────────┐    ┌──────────────┐
│ PIPELINE │    │   PIPELINE   │
│    A     │    │      B       │
│StyleGAN2 │    │SDXL+IP-Adapt │
│ (Local)  │    │  (Colab)     │
└──────────┘    └──────────────┘
      │               │
      └───────┬───────┘
              ▼
┌─────────────────────────────────────────────────────┐
│               GEMINI JUDGE LAYER                    │
│   Generated image vs Reference → structured scores  │
└─────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│                  MLFLOW DASHBOARD                   │
│      Both pipelines, same metrics, same scale       │
└─────────────────────────────────────────────────────┘
```

---

## Pipeline A — StyleGAN2 (Local Machine)

### What it does
- Inverts a celebrity photo into W+ latent space
- Optimizes W+ with CLIP guidance toward the edit prompt
- Uses Gemini-enriched prompt for better CLIP signal
- Gemini judges output every 50 steps → adjusts loss weights dynamically

### Components (already built or in progress)
| Component | Status | Description |
|---|---|---|
| `models/architecture.py` | ✅ Done | StyleGAN2 loader, generator, mean W |
| `models/encoder.py` | ✅ Done | Photo → W+ inversion |
| `models/clip_guidance.py` | ✅ Done | CLIP-guided optimization loop |
| `utils/metrics.py` | ✅ Done | LPIPS, CLIP score, sharpness, pixel std |
| `utils/gemini_enricher.py` | ✅ Done | Photo → enriched CLIP prompt |
| `utils/gemini_judge.py` | ✅ Done | Generated vs reference → structured scores |
| `utils/loss_scheduler.py` | 🔲 TODO | Judge scores → dynamic λ adjustment |
| MLflow logging | ✅ Done | Per-step metrics, final report |

### Known Limitations (the point)
- W+ inversion is lossy — identity partially lost before editing even starts
- CLIP and identity preservation fight each other during optimization
- StyleGAN distribution doesn't include specific celebrities
- Perceptual loss plateaus around 1.1 — ceiling of what the model can reconstruct

---

## Pipeline B — SDXL + IP-Adapter (Google Colab → FastAPI)

### What it does
- Takes the same celebrity photo + edit prompt
- IP-Adapter locks identity directly into the diffusion process
- No inversion step — identity preserved from the start
- Returns generated image to local machine via FastAPI

### Architecture
```
Google Colab:
┌─────────────────────────────────────┐
│  SDXL Base Model                    │
│       +                             │
│  IP-Adapter (face identity lock)    │
│       +                             │
│  FastAPI endpoint (ngrok tunnel)    │
│                                     │
│  POST /generate                     │
│    body: { prompt, image_base64 }   │
│    returns: { image_base64 }        │
└─────────────────────────────────────┘
          ▲              │
          │              ▼
Local Machine:
┌─────────────────────────────────────┐
│  sdxl_client.py                     │
│  - sends request to ngrok URL       │
│  - receives generated image         │
│  - passes to Gemini judge           │
│  - logs to MLflow                   │
└─────────────────────────────────────┘
```

### Components to build
| Component | Location | Description |
|---|---|---|
| `colab/sdxl_server.ipynb` | Colab | SDXL + IP-Adapter + FastAPI + ngrok |
| `models/sdxl_client.py` | Local | Calls Colab endpoint, gets image back |
| `utils/ngrok_config.py` | Local | Stores current ngrok URL |

### Why this works on 4GB VRAM
SDXL runs entirely on Colab's GPU (T4, 16GB). Local machine only runs the client — zero VRAM cost.

---

## Gemini Intelligence Layer

### Role 1: Prompt Enrichment (before generation)

Runs once per generation request, for both pipelines.

```
Input:   reference photo + raw edit ("red beard")
Process: Gemini analyzes facial geometry, features, proportions
Output:  "caucasian male, strong defined jawline, deep-set blue-grey 
          eyes, high cheekbones, straight nose, medium blonde hair, 
          red beard, sharp brow ridge, symmetrical face"
```

Both pipelines receive the same enriched prompt → fair comparison.

### Role 2: LLM as Judge (during + after generation)

Runs every 50 steps for StyleGAN (dynamic feedback), once at end for SDXL.

```
Input:   generated image + reference photo
Output:  {
           jaw:      8/10,
           eyes:     4/10,
           nose:     7/10,
           hair:     9/10,
           overall:  6/10,
           verdict:  "eye area lost identity, rest preserved well",
           attribute_success: "red beard added correctly"
         }
```

### Role 3: Dynamic Loss Adjustment (StyleGAN only)

```
Judge scores → loss_scheduler.py → adjusted λ weights
eyes: 4/10   → bump λ_identity up
overall: 6/10 → reduce λ_clip slightly
```

---

## MLflow Comparison Dashboard

Both pipelines log to the same MLflow instance. Metrics tracked:

| Metric | StyleGAN | SDXL | Source |
|---|---|---|---|
| `clip_similarity` | ✅ | ✅ | CLIP model |
| `lpips_from_mean` | ✅ | ✅ | LPIPS net |
| `sharpness` | ✅ | ✅ | Laplacian variance |
| `pixel_diversity` | ✅ | ✅ | Pixel std |
| `w_distance_from_mean` | ✅ | ❌ | W-space only |
| `gemini_jaw` | ✅ | ✅ | Gemini judge |
| `gemini_eyes` | ✅ | ✅ | Gemini judge |
| `gemini_overall` | ✅ | ✅ | Gemini judge |
| `gemini_verdict` | ✅ | ✅ | Gemini judge |
| `inversion_perceptual_loss` | ✅ | ❌ | Encoder only |

---

## Project File Structure

```
federateurBDAI/
│
├── models/
│   ├── architecture.py          ✅ StyleGAN2 loader + generator
│   ├── encoder.py               ✅ Photo → W+ inversion
│   ├── clip_guidance.py         ✅ CLIP optimization loop
│   └── sdxl_client.py           🔲 Calls Colab FastAPI endpoint
│
├── utils/
│   ├── config.py                ✅ All constants + API keys
│   ├── metrics.py               ✅ LPIPS, CLIP, sharpness etc.
│   ├── gemini_enricher.py       ✅ Prompt enrichment
│   ├── gemini_judge.py          ✅ LLM as judge
│   ├── loss_scheduler.py        🔲 Dynamic λ adjustment
│   └── ngrok_config.py          🔲 Stores Colab tunnel URL
│
├── colab/
│   └── sdxl_server.ipynb        🔲 Colab: SDXL + FastAPI + ngrok
│
├── inputs/
│   └── brad_pitt.webp           ✅ Reference photo
│
├── celebrity_db/
│   └── brad_pitt.pt             ✅ Saved W+ vector
│
├── outputs/                     ✅ Generated images
├── mlruns/                      ✅ MLflow data
└── mlflow.db                    ✅ MLflow SQLite
```

---

## Build Order

```
Phase 1 — Complete StyleGAN pipeline        (local, already mostly done)
    1.1  loss_scheduler.py                  wire Gemini scores → λ weights
    1.2  wire into optimize()               dynamic adjustment during loop
    1.3  run full StyleGAN generation       brad pitt + red beard
    1.4  verify MLflow logs all metrics

Phase 2 — Build SDXL server on Colab
    2.1  sdxl_server.ipynb                  SDXL + IP-Adapter + FastAPI
    2.2  ngrok tunnel                       expose endpoint publicly
    2.3  test /generate endpoint            curl test from local

Phase 3 — Wire SDXL into local project
    3.1  sdxl_client.py                     POST request + receive image
    3.2  ngrok_config.py                    store/load current tunnel URL
    3.3  run Gemini judge on SDXL output    same judge, same metrics
    3.4  log to MLflow                      same dashboard

Phase 4 — Comparison + Report
    4.1  run both pipelines on same input   brad pitt + red beard
    4.2  MLflow side by side comparison     all metrics same scale
    4.3  Gemini final verdict on both       structured identity scores
    4.4  write conclusion                   prove StyleGAN limitation
```

---

## The Academic Argument

```
Hypothesis:
StyleGAN2's W-space inversion is architecturally insufficient 
for identity-specific face manipulation.

Evidence:
1. Inversion perceptual loss plateaus at ~1.1 (theoretical min ~0.3)
2. Gemini identity scores: StyleGAN avg 5-6/10 vs SDXL avg 8-9/10
3. CLIP similarity improves but at cost of identity (tradeoff proven)
4. Dynamic loss adjustment helps but cannot overcome inversion ceiling

Conclusion:
The limitation is not hyperparameters or training — it is the 
architecture. StyleGAN was not designed for photo-conditioned generation.
SDXL + IP-Adapter solves this at the architecture level.
```

---

## Next Immediate Steps

1. `loss_scheduler.py` — finish the dynamic λ logic
2. Wire it into `optimize()` in `clip_guidance.py`
3. Run a full StyleGAN generation and verify all metrics log cleanly
4. Start `sdxl_server.ipynb` on Colab


*for claude : the most important is running SDXL to gneerate image more important than llm as a jduge or gemini or celebrity or whatevr that's the first and most important feature 
