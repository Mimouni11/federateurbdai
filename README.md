# Deepfake Face Detector

This project is a complete deepfake face detection system built around an **EfficientNet-B0** binary classifier, a **FastAPI** inference API, visual explanations with **Grad-CAM**, experiment tracking with **MLflow**, and multiple UI layers for demo and deployment. It also includes a **StyleGAN + CLIP** face generator so the project can both **detect** synthetic faces and **generate** them for comparison and analysis.

At a high level, the project answers one question:

**Can we train a detector that learns not only the classic artifacts of GAN-generated faces, but also the harder, more identity-aware artifacts introduced by modern diffusion-based portrait generation?**

## What Is a Deepfake Detector?

A deepfake detector is a computer vision model trained to distinguish between **real human face images** and **synthetically generated or manipulated ones**. In this repository, the detector is implemented as an **EfficientNet-B0 based classifier** trained on face images labeled as either:

- `real`
- `fake`

The system includes:

- a full training pipeline with preprocessing and data augmentation
- an inference API for uploading and classifying images
- Grad-CAM heatmaps to explain what regions influenced the prediction
- testing and deployment configuration

The detector does not just output a label. It also returns:

- a calibrated fake probability
- a confidence tier (`high`, `medium`, `low`)
- a Grad-CAM heatmap highlighting the image regions the model focused on

This makes the project useful both as an engineering system and as an explainable demo.

## Synthetic Data Sources

### StyleGAN

**StyleGAN** (Style-based Generative Adversarial Network) is the family of GAN architectures developed by NVIDIA that generated the fake face images in the core training dataset. It learns to synthesize photorealistic human faces from random noise while controlling attributes such as age, hair, pose, and lighting through style vectors injected at different layers of the network.

The **140k Real and Fake Faces** dataset used for training contains **StyleGAN-generated** fake faces as the fake class. Our EfficientNet detector therefore learned to recognize subtle visual artifacts often left by StyleGAN, including:

- unnatural blending around face boundaries
- irregular textures near the eyes and hairline
- inconsistent fine details in skin, teeth, or background transitions

These are exactly the kinds of cues highlighted by the Grad-CAM visualizations.

### SDXLrealvis + IP Adapter

**SDXLrealvis** is a fine-tuned version of **Stable Diffusion XL** optimized for photorealistic portrait generation. When combined with an **IP Adapter** (Image Prompt Adapter), it can take a reference face image and generate new images of the same person under different lighting, poses, and stylistic conditions while preserving identity.

In this project, **SDXLrealvis + IP Adapter** was used to create a more advanced second generation of fake images. These images are harder than pure StyleGAN outputs because they preserve identity and look closer to realistic portrait edits rather than fully synthetic GAN faces.

This matters because it pushes the detector beyond memorizing old GAN artifacts. Instead, it has to learn more subtle signals associated with:

- identity-preserving synthetic portrait generation
- diffusion-based texture inconsistencies
- harder, more realistic deepfake examples

### Why Both Matter

The fake data in this project comes from two different generations of synthesis:

- **StyleGAN** created the first generation of fake training data
- **SDXLrealvis + IP Adapter** created a more advanced, identity-preserving second generation

Our **EfficientNet detector** was trained to catch both. That means the model learns progressively harder examples of synthetic faces instead of overfitting to a single fake-image family, which helps explain its strong recall on the fake class and better generalization than a detector trained only on classic GAN outputs.

## Project Architecture

The repository combines training, inference, explainability, tracking, and UI layers in one place.

### Current application flow

```text
React app
  ├── Splash / tool picker
  ├── Detector UI
  └── Generator UI
          ↓
FastAPI
  ├── /health
  ├── /predict
  └── /generate
          ↓
Models
  ├── EfficientNet-B0 detector
  └── StyleGAN2 + CLIP generator
```

### Training and experimentation flow

```text
Dataset preprocessing
      ↓
EfficientNet-B0 training
      ↓
MLflow logging
      ↓
Threshold calibration
      ↓
FastAPI inference + Grad-CAM
```

## Core Features

### Deepfake detection

- EfficientNet-B0 binary classifier for real vs fake face detection
- Albumentations-based preprocessing and augmentation
- calibrated decision threshold with an `uncertain` zone
- Grad-CAM visual explanations for each prediction

### Generation

- StyleGAN2 face generation guided by CLIP text prompts
- prompt-based synthetic portrait creation
- useful for comparing older GAN-style fakes against detector behavior

### API

The FastAPI backend exposes:

- `GET /health`  
  Reports API status, runtime device, calibration values, and whether the generator has been loaded.

- `POST /predict`  
  Accepts a face image and returns:
  - `prediction`
  - `ai_probability`
  - `confidence`
  - `logit`
  - `gradcam_image`

- `POST /generate`  
  Accepts a text prompt and generation parameters and returns:
  - generated image as base64
  - CLIP similarity
  - generation time
  - prompt and optimization settings

### UI layers

The repo currently contains two frontends:

- [`frontend/`](./frontend)  
  A newer React + Vite interface with a splash screen, detector flow, and generator flow.

- [`streamlit_app/`](./streamlit_app)  
  An earlier Streamlit prototype kept in the repository for experimentation and demo convenience.

### Experiment tracking

MLflow is used for:

- detector training runs
- threshold calibration runs
- user-triggered generation runs

## Results

The detector achieved strong performance on the main benchmark dataset:

| Metric | Score |
|---|---:|
| Test Accuracy | 99.05% |
| Test AUC | 0.9995 |
| Fake Recall | 0.9940 |
| Fake F1 | 0.9905 |

These numbers come from a controlled benchmark with synthetic fake images. Real-world performance may differ on out-of-distribution manipulations or heavily compressed media.

## Repository Structure

```text
api/                    FastAPI backend and inference endpoints
deepfake_detector/      Detector model, training, calibration, Grad-CAM utilities
frontend/               React frontend (tool picker, detector, generator)
streamlit_app/          Legacy Streamlit interface
stylegan/               StyleGAN + CLIP generation pipeline
vendor/                 External StyleGAN2-ADA code required by the .pkl checkpoint
checkpoints/            StyleGAN checkpoints
tests/                  API integration tests
data/                   Dataset assets and splits
mlruns/                 MLflow run artifacts
mlartifacts/            MLflow artifact storage
```

## Key Components

### Detector model

The deepfake detector is based on **EfficientNet-B0**, chosen for its strong accuracy-to-efficiency tradeoff. It is lightweight enough to deploy comfortably while still learning high-quality visual representations for binary classification.

### Grad-CAM for explainability

The project includes Grad-CAM utilities adapted for a **single-neuron binary classifier**. Instead of only reporting a real/fake label, the system generates heatmaps showing where the model found suspicious evidence. This helps validate that the detector is looking at meaningful face regions rather than spurious background patterns.

### Threshold calibration

The detector output is not used as a raw sigmoid-only decision. A dedicated calibration step computes a threshold and margin written to [`deepfake_detector/threshold.json`](./deepfake_detector/threshold.json), enabling:

- better alignment between model score and final decision
- a three-way output: `real`, `fake`, `uncertain`
- more trustworthy confidence reporting

## Training Pipeline

The repository includes the full path from raw data to deployable detector:

1. preprocess and structure the dataset into train/validation/test splits
2. train EfficientNet-B0 with Albumentations-based augmentation
3. log runs and metrics to MLflow
4. calibrate the detector threshold and margin
5. serve the model through FastAPI with Grad-CAM explanations

## Testing

The test suite includes FastAPI integration coverage for:

- health checks
- valid image prediction requests
- invalid file type handling
- missing file handling

Run the tests with:

```bash
pytest
```

## Docker and Deployment

The repository includes Dockerfiles and compose configuration for containerized deployment. Depending on the branch or workflow you are using, the Docker setup may target:

- the FastAPI API
- the Streamlit interface
- the MLflow tracking server

The React frontend and CPU/GPU runtime choices may be handled separately depending on your local setup and demo needs.

In practice, there are two common ways to use this project:

- **Local GPU workflow**  
  Best for running StyleGAN generation at usable speed.

- **Docker CPU workflow**  
  Useful for packaging, sharing, and testing the stack, though generation will be slower.

## Stack

| Layer | Tools |
|---|---|
| Model training | PyTorch |
| Backbone | EfficientNet-B0 |
| Augmentation | Albumentations |
| Explainability | Grad-CAM |
| API | FastAPI |
| Frontend | React + Vite, Streamlit |
| Generation | StyleGAN2 + CLIP |
| Experiment tracking | MLflow |
| Testing | pytest |
| Deployment | Docker, docker-compose |

## Notes

- The StyleGAN `.pkl` checkpoint depends on the NVIDIA StyleGAN2-ADA vendor code in [`vendor/stylegan2-ada-pytorch/`](./vendor/stylegan2-ada-pytorch).
- The generator is significantly more practical on CUDA than on CPU.
- The detector was designed to catch both classic GAN artifacts and harder identity-preserving synthetic portraits.

## Summary

This repository is more than a simple binary classifier. It is a full deepfake analysis project that combines:

- synthetic data generation
- detector training
- explainable inference
- experiment tracking
- web-based interaction
- deployment tooling

By training on both **StyleGAN** and **SDXLrealvis + IP Adapter** style fake images, the project moves beyond narrow GAN artifact detection and toward a more realistic detector for modern synthetic face generation.
