"""Gemini-powered intelligence layer via OpenRouter.

Feature 1: Prompt enrichment — analyze face photo, return CLIP-friendly description
Feature 2: LLM as Judge — score identity preservation per facial feature
Feature 3: Dynamic loss adjustment — map judge scores to loss weights
Feature 4: Final evaluation report — structured analysis logged to MLflow
"""

import os
import json
import base64
import io

import numpy as np
import requests
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-2.5-flash"


def _get_api_key():
    # User's .env has a typo (OPENROUTER_PAI_KEY), support both
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_PAI_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found in .env")
    return key


def _np_to_data_url(image_np):
    """Convert numpy image [H, W, 3] in [0, 1] to data URL."""
    img = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def _file_to_data_url(image_path):
    """Load image file and convert to data URL."""
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}
    mime = mime_map.get(ext, "image/png")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _call_openrouter(messages, temperature=0.3):
    """Call OpenRouter chat completions and return response text."""
    headers = {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    if response.status_code != 200:
        raise RuntimeError(f"OpenRouter error {response.status_code}: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"]


def _build_content_with_images(text, image_urls):
    """Build OpenAI-style message content with images + text."""
    content = []
    for url in image_urls:
        content.append({"type": "image_url", "image_url": {"url": url}})
    content.append({"type": "text", "text": text})
    return content


# ── Feature 1: Prompt Enrichment ────────────────────────────────────────

def enrich_prompt(reference_image_np, edit_request, image_path=None):
    """Analyze a face photo and produce a CLIP-optimized prompt."""
    system_prompt = """You are a CLIP prompt engineer for face editing.
The user's face is ALREADY encoded into the model — you DO NOT need to describe it.
Your job: turn their edit request into a minimal, CLIP-friendly prompt that describes ONLY the change.

Rules:
1. Describe ONLY the requested edit — do NOT describe eyes, jaw, hair, skin, age, etc. unless they are the edit itself
2. Keep it under 10 words
3. Use concrete visual descriptors CLIP responds to (colors, textures, shapes)
4. One short phrase, no sentences

Examples:
Input: "with a mustache"        → "a thick black mustache"
Input: "red beard"              → "a vibrant red beard"
Input: "wearing glasses"        → "round black-framed glasses"
Input: "smile"                  → "a wide cheerful smile"
Input: "older"                  → "aged, deep wrinkles, grey hair"
"""

    user_text = f'Turn this edit request into a minimal CLIP prompt: "{edit_request}"'

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]

    text = _call_openrouter(messages, temperature=0.3)
    enriched = text.strip().strip('"')
    print(f"Enriched prompt: {enriched}")
    return enriched


# ── Feature 2: LLM as Judge ────────────────────────────────────────────

def judge_identity(generated_image_np, reference_image_np, step=None):
    """Score identity preservation between generated and reference face."""
    system_prompt = """You are a facial identity comparison expert.
Compare the generated face (image 1) against the reference face (image 2).
Score how well each facial feature's IDENTITY is preserved (not style, not quality — identity).

Respond ONLY with valid JSON, no markdown:
{
    "jawline": <1-10>,
    "eyes": <1-10>,
    "nose": <1-10>,
    "mouth": <1-10>,
    "hair": <1-10>,
    "skin_tone": <1-10>,
    "overall": <1-10>,
    "verdict": "<one sentence explaining what drifted and what held>"
}"""

    image_urls = [
        _np_to_data_url(generated_image_np),
        _np_to_data_url(reference_image_np),
    ]
    user_text = ("Image 1 is the generated face. Image 2 is the reference (original identity). "
                 "Score identity preservation per feature.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_content_with_images(user_text, image_urls)},
    ]

    text = _call_openrouter(messages, temperature=0.1)

    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        scores = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        print(f"WARNING: Could not parse judge response: {text[:200]}")
        scores = {"overall": 5, "verdict": "parse error"}

    if step is not None:
        print(f"  [Judge step {step}] overall: {scores.get('overall', '?')}/10 — {scores.get('verdict', '')}")

    return scores


# ── Feature 3: Dynamic Loss Adjustment ──────────────────────────────────

def adjust_loss_weights(judge_scores, current_lambda_id=0.5, current_lr=0.02):
    """Adjust loss weights based on judge feedback."""
    overall = judge_scores.get("overall", 5)
    eyes = judge_scores.get("eyes", 5)

    new_lambda_id = current_lambda_id
    new_lr = current_lr

    if overall <= 4:
        new_lambda_id = min(current_lambda_id * 1.5, 2.0)
        new_lr = current_lr * 0.8
    elif overall <= 6:
        new_lambda_id = min(current_lambda_id * 1.2, 1.5)
    elif overall >= 8:
        new_lambda_id = max(current_lambda_id * 0.8, 0.1)

    if eyes <= 3:
        new_lambda_id = min(new_lambda_id * 1.3, 2.0)

    if new_lambda_id != current_lambda_id or new_lr != current_lr:
        print(f"  [Adjust] lambda_id: {current_lambda_id:.3f} → {new_lambda_id:.3f}, "
              f"lr: {current_lr:.4f} → {new_lr:.4f}")

    return {"lambda_id": new_lambda_id, "lr": new_lr}


# ── Feature 4: Final Evaluation Report ─────────────────────────────────

def final_evaluation(generated_image_np, reference_image_np, edit_request, metrics_dict):
    """Generate final structured evaluation report."""
    metrics_str = "\n".join(f"  {k}: {v:.4f}" for k, v in metrics_dict.items())

    system_prompt = """You are an expert evaluator for AI face generation quality.
Given the generated face (image 1), the reference face (image 2), the edit request, and computed metrics,
produce a final evaluation.

Respond ONLY with valid JSON, no markdown:
{
    "gemini_final_score": <float 1-10>,
    "identity_preserved": ["<feature1>", "<feature2>"],
    "identity_lost": ["<feature1>"],
    "attribute_added": ["<description> — <success/partial/failed>"],
    "image_quality": "<excellent/good/fair/poor>",
    "recommendation": "<one sentence for improving results>",
    "summary": "<2-3 sentence overall assessment>"
}"""

    image_urls = [
        _np_to_data_url(generated_image_np),
        _np_to_data_url(reference_image_np),
    ]
    user_text = (f'Edit request: "{edit_request}"\n\nComputed metrics:\n{metrics_str}\n\n'
                 'Evaluate the result.')

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _build_content_with_images(user_text, image_urls)},
    ]

    text = _call_openrouter(messages, temperature=0.2)

    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        report = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        print(f"WARNING: Could not parse evaluation response: {text[:200]}")
        report = {"gemini_final_score": 0, "summary": "parse error"}

    print(f"\n{'='*60}")
    print(f"FINAL EVALUATION — Score: {report.get('gemini_final_score', '?')}/10")
    print(f"Quality: {report.get('image_quality', '?')}")
    print(f"Preserved: {report.get('identity_preserved', [])}")
    print(f"Lost: {report.get('identity_lost', [])}")
    print(f"Attribute: {report.get('attribute_added', [])}")
    print(f"Summary: {report.get('summary', '')}")
    print(f"Recommendation: {report.get('recommendation', '')}")
    print(f"{'='*60}\n")

    return report
