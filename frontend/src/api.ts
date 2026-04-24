const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8765";

export type Verdict = "real" | "fake" | "uncertain";
export type ConfidenceLevel = "high" | "medium" | "low";

export interface PredictResponse {
  prediction: Verdict;
  ai_probability: number;   // 0â€“1, calibrated P(fake)
  confidence: ConfidenceLevel;
  logit?: number;
  gradcam_image: string;
}

export interface GenerateRequest {
  prompt: string;
  steps?: number;
  lr?: number;
  seed?: number;
}

export interface GenerateResponse {
  prompt: string;
  image: string;
  final_clip_similarity: number;
  generation_time_s: number;
  num_steps: number;
  lr: number;
  seed: number;
}

export async function predict(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }

  return res.json();
}

export async function generate(payload: GenerateRequest): Promise<GenerateResponse> {
  const res = await fetch(`${API_URL}/generate`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`API ${res.status}: ${detail}`);
  }

  return res.json();
}
