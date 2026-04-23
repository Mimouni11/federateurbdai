const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8765";

export type Verdict = "real" | "fake" | "uncertain";
export type ConfidenceLevel = "high" | "medium" | "low";

export interface PredictResponse {
  prediction: Verdict;
  ai_probability: number;   // 0–1, calibrated P(fake)
  confidence: ConfidenceLevel;
  logit?: number;
  gradcam_image: string;
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
