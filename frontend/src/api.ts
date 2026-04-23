const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8765";

export type Verdict = "real" | "fake" | "uncertain";

export interface PredictResponse {
  prediction: Verdict;
  confidence: number;
  prob_real?: number;
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
