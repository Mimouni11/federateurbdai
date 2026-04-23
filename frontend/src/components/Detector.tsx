import { useState, useRef } from "react";
import { predict, type PredictResponse } from "../api";

interface Props {
  onHome: () => void;
}

export function Detector({ onHome }: Props) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const onFile = async (file: File) => {
    setError(null);
    setResult(null);
    setPreviewUrl(URL.createObjectURL(file));
    setLoading(true);
    try {
      const res = await predict(file);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <div className="min-h-full w-full bg-white">
      <Header onHome={onHome} />

      <main className="mx-auto w-full max-w-5xl px-6 py-10">
        {!previewUrl && (
          <UploadZone
            inputRef={inputRef}
            onFile={(f) => onFile(f)}
          />
        )}

        {previewUrl && (
          <div className="animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Panel title="Input image">
                <img
                  src={previewUrl}
                  alt="input"
                  className="w-full rounded-lg object-contain"
                />
              </Panel>
              <Panel title="Grad-CAM heatmap">
                {result?.gradcam_image ? (
                  <img
                    src={`data:image/png;base64,${result.gradcam_image}`}
                    alt="gradcam"
                    className="w-full rounded-lg object-contain"
                  />
                ) : (
                  <Skeleton />
                )}
              </Panel>
            </div>

            <div className="mt-6">
              {loading && <InfoBar>Running detector…</InfoBar>}
              {error && <ErrorBar>{error}</ErrorBar>}
              {result && <ResultBar result={result} />}
            </div>

            <div className="mt-8 flex justify-center">
              <button
                onClick={reset}
                className="rounded-md border-2 border-ink px-6 py-2 font-semibold text-ink hover:bg-ink hover:text-white transition"
              >
                Try another
              </button>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function Header({ onHome }: { onHome: () => void }) {
  return (
    <header className="border-b border-neutral-200">
      <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <button
          onClick={onHome}
          className="flex items-center gap-3 transition hover:opacity-70"
        >
          <div className="h-3 w-3 rounded-full bg-brand animate-siren-pulse" />
          <span className="font-extrabold tracking-tight text-ink">
            AI <span className="text-brand">FORENSICS</span>
          </span>
        </button>
        <span className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
          Detector
        </span>
      </div>
    </header>
  );
}

function UploadZone({
  inputRef,
  onFile,
}: {
  inputRef: React.RefObject<HTMLInputElement | null>;
  onFile: (f: File) => void;
}) {
  const [dragging, setDragging] = useState(false);

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const f = e.dataTransfer.files?.[0];
        if (f) onFile(f);
      }}
      className={`cursor-pointer border-2 border-dashed rounded-xl p-16 text-center transition ${
        dragging
          ? "border-brand bg-brand-light"
          : "border-neutral-300 hover:border-brand hover:bg-brand-light"
      }`}
    >
      <div className="mx-auto mb-5 flex h-14 w-14 items-center justify-center rounded-full bg-brand-light">
        <svg
          viewBox="0 0 24 24"
          width="28"
          height="28"
          fill="none"
          stroke="#F7820F"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <h2 className="text-xl font-bold text-ink">Upload a face image</h2>
      <p className="mt-2 text-sm text-neutral-600">
        Drag & drop or click to browse — JPEG/PNG
      </p>
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/jpg"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
        }}
      />
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-neutral-200 p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-neutral-500">
        {title}
      </h3>
      {children}
    </div>
  );
}

function Skeleton() {
  return <div className="aspect-square w-full rounded-lg bg-neutral-100 animate-pulse" />;
}

function InfoBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg bg-brand-light px-4 py-3 text-ink">{children}</div>
  );
}

function ErrorBar({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-red-800">
      {children}
    </div>
  );
}

function ResultBar({ result }: { result: PredictResponse }) {
  const aiPct = (result.ai_probability * 100).toFixed(1);
  const style = VERDICT_STYLES[result.prediction];
  const confBadge = CONFIDENCE_BADGE[result.confidence];

  return (
    <div className={`rounded-xl border-2 p-5 ${style.border} ${style.bg}`}>
      <div className="flex items-start justify-between gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
            Verdict
          </p>
          <p className={`text-3xl font-extrabold ${style.text}`}>{style.label}</p>
        </div>
        <div className="text-right">
          <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
            AI probability
          </p>
          <p className="font-mono text-2xl font-bold text-ink">{aiPct}%</p>
        </div>
        <div className="text-right">
          <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
            Confidence
          </p>
          <span
            className={`inline-block mt-1 rounded-full px-3 py-1 text-sm font-bold uppercase tracking-wider ${confBadge}`}
          >
            {result.confidence}
          </span>
        </div>
      </div>
      <div className="mt-4 h-2 w-full rounded-full bg-neutral-200 overflow-hidden">
        <div
          className={`h-full ${style.bar} transition-all`}
          style={{ width: `${aiPct}%` }}
        />
      </div>
    </div>
  );
}

const CONFIDENCE_BADGE: Record<"high" | "medium" | "low", string> = {
  high: "bg-ink text-white",
  medium: "bg-brand text-white",
  low: "bg-neutral-300 text-neutral-700",
};

const VERDICT_STYLES = {
  real: {
    border: "border-ink",
    bg: "bg-white",
    text: "text-ink",
    bar: "bg-ink",
    label: "REAL",
  },
  fake: {
    border: "border-brand",
    bg: "bg-brand-light",
    text: "text-brand",
    bar: "bg-brand",
    label: "FAKE",
  },
  uncertain: {
    border: "border-neutral-300",
    bg: "bg-neutral-50",
    text: "text-neutral-600",
    bar: "bg-neutral-400",
    label: "UNCERTAIN",
  },
} as const;
