import { useEffect, useState, type HTMLAttributes, type ReactNode } from "react";
import { generate, type GenerateResponse } from "../api";

interface Props {
  onHome: () => void;
}

const DEFAULT_STEPS = 300;
const DEFAULT_LR = "0.02";
const DEFAULT_SEED = "42";

export function Generator({ onHome }: Props) {
  const [prompt, setPrompt] = useState("");
  const [steps, setSteps] = useState(String(DEFAULT_STEPS));
  const [lr, setLr] = useState(DEFAULT_LR);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateResponse | null>(null);

  useEffect(() => {
    if (!loading) return;

    const timer = window.setInterval(() => {
      setElapsedSeconds((current) => current + 1);
    }, 1000);

    return () => window.clearInterval(timer);
  }, [loading]);

  const submitDisabled = loading || prompt.trim().length === 0;

  const resetAll = () => {
    setPrompt("");
    setSteps(String(DEFAULT_STEPS));
    setLr(DEFAULT_LR);
    setSeed(DEFAULT_SEED);
    setAdvancedOpen(false);
    setElapsedSeconds(0);
    setError(null);
    setResult(null);
  };

  const handleGenerate = async () => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) return;

    const parsedSteps = Number.parseInt(steps, 10);
    const parsedLr = Number.parseFloat(lr);
    const parsedSeed = Number.parseInt(seed, 10);

    if (Number.isNaN(parsedSteps) || parsedSteps < 20 || parsedSteps > 1000) {
      setError("Steps must be a whole number between 20 and 1000.");
      return;
    }

    if (Number.isNaN(parsedLr) || parsedLr <= 0 || parsedLr > 0.5) {
      setError("Learning rate must be greater than 0 and at most 0.5.");
      return;
    }

    if (Number.isNaN(parsedSeed)) {
      setError("Seed must be a whole number.");
      return;
    }

    setLoading(true);
    setElapsedSeconds(0);
    setError(null);
    setResult(null);

    try {
      const response = await generate({
        prompt: trimmedPrompt,
        steps: parsedSteps,
        lr: parsedLr,
        seed: parsedSeed,
      });
      setResult(response);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-full w-full bg-white">
      <Header onHome={onHome} />

      <main className="mx-auto w-full max-w-5xl px-6 py-10">
        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <section className="rounded-2xl border border-neutral-200 p-6">
            <div className="flex items-start justify-between gap-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-neutral-500">
                  Generator
                </p>
                <h1 className="mt-2 text-3xl font-extrabold tracking-tight text-ink">
                  Prompt a face into existence
                </h1>
                <p className="mt-3 max-w-2xl text-sm leading-6 text-neutral-600">
                  StyleGAN2 + CLIP will optimize a synthetic portrait from your text
                  prompt. Expect a painterly result and a wait of roughly 30 to 60
                  seconds on GPU.
                </p>
              </div>
              <div className="hidden rounded-full bg-brand-light px-3 py-1 text-xs font-bold uppercase tracking-wider text-brand sm:block">
                CUDA-friendly demo
              </div>
            </div>

            <label className="mt-8 block">
              <span className="text-sm font-semibold uppercase tracking-wider text-neutral-500">
                Prompt
              </span>
              <textarea
                value={prompt}
                onChange={(e) => {
                  setPrompt(e.target.value);
                  if (error) setError(null);
                }}
                rows={6}
                maxLength={300}
                placeholder="A cinematic studio portrait of a woman with short silver hair, soft rim light, confident expression"
                className="mt-3 w-full rounded-xl border border-neutral-300 px-4 py-4 text-base text-ink outline-none transition placeholder:text-neutral-400 focus:border-brand focus:ring-2 focus:ring-brand/20"
              />
            </label>

            <div className="mt-3 flex items-center justify-between text-xs text-neutral-500">
              <span>Be specific about mood, lighting, and face details.</span>
              <span>{prompt.length}/300</span>
            </div>

            <div className="mt-6 rounded-xl border border-neutral-200">
              <button
                type="button"
                onClick={() => setAdvancedOpen((open) => !open)}
                className="flex w-full items-center justify-between px-4 py-3 text-left transition hover:bg-neutral-50"
              >
                <span className="text-sm font-semibold uppercase tracking-wider text-neutral-500">
                  Advanced
                </span>
                <span className="text-sm font-bold text-ink">
                  {advancedOpen ? "Hide" : "Show"}
                </span>
              </button>

              {advancedOpen && (
                <div className="grid gap-4 border-t border-neutral-200 px-4 py-4 md:grid-cols-3">
                  <Field
                    label="Steps"
                    hint="20 to 1000"
                    value={steps}
                    onChange={setSteps}
                    inputMode="numeric"
                  />
                  <Field
                    label="Learning rate"
                    hint="0.001 to 0.5"
                    value={lr}
                    onChange={setLr}
                    inputMode="decimal"
                  />
                  <Field
                    label="Seed"
                    hint="Whole number"
                    value={seed}
                    onChange={setSeed}
                    inputMode="numeric"
                  />
                </div>
              )}
            </div>

            <div className="mt-6 flex flex-col gap-3 sm:flex-row">
              <button
                type="button"
                onClick={handleGenerate}
                disabled={submitDisabled}
                className="rounded-xl border-2 border-brand bg-brand px-6 py-3 text-sm font-bold uppercase tracking-wider text-white transition hover:brightness-95 disabled:cursor-not-allowed disabled:border-neutral-300 disabled:bg-neutral-300"
              >
                {loading ? "Generating..." : "Generate face"}
              </button>

              <button
                type="button"
                onClick={resetAll}
                disabled={loading}
                className="rounded-xl border-2 border-ink px-6 py-3 text-sm font-bold uppercase tracking-wider text-ink transition hover:bg-ink hover:text-white disabled:cursor-not-allowed disabled:border-neutral-300 disabled:text-neutral-400 disabled:hover:bg-transparent"
              >
                Reset
              </button>
            </div>

            <div className="mt-6 space-y-3">
              {loading && (
                <InfoBar>
                  <div className="flex items-center justify-between gap-4">
                    <div className="flex items-center gap-3">
                      <Spinner />
                      <div>
                        <p className="font-semibold text-ink">
                          Generating... this usually takes around 60 seconds.
                        </p>
                        <p className="text-sm text-neutral-600">
                          First run may be slower while the StyleGAN pipeline warms up.
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                        Elapsed
                      </p>
                      <p className="font-mono text-xl font-bold text-ink">
                        {formatElapsed(elapsedSeconds)}
                      </p>
                    </div>
                  </div>
                </InfoBar>
              )}

              {error && <ErrorBar>{error}</ErrorBar>}
            </div>
          </section>

          <section className="rounded-2xl border border-neutral-200 p-6">
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-neutral-500">
              Result
            </p>

            {!result && !loading && (
              <div className="mt-4 flex min-h-[28rem] flex-col items-center justify-center rounded-2xl border border-dashed border-neutral-300 bg-neutral-50 px-8 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-brand-light">
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
                    <path d="M12 3v18" />
                    <path d="M3 12h18" />
                    <path d="M5.6 5.6l12.8 12.8" />
                    <path d="M18.4 5.6 5.6 18.4" />
                  </svg>
                </div>
                <h2 className="mt-5 text-xl font-bold text-ink">No image yet</h2>
                <p className="mt-2 max-w-sm text-sm leading-6 text-neutral-600">
                  Enter a prompt, tune the advanced parameters if you want, and run a
                  generation to see the portrait here.
                </p>
              </div>
            )}

            {loading && (
              <div className="mt-4">
                <Skeleton />
              </div>
            )}

            {result && (
              <div className="mt-4 animate-fade-in space-y-4">
                <div className="overflow-hidden rounded-2xl bg-neutral-100">
                  <img
                    src={`data:image/png;base64,${result.image}`}
                    alt={result.prompt}
                    className="w-full object-cover"
                  />
                </div>

                <div className="rounded-xl border border-neutral-200 bg-white p-4">
                  <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
                    Prompt used
                  </p>
                  <p className="mt-2 text-sm leading-6 text-ink">{result.prompt}</p>
                </div>

                <div className="grid gap-4 md:grid-cols-3">
                  <MetricCard
                    label="CLIP similarity"
                    value={result.final_clip_similarity.toFixed(4)}
                  />
                  <MetricCard
                    label="Generation time"
                    value={`${result.generation_time_s.toFixed(2)}s`}
                  />
                  <MetricCard
                    label="Settings"
                    value={`${result.num_steps} / ${result.lr} / ${result.seed}`}
                  />
                </div>
              </div>
            )}
          </section>
        </div>
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
          Generator
        </span>
      </div>
    </header>
  );
}

function Field({
  label,
  hint,
  value,
  onChange,
  inputMode,
}: {
  label: string;
  hint: string;
  value: string;
  onChange: (value: string) => void;
  inputMode: HTMLAttributes<HTMLInputElement>["inputMode"];
}) {
  return (
    <label className="block">
      <span className="text-sm font-semibold text-ink">{label}</span>
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        inputMode={inputMode}
        className="mt-2 w-full rounded-lg border border-neutral-300 px-3 py-2 text-sm text-ink outline-none transition focus:border-brand focus:ring-2 focus:ring-brand/20"
      />
      <span className="mt-2 block text-xs text-neutral-500">{hint}</span>
    </label>
  );
}

function InfoBar({ children }: { children: ReactNode }) {
  return <div className="rounded-lg bg-brand-light px-4 py-4 text-ink">{children}</div>;
}

function ErrorBar({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-red-800">
      {children}
    </div>
  );
}

function Spinner() {
  return (
    <div className="h-5 w-5 animate-spin rounded-full border-2 border-brand border-t-transparent" />
  );
}

function Skeleton() {
  return <div className="aspect-square w-full rounded-2xl bg-neutral-100 animate-pulse" />;
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-neutral-200 p-4">
      <p className="text-xs font-semibold uppercase tracking-wider text-neutral-500">
        {label}
      </p>
      <p className="mt-2 text-lg font-bold text-ink">{value}</p>
    </div>
  );
}

function formatElapsed(seconds: number) {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}
