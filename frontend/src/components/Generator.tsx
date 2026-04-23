interface Props {
  onHome: () => void;
}

export function Generator({ onHome }: Props) {
  return (
    <div className="min-h-full w-full bg-white">
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

      <main className="mx-auto w-full max-w-5xl px-6 py-16 text-center">
        <h1 className="text-2xl font-bold text-ink">StyleGAN2 + CLIP generator</h1>
        <p className="mt-3 text-neutral-600">Coming in the next commit.</p>
      </main>
    </div>
  );
}
