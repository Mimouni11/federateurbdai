interface Props {
  onPick: (tool: "detector" | "generator") => void;
}

export function ToolPicker({ onPick }: Props) {
  return (
    <div className="flex h-full w-full items-center justify-center bg-white">
      <div className="w-full max-w-3xl px-6 animate-fade-in">
        <div className="mb-10 text-center">
          <div className="mb-3 flex items-center justify-center gap-3">
            <SirenDot />
            <h1 className="text-4xl font-extrabold text-ink">
              AI <span className="text-brand">FORENSICS</span>
            </h1>
          </div>
          <p className="text-neutral-600">Choose a tool.</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <ToolCard
            title="Detect fakes"
            description="Upload a face photo. See if the detector classifies it as real or AI-generated, plus a Grad-CAM heatmap."
            onClick={() => onPick("detector")}
            icon={<ShieldIcon />}
          />
          <ToolCard
            title="Generate faces"
            description="Type a prompt — StyleGAN2 + CLIP will produce a new face from scratch. Then pipe it to the detector."
            onClick={() => onPick("generator")}
            icon={<SparkleIcon />}
          />
        </div>
      </div>
    </div>
  );
}

function ToolCard({
  icon,
  title,
  description,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group rounded-xl border-2 border-neutral-200 bg-white p-7 text-left transition hover:border-brand hover:bg-brand-light"
    >
      <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-brand-light transition group-hover:bg-white">
        {icon}
      </div>
      <h2 className="mb-2 text-xl font-bold text-ink">{title}</h2>
      <p className="text-sm text-neutral-600">{description}</p>
    </button>
  );
}

function SirenDot() {
  return <span className="h-3 w-3 rounded-full bg-brand animate-siren-pulse" />;
}

function ShieldIcon() {
  return (
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
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="m9 12 2 2 4-4" />
    </svg>
  );
}

function SparkleIcon() {
  return (
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
      <path d="M12 3v4" />
      <path d="M12 17v4" />
      <path d="M5.6 5.6l2.8 2.8" />
      <path d="M15.6 15.6l2.8 2.8" />
      <path d="M3 12h4" />
      <path d="M17 12h4" />
      <path d="M5.6 18.4l2.8-2.8" />
      <path d="M15.6 8.4l2.8-2.8" />
    </svg>
  );
}
