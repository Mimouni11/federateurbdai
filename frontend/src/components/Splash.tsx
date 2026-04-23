import { useEffect } from "react";

interface Props {
  onDone: () => void;
}

export function Splash({ onDone }: Props) {
  useEffect(() => {
    const t = setTimeout(onDone, 2000);
    return () => clearTimeout(t);
  }, [onDone]);

  return (
    <div className="flex h-full w-full items-center justify-center bg-white">
      <div className="flex items-center gap-5 animate-fade-in">
        <SirenIcon />
        <h1 className="text-5xl font-extrabold tracking-tight text-ink">
          AI <span className="text-brand">FORENSICS</span>
        </h1>
      </div>
    </div>
  );
}

function SirenIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      width="72"
      height="72"
      fill="none"
      stroke="#F7820F"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="animate-siren-pulse"
    >
      <path d="M7 18v-6a5 5 0 0 1 5-5v0a5 5 0 0 1 5 5v6" />
      <path d="M5 21h14" />
      <path d="M12 2v2" />
      <path d="M4.93 5.93 6.34 7.34" />
      <path d="M19.07 5.93 17.66 7.34" />
      <path d="M9 18v3" />
      <path d="M15 18v3" />
    </svg>
  );
}
