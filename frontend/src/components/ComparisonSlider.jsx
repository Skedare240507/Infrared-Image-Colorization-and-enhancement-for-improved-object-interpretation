import { useState } from "react";

export default function ComparisonSlider({ originalUrl, colorizedUrl, heightClass = "aspect-[4/3] md:aspect-video" }) {
  const [sliderPosition, setSliderPosition] = useState(50);

  function handleSliderChange(e) {
    setSliderPosition(Number(e.target.value));
  }

  return (
    <div className={`relative w-full ${heightClass} overflow-hidden rounded-xl border border-white/10 shadow-2xl select-none`}>
      {/* Base Layer: Colorized (RGB Output) */}
      <img
        src={colorizedUrl}
        alt="Colorized Output"
        className="absolute inset-0 h-full w-full object-cover"
        draggable="false"
      />

      {/* Overlay Layer: Original Grayscale Thermal (clipped to sliderPosition) */}
      <img
        src={originalUrl}
        alt="Original Grayscale"
        className="absolute inset-0 h-full w-full object-cover"
        style={{
          clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`,
        }}
        draggable="false"
      />

      {/* Vertical Divider Line */}
      <div
        className="absolute bottom-0 top-0 w-0.5 bg-teal-400 shadow-[0_0_10px_rgba(20,184,166,0.8)] pointer-events-none"
        style={{ left: `${sliderPosition}%` }}
      >
        {/* Handle circle */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 flex h-10 w-10 items-center justify-center rounded-full bg-slate-900 border-2 border-teal-400 text-teal-400 shadow-lg pulse-step-glow">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2.5}
            stroke="currentColor"
            className="w-5 h-5"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M8.25 15L12 18.75 15.75 15m-7.5-6L12 5.25 15.75 9"
              transform="rotate(90 12 12)"
            />
          </svg>
        </div>
      </div>

      {/* Label Tags */}
      <div className="absolute top-4 left-4 rounded-md bg-slate-950/80 px-2.5 py-1 text-xs font-semibold uppercase tracking-wider text-slate-300 border border-white/10 backdrop-blur-sm pointer-events-none">
        Original Thermal
      </div>
      <div className="absolute top-4 right-4 rounded-md bg-slate-950/80 px-2.5 py-1 text-xs font-semibold uppercase tracking-wider text-teal-300 border border-teal-500/20 backdrop-blur-sm pointer-events-none">
        AI Colorized
      </div>

      {/* Invisible range slider overlay covering the entire box for control */}
      <input
        type="range"
        min="0"
        max="100"
        value={sliderPosition}
        onChange={handleSliderChange}
        className="absolute inset-0 h-full w-full cursor-ew-resize opacity-0 slider-input z-10"
      />
    </div>
  );
}
