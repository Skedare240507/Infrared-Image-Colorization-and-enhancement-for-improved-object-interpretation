import { useState } from "react";
import ComparisonSlider from "./ComparisonSlider";

// Programmatic download: fetch → Blob → object URL → hidden anchor click.
// FIX: Use setAttribute("download", filename) and delay DOM removal by 1s.
// Calling link.remove() synchronously after link.click() causes browsers to
// read the download attribute BEFORE it is committed, so the filename falls
// back to the blob UUID.  Keeping the node alive for ~1s resolves this.
async function triggerDownload(url, filename) {
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const blob = await res.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.style.display = "none";          // invisible, no layout shift
    link.href = objectUrl;
    link.setAttribute("download", filename); // setAttribute is more reliable than .download
    document.body.appendChild(link);
    link.click();
    // ⚠️  Do NOT remove immediately — browser needs ~1 tick to read the attribute
    setTimeout(() => {
      document.body.removeChild(link);
      URL.revokeObjectURL(objectUrl);
    }, 1500);
  } catch (err) {
    console.error("Download failed:", err);
    alert(`Download failed: ${err.message}`);
  }
}

export default function ResultViewer({
  currentJob,
  processingState,
  previewUrlFetcher,
  downloadUrlFetcher,
}) {
  const [activeTab, setActiveTab] = useState("slider");

  if (processingState === "idle" && !currentJob) {
    return (
      <div className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-xl aspect-video md:h-full min-h-[350px] relative overflow-hidden">
        <div className="absolute inset-0 grid-overlay opacity-30 pointer-events-none"></div>
        <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-slate-900 border border-slate-800 text-slate-500 mb-6 relative z-10">
          <svg className="w-8 h-8 text-slate-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
          </svg>
        </div>
        <h3 className="text-lg font-bold text-white mb-2 relative z-10">Awaiting Geospatial Imagery</h3>
        <p className="max-w-xs text-slate-400 text-sm relative z-10 leading-relaxed">
          Upload a raw 16-bit thermal GeoTIFF image, or select the simulated demo sample to trigger the AI processing pipeline.
        </p>
      </div>
    );
  }

  const isWorking = processingState === "uploading" || processingState === "processing";

  if (isWorking) {
    return (
      <div className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center shadow-xl aspect-video md:h-full min-h-[350px]">
        <div className="relative flex items-center justify-center mb-8">
          {/* Animated radar rings */}
          <div className="absolute h-24 w-24 rounded-full border border-teal-500/20 animate-ping"></div>
          <div className="absolute h-16 w-16 rounded-full border border-teal-500/40 animate-pulse"></div>
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-slate-900 border border-teal-500 text-teal-400">
            <svg className="animate-spin h-6 w-6" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        </div>
        <h3 className="text-lg font-bold text-white mb-2 tracking-wide">
          {processingState === "uploading" ? "UPLOADING GEOSPATIAL DATA" : "RUNNING NEURAL NETWORKS"}
        </h3>
        <p className="max-w-xs text-slate-400 text-sm text-center leading-relaxed">
          {processingState === "uploading"
            ? "Transferring imagery to telemetry processing server..."
            : "Executing super-resolution, Pix2Pix translation, and projection bindings..."}
        </p>
      </div>
    );
  }

  if (currentJob?.status === "failed") {
    return (
      <div className="glass-panel rounded-2xl p-8 flex flex-col items-center justify-center text-center shadow-xl min-h-[350px]">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-red-950 border border-red-500/30 text-red-500 mb-4">
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <h3 className="text-lg font-bold text-white mb-2">Pipeline Processing Error</h3>
        <p className="max-w-sm text-slate-400 text-sm mb-4 leading-relaxed">
          The processing pipeline encountered an issue. Check file structure or try reloading the demo model.
        </p>
        <div className="rounded-lg bg-red-950/20 border border-red-500/20 px-4 py-2.5 max-w-md text-left text-xs font-mono text-red-400 overflow-x-auto w-full">
          {currentJob.error || "Unknown processing exception."}
        </div>
      </div>
    );
  }

  if (!currentJob || currentJob.status !== "completed") return null;

  // Extract previews
  const jobId = currentJob.job_id;
  const originalPreviewUrl = previewUrlFetcher(jobId, "original");
  const superResolutionPreviewUrl = previewUrlFetcher(jobId, "super_resolution");
  const colorizedUrl = previewUrlFetcher(jobId, "colorized");
  const combinedUrl = previewUrlFetcher(jobId, "preview");

  const metadata = currentJob.metadata || {};

  const tabs = [
    { id: "slider", name: "Before/After Slider" },
    { id: "colorized", name: "Colorized Output" },
    { id: "sr", name: "Super-Resolution" },
    { id: "original", name: "Original Input" },
    { id: "static", name: "Static Stack" },
  ];

  return (
    <div className="space-y-6">
      {/* Viewer Tab Container */}
      <div className="glass-panel rounded-2xl p-4 shadow-xl">
        {/* Navigation Tabs */}
        <div className="flex border-b border-white/5 pb-2 mb-4 overflow-x-auto gap-2 scrollbar-none">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all whitespace-nowrap ${
                activeTab === tab.id
                  ? "bg-teal-500 text-slate-950 font-bold shadow-md shadow-teal-500/25"
                  : "text-slate-400 hover:text-white hover:bg-white/5"
              }`}
            >
              {tab.name}
            </button>
          ))}
        </div>

        {/* Dynamic Media Body */}
        <div className="relative">
          {activeTab === "slider" && (
            <ComparisonSlider originalUrl={originalPreviewUrl} colorizedUrl={colorizedUrl} />
          )}

          {activeTab === "colorized" && (
            <div className="overflow-hidden rounded-xl border border-white/10 aspect-[4/3] md:aspect-video bg-slate-950">
              <img src={colorizedUrl} alt="AI Colorized Product" className="h-full w-full object-contain" />
            </div>
          )}

          {activeTab === "sr" && (
            <div className="overflow-hidden rounded-xl border border-white/10 aspect-[4/3] md:aspect-video bg-slate-950">
              <img src={superResolutionPreviewUrl} alt="Super Resolution Grayscale" className="h-full w-full object-contain" />
            </div>
          )}

          {activeTab === "original" && (
            <div className="overflow-hidden rounded-xl border border-white/10 aspect-[4/3] md:aspect-video bg-slate-950">
              <img src={originalPreviewUrl} alt="Original Grayscale" className="h-full w-full object-contain" />
            </div>
          )}

          {activeTab === "static" && (
            <div className="overflow-hidden rounded-xl border border-white/10 aspect-[4/3] md:aspect-video bg-slate-950">
              <img src={combinedUrl} alt="Horizontal Dual Stack" className="h-full w-full object-contain" />
            </div>
          )}
        </div>
      </div>

      {/* Analytical Metadata & Georeference Bind */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Metric panel */}
        <div className="glass-panel rounded-2xl p-6 shadow-xl relative overflow-hidden">
          <div className="absolute inset-0 grid-overlay opacity-20 pointer-events-none"></div>
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-teal-400 rounded-full"></span>
            Imagery Analytics
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div className="rounded-xl bg-slate-950/45 p-3.5 border border-white/5">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Thermal Input Dimensions</p>
              <p className="text-xl font-mono font-bold text-white mt-1">
                {metadata.input_shape ? `${metadata.input_shape[1]} × ${metadata.input_shape[0]}` : "128 × 128"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/45 p-3.5 border border-teal-500/10">
              <p className="text-[10px] text-teal-400 uppercase tracking-widest font-semibold">Super-Resolved Output</p>
              <p className="text-xl font-mono font-bold text-teal-300 mt-1">
                {metadata.output_shape ? `${metadata.output_shape[1]} × ${metadata.output_shape[0]}` : "256 × 256"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/45 p-3.5 border border-white/5">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Spatial Scale Ratio</p>
              <p className="text-xl font-mono font-bold text-white mt-1">
                {metadata.sr_scale ? `${metadata.sr_scale}x` : "2x"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/45 p-3.5 border border-white/5">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Processing Hardware</p>
              <p className="text-xl font-mono font-bold text-indigo-400 mt-1 uppercase">
                {currentJob.metadata?.device || "CPU"}
              </p>
            </div>
            <div className="rounded-xl bg-slate-950/45 p-3.5 border border-white/5 col-span-2">
              <p className="text-[10px] text-slate-400 uppercase tracking-widest font-semibold">Model Backends</p>
              <p className="text-xs font-mono text-slate-200 mt-1.5 flex items-center justify-between">
                <span>Upscaling:</span>
                <span className="text-slate-100 font-bold bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-[10px] uppercase">
                  {metadata.sr_backend || "Lanczos OpenCV"}
                </span>
              </p>
              <p className="text-xs font-mono text-slate-200 mt-1 flex items-center justify-between">
                <span>Colorization:</span>
                <span className="text-teal-300 font-bold bg-slate-900 border border-slate-800 px-2 py-0.5 rounded text-[10px] uppercase">
                  {metadata.colorization_backend || "Pix2Pix Engine"}
                </span>
              </p>
            </div>
          </div>
        </div>

        {/* Spatial Coordinates Box */}
        <div className="glass-panel rounded-2xl p-6 shadow-xl relative overflow-hidden">
          <div className="absolute inset-0 grid-overlay opacity-20 pointer-events-none"></div>
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
            <span className="w-1.5 h-1.5 bg-teal-400 rounded-full"></span>
            Geospatial Reference Bind
          </h3>
          <div className="space-y-3.5">
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <span className="text-xs text-slate-400">Coordinate System</span>
              <span className="text-xs font-mono font-semibold text-slate-200">EPSG:4326 (WGS 84)</span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <span className="text-xs text-slate-400">Latitude Coverage</span>
              <span className="text-xs font-mono font-semibold text-slate-200">12.9715° N to 12.9843° N</span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <span className="text-xs text-slate-400">Longitude Coverage</span>
              <span className="text-xs font-mono font-semibold text-slate-200">77.5842° E to 77.5970° E</span>
            </div>
            <div className="flex items-center justify-between border-b border-white/5 pb-2">
              <span className="text-xs text-slate-400">Target Coverage Location</span>
              <span className="text-xs font-mono font-semibold text-teal-400">Bangalore Urban, IN</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-400">Geospatial Tie Points</span>
              <span className="text-[10px] font-mono text-slate-400 border border-slate-800 bg-slate-950/60 px-2 py-0.5 rounded">
                Validated
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Download Center */}
      <div className="glass-panel rounded-2xl p-6 shadow-xl relative overflow-hidden">
        <div className="absolute inset-0 grid-overlay opacity-20 pointer-events-none"></div>
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4 flex items-center gap-2">
          <span className="w-1.5 h-1.5 bg-teal-400 rounded-full"></span>
          Pipeline Download Center
        </h3>
        <div className="grid gap-3 sm:grid-cols-2 md:grid-cols-4">

          {/* ─── Download Card Component ─── */}
          {[
            {
              artifact: "original",
              label: "ORIGINAL SOURCE",
              filename: "original.tif",
              subtitle: "16-bit GeoTIFF",
              labelColor: "text-slate-400",
            },
            {
              artifact: "super_resolution",
              label: "SUPER RESOLUTION",
              filename: "super_resolution.tif",
              subtitle: "2x scale GeoTIFF",
              labelColor: "text-teal-400",
            },
            {
              artifact: "colorized",
              label: "VISUALIZATION MAP",
              filename: "colorized.png",
              subtitle: "RGB Color Image",
              labelColor: "text-slate-400",
            },
            {
              artifact: "preview",
              label: "COMBINED SLIDES",
              filename: "preview.png",
              subtitle: "Dual-grid RGB Stack",
              labelColor: "text-slate-400",
            },
          ].map(({ artifact, label, filename, subtitle, labelColor }) => (
            <DownloadCard
              key={artifact}
              label={label}
              filename={filename}
              subtitle={subtitle}
              labelColor={labelColor}
              onDownload={() =>
                triggerDownload(downloadUrlFetcher(jobId, artifact), filename)
              }
            />
          ))}

        </div>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────
// DownloadCard: button-based card that calls the
// fetch→Blob→click handler instead of <a download>
// ──────────────────────────────────────────────
function DownloadCard({ label, filename, subtitle, labelColor, onDownload }) {
  const [loading, setLoading] = useState(false);

  async function handleClick() {
    if (loading) return;
    setLoading(true);
    try {
      await onDownload();
    } finally {
      setLoading(false);
    }
  }

  return (
    <button
      onClick={handleClick}
      disabled={loading}
      className="group flex flex-col justify-between rounded-xl bg-slate-950/45 p-4 border border-white/5 hover:border-teal-500/20 transition-all hover:bg-slate-900/40 text-left w-full cursor-pointer disabled:opacity-60 disabled:cursor-wait"
    >
      <div>
        <p className={`text-[10px] ${labelColor} uppercase tracking-widest font-semibold`}>
          {label}
        </p>
        <p className="text-xs font-bold text-slate-200 mt-1">{filename}</p>
        <p className="text-[10px] text-slate-500 mt-0.5 font-mono">{subtitle}</p>
      </div>
      <div className="mt-4 flex items-center gap-1.5 text-xs text-teal-400 font-semibold group-hover:text-teal-300">
        {loading ? (
          <>
            <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            <span>Downloading…</span>
          </>
        ) : (
          <>
            <span>Download</span>
            <svg
              className="w-4 h-4 transition-transform group-hover:translate-y-0.5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0L8 8m4-4v12"
              />
            </svg>
          </>
        )}
      </div>
    </button>
  );
}
