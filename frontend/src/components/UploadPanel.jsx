import { useState, useRef } from "react";

export default function UploadPanel({
  onProcessFile,
  onLoadDemo,
  onReset,
  currentJob,
  processingState,
}) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const fileInputRef = useRef(null);

  function handleDrag(e) {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  }

  function handleFileChange(e) {
    if (e.target.files && e.target.files[0]) {
      validateAndSetFile(e.target.files[0]);
    }
  }

  function validateAndSetFile(file) {
    const name = file.name.toLowerCase();
    if (name.endsWith(".tif") || name.endsWith(".tiff") || name.endsWith(".png")) {
      setSelectedFile(file);
    } else {
      alert("Invalid file format. Please upload a georeferenced thermal TIFF (.tif, .tiff) or PNG (.png) image.");
    }
  }

  function triggerFileInput() {
    fileInputRef.current.click();
  }

  function handleProcess() {
    if (selectedFile) {
      onProcessFile(selectedFile);
    }
  }

  function handleClear() {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
    onReset();
  }

  // Determine current active index for stepper
  // States: idle, uploading, processing, completed, failed
  let activeStep = 0;
  if (processingState === "uploading") activeStep = 1;
  else if (processingState === "processing") activeStep = 2;
  else if (currentJob?.status === "completed") activeStep = 4;
  else if (currentJob?.status === "failed" || processingState === "failed") activeStep = -1;

  const steps = [
    { title: "TIFF Upload", desc: "Select 16-bit thermal file" },
    { title: "Super-Resolution", desc: "2x Geospatial upscaling" },
    { title: "AI Colorization", desc: "Pix2Pix RGB translation" },
    { title: "Georeference Bind", desc: "Co-register coordinates" },
  ];

  const formatSize = (bytes) => {
    if (!bytes) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const isWorking = processingState === "uploading" || processingState === "processing";

  return (
    <div className="space-y-6">
      {/* Upload Box */}
      <div className="glass-panel rounded-2xl p-6 shadow-xl relative overflow-hidden">
        {/* Subtle grid pattern overlay */}
        <div className="absolute inset-0 grid-overlay opacity-30 pointer-events-none"></div>

        <h2 className="text-xl font-bold tracking-tight text-white mb-4 flex items-center gap-2 relative z-10">
          <svg className="w-5 h-5 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
          </svg>
          Sensor Data Input
        </h2>

        {!selectedFile && !currentJob ? (
          <div
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-300 relative z-10 ${
              dragActive
                ? "border-teal-400 bg-teal-500/10 shadow-[0_0_15px_rgba(20,184,166,0.2)]"
                : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
            }`}
            onClick={triggerFileInput}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".tif,.tiff,.png"
              className="hidden"
            />
            <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-slate-900 border border-slate-800 text-slate-400 mb-4">
              <svg className="w-7 h-7 text-teal-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <p className="text-slate-200 font-medium mb-1">Drag & Drop Imagery</p>
            <p className="text-slate-400 text-xs mb-4">Supports TIFF (.tif, .tiff) and PNG (.png) formats</p>
            <button
              type="button"
              className="rounded-lg bg-teal-500/10 hover:bg-teal-500/20 text-teal-400 text-xs px-4 py-2 border border-teal-500/20 font-semibold tracking-wider transition-all"
            >
              BROWSE FILE
            </button>
          </div>
        ) : (
          <div className="rounded-xl border border-slate-800/80 bg-slate-950/40 p-4 relative z-10 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-slate-900 border border-slate-800 text-teal-400">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </div>
              <div className="max-w-[200px] md:max-w-xs overflow-hidden">
                <p className="text-slate-200 text-sm font-medium truncate">
                  {selectedFile ? selectedFile.name : currentJob.original_filename}
                </p>
                <p className="text-slate-400 text-xs">
                  {selectedFile ? formatSize(selectedFile.size) : "Uploaded Image"}
                </p>
              </div>
            </div>
            {!isWorking && (
              <button
                onClick={handleClear}
                className="text-slate-400 hover:text-red-400 transition-colors p-1.5 rounded bg-slate-900 border border-slate-800 hover:border-red-500/20"
                title="Remove file"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
          </div>
        )}

        {/* Action Panel */}
        <div className="mt-6 flex flex-col sm:flex-row gap-3 relative z-10">
          {selectedFile && !currentJob && (
            <button
              onClick={handleProcess}
              disabled={isWorking}
              className="flex-1 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold px-4 py-3 tracking-wide transition-all shadow-[0_0_15px_rgba(20,184,166,0.3)] disabled:opacity-50 flex items-center justify-center gap-2"
            >
              {isWorking ? (
                <>
                  <svg className="animate-spin h-5 w-5 text-slate-950" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  PROCESSING PILES...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  PROCESS IMAGERY
                </>
              )}
            </button>
          )}

          {!selectedFile && !currentJob && (
            <button
              onClick={onLoadDemo}
              disabled={isWorking}
              className="flex-1 rounded-xl border border-teal-500/30 bg-teal-500/10 hover:bg-teal-500/20 text-teal-300 font-bold px-4 py-3 tracking-wide transition-all disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <svg className="w-5 h-5 text-teal-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
              LOAD DEMO SAMPLE
            </button>
          )}

          {currentJob && !isWorking && (
            <button
              onClick={handleClear}
              className="w-full rounded-xl border border-slate-800 bg-slate-900/60 hover:bg-slate-900 hover:text-white text-slate-300 font-bold px-4 py-3 tracking-wide transition-all flex items-center justify-center gap-2"
            >
              RESET DASHBOARD
            </button>
          )}
        </div>
      </div>

      {/* Stepper Pipeline */}
      {(isWorking || currentJob) && (
        <div className="glass-panel rounded-2xl p-6 shadow-xl">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-5">
            Geospatial Pipeline Stages
          </h3>

          <div className="relative pl-6 space-y-6 border-l border-slate-800">
            {steps.map((step, idx) => {
              const stepNum = idx + 1;
              let stepState = "pending"; // pending, active, completed, error

              if (activeStep === -1) {
                stepState = idx === 0 ? "completed" : idx === 1 ? "error" : "pending";
              } else if (activeStep >= stepNum) {
                stepState = "completed";
              } else if (activeStep === idx) {
                stepState = "active";
              }

              return (
                <div key={idx} className="relative">
                  {/* Stepper Badge */}
                  <span
                    className={`absolute -left-[35px] top-0 flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-bold border transition-all duration-300 ${
                      stepState === "completed"
                        ? "bg-teal-500 border-teal-500 text-slate-950"
                        : stepState === "active"
                        ? "bg-slate-900 border-teal-400 text-teal-400 pulse-step-glow"
                        : stepState === "error"
                        ? "bg-red-950 border-red-500 text-red-400"
                        : "bg-slate-950 border-slate-850 text-slate-500"
                    }`}
                  >
                    {stepState === "completed" ? (
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                      </svg>
                    ) : stepState === "error" ? (
                      "!"
                    ) : (
                      stepNum
                    )}
                  </span>

                  {/* Step Description */}
                  <div>
                    <h4
                      className={`text-sm font-semibold transition-colors duration-300 ${
                        stepState === "completed"
                          ? "text-slate-200"
                          : stepState === "active"
                          ? "text-teal-400"
                          : stepState === "error"
                          ? "text-red-400"
                          : "text-slate-500"
                      }`}
                    >
                      {step.title}
                    </h4>
                    <p className="text-xs text-slate-400">{step.desc}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
