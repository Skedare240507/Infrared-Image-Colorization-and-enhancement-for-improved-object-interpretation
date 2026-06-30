import { useState, useEffect } from "react";
import UploadPanel from "./components/UploadPanel";
import ResultViewer from "./components/ResultViewer";
import {
  checkHealth,
  uploadThermal,
  processThermal,
  getJobStatus,
  loadDemoSample,
  getPreviewUrl,
  getDownloadUrl,
} from "./services/api";

export default function App() {
  const [currentJob, setCurrentJob] = useState(null);
  const [processingState, setProcessingState] = useState("idle"); // idle, uploading, processing, completed, failed
  const [healthStatus, setHealthStatus] = useState("checking"); // checking, online, offline
  const [pollIntervalId, setPollIntervalId] = useState(null);

  // Run periodic health check
  useEffect(() => {
    async function verifyBackendConnection() {
      const health = await checkHealth();
      if (health && health.status === "ok") {
        setHealthStatus("online");
      } else {
        setHealthStatus("offline");
      }
    }

    verifyBackendConnection();
    const interval = setInterval(verifyBackendConnection, 10000);
    return () => clearInterval(interval);
  }, []);

  // Cleanup polling interval if component unmounts
  useEffect(() => {
    return () => {
      if (pollIntervalId) clearInterval(pollIntervalId);
    };
  }, [pollIntervalId]);

  // Handle uploading and processing of custom TIFF file
  async function handleProcessFile(file) {
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
      setPollIntervalId(null);
    }

    setProcessingState("uploading");
    setCurrentJob(null);

    try {
      // 1. Upload TIFF file
      const uploadResult = await uploadThermal(file);
      setCurrentJob(uploadResult);
      setProcessingState("processing");

      // 2. Trigger pipeline processing
      const processResult = await processThermal(uploadResult.job_id);
      setCurrentJob(processResult);

      if (processResult.status === "completed") {
        setProcessingState("completed");
      } else if (processResult.status === "processing" || processResult.status === "uploaded") {
        // Polling if asynchronous
        startPolling(processResult.job_id);
      } else {
        setProcessingState("failed");
      }
    } catch (err) {
      console.error("Upload & Processing failed:", err);
      setProcessingState("failed");
      setCurrentJob({
        status: "failed",
        error: err.message || "Failed to upload or process thermal image.",
      });
    }
  }

  // Handle loading programmatically generated sample
  async function handleLoadDemo() {
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
      setPollIntervalId(null);
    }

    setProcessingState("processing");
    setCurrentJob(null);

    try {
      const demoResult = await loadDemoSample();
      setCurrentJob(demoResult);
      if (demoResult.status === "completed") {
        setProcessingState("completed");
      } else {
        setProcessingState("failed");
      }
    } catch (err) {
      console.error("Demo loading failed:", err);
      setProcessingState("failed");
      setCurrentJob({
        status: "failed",
        error: err.message || "Failed to generate simulated thermal geo-sample.",
      });
    }
  }

  // Start polling backend for job updates
  function startPolling(jobId) {
    const interval = setInterval(async () => {
      try {
        const job = await getJobStatus(jobId);
        setCurrentJob(job);
        if (job.status === "completed") {
          setProcessingState("completed");
          clearInterval(interval);
          setPollIntervalId(null);
        } else if (job.status === "failed") {
          setProcessingState("failed");
          clearInterval(interval);
          setPollIntervalId(null);
        }
      } catch (err) {
        console.error("Polling status failed:", err);
        setProcessingState("failed");
        clearInterval(interval);
        setPollIntervalId(null);
      }
    }, 1500);

    setPollIntervalId(interval);
  }

  function handleReset() {
    if (pollIntervalId) {
      clearInterval(pollIntervalId);
      setPollIntervalId(null);
    }
    setCurrentJob(null);
    setProcessingState("idle");
  }

  return (
    <div className="min-h-screen relative flex flex-col font-sans">
      {/* Background Matrix Mesh */}
      <div className="absolute inset-0 grid-overlay opacity-30 pointer-events-none z-0"></div>

      {/* Header Bar */}
      <header className="glass-panel border-b border-white/5 relative z-10 px-6 py-4 flex items-center justify-between shadow-lg">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-400 shadow-[0_0_10px_rgba(20,184,166,0.1)]">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </div>
          <div>
            <h1 className="text-lg font-bold text-white tracking-tight flex items-center gap-2">
              GeoThermal AI Dashboard
              <span className="text-[10px] bg-teal-500/10 text-teal-400 font-bold border border-teal-500/20 px-2 py-0.5 rounded uppercase tracking-wider">
                V1.0.0
              </span>
            </h1>
            <p className="text-xs text-slate-400">
              Super-Resolution & Chromatic Interpretation Pipeline
            </p>
          </div>
        </div>

        {/* Connection status indicator */}
        <div className="flex items-center gap-2 rounded-full bg-slate-900 border border-white/5 px-3 py-1.5 backdrop-blur-md">
          <span
            className={`h-2 w-2 rounded-full shadow-[0_0_8px] ${
              healthStatus === "online"
                ? "bg-emerald-500 shadow-emerald-500"
                : healthStatus === "checking"
                ? "bg-amber-500 shadow-amber-500 animate-pulse"
                : "bg-red-500 shadow-red-500 animate-pulse"
            }`}
          ></span>
          <span className="text-[10px] font-bold text-slate-300 uppercase tracking-widest">
            {healthStatus === "online" ? "SYSTEMS ONLINE" : healthStatus === "checking" ? "DIAGNOSTIC" : "DISCONNECTED"}
          </span>
        </div>
      </header>

      {/* Main Workspace Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto grid gap-6 p-6 md:grid-cols-3 relative z-10">
        {/* Left Column: upload controls and pipeline timeline */}
        <div className="md:col-span-1 space-y-6">
          <UploadPanel
            onProcessFile={handleProcessFile}
            onLoadDemo={handleLoadDemo}
            onReset={handleReset}
            currentJob={currentJob}
            processingState={processingState}
          />
        </div>

        {/* Right Columns: main preview screens and details dashboard */}
        <div className="md:col-span-2">
          <ResultViewer
            currentJob={currentJob}
            processingState={processingState}
            previewUrlFetcher={getPreviewUrl}
            downloadUrlFetcher={getDownloadUrl}
          />
        </div>
      </main>

      {/* Footer Branding */}
      <footer className="glass-panel border-t border-white/5 py-4 text-center text-[10px] text-slate-500 relative z-10 uppercase tracking-widest font-mono">
        Telemetry Service Bind // Pix2Pix Generative Network // ISRO Hackathon Demo UI
      </footer>
    </div>
  );
}
