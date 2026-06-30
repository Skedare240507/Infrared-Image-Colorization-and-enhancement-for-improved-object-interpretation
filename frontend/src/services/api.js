const API_BASE = import.meta.env.VITE_API_URL ?? "";

export async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (!res.ok) throw new Error("Health check failed");
    return await res.json();
  } catch (error) {
    console.error("Health check error:", error);
    return { status: "error", message: error.message };
  }
}

export async function uploadThermal(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${API_BASE}/api/v1/thermal/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ message: "Failed to upload file" }));
    throw new Error(errorData.message || "Failed to upload file");
  }
  return await res.json();
}

export async function processThermal(jobId) {
  const res = await fetch(`${API_BASE}/api/v1/thermal/${jobId}/process`, {
    method: "POST",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ message: "Failed to process job" }));
    throw new Error(errorData.message || "Failed to process job");
  }
  return await res.json();
}

export async function getJobStatus(jobId) {
  const res = await fetch(`${API_BASE}/api/v1/thermal/${jobId}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ message: "Failed to fetch job status" }));
    throw new Error(errorData.message || "Failed to fetch job status");
  }
  return await res.json();
}

export async function loadDemoSample() {
  const res = await fetch(`${API_BASE}/api/v1/thermal/sample`, {
    method: "POST",
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({ message: "Failed to generate demo sample" }));
    throw new Error(errorData.message || "Failed to generate demo sample");
  }
  return await res.json();
}

export function getPreviewUrl(jobId, artifact) {
  return `${API_BASE}/api/v1/thermal/${jobId}/preview/${artifact}`;
}

export function getDownloadUrl(jobId, artifact) {
  return `${API_BASE}/api/v1/thermal/${jobId}/download/${artifact}`;
}
