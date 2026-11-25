// Small wrapper around the Batik Checker API.

import { API_BASE } from "./config";

/**
 * Send an image file to POST /predict.
 * @param {File} file
 * @returns {Promise<object>} the JSON response
 */
export async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }

  return response.json();
}

/** Fetch the list of supported motifs. */
export async function fetchMotifs() {
  const response = await fetch(`${API_BASE}/motifs`);
  if (!response.ok) throw new Error("Failed to load motifs");
  const data = await response.json();
  return data.motifs;
}
