// Central configuration for the frontend.
// In dev, Vite proxies /api -> http://localhost:8000 (see vite.config.js).
export const API_BASE = "/api";

// Accepted image types for the file input.
export const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/bmp", "image/webp"];

// Maximum upload size in bytes (must match the backend).
export const MAX_FILE_SIZE = 5 * 1024 * 1024;
