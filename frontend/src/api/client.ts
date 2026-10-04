/**
 * Cheel Nazar — typed backend API client (Phase 7).
 *
 * Single place that knows the backend base URL and the existing job API
 * contracts. Every function calls the endpoints exactly as implemented in
 * `backend/app/api/v1/endpoints/*` — no backend contracts are changed here.
 */

import type { JobResults, JobStatusResponse, YoloDetection } from '../types/job.ts';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

/** Backend base URL. Configure with `VITE_API_BASE_URL`; defaults to local FastAPI. */
export const API_BASE_URL: string = (() => {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (typeof configured === 'string' && configured.trim().length > 0) {
    return configured.trim().replace(/\/+$/, '');
  }
  return DEFAULT_API_BASE_URL;
})();

/** Error thrown for any non-2xx response, or a network failure (status 0). */
export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

export interface HealthResponse {
  status: string;
  app: string;
  version: string;
  timestamp: string;
}

export interface ProcessStartResponse {
  job_id: string;
  status: string;
  message: string;
}

async function request<T>(path: string, method: 'GET' | 'POST' = 'GET'): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, {
      method,
      headers: { Accept: 'application/json' },
    });
  } catch {
    throw new ApiError(0, 'Network error: unable to reach the backend server.');
  }

  if (!response.ok) {
    let detail = `Request failed with status ${response.status}.`;
    try {
      const body: unknown = await response.json();
      const maybe = body as { detail?: unknown };
      if (maybe && typeof maybe.detail === 'string') {
        detail = maybe.detail;
      }
    } catch {
      // Non-JSON error body — keep the generic message.
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

/** GET /health */
export async function healthCheck(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

/** POST /api/v1/jobs/{job_id}/process — returns 202 Accepted when queued. */
export async function startProcessing(jobId: string): Promise<ProcessStartResponse> {
  return request<ProcessStartResponse>(
    `/api/v1/jobs/${encodeURIComponent(jobId)}/process`,
    'POST',
  );
}

/** GET /api/v1/jobs/{job_id}/status */
export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  return request<JobStatusResponse>(`/api/v1/jobs/${encodeURIComponent(jobId)}/status`);
}

/** GET /api/v1/jobs/{job_id}/results — 404 until processing completes. */
export async function getJobResults(jobId: string): Promise<JobResults> {
  return request<JobResults>(`/api/v1/jobs/${encodeURIComponent(jobId)}/results`);
}

/** GET /api/v1/jobs/{job_id}/detections — real YOLO detections; 404 until ready. */
export async function getJobDetections(jobId: string): Promise<YoloDetection[]> {
  return request<YoloDetection[]>(`/api/v1/jobs/${encodeURIComponent(jobId)}/detections`);
}

/**
 * The backend `results.json` is still produced by `_build_mock_results()`
 * (reconstruction + georeferencing are sleep placeholders) and always reports
 * `model_url: null`. Until a real reconstruction writes a model URL, results
 * must be treated as unavailable rather than presented as real measurements.
 */
export function isPlaceholderResults(results: JobResults | null | undefined): boolean {
  return !results || results.model_url == null;
}
