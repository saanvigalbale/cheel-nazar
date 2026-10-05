/**
 * Cheel Nazar — typed backend API client (Phase 7).
 *
 * Single place that knows the backend base URL and the existing job API
 * contracts. Every function calls the endpoints exactly as implemented in
 * `backend/app/api/v1/endpoints/*` — no backend contracts are changed here.
 */

import type {
  JobReconstruction,
  JobResults,
  JobStatusResponse,
  YoloDetection,
} from '../types/job.ts';

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
 * GET /api/v1/reconstruction/{job_id} — real 3D reconstruction (Phases 8.1-8.5).
 *
 * Always resolves with a payload: when nothing has been reconstructed the
 * backend reports `status: 'not_available'` rather than 404-ing, so the viewer
 * can explain the state instead of erroring.
 */
export async function getJobReconstruction(jobId: string): Promise<JobReconstruction> {
  return request<JobReconstruction>(
    `/api/v1/reconstruction/${encodeURIComponent(jobId)}`,
  );
}

/**
 * Absolute URL for a servable reconstruction artifact (GLB/PLY/NPZ/JSON).
 *
 * The backend returns relative URLs like
 * `/api/v1/reconstruction/<job>/artifact/<name>`; Three.js needs an absolute
 * one, and the API base is configurable via `VITE_API_BASE_URL`, so it must be
 * resolved here rather than hardcoded.
 */
export function reconstructionArtifactUrl(
  artifactPath: string | null | undefined,
): string | null {
  if (!artifactPath) return null;
  if (/^https?:\/\//i.test(artifactPath)) return artifactPath;
  const base = API_BASE_URL.replace(/\/+$/, '');
  const suffix = artifactPath.startsWith('/') ? artifactPath : `/${artifactPath}`;
  return `${base}${suffix}`;
}

/**
 * Whether METRIC geospatial results are usable.
 *
 * This used to key off `model_url == null`, but the backend hardcodes
 * `model_url: null` for every job, so the check was permanently true and every
 * panel read as a placeholder. Metric output is governed by the reconstruction:
 * the cloud is up-to-scale and un-georeferenced, so real-world units are never
 * available regardless of `model_url`.
 */
export function isPlaceholderResults(results: JobResults | null | undefined): boolean {
  if (!results) return true;
  const reconstruction = results.reconstruction;
  if (!reconstruction) return true;
  // Metric values would need both a reconstruction AND a georeference.
  return !reconstruction.georeferenced;
}

/** True when a real reconstructed point cloud exists for this job. */
export function hasReconstruction(
  results: JobResults | null | undefined,
): boolean {
  const reconstruction = results?.reconstruction;
  if (!reconstruction) return false;
  return (
    reconstruction.status === 'success' ||
    reconstruction.status === 'partial'
  );
}
