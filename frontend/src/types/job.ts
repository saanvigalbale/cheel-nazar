/**
 * Shared TypeScript types for the Cheel Nazar job pipeline — Phase 6A.
 * These mirror the backend response shapes from jobs.py / ai/results_builder.py.
 */

// ─── Job Status ────────────────────────────────────────────────────────────────

export type JobStatusValue =
  | 'uploaded'
  | 'queued'
  | 'extracting_frames'
  | 'detecting_objects'
  | 'segmenting_scene'
  | 'estimating_depth'
  | 'analyzing_hazards'
  | 'reconstructing'
  | 'georeferencing'
  | 'done'
  | 'failed';

/** Shape returned by GET /api/v1/jobs/{job_id}/status */
export interface JobStatusResponse {
  job_id: string;
  status: JobStatusValue;
  progress: number;             // 0–100
  error: string | null;
  frames_extracted?: number;
  detections_count?: number;
  segmentation_frames_count?: number;
  depth_maps_count?: number;
  hazards_analyzed?: string[];
}

// ─── YOLO Detections ───────────────────────────────────────────────────────────

/** Single entry from GET /api/v1/jobs/{job_id}/detections */
export interface YoloDetection {
  frame: string;
  label: string;
  confidence: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

// ─── Job Results (Phase 6A, real outputs only) ─────────────────────────────────

/** Aggregated real YOLO detections (results.categories.people / vehicles). */
export interface DetectedCategory {
  nature: 'detected';
  count: number;
  avg_confidence: number | null;
  max_confidence: number | null;
  min_confidence: number | null;
  frames_with_detections: number;
  source: string;
  breakdown: Record<string, number>;
}

/** Semantic coverage from SegFormer (results.categories.*). */
export interface SemanticCategory {
  nature: 'semantic_estimate';
  coverage_pct: number;
  source: string;
  disclaimer?: string;
}

/** Relative (non-metric) depth summary from Depth Anything V2. */
export interface RelativeDepthInfo {
  nature: 'relative';
  metric_depth_available: boolean;
  source: string;
  maps_generated?: number;
  units?: string;
  mean_relative_depth?: number | null;
  min_relative_depth?: number | null;
  max_relative_depth?: number | null;
  disclaimer?: string;
}

// ─── Disaster intelligence (Phase 6B.1) ─────────────────────────────────────────

/** Relative (non-metric) depth evidence for a hazard finding. */
export interface FloodRelativeDepthEvidence {
  frames_with_water_depth_evidence: number;
  mean_relative_depth_in_water: number | null;
  mean_relative_depth_scene: number | null;
  mean_within_frame_depth_separation: number | null;
  metric_depth_available: boolean;
  within_frame_only: boolean;
  interpretation?: string;
}

/** Temporal trend label emitted by the hazard analyzer. */
export interface FloodTemporalTrend {
  label: string;                       // increasing | stable | decreasing | insufficient_data
  frames_compared?: number;
  first_half_mean_water_pct?: number;
  second_half_mean_water_pct?: number;
  absolute_change_pct_points?: number;
  relative_change?: number | null;
  relative_change_threshold?: number;
  min_frames_required?: number;
  note?: string;
}

/**
 * Aggregate of a flood proxy analysis.
 * Fields are optional because the analyzer returns a reduced payload
 * (`frames_analyzed: 0`, `note`) when no artifacts are available.
 */
export interface FloodAggregate {
  frames_analyzed: number;
  note?: string;
  water_extent_pct?: number;
  water_extent_semantics?: string;
  water_present_frames?: number;
  water_present_frame_ratio?: number;
  mean_water_coverage_pct?: number;
  median_water_coverage_pct?: number;
  max_water_coverage_pct?: number;
  road_coverage_pct?: number;
  /**
   * Spatial-proximity proxy (Phase 6B.3): share of road-classified pixels that
   * lie within the configured water buffer. Null when no road pixels exist.
   * NOT a confirmed flooded-road measurement.
   */
  possible_water_near_road_pct?: number | null;
  road_pixels_near_water?: number;
  water_near_road_semantics?: string;
  people_frame_instances_in_water?: number;
  people_frame_instances_near_water?: number;
  people_in_water_breakdown?: Record<string, number>;
  people_near_water_breakdown?: Record<string, number>;
  vehicles_frame_instances_in_water?: number;
  vehicles_frame_instances_near_water?: number;
  vehicles_in_water_breakdown?: Record<string, number>;
  vehicles_near_water_breakdown?: Record<string, number>;
  count_semantics?: string;
  relative_depth_evidence?: FloodRelativeDepthEvidence;
  temporal_trend?: FloodTemporalTrend;
  confidence: null;                    // never invented
}

/** One hazard finding under results.disasters.* (currently only `flood`). */
export interface DisasterFloodFinding {
  hazard: string;
  nature: string;
  job_id: string;
  source: string;
  segmentation_model: string | null;
  confidence: null;                    // never invented
  confidence_note: string;
  count_semantics: string;
  parameters: Record<string, number>;
  disclaimer: string;
  not_measured: string[];
  aggregate: FloodAggregate;
  /** Large per-frame evidence records; present but not rendered. */
  frames?: unknown[];
}

export interface Disasters {
  flood?: DisasterFloodFinding;
  [hazard: string]: DisasterFloodFinding | undefined;
}

/** Shape returned by GET /api/v1/jobs/{job_id}/results */
export interface JobResults {
  job_id: string;
  models: {
    detector: string;
    segmentation: string;
    depth: string;
  };
  categories: {
    people: DetectedCategory;
    vehicles: DetectedCategory;
    structures: SemanticCategory;
    roads: SemanticCategory;
    vegetation: SemanticCategory;
    terrain: SemanticCategory;
    possible_water_extent: SemanticCategory;
  };
  relative_depth?: RelativeDepthInfo;
  disasters?: Disasters;
  disaster_modules_available?: string[];
  disaster_modules_not_implemented?: string[];
  not_measured: string[];
  model_url: string | null;

  // ─── Legacy / optional (no longer produced; kept for guardrail compatibility)
  georeference?: { lat: number; lon: number; altitude_m: number };
  measurements?: {
    area_sq_m: number;
    max_building_height_m: number;
    scene_width_m: number;
    scene_length_m: number;
  };
  damage_analysis?: { damaged_structures: number; severity: string; affected_area_pct: number };
  uncertainty?: { overall_confidence: number; low_confidence_regions: number };
}
