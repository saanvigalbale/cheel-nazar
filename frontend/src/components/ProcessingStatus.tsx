import React from 'react';
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Loader2,
  Play,
  Terminal,
  XCircle,
} from 'lucide-react';
import type { JobStatusResponse, JobStatusValue } from '../types/job.ts';

type StageState = 'pending' | 'active' | 'completed' | 'failed';

interface Stage {
  id: string;
  name: string;
  module: string;
  description: string;
  /** Backend statuses that map to this stage. */
  match: JobStatusValue[];
  /** Approx. progress value at which this stage begins (used for failure attribution). */
  enterProgress: number;
}

interface ProcessingStatusProps {
  jobStatus: JobStatusResponse | null;
  hasActiveJob: boolean;
  isStarting: boolean;
  isPolling: boolean;
  errorMessage: string | null;
  onStartProcessing: () => void;
}

const PIPELINE_ORDER: JobStatusValue[] = [
  'uploaded',
  'queued',
  'extracting_frames',
  'detecting_objects',
  'segmenting_scene',
  'estimating_depth',
  'analyzing_hazards',
  'reconstructing',
  'georeferencing',
  'done',
];

const STAGES: Stage[] = [
  {
    id: '01',
    name: 'Upload & Job Queue',
    module: 'Ingestion',
    description: 'Validate uploaded drone footage and stage the processing job',
    match: ['uploaded', 'queued'],
    enterProgress: 0,
  },
  {
    id: '02',
    name: 'Frame Extraction',
    module: 'OpenCV',
    description: 'Sample representative frames from the drone video for analysis',
    match: ['extracting_frames'],
    enterProgress: 10,
  },
  {
    id: '03',
    name: 'Object & Hazard Detection',
    module: 'YOLO11n',
    description: 'Detect objects, vehicles and structural hazards in every frame',
    match: ['detecting_objects'],
    enterProgress: 35,
  },
  {
    id: '04',
    name: 'Semantic Scene Segmentation',
    module: 'SegFormer-B0 (ADE20K)',
    description: 'Per-pixel scene classes mapped to structures, roads, vegetation, terrain, water proxy',
    match: ['segmenting_scene'],
    enterProgress: 60,
  },
  {
    id: '05',
    name: 'Monocular Depth Estimation',
    module: 'Depth Anything V2 (relative)',
    description: 'Estimate relative depth maps for each extracted frame (not metric depth)',
    match: ['estimating_depth'],
    enterProgress: 78,
  },
  {
    id: '06',
    name: 'Disaster Intelligence',
    module: 'ai.disaster registry (proxy indicators)',
    description: 'Flood / hazard analysis over existing artifacts — estimates only, no model re-run',
    match: ['analyzing_hazards'],
    enterProgress: 91,
  },
  {
    id: '07',
    name: '3D Reconstruction',
    module: 'Placeholder (not implemented)',
    description: 'Sparse & dense reconstruction — not yet implemented in this prototype',
    match: ['reconstructing'],
    enterProgress: 94,
  },
  {
    id: '08',
    name: 'Georeferencing & Export',
    module: 'Placeholder (not implemented)',
    description: 'Projection to UTM/WGS84 and export — not yet implemented in this prototype',
    match: ['georeferencing'],
    enterProgress: 97,
  },
];

const STATUS_LABELS: Record<JobStatusValue, string> = {
  uploaded: 'UPLOADED',
  queued: 'QUEUED',
  extracting_frames: 'EXTRACTING FRAMES',
  detecting_objects: 'DETECTING OBJECTS',
  segmenting_scene: 'SEGMENTING SCENE',
  estimating_depth: 'ESTIMATING DEPTH',
  analyzing_hazards: 'ANALYZING HAZARDS',
  reconstructing: 'RECONSTRUCTING',
  georeferencing: 'GEOREFERENCING',
  done: 'COMPLETED',
  failed: 'FAILED',
};

function failedStageIndex(progress: number): number {
  let index = 0;
  STAGES.forEach((stage, i) => {
    if (progress >= stage.enterProgress) index = i;
  });
  return index;
}

function resolveStageState(stage: Stage, status: JobStatusValue | null, progress: number): StageState {
  if (!status) return 'pending';
  if (status === 'done') return 'completed';
  if (status === 'failed') {
    const failedIndex = failedStageIndex(progress);
    const index = STAGES.indexOf(stage);
    if (index < failedIndex) return 'completed';
    if (index === failedIndex) return 'failed';
    return 'pending';
  }
  if (stage.match.includes(status)) return 'active';
  const current = PIPELINE_ORDER.indexOf(status);
  const stageOrder = PIPELINE_ORDER.indexOf(stage.match[0]);
  return current > stageOrder ? 'completed' : 'pending';
}

function stageIcon(state: StageState) {
  switch (state) {
    case 'completed':
      return <CheckCircle2 className="w-4 h-4 text-emerald-400" />;
    case 'active':
      return <Loader2 className="w-4 h-4 text-cyan-400 animate-spin" />;
    case 'failed':
      return <XCircle className="w-4 h-4 text-rose-400" />;
    default:
      return <CircleDashed className="w-4 h-4 text-slate-600" />;
  }
}

export const ProcessingStatus: React.FC<ProcessingStatusProps> = ({
  jobStatus,
  hasActiveJob,
  isStarting,
  isPolling,
  errorMessage,
  onStartProcessing,
}) => {
  const status: JobStatusValue | null = jobStatus?.status ?? null;
  const progress = jobStatus?.progress ?? 0;

  const badgeLabel = status ? STATUS_LABELS[status] : 'STANDBY / IDLE';
  const badgeClasses =
    status === 'failed'
      ? 'text-rose-300 bg-rose-950/40 border-rose-500/40'
      : status === 'done'
      ? 'text-emerald-300 bg-emerald-950/40 border-emerald-500/40'
      : status
      ? 'text-cyan-300 bg-cyan-950/40 border-cyan-500/40'
      : 'text-amber-400 bg-amber-950/40 border-amber-500/30';

  const badgeIcon =
    status === 'failed' ? (
      <XCircle className="w-3 h-3" />
    ) : status === 'done' ? (
      <CheckCircle2 className="w-3 h-3" />
    ) : isPolling ? (
      <Loader2 className="w-3 h-3 animate-spin" />
    ) : (
      <CircleDashed className="w-3 h-3" />
    );

  const startDisabled = !hasActiveJob || isStarting || isPolling || status === 'done';

  const logLines: { level: string; levelClass: string; text: string }[] = [];
  if (!hasActiveJob) {
    logLines.push({
      level: 'WAIT',
      levelClass: 'text-slate-400',
      text: 'Awaiting drone video ingestion from operator...',
    });
  } else {
    logLines.push({
      level: 'INFO',
      levelClass: 'text-emerald-400',
      text: `Job ${jobStatus?.job_id ?? ''} staged for processing.`,
    });
    if (status) {
      logLines.push({
        level: 'STAT',
        levelClass: 'text-cyan-400',
        text: `Pipeline status: ${STATUS_LABELS[status]} (${progress}%).`,
      });
    }
    if (jobStatus?.frames_extracted != null) {
      logLines.push({
        level: 'INFO',
        levelClass: 'text-emerald-400',
        text: `Frames extracted: ${jobStatus.frames_extracted}.`,
      });
    }
    if (jobStatus?.detections_count != null) {
      logLines.push({
        level: 'INFO',
        levelClass: 'text-emerald-400',
        text: `Detections recorded: ${jobStatus.detections_count}.`,
      });
    }
    if (jobStatus?.depth_maps_count != null) {
      logLines.push({
        level: 'INFO',
        levelClass: 'text-emerald-400',
        text: `Depth maps generated: ${jobStatus.depth_maps_count}.`,
      });
    }
    if (status === 'failed') {
      logLines.push({
        level: 'ERR',
        levelClass: 'text-rose-400',
        text: errorMessage ?? 'Processing failed.',
      });
    } else if (status === 'done') {
      logLines.push({
        level: 'DONE',
        levelClass: 'text-emerald-400',
        text: 'Pipeline completed successfully.',
      });
    }
  }

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            Processing Status
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center gap-1.5 text-xs font-mono px-2 py-0.5 rounded border ${badgeClasses}`}
          >
            {badgeIcon}
            {badgeLabel}
          </span>
        </div>
      </div>

      {/* Real progress bar */}
      {jobStatus && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-[11px] font-mono text-slate-400 mb-1">
            <span>
              JOB: <span className="text-cyan-400">{jobStatus.job_id.substring(0, 8)}...</span>
            </span>
            <span className="text-slate-300">{progress}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden border border-slate-700">
            <div
              className={`h-full transition-all duration-300 ${
                status === 'failed'
                  ? 'bg-rose-500'
                  : status === 'done'
                  ? 'bg-emerald-400'
                  : 'bg-gradient-to-r from-cyan-500 to-emerald-400'
              }`}
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Real pipeline counters (shown only when provided by the backend) */}
      {(jobStatus?.frames_extracted != null ||
        jobStatus?.detections_count != null ||
        jobStatus?.depth_maps_count != null ||
        (jobStatus?.hazards_analyzed?.length ?? 0) > 0) && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4 text-center font-mono text-xs">
          <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
            <div className="text-slate-500 text-[10px]">FRAMES</div>
            <div className="text-slate-100 font-semibold">{jobStatus?.frames_extracted ?? '--'}</div>
          </div>
          <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
            <div className="text-slate-500 text-[10px]">DETECTIONS</div>
            <div className="text-slate-100 font-semibold">{jobStatus?.detections_count ?? '--'}</div>
          </div>
          <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
            <div className="text-slate-500 text-[10px]">DEPTH MAPS</div>
            <div className="text-slate-100 font-semibold">{jobStatus?.depth_maps_count ?? '--'}</div>
          </div>
          <div className="p-2 rounded bg-slate-950/60 border border-slate-800">
            <div className="text-slate-500 text-[10px]">HAZARDS</div>
            <div className="text-slate-100 font-semibold">
              {jobStatus?.hazards_analyzed?.length ?? '--'}
            </div>
          </div>
        </div>
      )}

      {/* Start / status action row */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-4">
        <span className="text-[11px] font-mono text-slate-500">
          {!hasActiveJob
            ? 'Upload a drone video to begin.'
            : status === 'done'
            ? 'Processing complete.'
            : isPolling
            ? `Processing... (${progress}%)`
            : 'Ready to begin pipeline processing.'}
        </span>
        <button
          onClick={onStartProcessing}
          disabled={startDisabled}
          className={`px-4 py-2 rounded-lg font-mono text-xs font-semibold uppercase tracking-wider transition-all flex items-center justify-center gap-2 ${
            startDisabled
              ? 'bg-slate-800 border border-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950 shadow-[0_0_15px_rgba(6,182,212,0.3)] active:scale-95'
          }`}
        >
          {isStarting ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
          <span>{isStarting ? 'Starting...' : status === 'done' ? 'Completed' : 'Start Processing'}</span>
        </button>
      </div>

      {/* Error banner */}
      {errorMessage && (
        <div className="mb-4 p-3 rounded-lg bg-rose-950/40 border border-rose-500/50 flex items-start gap-2.5 text-xs text-rose-300">
          <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
          <div className="flex-1 font-mono">
            <span className="font-semibold block mb-0.5">PIPELINE ERROR</span>
            <span>{errorMessage}</span>
          </div>
        </div>
      )}

      {/* Stepper Pipeline */}
      <div className="space-y-3">
        {STAGES.map((stage) => {
          const state = resolveStageState(stage, status, progress);
          const cardClasses =
            state === 'active'
              ? 'bg-cyan-950/20 border-cyan-500/40'
              : state === 'failed'
              ? 'bg-rose-950/30 border-rose-500/50'
              : 'bg-slate-950/40 border-slate-800/80 hover:border-slate-700';
          const idClasses =
            state === 'completed'
              ? 'text-emerald-400 border-emerald-500/40'
              : state === 'active'
              ? 'text-cyan-400 border-cyan-500/40'
              : state === 'failed'
              ? 'text-rose-400 border-rose-500/40'
              : 'text-slate-400 border-slate-700';

          return (
            <div
              key={stage.id}
              className={`flex items-start gap-3 p-3 rounded-lg border transition-colors ${cardClasses}`}
            >
              <div
                className={`flex items-center justify-center w-7 h-7 rounded-md bg-slate-900 border font-mono text-xs font-bold shrink-0 mt-0.5 ${idClasses}`}
              >
                {stage.id}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-1">
                  <span className="text-sm font-medium text-slate-200">{stage.name}</span>
                  <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-cyan-400 w-fit">
                    {stage.module}
                  </span>
                </div>
                <p className="text-xs text-slate-500">{stage.description}</p>
              </div>

              <div className="shrink-0 flex items-center">{stageIcon(state)}</div>
            </div>
          );
        })}
      </div>

      {/* Execution Console Log */}
      <div className="mt-4 p-3 rounded-lg bg-slate-950 border border-slate-800 font-mono text-xs text-slate-400">
        <div className="flex items-center justify-between text-slate-500 text-[11px] pb-1 border-b border-slate-800 mb-2">
          <div className="flex items-center gap-1.5">
            <Terminal className="w-3.5 h-3.5 text-cyan-400" />
            <span>MISSION EXECUTION LOG</span>
          </div>
          <span>[LIVE STATUS]</span>
        </div>
        {logLines.map((line, idx) => (
          <p key={idx} className="text-slate-500">
            <span className={line.levelClass}>[{line.level}]</span> {line.text}
          </p>
        ))}
      </div>
    </div>
  );
};
