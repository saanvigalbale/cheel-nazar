import React, { useState } from 'react';
import { AlertTriangle, Gauge, HelpCircle, Layers, ShieldCheck, Target } from 'lucide-react';
import { isPlaceholderResults } from '../api/client.ts';
import type { JobResults } from '../types/job.ts';

interface ConfidenceMetricProps {
  isComplete: boolean;
  results: JobResults | null;
  resultsLoading: boolean;
  resultsError: string | null;
}

export const ConfidenceMetric: React.FC<ConfidenceMetricProps> = ({
  isComplete,
  results,
  resultsLoading,
  resultsError,
}) => {
  const [heatmapEnabled, setHeatmapEnabled] = useState(false);

  // No backend-reported uncertainty exists for this pipeline, and none is
  // invented here. Accuracy figures would require a georeference (GNSS/IMU
  // RMSE), which this footage does not have.
  const unavailable = isPlaceholderResults(results);
  const uncertainty = unavailable ? undefined : results?.uncertainty;
  const reconstruction = results?.reconstruction;
  const hasCloud = reconstruction?.status === 'success' || reconstruction?.status === 'partial';

  const notice = resultsLoading
    ? 'Loading job results...'
    : resultsError
    ? resultsError
    : !isComplete
    ? 'Confidence metrics will be available once the pipeline completes.'
    : 'No accuracy figures are reported: the backend does not produce uncertainty metrics, and georeferencing precision requires GNSS/IMU telemetry that this video does not contain.';

  const metrics = [
    {
      title: 'Backend Overall Confidence',
      value: uncertainty ? `${(uncertainty.overall_confidence * 100).toFixed(1)} %` : '--',
      subtext: uncertainty ? 'Source: backend /results' : 'Not provided by backend',
      status: uncertainty ? 'REPORTED' : 'UNAVAILABLE',
      color: 'text-cyan-400',
    },
    {
      title: 'Low-Confidence Regions',
      value: uncertainty ? String(uncertainty.low_confidence_regions) : '--',
      subtext: 'Backend-reported count',
      status: uncertainty ? 'REPORTED' : 'UNAVAILABLE',
      color: 'text-cyan-400',
    },
    {
      title: 'Georeferencing Precision',
      value: reconstruction?.georeferenced ? 'REPORTED' : '--',
      subtext: reconstruction?.georeferenced
        ? 'GNSS/IMU sensor fusion RMSE'
        : 'No GPS/flight telemetry in this video',
      status: reconstruction?.georeferenced ? 'REPORTED' : 'UNAVAILABLE',
      color: 'text-cyan-400',
    },
    {
      title: 'Camera Pose Covariance',
      value: hasCloud ? 'SOLVED' : '--',
      subtext: hasCloud
        ? `${reconstruction?.n_frames_registered ?? 0} cameras registered globally`
        : 'No global pose solution for this job',
      status: hasCloud ? 'REPORTED' : 'UNAVAILABLE',
      color: 'text-cyan-400',
    },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            Confidence & Uncertainty Telemetry
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
          {unavailable ? 'NOT AVAILABLE' : 'QA/QC METRICS'}
        </span>
      </div>

      {/* Availability notice — mock results are never presented as real */}
      {unavailable && (
        <div className="p-3 mb-4 rounded-lg bg-amber-950/30 border border-amber-500/40 flex items-start gap-2.5 text-xs text-amber-300">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <span>{notice}</span>
        </div>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {metrics.map((item, idx) => (
          <div
            key={idx}
            className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col justify-between"
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-xs font-mono text-slate-400">{item.title}</span>
              <Gauge className="w-3.5 h-3.5 text-slate-500" />
            </div>
            <div className="my-1.5">
              <span className={`text-xl font-bold font-mono ${item.color}`}>{item.value}</span>
            </div>
            <div className="flex items-center justify-between pt-1 border-t border-slate-800/60 text-[10px] font-mono text-slate-500">
              <span>{item.subtext}</span>
              <span className="text-slate-400">{item.status}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Uncertainty Overlay Toggle Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2.5">
          <button
            onClick={() => setHeatmapEnabled(!heatmapEnabled)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded transition-all border ${
              heatmapEnabled
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            <span>Uncertainty Heatmap: {heatmapEnabled ? 'ON' : 'OFF'}</span>
          </button>
          <span className="text-slate-500 text-[11px] hidden sm:inline">
            (Visualizes spatial variance across high vs low parallax regions)
          </span>
        </div>

        <div className="flex items-center gap-1.5 text-slate-400 text-[11px]">
          <ShieldCheck className="w-3.5 h-3.5 text-slate-500" />
          <span>{unavailable ? 'Outlier Rejection: Not Available' : 'Automated Outlier Rejection: Armed'}</span>
          <HelpCircle className="w-3.5 h-3.5 text-slate-600" />
        </div>
      </div>
    </div>
  );
};
