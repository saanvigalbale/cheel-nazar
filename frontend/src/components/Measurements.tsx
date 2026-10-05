import React from 'react';
import { AlertTriangle, MapPin, Maximize, Mountain, Ruler, Box } from 'lucide-react';
import { hasReconstruction, isPlaceholderResults } from '../api/client.ts';
import type { JobResults } from '../types/job.ts';

interface MeasurementsProps {
  isComplete: boolean;
  results: JobResults | null;
  resultsLoading: boolean;
  resultsError: string | null;
}

const countFormatter = new Intl.NumberFormat('en-US');

function formatMetric(value: number | undefined, unit: string): string {
  if (value == null || Number.isNaN(value)) return '--';
  return `${value.toFixed(2)} ${unit}`;
}

export const Measurements: React.FC<MeasurementsProps> = ({
  isComplete,
  results,
  resultsLoading,
  resultsError,
}) => {
  // The reconstruction is real but UP-TO-SCALE and not georeferenced, so real
  // -world units (m / m²) genuinely do not exist. This is a property of the
  // data, not a missing feature.
  const metricUnavailable = isPlaceholderResults(results);
  const reconstructed = hasReconstruction(results);
  const reconstruction = results?.reconstruction;
  const measurements = metricUnavailable ? undefined : results?.measurements;

  const notice = resultsLoading
    ? 'Loading job results...'
    : resultsError
    ? resultsError
    : !isComplete
    ? 'Reconstruction facts will be available once the pipeline completes.'
    : reconstructed
    ? 'Metric measurements are unavailable: the reconstruction is up-to-scale and this video contains no GPS or flight telemetry, so no real-world units exist.'
    : 'No reconstructed geometry for this job yet.';

  // Real-world units cannot be derived from an up-to-scale, un-georeferenced
  // cloud, so this panel reports reconstruction facts instead of fake m / m².
  const facts = [
    {
      name: 'Cameras Registered',
      icon: Ruler,
      value: countFormatter.format(reconstruction?.n_frames_registered ?? 0),
      desc: 'Frames with a solved global pose',
    },
    {
      name: 'Sparse Points',
      icon: Box,
      value: countFormatter.format(reconstruction?.sparse_point_count ?? 0),
      desc: 'Triangulated SfM points',
    },
    {
      name: 'Denser Points',
      icon: Box,
      value: countFormatter.format(reconstruction?.dense_point_count ?? 0),
      desc: 'Depth-anchored back-projected points',
    },
    {
      name: 'Georeferenced',
      icon: MapPin,
      value: reconstruction?.georeferenced ? 'YES' : 'NO',
      desc: reconstruction?.georeferenced
        ? 'Placed on WGS84 from flight telemetry'
        : 'No GPS/flight telemetry in this video',
    },
  ];

  const tools = [
    {
      name: 'Scene Area',
      icon: Maximize,
      unit: 'm²',
      value: measurements?.area_sq_m,
      desc: 'Footprint area from reconstruction',
    },
    {
      name: 'Scene Width',
      icon: Ruler,
      unit: 'm',
      value: measurements?.scene_width_m,
      desc: 'Reconstructed scene width',
    },
    {
      name: 'Scene Length',
      icon: Ruler,
      unit: 'm',
      value: measurements?.scene_length_m,
      desc: 'Reconstructed scene length',
    },
    {
      name: 'Max Building Height',
      icon: Mountain,
      unit: 'm',
      value: measurements?.max_building_height_m,
      desc: 'Tallest structure height',
    },
  ];

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg backdrop-blur-sm">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Ruler className="w-5 h-5 text-cyan-400" />
          <h2 className="text-base font-semibold text-slate-100 tracking-wide font-mono uppercase">
            Geospatial & Volumetric Measurements
          </h2>
        </div>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300">
          {reconstruction?.georeferenced ? 'METRIC WGS84' : 'UP-TO-SCALE / LOCAL'}
        </span>
      </div>

      {/* Availability notice — never presents non-metric data as metric */}
      <div className="p-3 mb-4 rounded-lg bg-amber-950/30 border border-amber-500/40 flex items-start gap-2.5 text-xs text-amber-300">
        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <span>{notice}</span>
      </div>

      {/* Real reconstruction facts */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
        {facts.map((fact) => {
          const Icon = fact.icon;
          return (
            <div
              key={fact.name}
              className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 hover:border-slate-700 transition-colors flex flex-col justify-between"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono font-medium text-slate-300">{fact.name}</span>
                <Icon className="w-4 h-4 text-cyan-400" />
              </div>
              <div className="my-2">
                <span className="text-xl font-bold font-mono text-slate-100">{fact.value}</span>
              </div>
              <p className="text-[10px] text-slate-500 leading-tight">{fact.desc}</p>
            </div>
          );
        })}
      </div>

      {/* Metric tools — only meaningful once a georeference exists */}
      {tools.some((tool) => tool.value != null) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 mb-4">
          {tools.map((tool) => {
            const Icon = tool.icon;
            return (
              <div
                key={tool.name}
                className="p-3.5 rounded-lg bg-slate-950/60 border border-slate-800 flex flex-col justify-between"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-medium text-slate-300">{tool.name}</span>
                  <Icon className="w-4 h-4 text-cyan-400" />
                </div>
                <div className="my-2">
                  <span className="text-xl font-bold font-mono text-slate-100">
                    {formatMetric(tool.value, tool.unit)}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 leading-tight">{tool.desc}</p>
              </div>
            );
          })}
        </div>
      )}

      {/* Interactive Tool Selector Bar */}
      <div className="flex flex-wrap items-center justify-between gap-3 p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-xs font-mono">
        <div className="flex items-center gap-2 text-slate-400">
          <MapPin className="w-3.5 h-3.5 text-cyan-400" />
          <span>
            TOOL: <span className="text-slate-200">Relative Distance (3D Viewer)</span>
          </span>
        </div>
        <span className="text-amber-400/90 text-[11px]">
          {reconstructed
            ? 'Measure relative spans in reconstruction units using the ruler tool in the 3D viewer. Real-world metres require georeferencing.'
            : 'Available once the job has a reconstructed point cloud.'}
        </span>
      </div>
    </div>
  );
};
